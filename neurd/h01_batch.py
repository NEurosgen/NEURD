"""Resumable H01 branch resegmentation into a separate, versioned dataset.

Source meshes are never modified. A local geometric axis is distinct from a
topological skeleton: branch replacements must retain all original endpoints
and junctions. Failed repairs retain the original graph and an auxiliary axis.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from threading import Event, Lock
import uuid

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_json(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def dependency_signature():
    versions = {}
    for name in ('numpy','scipy','trimesh','networkx','meshparty','mesh-processing-tools','datasci-stdlib-tools'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    files = {}
    for name in ('cgal_Segmentation_Module','mesh_tools'):
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.origin:
            raise RuntimeError(f'Missing required dependency: {name}')
        path = Path(spec.origin)
        files[str(path)] = sha(path)
        if name=='mesh_tools':
            for basename in ('mesh_utils.py','trimesh_utils.py','skeleton_utils.py','meshparty_skeletonize.py'):
                module = path.parent/basename
                if module.is_file():
                    files[str(module)] = sha(module)
    return dict(python=sys.version,executable=sys.executable,versions=versions,source_files=files)


def save(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def separate_roots(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Input and output roots must be separate, non-nested directories')
    return source, output


def inventory(source):
    rows = []
    for mesh in sorted(Path(source).glob('neuron_*/limb_*/branch_*/branch_mesh.off')):
        branch, limb, neuron = mesh.parent.name, mesh.parent.parent.name, mesh.parent.parent.parent.name
        skeleton = mesh.with_name('branch_skeleton.npy')
        rows.append(dict(key=f'{neuron}_{limb}_{branch}', relative=str(mesh.parent.relative_to(source)),
                         source=str(mesh.resolve()), skeleton=str(skeleton.resolve()),
                         archived_spines=sum(1 for _ in (mesh.parent / 'spines').glob('*.off')),
                         skeleton_exists=skeleton.is_file()))
    if not rows:
        raise ValueError('No exported H01 branch meshes found')
    return rows


def inputs(row):
    mesh = Path(row['source'])
    files = dict(mesh=mesh, skeleton=Path(row['skeleton']))
    soma = mesh.parents[2] / 'soma/soma_mesh.off'
    if soma.is_file():
        files['soma'] = soma
    return {name:dict(path=str(path), sha256=sha(path)) for name,path in files.items()}


def coverage(vertices, segments, sample=2048):
    """Uniform vertex coverage screen; point spacing adds at most 100 nm."""
    import numpy as np
    from scipy.spatial import cKDTree
    points = np.asarray(vertices)
    segments = np.asarray(segments, dtype=float)
    if segments.ndim != 3 or segments.shape[1:] != (2,3) or not np.isfinite(segments).all():
        raise ValueError('Invalid skeleton segments')
    if not len(points) or not len(segments):
        raise ValueError('Empty mesh or skeleton')
    cloud = np.concatenate([np.linspace(a,b,max(2,int(np.ceil(np.linalg.norm(b-a)/200))+1)) for a,b in segments])
    ids = np.unique(np.linspace(0,len(points)-1,min(sample,len(points)),dtype=int))
    distance = cKDTree(cloud).query(points[ids], workers=1)[0]
    return dict(q95_nm=float(np.percentile(distance,95)), fraction_over_10um=float(np.mean(distance>10000)),
                sampled_vertices=len(ids), segments=len(segments),
                length_nm=float(np.linalg.norm(segments[:,1]-segments[:,0],axis=1).sum()))


def graph_data(segments):
    import numpy as np
    segments = np.asarray(segments, dtype=float)
    if segments.ndim != 3 or segments.shape[1:] != (2,3) or not len(segments):
        raise ValueError('Invalid or empty skeleton segments')
    if not np.isfinite(segments).all():
        raise ValueError('Nonfinite skeleton coordinates')
    vertices, inverse = np.unique(segments.reshape(-1,3),axis=0,return_inverse=True)
    edges = inverse.reshape(-1,2)
    if np.any(edges[:,0] == edges[:,1]):
        raise ValueError('Zero length skeleton edges')
    degree = np.bincount(edges.ravel(),minlength=len(vertices))
    return vertices, edges, degree


def _landmark_topology(segments):
    """Contract degree-2 paths, retaining exact landmark labels and multiedges.

    Edge identities matter: sets would hide parallel paths and cycles returning
    to the same landmark. Components with no landmarks are recorded separately;
    anchoring conservatively rejects these cycles rather than inventing labels.
    """
    vertices, edges, degree = graph_data(segments)
    adjacency = [[] for _ in vertices]
    for edge_id, (a,b) in enumerate(edges):
        adjacency[a].append(edge_id)
        adjacency[b].append(edge_id)
    component_count, seen_vertices = 0, set()
    for start in range(len(vertices)):
        if start in seen_vertices:
            continue
        component_count += 1
        queue = [start]
        seen_vertices.add(start)
        while queue:
            vertex = queue.pop()
            for edge_id in adjacency[vertex]:
                a,b = edges[edge_id]
                neighbor = int(b if a == vertex else a)
                if neighbor not in seen_vertices:
                    seen_vertices.add(neighbor)
                    queue.append(neighbor)
    labels = {i:tuple(float(x) for x in point) for i,point in enumerate(vertices) if degree[i] != 2}
    paths, used_edges = [], set()
    for start in labels:
        for edge_id in adjacency[start]:
            if edge_id in used_edges:
                continue
            current = start
            while True:
                used_edges.add(edge_id)
                a,b = edges[edge_id]
                current = int(b if a == current else a)
                if current in labels:
                    paths.append(tuple(sorted((labels[start],labels[current]))))
                    break
                edge_id = next(i for i in adjacency[current] if i != edge_id)
    unlabelled_cycles = 0
    for first in range(len(edges)):
        if first in used_edges:
            continue
        unlabelled_cycles += 1
        queue = [first]
        used_edges.add(first)
        while queue:
            edge_id = queue.pop()
            for vertex in edges[edge_id]:
                for neighbor in adjacency[vertex]:
                    if neighbor not in used_edges:
                        used_edges.add(neighbor)
                        queue.append(neighbor)
    return dict(landmarks=tuple((labels[i],int(degree[i])) for i in labels),
                paths=tuple(sorted(paths)),components=component_count,
                cycle_rank=len(edges)-len(vertices)+component_count,
                unlabelled_cycles=unlabelled_cycles)


def triangle_keys(mesh):
    """Exact triangle geometry, independent of face/vertex order and winding."""
    import numpy as np
    rows = np.asarray(mesh.triangles,dtype=np.float64).copy().view(np.dtype((np.void,24))).reshape(-1,3)
    rows.sort(axis=1)
    return np.sort(rows.copy().view(np.dtype((np.void,72))).reshape(-1))


def anchor_skeleton(original, fresh, maximum_connector_nm=2000):
    """Prune a fresh tree to original graph landmarks, retaining exact positions.

    Return None if an endpoint/junction lies far from the fresh axis, if either
    graph is disconnected/cyclic, or if the labelled landmark topology changes.
    Geometry coverage is checked separately after anchoring, before acceptance.
    """
    import numpy as np
    old_v, _, old_d = graph_data(original)
    old_topology = _landmark_topology(original)
    if old_topology['components'] != 1 or old_topology['cycle_rank'] != 0:
        return None, 'original_graph_is_not_a_connected_tree'
    fresh_topology = _landmark_topology(fresh)
    if fresh_topology['components'] != 1 or fresh_topology['cycle_rank'] != 0:
        return None, 'fresh_axis_is_not_a_connected_tree'
    verts, edges, _ = graph_data(fresh)
    landmarks = np.flatnonzero(old_d != 2)
    if len(landmarks) < 2:
        return None, 'no_original_graph_landmarks'
    verts, edges = list(verts), [tuple(e) for e in edges]
    anchors = {}
    for i in landmarks:
        point = old_v[i]
        a = np.asarray([verts[e[0]] for e in edges]); b = np.asarray([verts[e[1]] for e in edges])
        vector = b-a
        t = np.clip(np.einsum('ij,ij->i',point-a,vector)/np.einsum('ij,ij->i',vector,vector),0,1)
        projections = a + t[:,None]*vector
        distance = np.linalg.norm(projections-point,axis=1)
        j = int(np.argmin(distance))
        if distance[j] > maximum_connector_nm:
            return None, 'original_graph_landmark_far_from_fresh_axis'
        exact = [k for k,v in enumerate(verts) if np.array_equal(point,v)]
        if exact:
            anchor = exact[0]
        else:
            u,v = edges[j]
            if t[j] <= 1e-12:
                join = u
            elif t[j] >= 1-1e-12:
                join = v
            else:
                join = len(verts); verts.append(projections[j])
                edges.pop(j); edges += [(u,join),(join,v)]
            if distance[j] <= 1e-10:
                # A landmark already on the new axis must not introduce a
                # zero-length connector when splitting that axis segment.
                verts[join] = point
                anchor = join
            else:
                anchor = len(verts); verts.append(point)
                edges.append((join,anchor))
        anchors[anchor] = int(old_d[i])
    adjacency = {i:set() for i in range(len(verts))}
    for a,b in edges:
        adjacency[a].add(b); adjacency[b].add(a)
    queue = [i for i,n in adjacency.items() if len(n)==1 and i not in anchors]
    while queue:
        i = queue.pop()
        if len(adjacency[i]) != 1 or i in anchors:
            continue
        j = adjacency[i].pop(); adjacency[j].remove(i)
        if len(adjacency[j]) == 1 and j not in anchors:
            queue.append(j)
    if any(len(adjacency[i]) != degree for i,degree in anchors.items()):
        return None, 'anchoring_changes_original_landmark_degree'
    kept = [(a,b) for a,n in adjacency.items() for b in n if a < b]
    if not kept:
        return None, 'anchoring_collapses_axis'
    result = np.asarray(verts)[np.asarray(kept)]
    if _landmark_topology(result) != old_topology:
        return None, 'anchoring_changes_original_landmark_topology'
    return result, 'exact_original_graph_landmark_topology_preserved'


def repair_skeleton(parent, original, settings):
    import numpy as np
    before = coverage(parent.vertices, original)
    record = dict(status='original_covered', before=before, topological_replacement=False)
    # Short skeletons on genuinely tiny meshes are not a coverage failure.
    flagged = before['q95_nm'] > settings['coverage_q95_nm']
    if not flagged or not settings['repair']:
        if flagged:
            record['status'] = 'repair_disabled'
        return original, None, record
    from mesh_tools import meshparty_skeletonize as msk
    started = time.perf_counter()
    try:
        obj = msk.skeletonize_mesh_largest_component(parent,filter_mesh=False,
                 invalidation_d=settings['invalidation_d_nm'],smooth_neighborhood=1)
        fresh = np.asarray(obj.vertices)[np.asarray(obj.edges)]
        after = coverage(parent.vertices,fresh)
        record.update(after=after,build_seconds=time.perf_counter()-started)
        if after['q95_nm'] >= .5*before['q95_nm'] or after['q95_nm'] > settings['coverage_q95_nm']:
            record.update(status='repair_rejected',reason='coverage_did_not_improve_sufficiently')
            return original, None, record
        if not settings.get('allow_topological_replacement', False):
            record.update(status='auxiliary_axis_only',reason='topological_replacement_disabled')
            return original, fresh, record
        anchored, reason = anchor_skeleton(original,fresh,settings['maximum_connector_nm'])
        record['topology_reason'] = reason
        if anchored is not None:
            anchored_quality = coverage(parent.vertices,anchored)
            record['anchored_coverage'] = anchored_quality
            if anchored_quality['q95_nm'] < .5*before['q95_nm'] and anchored_quality['q95_nm'] <= settings['coverage_q95_nm']:
                record.update(status='repaired_with_original_connections',topological_replacement=True)
                return anchored, fresh, record
        record.update(status='auxiliary_axis_only',reason='original_graph_retained')
        return original, fresh, record
    except Exception as exc:
        record.update(status='repair_failed',reason=f'{type(exc).__name__}: {str(exc)[:300]}',
                      build_seconds=time.perf_counter()-started)
        return original, None, record


def complete_valid(dest, run_hash, source_inputs):
    result_path = Path(dest) / 'result.json'
    if not result_path.is_file():
        return False
    record = json.loads(result_path.read_text())
    if not isinstance(record, dict):
        raise ValueError(f'Malformed completion record: {result_path}')
    if record.get('status') != 'complete' or record.get('run_hash') != run_hash or record.get('inputs') != source_inputs:
        raise ValueError(f'Existing result has changed inputs or settings: {dest}')
    outputs = record.get('outputs_sha256')
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError(f'Completion record requires a nonempty output hash map: {result_path}')
    excluded = {'run.log', 'result.json'}
    for relative, expected in outputs.items():
        if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
                or '..' in Path(relative).parts or Path(relative).name in excluded
                or not isinstance(expected, str) or len(expected) != 64
                or any(character not in '0123456789abcdef' for character in expected)):
            raise ValueError(f'Malformed completed output hash entry: {result_path}')
        path = Path(dest)/relative
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'Completed output is missing or changed: {path}')
    actual = {str(path.relative_to(dest)) for path in Path(dest).rglob('*')
              if (path.is_file() or path.is_symlink()) and path.name not in excluded}
    if set(outputs) != actual:
        raise ValueError(f'Completed output hash inventory is incomplete or changed: {dest}')
    return True


def dispatch(rows,job,workers,stop,on_result):
    """Bound queued work and stop submitting branches after interruption."""
    rows = iter(rows)
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        for _ in range(workers):
            row = next(rows,None)
            if row is not None:
                pending.add(pool.submit(job,row))
        while pending:
            done,pending = wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result();results.append(result);on_result(result,len(results))
                row = None if stop.is_set() else next(rows,None)
                if row is not None:
                    pending.add(pool.submit(job,row))
    return results


def _record_width_failure(staging, manifest, row, source_inputs, parent,
                          original, skeleton, auxiliary, skeleton_qc,
                          distances, width, error=None):
    """Best-effort evidence, written before unresolved eligibility is raised."""
    import numpy as np
    errors, artifacts, axes = [], {}, {}
    def attempt(label, operation):
        try:
            return operation()
        except Exception as exc:
            errors.append(dict(artifact=label, error=f'{type(exc).__name__}: {str(exc)[:300]}'))
            return None
    def store_array(filename, values):
        np.save(staging/filename, values, allow_pickle=False)
        artifacts[filename] = sha(staging/filename)
        return dict(file=filename, sha256=artifacts[filename], shape=list(np.shape(values)),
                    dtype=str(np.asarray(values).dtype))
    def store_json(filename, value):
        save(staging/filename, value)
        artifacts[filename] = sha(staging/filename)

    # Prioritize the actual axis passed to the failed width calculation. Each
    # write is independent, so one unavailable artifact does not hide the rest.
    axes['width_eligibility'] = attempt('branch_skeleton.npy', lambda:
        store_array('branch_skeleton.npy', skeleton))
    axes['original'] = attempt('branch_skeleton_original.npy', lambda:
        store_array('branch_skeleton_original.npy', original))
    axes['auxiliary'] = (attempt('segmentation_axis.npy', lambda:
        store_array('segmentation_axis.npy', auxiliary)) if auxiliary is not None else None)
    attempt('skeleton_repair.json', lambda: store_json('skeleton_repair.json', skeleton_qc))
    observed = dict(average_nm=None, average_state='not_returned',
                    values_source='exact_postprocessed_calculate_new_width_return',
                    active_backend='owned' if os.environ.get('NEURD_OWNED_SKELETAL_DISTANCE') == '1' else 'mesh_tools')
    if width is not None:
        value = float(width)
        observed.update(average_nm=value if np.isfinite(value) else None,
                        average_state='nonfinite' if not np.isfinite(value) else
                                      'zero' if value == 0 else 'nonpositive' if value < 0 else 'positive')
    if distances is not None:
        values = np.asarray(distances)
        observed.update(returned_distance_count=int(values.size),
                        positive_finite_count=int(np.count_nonzero(np.isfinite(values) & (values > 0))),
                        zero_count=int(np.count_nonzero(values == 0)),
                        nonfinite_count=int(np.count_nonzero(~np.isfinite(values))),
                        returned_distances=attempt('width_distances.npy', lambda:
                            store_array('width_distances.npy', values)))
    eligibility = dict(eligible=None, reason='eligibility_unresolved',
                       threshold_nm=115, comparison='>', median_mesh_center_nm=observed['average_nm'])
    evidence = dict(schema_version=1, status='unresolved', units='nm',
                    branch_eligibility=eligibility, observed_width=observed,
                    axes=axes, width_axis='width_eligibility',
                    auxiliary_used_for_width=False, skeleton_repair_file='skeleton_repair.json',
                    width_parameters=dict(no_spines=False, summary_measure='median',
                                          skeleton_segment_size_nm=1000, distance_by_mesh_center=True,
                                          distance_threshold_nm=3000),
                    support_sampler=dict(status='pending', diagnostic_only=True))
    if error is not None:
        evidence['calculation_error'] = f'{type(error).__name__}: {str(error)[:300]}'
    record = dict(status='failed', reason='eligibility_unresolved', key=row['key'],
                  run_hash=manifest['run_hash'], inputs=source_inputs, units='nm',
                  skeleton=skeleton_qc, branch_eligibility=eligibility,
                  width_evidence_file='width_evidence.json')
    def publish():
        evidence['diagnostic_errors'] = list(errors)
        attempt('width_evidence.json', lambda: store_json('width_evidence.json', evidence))
        record.update(outputs_sha256=dict(artifacts), diagnostic_errors=list(errors))
        attempt('result.json', lambda: save(staging/'result.json', record))
    # Persist axes and the actual return before a diagnostic rerun: a timeout
    # during that rerun still leaves the failed calculation's basic evidence.
    publish()
    def support_sample():
        from neurd.width_diagnostics import sample_width_support
        return sample_width_support(parent, skeleton)
    sampler = attempt('support_sampler', support_sample)
    evidence['support_sampler'] = sampler or dict(status='failed', diagnostic_only=True)
    publish()
    for failure in errors:
        print(f"Width diagnostic unavailable ({failure['artifact']}): {failure['error']}", file=sys.stderr)


def _record_width_failure_safely(*args, **kwargs):
    """Diagnostic serialization must never replace the original width error."""
    try:
        _record_width_failure(*args, **kwargs)
    except Exception as exc:
        print(f'Width diagnostics failed: {type(exc).__name__}: {str(exc)[:300]}', file=sys.stderr)


def branch_worker(output, key, staging):
    import numpy as np
    from types import SimpleNamespace
    # This script is launched from the pinned package snapshot.
    sys.path.insert(0, str(Path(output)/'runtime'))
    import neurd
    from neurd.mesh_io import load_mesh
    from neurd.run_config import RunConfig, runtime_environment
    from neurd import parameters, spine_utils as spu
    from neurd.cgal_labels import phase as cgal_phase
    from neurd.spine_volume import reuse_volume_quality
    import cgal_Segmentation_Module as csm
    np.random.seed(0)
    if Path(csm.__file__).suffix != '.so':
        raise RuntimeError('Real compiled CGAL segmentation is required')
    manifest = json.loads((Path(output)/'manifest.json').read_text())
    if dependency_signature() != manifest['dependencies']:
        raise ValueError('Installed numerical or mesh dependencies changed')
    for relative, expected in manifest['code_sha256'].items():
        if sha(Path(output)/'runtime'/relative) != expected:
            raise ValueError(f'Pinned runtime changed: {relative}')
    row = next(r for r in json.loads((Path(output)/'plan.json').read_text()) if r['key']==key)
    source_inputs = inputs(row)
    staging = Path(staging).resolve()
    if (Path(output).resolve()/'.work') not in staging.parents:
        raise ValueError('Worker staging must be inside this run output directory')
    os.chdir(staging)
    runtime = runtime_environment(RunConfig(dataset='h01',calculate_spines=True,sdf_rays=manifest['sdf_rays']))
    if manifest.get('runtime_environment') is not None and runtime != manifest['runtime_environment']:
        raise ValueError('Worker behavior flags or numerical thread environment changed')
    parent = load_mesh(row['source'])
    if len(parent.faces) > manifest['maximum_parent_faces']:
        raise ValueError('Branch exceeds configured face budget; explicit larger budget required')
    if not len(parent.faces) or not np.isfinite(parent.vertices).all():
        raise ValueError('Empty or nonfinite parent surface')
    original = np.load(row['skeleton'],allow_pickle=False)
    skeleton, auxiliary, skeleton_qc = repair_skeleton(parent,original,manifest['skeleton'])
    parameters.params.use('h01')
    os.environ['NEURD_SDF_RAYS'] = str(manifest['sdf_rays'])
    soma = source_inputs.get('soma')
    soma_vertices = load_mesh(soma['path']).vertices if soma else None
    diagnostics, masks, volume_quality = {}, [], []
    eligibility = dict(eligible=True,reason='previously_spine_positive')
    if row['archived_spines'] == 0:
        if len(parent.faces) <= 100:
            eligibility = dict(eligible=False,reason='neurd_minimum_branch_faces')
        else:
            # Retain NEURD's branch-width restriction for previously empty
            # branches, rather than treating every thin axon as a dendrite.
            from neurd import width_utils
            width_distances, width = None, None
            try:
                width_distances,width = width_utils.calculate_new_width(
                    SimpleNamespace(mesh=parent,skeleton=skeleton,width=0),no_spines=False,
                    summary_measure='median',return_average=True)
            except Exception as exc:
                _record_width_failure_safely(staging,manifest,row,source_inputs,parent,
                    original,skeleton,auxiliary,skeleton_qc,width_distances,width,error=exc)
                raise
            if not np.isfinite(width) or width <= 0:
                _record_width_failure_safely(staging,manifest,row,source_inputs,parent,
                    original,skeleton,auxiliary,skeleton_qc,width_distances,width)
                raise ValueError('Cannot establish original NEURD branch-width eligibility')
            eligibility = dict(eligible=bool(width>115),median_mesh_center_nm=float(width),
                               reason='neurd_median_mesh_center_gt_115')
    started = time.perf_counter()
    if eligibility['eligible']:
        meshes, volumes = spu.calculate_spines_on_branch(
            SimpleNamespace(mesh=parent,skeleton=original),geometric_refinement=True,
            geometric_refinement_config=manifest['refinement_config'],
            geometric_refinement_diagnostics=diagnostics,geometric_refinement_face_indices=masks,
            spine_volume_quality=volume_quality,
            geometric_refinement_head_skeleton=(auxiliary if manifest['skeleton']['use_auxiliary_for_heads'] else None),
            soma_verts=soma_vertices,soma_vertex_nullification=soma_vertices is not None,
            cgal_folder=staging/'cgal',print_flag=False)
    else:
        meshes,volumes = [],np.empty(0)
        diagnostics = dict(candidates=[],counts={},branch_eligibility=eligibility)
    elapsed = time.perf_counter()-started
    with cgal_phase("h01_surface_provenance_check", input_meshes=lambda: len(meshes),
                    input_faces=lambda: sum(len(g) for g in masks)):
        if len(meshes)!=len(volume_quality):
            raise ValueError("Invalid final volume quality alignment")
        if len(meshes)!=len(masks) or len(meshes)!=len(volumes) or not np.isfinite(volumes).all() or (np.asarray(volumes)<=0).any():
            raise ValueError('Invalid final mesh/mask/volume alignment')
        _, canonical = np.unique(np.sort(parent.faces,axis=1),axis=0,return_inverse=True)
        occupied = np.concatenate([np.unique(canonical[g]) for g in masks]) if masks else np.empty(0,int)
        if len(occupied) != len(np.unique(occupied)):
            raise ValueError('Final spine masks overlap on parent surface')
        identities = {}
        for candidate in diagnostics['candidates']:
            for mapping in candidate['final_output_mapping']:
                i = mapping['final_output_index']
                if i is not None:
                    identities[i] = dict(raw_id=candidate['raw_input_index'],part_id=mapping['part_index'],action=candidate['action'])
        if set(identities) != set(range(len(meshes))):
            raise ValueError('Incomplete raw-owner provenance')
    (staging/'spines').mkdir()
    spine_records = []
    for i,(mesh,mask,volume) in enumerate(zip(meshes,masks,volumes)):
        with cgal_phase("h01_mesh_mask_validation", spine_index=i, input_faces=lambda: len(mask)):
            if not np.array_equal(triangle_keys(mesh),triangle_keys(parent.submesh([mask],append=True))):
                raise ValueError('Final mesh does not match parent face mask')
        relative = f'spines/spine_{i:05d}.off'
        with cgal_phase("h01_output_write", output_role="spine_mesh", spine_index=i,
                        input_faces=lambda: len(mask)):
            mesh.export(staging/relative)
        with cgal_phase("h01_volume_quality", spine_index=i, input_faces=lambda: len(mask)):
            quality = reuse_volume_quality(volume_quality[i], mesh)
            if not quality['valid'] or not np.isclose(quality['volume_nm3'],volume,rtol=1e-6):
                raise ValueError('Exported volume provenance differs from the final filter')
        spine_records.append(dict(index=i,file=relative,faces=len(mask),volume_nm3=float(volume),volume_quality=quality,**identities[i]))
    with cgal_phase("h01_output_write", output_role="branch_finalize", input_meshes=lambda: len(meshes),
                    input_faces=lambda: sum(len(g) for g in masks)):
        np.savez_compressed(staging/'spine_faces.npz',faces=np.concatenate(masks) if masks else np.empty(0,int),
                            offsets=np.r_[0,np.cumsum([len(g) for g in masks])],
                            raw_ids=[identities[i]['raw_id'] for i in range(len(meshes))],
                            part_ids=[identities[i]['part_id'] for i in range(len(meshes))])
        np.save(staging/'branch_skeleton.npy',skeleton)
        np.save(staging/'branch_skeleton_original.npy',original)
        if auxiliary is not None:
            np.save(staging/'segmentation_axis.npy',auxiliary)
        (staging/'branch_mesh.off').symlink_to(Path(row['source']))
        save(staging/'refinement.json',diagnostics)
        if inputs(row) != source_inputs:
            raise ValueError('Input changed during processing')
        # Intermediate CGAL/length files have no consumer-facing role.
        for path in list(staging.iterdir()):
            if path.is_dir() and path.name != 'spines':
                shutil.rmtree(path)
            elif path.suffix in ('.off','.csv') and path.name != 'branch_mesh.off':
                path.unlink()
        output_files = [p for p in staging.rglob('*') if p.is_file() and p.name not in ('run.log','result.json')]
        record = dict(status='complete',key=key,run_hash=manifest['run_hash'],inputs=source_inputs,
                      archived_spines=row['archived_spines'],spine_count=len(meshes),spines=spine_records,
                      skeleton=skeleton_qc,segmentation_seconds=elapsed,units='nm',
                      branch_eligibility=eligibility,
                      endpoint_filter_axis='original_skeleton',
                      head_completion_axis='auxiliary' if auxiliary is not None and manifest['skeleton']['use_auxiliary_for_heads'] else 'original',
                      compiled_cgal=dict(path=csm.__file__,sha256=sha(csm.__file__)),
                      outputs_sha256={str(p.relative_to(staging)):sha(p) for p in output_files},
                      context_mesh='reference_to_source',
                      checks=dict(real_cgal=eligibility['eligible'],real_neurd_filters=True,positive_finite_volumes=True,
                                  masks_match_meshes=True,no_surface_overlap=True,source_unmodified=True))
        save(staging/'result.json',record)


def update_limb_axes(source, output, processed):
    """Keep each exported limb skeleton consistent with its branch replacements."""
    import numpy as np
    for limb_rel in sorted({str(Path(r['relative']).parent) for r in processed}):
        original_limb, dest = Path(source)/limb_rel, Path(output)/'dataset'/limb_rel
        dest.mkdir(parents=True,exist_ok=True)
        branches = sorted(original_limb.glob('branch_*/branch_skeleton.npy'))
        groups, count = [], 0
        for path in branches:
            replacement = dest/path.parent.name/'branch_skeleton.npy'
            if replacement.is_file():
                count += 1
            groups.append(np.load(replacement if replacement.is_file() else path,allow_pickle=False))
        combined = np.concatenate(groups) if groups else np.empty((0,2,3))
        temp = dest/'limb_skeleton.tmp.npy'
        np.save(temp,combined); temp.replace(dest/'limb_skeleton.npy')
        for name in ('limb_mesh.off','connectivity.json','branch_nodes.npy','branch_edges.npy','branch_index_map.json'):
            target = original_limb/name
            if target.is_file() and not (dest/name).exists():
                (dest/name).symlink_to(target)
        save(dest/'skeleton_provenance.json',dict(processed_branches=count,total_branches=len(branches),
             complete=count==len(branches),unprocessed_branches_use_original_axis=True,
             connectivity='original_graph_connections_preserved'))


def update_neuron_summaries(source,output,processed):
    for neuron in sorted({Path(r['relative']).parts[0] for r in processed}):
        origin, dest = Path(source)/neuron, Path(output)/'dataset'/neuron
        limbs = []
        for limb in sorted(origin.glob('limb_*')):
            if not limb.is_dir():
                continue
            counts = []
            for mesh in sorted(limb.glob('branch_*/branch_mesh.off')):
                result = dest/limb.name/mesh.parent.name/'result.json'
                counts.append(json.loads(result.read_text())['spine_count'] if result.is_file() else None)
            limbs.append(dict(name=limb.name,n_branches=len(counts),spines_per_branch=counts,
                              total_spines=sum(x for x in counts if x is not None),
                              complete=all(x is not None for x in counts)))
        if (origin/'soma').is_dir() and not (dest/'soma').exists():
            (dest/'soma').symlink_to(origin/'soma',target_is_directory=True)
        save(dest/'summary.json',dict(data_type='h01',n_limbs=len(limbs),limbs=limbs,
              total_spines=sum(l['total_spines'] for l in limbs),complete=all(l['complete'] for l in limbs),
              source_neuron=str(origin)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--keys',type=Path,help='JSON list of exact branch keys for a limited run')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--timeout',type=int,default=2700)
    parser.add_argument('--max-faces',type=int,default=2000000)
    parser.add_argument('--rays',type=int,default=25)
    parser.add_argument('--no-skeleton-repair',action='store_true')
    parser.add_argument('--experimental-skeleton-replacement',action='store_true',
                        help='Experimental: allow topology-preserving replacement of the exported skeleton')
    parser.add_argument('--auxiliary-heads',action='store_true',help='Experimental: use rebuilt axis for head completion')
    parser.add_argument('--worker')
    parser.add_argument('--staging',type=Path)
    args = parser.parse_args(argv)
    if args.worker:
        return branch_worker(args.output,args.worker,args.staging)
    if not args.source or not args.config:
        parser.error('--source and --config are required')
    if args.workers < 1 or args.timeout < 1 or args.max_faces < 1 or args.rays < 1 or (args.limit is not None and args.limit < 1):
        parser.error('Worker, timeout, face, ray and limit values must be positive')
    if args.no_skeleton_repair and args.experimental_skeleton_replacement:
        parser.error('--experimental-skeleton-replacement requires skeleton repair')
    source, output = separate_roots(args.source,args.output)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another batch process owns this output root')
        return run(args,source,output)


def worker_environment(output):
    """Apply the same numerical thread limits to planning and child execution."""
    return dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='4',MPLCONFIGDIR=str(output/'.cache/matplotlib'),
                NUMBA_CACHE_DIR=str(output/'.cache/numba'),TF_CPP_MIN_LOG_LEVEL='2')


def run(args,source,output):
    from neurd.run_config import RunConfig, runtime_environment

    config = json.loads(args.config.read_text())
    repo = Path(__file__).resolve().parents[1]
    code = {str(p.relative_to(repo)):sha(p) for p in (repo/'neurd').rglob('*') if p.is_file() and p.suffix in ('.py','.json')}
    settings = dict(source=str(source),refinement_config=config,sdf_rays=args.rays,maximum_parent_faces=args.max_faces,
                    skeleton=dict(repair=not args.no_skeleton_repair,
                                  allow_topological_replacement=args.experimental_skeleton_replacement,coverage_q95_nm=10000,
                                  invalidation_d_nm=12000,maximum_connector_nm=2000,
                                  use_auxiliary_for_heads=args.auxiliary_heads),code_sha256=code,
                    dependencies=dependency_signature(),
                    runtime_environment=runtime_environment(
                        RunConfig(dataset='h01',calculate_spines=True,sdf_rays=args.rays),
                        environment=worker_environment(output)))
    run_hash = digest_json(settings)
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old['run_hash'] != run_hash:
            raise ValueError('Settings or code changed: use a separate output directory')
        rows = json.loads((output/'plan.json').read_text())
        if rows != inventory(source):
            raise ValueError('Source branch inventory changed: use a separate output directory')
    else:
        foreign = [p for p in output.iterdir() if p.name != '.lock']
        if foreign:
            raise ValueError('Output directory contains unrelated files')
        rows = inventory(source)
        settings.update(run_hash=run_hash,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                        total_branches=len(rows),total_neurons=len({r['relative'].split('/')[0] for r in rows}),
                        original_zero_spine_branches=sum(r['archived_spines']==0 for r in rows))
        shutil.copytree(repo/'neurd',output/'runtime/neurd',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        save(output/'plan.json',rows); save(manifest_path,settings)
    chosen = rows
    if args.keys:
        wanted = set(json.loads(args.keys.read_text()))
        chosen = [r for r in rows if r['key'] in wanted]
        if {r['key'] for r in chosen} != wanted:
            raise ValueError('Requested branch keys not found')
    if args.limit:
        chosen = chosen[:args.limit]
    print(json.dumps(dict(event='plan',total_branches=len(rows),selected=len(chosen),
                          zero_archived_spines=sum(r['archived_spines']==0 for r in rows),run_hash=run_hash)),flush=True)
    if args.plan_only:
        return
    (output/'.work').mkdir(exist_ok=True); (output/'failures').mkdir(exist_ok=True)
    stop,active_lock = Event(),Lock()
    active = {}
    def interrupt(signum,frame):
        stop.set()
        with active_lock:
            running = list(active.values())
        for child in running:
            try:
                os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
    old_handlers = {sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    for sig in old_handlers:
        signal.signal(sig,interrupt)
    def job(row):
        dest = output/'dataset'/row['relative']
        try:
            if stop.is_set():
                raise InterruptedError('Run interrupted')
            source_inputs = inputs(row)
            if dest.exists():
                if complete_valid(dest,run_hash,source_inputs):
                    return dict(key=row['key'],status='resumed',spines=json.loads((dest/'result.json').read_text())['spine_count'])
                raise ValueError('Incomplete destination exists; refusing overwrite')
            staging = output/'.work'/f"{row['key']}_{uuid.uuid4().hex}"
            staging.mkdir()
            env = worker_environment(output)
            with (staging/'run.log').open('w') as log:
                try:
                    proc = subprocess.Popen([sys.executable,str(output/'runtime/neurd/h01_batch.py'),
                         '--output',str(output),'--worker',row['key'],'--staging',str(staging)],
                         cwd=staging,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    with active_lock:
                        active[proc.pid] = proc
                    try:
                        if stop.is_set():
                            os.killpg(proc.pid,signal.SIGTERM)
                        proc.wait(timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                        raise
                    finally:
                        with active_lock:
                            active.pop(proc.pid,None)
                    if proc.returncode:
                        raise RuntimeError(f'Worker exited {proc.returncode}')
                except Exception:
                    failed = output/'failures'/staging.name
                    staging.rename(failed)
                    raise
            if not complete_valid(staging,run_hash,source_inputs):
                raise RuntimeError('Worker did not produce a complete validated result')
            dest.parent.mkdir(parents=True,exist_ok=True)
            staging.rename(dest)
            result = json.loads((dest/'result.json').read_text())
            return dict(key=row['key'],status='complete',spines=result['spine_count'],skeleton=result['skeleton']['status'])
        except Exception as exc:
            return dict(key=row['key'],status='interrupted' if stop.is_set() else 'failed',reason=f'{type(exc).__name__}: {exc}')
    try:
        with (output/'last_run.jsonl').open('w') as ledger:
            def on_result(result,count):
                ledger.write(json.dumps(result,allow_nan=False)+'\n');ledger.flush()
                save(output/'progress.json',dict(finished=count,selected=len(chosen),last=result))
                print(json.dumps(dict(completed=count,selected=len(chosen),**result)),flush=True)
            results = dispatch(chosen,job,args.workers,stop,on_result)
    finally:
        for sig,handler in old_handlers.items():
            signal.signal(sig,handler)
    save(output/'last_run.json',results)
    successful = {r['key'] for r in results if r['status'] in ('complete','resumed')}
    update_limb_axes(source,output,[r for r in chosen if r['key'] in successful])
    update_neuron_summaries(source,output,[r for r in chosen if r['key'] in successful])
    summary = dict(selected=len(chosen),completed=len(successful),failed=sum(r['status']=='failed' for r in results),
                   interrupted=sum(r['status']=='interrupted' for r in results),unattempted=len(chosen)-len(results),
                   total_branches=len(rows),full_dataset_complete=len(chosen)==len(rows) and len(successful)==len(rows),
                   total_selected_spines=sum(r.get('spines',0) for r in results),run_hash=run_hash)
    save(output/'last_run_summary.json',summary)
    if stop.is_set():
        raise KeyboardInterrupt('Stopped; completed branches will be reused on resume')
    if summary['failed']:
        raise RuntimeError(f"{summary['failed']} branches failed; see last_run.json and failures/")


if __name__ == '__main__':
    main()
