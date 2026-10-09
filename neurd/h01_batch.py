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
    for name in ('numpy','scipy','trimesh','networkx','meshparty','mesh-tools','datasci-tools'):
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
    if not np.isfinite(segments).all():
        raise ValueError('Nonfinite skeleton coordinates')
    vertices, inverse = np.unique(np.asarray(segments).reshape(-1,3),axis=0,return_inverse=True)
    edges = inverse.reshape(-1,2)
    if np.any(edges[:,0] == edges[:,1]):
        raise ValueError('Zero length skeleton edges')
    degree = np.bincount(edges.ravel(),minlength=len(vertices))
    return vertices, edges, degree


def triangle_keys(mesh):
    """Exact triangle geometry, independent of face/vertex order and winding."""
    import numpy as np
    rows = np.asarray(mesh.triangles,dtype=np.float64).copy().view(np.dtype((np.void,24))).reshape(-1,3)
    rows.sort(axis=1)
    return np.sort(rows.copy().view(np.dtype((np.void,72))).reshape(-1))


def anchor_skeleton(original, fresh, maximum_connector_nm=2000):
    """Prune a fresh tree to original graph landmarks, retaining exact positions.

    Return None if an endpoint/junction lies far from the fresh axis, if the
    fresh graph is disconnected/cyclic, or if any landmark degree changes.
    Geometry coverage is checked separately after anchoring, before acceptance.
    """
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    old_v, old_e, old_d = graph_data(original)
    verts, edges, _ = graph_data(fresh)
    graph = coo_matrix((np.ones(len(edges)), (edges[:,0],edges[:,1])),shape=(len(verts),len(verts)))
    if connected_components(graph,directed=False)[0] != 1 or len(edges) != len(verts)-1:
        return None, 'fresh_axis_is_not_a_connected_tree'
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
    return result, 'exact_original_graph_landmarks_preserved'


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
    if record.get('status') != 'complete' or record.get('run_hash') != run_hash or record.get('inputs') != source_inputs:
        raise ValueError(f'Existing result has changed inputs or settings: {dest}')
    for relative, expected in record['outputs_sha256'].items():
        path = Path(dest)/relative
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'Completed output is missing or changed: {path}')
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


def branch_worker(output, key, staging):
    import numpy as np
    import trimesh
    from types import SimpleNamespace
    # This script is launched from the pinned package snapshot.
    sys.path.insert(0, str(Path(output)/'runtime'))
    import neurd
    from neurd import parameters, spine_utils as spu
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
    parent = trimesh.load(row['source'],process=False,force='mesh')
    if len(parent.faces) > manifest['maximum_parent_faces']:
        raise ValueError('Branch exceeds configured face budget; explicit larger budget required')
    if not len(parent.faces) or not np.isfinite(parent.vertices).all():
        raise ValueError('Empty or nonfinite parent surface')
    original = np.load(row['skeleton'],allow_pickle=False)
    skeleton, auxiliary, skeleton_qc = repair_skeleton(parent,original,manifest['skeleton'])
    parameters.params.use('h01')
    os.environ['NEURD_SDF_RAYS'] = str(manifest['sdf_rays'])
    soma = source_inputs.get('soma')
    soma_vertices = trimesh.load(soma['path'],process=False,force='mesh').vertices if soma else None
    diagnostics, masks = {}, []
    eligibility = dict(eligible=True,reason='previously_spine_positive')
    if row['archived_spines'] == 0:
        if len(parent.faces) <= 100:
            eligibility = dict(eligible=False,reason='neurd_minimum_branch_faces')
        else:
            # Retain NEURD's branch-width restriction for previously empty
            # branches, rather than treating every thin axon as a dendrite.
            from neurd import width_utils
            _,width = width_utils.calculate_new_width(
                SimpleNamespace(mesh=parent,skeleton=skeleton,width=0),no_spines=False,
                summary_measure='median',return_average=True)
            if not np.isfinite(width) or width <= 0:
                raise ValueError('Cannot establish original NEURD branch-width eligibility')
            eligibility = dict(eligible=bool(width>115),median_mesh_center_nm=float(width),
                               reason='neurd_median_mesh_center_gt_115')
    started = time.perf_counter()
    if eligibility['eligible']:
        meshes, volumes = spu.calculate_spines_on_branch(
            SimpleNamespace(mesh=parent,skeleton=original),geometric_refinement=True,
            geometric_refinement_config=manifest['refinement_config'],
            geometric_refinement_diagnostics=diagnostics,geometric_refinement_face_indices=masks,
            geometric_refinement_head_skeleton=(auxiliary if manifest['skeleton']['use_auxiliary_for_heads'] else None),
            soma_verts=soma_vertices,soma_vertex_nullification=soma_vertices is not None,
            cgal_folder=staging/'cgal',print_flag=False)
    else:
        meshes,volumes = [],np.empty(0)
        diagnostics = dict(candidates=[],counts={},branch_eligibility=eligibility)
    elapsed = time.perf_counter()-started
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
        if not np.array_equal(triangle_keys(mesh),triangle_keys(parent.submesh([mask],append=True))):
            raise ValueError('Final mesh does not match parent face mask')
        relative = f'spines/spine_{i:05d}.off'
        mesh.export(staging/relative)
        quality = spu.estimate_mesh_volume(mesh)
        if not quality['valid'] or not np.isclose(quality['volume_nm3'],volume,rtol=1e-6):
            raise ValueError('Exported volume provenance differs from the final filter')
        spine_records.append(dict(index=i,file=relative,faces=len(mask),volume_nm3=float(volume),volume_quality=quality,**identities[i]))
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
    source, output = separate_roots(args.source,args.output)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another batch process owns this output root')
        return run(args,source,output)


def run(args,source,output):
    config = json.loads(args.config.read_text())
    repo = Path(__file__).resolve().parents[1]
    code = {str(p.relative_to(repo)):sha(p) for p in (repo/'neurd').rglob('*') if p.is_file() and p.suffix in ('.py','.json')}
    settings = dict(source=str(source),refinement_config=config,sdf_rays=args.rays,maximum_parent_faces=args.max_faces,
                    skeleton=dict(repair=not args.no_skeleton_repair,coverage_q95_nm=10000,
                                  invalidation_d_nm=12000,maximum_connector_nm=2000,
                                  use_auxiliary_for_heads=args.auxiliary_heads),code_sha256=code,
                    dependencies=dependency_signature())
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
            env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='4',MPLCONFIGDIR=str(output/'.cache/matplotlib'),
                       NUMBA_CACHE_DIR=str(output/'.cache/numba'),TF_CPP_MIN_LOG_LEVEL='2')
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
