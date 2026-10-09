#!/usr/bin/env python3
"""Refine encoder-selected archived H01 objects with exact parent-face identity.

This compares the geometry stage on every archived spine on each selected
branch. It does not rerun CGAL, length or final volume filters. Selected objects
are a targeted diagnostic sample, not an estimate of H01 error prevalence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def job(args, row):
    import numpy as np
    from scipy.spatial import cKDTree
    import trimesh

    module = args.code_snapshot/'neurd/spine_refinement.py'
    spec = importlib.util.spec_from_file_location('target_refinement', module)
    sr = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sr
    spec.loader.exec_module(sr)
    assert sha(row['parent']) == row['parent_sha256']
    assert sha(row['skeleton']) == row['skeleton_sha256']
    parent = trimesh.load(row['parent'], process=False, force='mesh')
    skeleton = np.load(row['skeleton'])
    tree = cKDTree(parent.triangles_center)
    groups, sources, source_ids = [], [], []

    def ordered(triangles):
        order = np.lexsort((triangles[:,:,2], triangles[:,:,1], triangles[:,:,0]), axis=1)
        return np.take_along_axis(triangles, order[:,:,None], axis=1)

    for path in sorted((Path(row['parent']).parent/'spines').glob('*.off')):
        mesh = trimesh.load(path, process=False, force='mesh')
        distance, indices = tree.query(mesh.triangles_center)
        assert distance.max() < .001, str(path)
        actual = ordered(mesh.triangles)
        for face in np.flatnonzero(np.max(abs(actual-ordered(parent.triangles[indices])), axis=(1,2)) > .001):
            candidates = np.asarray(tree.query_ball_point(mesh.triangles_center[face], .001), int)
            exact = np.max(abs(ordered(parent.triangles[candidates])-actual[face]), axis=(1,2)) <= .001
            assert exact.any(), (str(path), int(face), 'No identical parent triangle')
            indices[face] = candidates[np.flatnonzero(exact)[0]]
        assert np.allclose(actual, ordered(parent.triangles[indices]), atol=.001, rtol=0)
        groups.append(np.unique(indices))
        source_ids.append(int(path.stem.split('_')[-1]))
        sources.append(dict(path=str(path), sha256=sha(path), faces=len(mesh.faces)))
    assert len(groups) == row['archived_spines'] and len(set(source_ids)) == len(source_ids)
    target_position = source_ids.index(row['source_spine_index'])
    assert sources[target_position]['sha256'] == row['mesh_sha256']
    target_file = args.selection.parent/row['case_id']/'parent_face_indices.npy'
    assert np.array_equal(groups[target_position], np.load(target_file))
    output = args.output/'branches'/row['key']
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'raw_faces.npz', faces=np.concatenate(groups),
                       offsets=np.r_[0, np.cumsum([len(m) for m in groups])],
                       raw_ids=source_ids, part_ids=np.zeros(len(groups), int))

    def physical(mask):
        return np.unique(np.sort(parent.faces[mask], axis=1), axis=0)

    exports, diagnostics, timings = {}, {}, {}
    for mode, config in [('previous', args.previous_config), ('candidate', args.candidate_config)]:
        start = perf_counter()
        results, d = sr.refine_spine_face_indices(parent, skeleton, groups, config)
        timings[mode] = dict(geometry_seconds=perf_counter()-start)
        mapped = {(source_ids[r['input_index']], part): results[index]
                  for r in d['candidates'] for part, index in enumerate(r['output_indices'])}
        for r in d['candidates']:
            r['raw_input_index'] = source_ids[r['input_index']]
        masks = list(mapped.values())
        np.savez_compressed(output/(mode+'_faces.npz'),
                           faces=np.concatenate(masks) if masks else np.empty(0, int),
                           offsets=np.r_[0, np.cumsum([len(m) for m in masks])],
                           raw_ids=[k[0] for k in mapped], part_ids=[k[1] for k in mapped])
        # A split must give distinct physical triangles to each of its parts.
        for raw in source_ids:
            parts = [physical(m) for (r,p),m in mapped.items() if r == raw]
            if parts:
                combined = np.concatenate(parts)
                assert len(combined) == len(np.unique(combined, axis=0)), (raw, 'Overlapping split')
        exports[mode], diagnostics[mode] = mapped, d
    before, after = exports['previous'], exports['candidate']
    changed = sorted({raw for raw,part in before.keys() | after.keys()
                      if (raw,part) not in before or (raw,part) not in after
                      or not np.array_equal(physical(before[(raw,part)]), physical(after[(raw,part)]))})
    changes = []
    for raw in changed:
        old = [mask for (r,p),mask in before.items() if r == raw]
        new = [mask for (r,p),mask in after.items() if r == raw]
        a = np.unique(np.concatenate(old)) if old else np.empty(0,int)
        b = np.unique(np.concatenate(new)) if new else np.empty(0,int)
        changes.append(dict(raw_id=raw, previous_faces=len(a), candidate_faces=len(b),
                            added_faces=len(np.setdiff1d(b,a)), removed_faces=len(np.setdiff1d(a,b))))
    selected = diagnostics['candidate']['candidates'][target_position]
    result = dict(status='passed', key=row['key'], dataset='h01', validation_stage='geometry_only',
                  input_origin='archived_neurd_outputs', raw_id_semantics='archived_spine_filename_index',
                  selection=dict(source=row['parent'], skeleton=row['skeleton']),
                  source_sha256=row['parent_sha256'], skeleton_sha256=row['skeleton_sha256'],
                  archived_spines=len(groups), previous_spines=len(before), candidate_spines=len(after),
                  changed_raw_ids=changed, changes=changes, timings=timings,
                  previous_refinement=diagnostics['previous'], refinement=diagnostics['candidate'],
                  selected_target=dict(case_id=row['case_id'], raw_id=row['source_spine_index'],
                                       encoder_distance=row['encoder_distance'], changed=row['source_spine_index'] in changed,
                                       decision=selected),
                  seed_sources=sources, code_sha256=sha(module),
                  selected_mask_sha256=sha(target_file),
                  versions={'numpy':np.__version__, 'trimesh':trimesh.__version__})
    save(output/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--code-snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous-config', type=json.loads,
                        default={'trim_shaft_connections':True, 'grow_ragged_heads':True})
    parser.add_argument('--candidate-config', type=json.loads,
                        default={'trim_shaft_connections':True, 'grow_ragged_heads':True,
                                 'trim_single_protrusions':True, 'filter_low_relief_sheets':True})
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--job-key')
    args = parser.parse_args()
    args.selection, args.code_snapshot, args.output = (p.resolve() for p in
                                                      (args.selection, args.code_snapshot, args.output))
    rows = json.loads(args.selection.read_text())
    assert len({r['key'] for r in rows}) == len(rows), 'One target per branch required'
    args.output.mkdir(parents=True, exist_ok=True)
    if args.job_key:
        job(args, next(r for r in rows if r['key'] == args.job_key))
        return
    save(args.output/'manifest.json', dict(selection=str(args.selection), selection_sha256=sha(args.selection),
         code_snapshot=str(args.code_snapshot), script_sha256=sha(__file__), stage='geometry_only',
         previous_config=args.previous_config, candidate_config=args.candidate_config,
         targeted_sample=True, branches=[r['key'] for r in rows],
         input_origin='archived_neurd_outputs', raw_id_semantics='archived_spine_filename_index'))

    def execute(row):
        folder = args.output/'branches'/row['key']
        folder.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, __file__, '--selection', str(args.selection),
               '--code-snapshot', str(args.code_snapshot), '--output', str(args.output),
               '--previous-config', json.dumps(args.previous_config),
               '--candidate-config', json.dumps(args.candidate_config), '--job-key', row['key']]
        with (folder/'run.log').open('w') as log:
            process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            return dict(status='failed', key=row['key'], log=str(folder/'run.log'))
        return json.loads((folder/'result.json').read_text())

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(execute, rows):
            results.append(result)
            save(args.output/'results.json', results)
            print(json.dumps({k:result[k] for k in ['key','status','previous_spines','candidate_spines']
                              if k in result}), flush=True)
    assert all(r['status'] == 'passed' for r in results)


if __name__ == '__main__':
    main()
