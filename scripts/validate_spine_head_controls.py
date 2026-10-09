#!/usr/bin/env python3
"""Compare head refinement on frozen archived controls, at the geometry stage.

This does not rerun CGAL or final branch filters. Full-pipeline H01 validation
is provided by validate_spine_head_refinement.py. Parent, skeleton, physical
triangle identity and the frozen baseline are checked before comparison.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def job(args, row):
    import numpy as np
    from scipy.spatial import cKDTree
    import trimesh

    module = args.code_snapshot/'neurd/spine_refinement.py'
    spec = importlib.util.spec_from_file_location('control_refinement', module)
    sr = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sr
    spec.loader.exec_module(sr)
    assert sha(row['source']) == row['source_sha256']
    assert sha(row['skeleton']) == row['skeleton_sha256']
    parent = trimesh.load(row['source'], process=False, force='mesh')
    skeleton = np.load(row['skeleton'])
    tree = cKDTree(parent.triangles_center)
    groups, seed_sources = [], []

    def ordered(triangles):
        order = np.lexsort((triangles[:, :, 2], triangles[:, :, 1], triangles[:, :, 0]), axis=1)
        return np.take_along_axis(triangles, order[:, :, None], axis=1)

    for path in sorted((Path(row['source']).parent/'spines').glob('*.off')):
        mesh = trimesh.load(path, process=False, force='mesh')
        distance, indices = tree.query(mesh.triangles_center)
        assert np.max(distance) < .001, path
        actual = ordered(mesh.triangles)
        matched = ordered(parent.triangles[indices])
        for face in np.flatnonzero(np.max(abs(actual-matched), axis=(1, 2)) > .001):
            # Coincident centroids can belong to different triangles. Resolve
            # them by vertex coordinates rather than silently taking a neighbour.
            candidates = np.asarray(tree.query_ball_point(mesh.triangles_center[face], .001), dtype=int)
            exact = np.max(abs(ordered(parent.triangles[candidates])-actual[face]), axis=(1, 2)) <= .001
            assert exact.any(), (str(path), int(face), 'No identical parent triangle')
            indices[face] = candidates[np.flatnonzero(exact)[0]]
        assert np.allclose(actual, ordered(parent.triangles[indices]), atol=.001, rtol=0), path
        groups.append(np.unique(indices))
        seed_sources.append({'path': str(path), 'sha256': sha(path)})
    assert len(groups) == row['archived_spines']
    with np.load(args.baseline/row['key']/'candidate_faces.npz') as z:
        frozen_masks = (np.split(z['faces'], z['offsets'][1:-1].astype(int))
                        if len(z['faces']) else [])
    cached_previous = None
    if args.validated_baseline:
        ref = args.validated_baseline/row['key']
        reference = json.loads((ref/'result.json').read_text())
        assert reference['status']=='passed' and reference['source_sha256']==row['source_sha256']
        assert reference['skeleton_sha256']==row['skeleton_sha256']
        if args.validated_baseline_mask == 'candidate':
            assert all(reference['refinement']['config'].get(k)==v for k,v in args.previous_config.items())
        with np.load(ref/(args.validated_baseline_mask+'_faces.npz')) as z:
            masks=np.split(z['faces'],z['offsets'][1:-1].astype(int)) if len(z['faces']) else []
            cached_previous={(int(raw),int(part)):mask for raw,part,mask in zip(z['raw_ids'],z['part_ids'],masks)}

    def physical(mask):
        return np.unique(np.sort(parent.faces[mask], axis=1), axis=0)

    output = args.output/row['key']
    output.mkdir(parents=True, exist_ok=True)
    timings, diagnostics, exports = {}, {}, {}
    for mode, config in [('previous', args.previous_config), ('candidate', args.candidate_config)]:
        start = perf_counter()
        if mode=='previous' and cached_previous is not None:
            rows, d = cached_previous, {'cached': True, 'source': str(args.validated_baseline/row['key'])}
        else:
            result, d = sr.refine_spine_face_indices(parent, skeleton, groups, config)
            rows = {}
            for record in d['candidates']:
                for part, index in enumerate(record['output_indices']):
                    rows[(record['input_index'], part)] = result[index]
        timings[mode] = perf_counter()-start
        if mode == 'previous':
            # Historical control masks stored ordered outputs without raw IDs.
            # Recover their mapping only after reproducing every physical mask
            # in the same order; the new exports persist both raw and part IDs.
            assert len(rows) == len(frozen_masks), 'Frozen baseline output count differs'
            assert all(np.array_equal(physical(mask), physical(old))
                       for mask, old in zip(rows.values(), frozen_masks)), 'Frozen baseline physical triangles differ'
            frozen = dict(zip(rows, frozen_masks))
        masks = list(rows.values())
        np.savez_compressed(output/(mode+'_faces.npz'),
                            faces=np.concatenate(masks) if masks else np.empty(0, dtype=int),
                            offsets=np.r_[0, np.cumsum([len(g) for g in masks])],
                            raw_ids=[key[0] for key in rows], part_ids=[key[1] for key in rows])
        diagnostics[mode], exports[mode] = d, rows
    changed = sorted({raw for raw, part in frozen.keys() | exports['candidate'].keys()
                      if (raw, part) not in frozen or (raw, part) not in exports['candidate']
                      or not np.array_equal(physical(frozen[(raw, part)]), physical(exports['candidate'][(raw, part)]))})
    examples = []
    for example in row['examples']:
        raw = example['decision']['input_index']
        examples.append(dict(key=example['key'], group=example['group'], raw_id=raw,
                             changed=raw in changed, decision=diagnostics['candidate']['candidates'][raw]))
    result = dict(status='passed', key=row['key'], dataset=row['dataset'],
                  validation_stage='geometry_only', source=row['source'], skeleton=row['skeleton'],
                  source_sha256=row['source_sha256'], skeleton_sha256=row['skeleton_sha256'],
                  archived_spines=len(groups), previous_spines=len(frozen),
                  candidate_spines=len(exports['candidate']), changed_raw_ids=changed,
                  examples=examples, timings=timings, refinement=diagnostics['candidate'],
                  seed_sources=seed_sources, code_sha256=sha(module),
                  versions={'numpy': np.__version__, 'trimesh': trimesh.__version__})
    save(output/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--code-snapshot', type=Path, required=True)
    parser.add_argument('--validated-baseline', type=Path, help='Reuse verified baseline masks without repeating geometry work')
    parser.add_argument('--validated-baseline-mask', choices=['previous','candidate'], default='previous')
    parser.add_argument('--previous-config',type=json.loads,default={'trim_shaft_connections':True})
    parser.add_argument('--candidate-config',type=json.loads,default={'trim_shaft_connections':True,'grow_ragged_heads':True})
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--job-key')
    args = parser.parse_args()
    args.baseline, args.code_snapshot, args.output = (p.resolve() for p in
                                                    (args.baseline, args.code_snapshot, args.output))
    source = args.baseline/'results.json'
    if args.validated_baseline: args.validated_baseline=args.validated_baseline.resolve()
    rows = json.loads(source.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.job_key:
        job(args, next(row for row in rows if row['key'] == args.job_key))
        return
    save(args.output/'manifest.json', dict(baseline=str(source), baseline_sha256=sha(source),
         code_snapshot=str(args.code_snapshot), script_sha256=sha(__file__), stage='geometry_only',
         validated_baseline=str(args.validated_baseline) if args.validated_baseline else None,
         validated_baseline_mask=args.validated_baseline_mask,previous_config=args.previous_config,
         candidate_config=args.candidate_config,
         branches=[r['key'] for r in rows], named_examples=sum(len(r['examples']) for r in rows)))

    def execute(row):
        folder = args.output/row['key']
        folder.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, __file__, '--baseline', str(args.baseline),
                   '--code-snapshot', str(args.code_snapshot), '--output', str(args.output),
                   '--job-key', row['key']]
        command+=['--previous-config',json.dumps(args.previous_config),'--candidate-config',json.dumps(args.candidate_config),
                  '--validated-baseline-mask',args.validated_baseline_mask]
        if args.validated_baseline: command+=['--validated-baseline',str(args.validated_baseline)]
        with (folder/'run.log').open('w') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            return dict(status='failed', key=row['key'], log=str(folder/'run.log'))
        return json.loads((folder/'result.json').read_text())

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(execute, rows):
            results.append(row)
            save(args.output/'results.json', results)
            print(json.dumps({k: row[k] for k in ('key', 'status', 'changed_raw_ids') if k in row}), flush=True)
    assert all(r['status'] == 'passed' for r in results)


if __name__ == '__main__':
    main()
