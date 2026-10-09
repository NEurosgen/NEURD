"""Replay frozen CGAL/length results with real final filters; no held-out claim."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def unpack(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        groups = np.split(z['faces'], z['offsets'][1:-1]) if len(z['raw_ids']) else []
        return {(int(r), int(p)): g.copy() for r, p, g in zip(z['raw_ids'], z['part_ids'], groups)}


def worker(args):
    sys.path.insert(0, str(args.output / 'code_snapshot'))
    import neurd
    import numpy as np
    import trimesh
    from neurd import parameters, spine_utils as spu
    manifest = json.loads((args.output / 'manifest.json').read_text())
    row = next(r for r in json.loads((args.frozen / 'selection.json').read_text())['selection'] if r['key'] == args.key)
    cached = args.frozen / 'comparison/branches' / args.key
    previous = json.loads((cached / 'result.json').read_text())
    assert sha(row['source']) == previous['source_sha256']
    assert sha(row['skeleton']) == previous['skeleton_sha256']
    for name, expected in manifest['code_sha256'].items():
        assert sha(args.output / 'code_snapshot/neurd' / name) == expected
    dest = args.output / 'branches' / args.key
    dest.mkdir(parents=True, exist_ok=True)
    os.chdir(dest)
    parent = trimesh.load(row['source'], process=False, force='mesh')
    axis = np.load(row['skeleton'], allow_pickle=False)
    raw = unpack(cached / 'raw_faces.npz')
    groups = [raw[(i, 0)] for i in range(len(raw))]
    meshes = [parent.submesh([g], append=True) for g in groups]
    prefix = json.loads((cached / 'shared_prefix.json').read_text())
    passing = [meshes[i] for i in prefix['length_raw_ids']]
    original_filter = spu.filter_spine_meshes
    def length(inputs, *a, **kw):
        if len(inputs) == len(meshes) and all(x is y for x, y in zip(inputs, meshes)):
            return passing
        return original_filter(inputs, *a, **kw)
    spu.filter_spine_meshes = length
    spu.get_spine_meshes_unfiltered_from_mesh = lambda *a, **kw: (meshes, groups) if kw.get('return_mesh_idx') else meshes
    parameters.params.use('h01')
    assert parameters.params._data == previous['parameters']
    soma_path = Path(row['source']).parents[2] / 'soma/soma_mesh.off'
    soma = trimesh.load(soma_path, process=False, force='mesh').vertices if soma_path.exists() else None
    before = unpack(cached / 'candidate_faces.npz')
    results = {}
    _, canonical = np.unique(np.sort(parent.faces, axis=1), axis=0, return_inverse=True)
    for mode, config in manifest['configs'].items():
        if (mode == 'control' and args.control_cache
                and (args.control_cache/'branches'/args.key/'result.json').is_file()):
            cached_control = args.control_cache/'branches'/args.key
            saved = json.loads((cached_control/'result.json').read_text())
            assert saved['status']=='passed' and saved['modes']['control']['changes']==[]
            shutil.copy2(cached_control/'control_faces.npz',dest/'control_faces.npz')
            results[mode] = saved['modes']['control']
            continue
        np.random.seed(0)
        stage, final_faces = {}, []
        started = perf_counter()
        outputs, volumes = spu.calculate_spines_on_branch(
            SimpleNamespace(mesh=parent, skeleton=axis), geometric_refinement=True,
            geometric_refinement_config=config, geometric_refinement_diagnostics=stage,
            geometric_refinement_face_indices=final_faces,
            soma_verts=soma, soma_vertex_nullification=soma is not None,
            cgal_folder=dest / 'cgal', print_flag=False)
        elapsed = perf_counter() - started
        mapping = {}
        for record in stage['candidates']:
            for item in record['final_output_mapping']:
                i = item['final_output_index']
                if i is not None:
                    mapping[(record['raw_input_index'], item['part_index'])] = final_faces[i]
        assert len(mapping) == len(outputs) == len(final_faces) == len(volumes)
        assert np.isfinite(volumes).all() and (np.asarray(volumes) > 0).all()
        for mesh, face_ids in zip(outputs, final_faces):
            assert np.array_equal(mesh.triangles, parent.submesh([face_ids], append=True).triangles)
        physical = np.concatenate([np.unique(canonical[g]) for g in mapping.values()]) if mapping else np.empty(0, int)
        assert len(physical) == len(np.unique(physical)), 'Final masks overlap'
        changes = []
        for identity in sorted(set(before) | set(mapping)):
            old, new = before.get(identity, np.empty(0, int)), mapping.get(identity, np.empty(0, int))
            if np.array_equal(old, new):
                continue
            changes.append(dict(raw_id=identity[0], part_id=identity[1], before_faces=len(old), after_faces=len(new),
                                added_faces=len(np.setdiff1d(new, old)), removed_faces=len(np.setdiff1d(old, new))))
        if mode == 'control':
            assert not changes, 'Replay changed the previously validated result'
        np.savez_compressed(dest / (mode + '_faces.npz'),
                            faces=np.concatenate(list(mapping.values())) if mapping else np.empty(0, int),
                            offsets=np.r_[0, np.cumsum([len(g) for g in mapping.values()])],
                            raw_ids=[k[0] for k in mapping], part_ids=[k[1] for k in mapping])
        results[mode] = dict(count=len(outputs), seconds=elapsed, changes=changes, refinement=stage)
    assert sha(row['source']) == previous['source_sha256'] and sha(row['skeleton']) == previous['skeleton_sha256']
    result = dict(status='passed', key=args.key, source=row['source'], modes=results,
                  checks=dict(cached_real_cgal_and_initial_length=True, real_changed_length_and_final_filters=True,
                              exact_control_replay=args.control_cache is None,
                              control_reused_from_verified_replay=args.control_cache is not None,valid_positive_volumes=True,
                              masks_match_outputs=True, no_overlap=True, sources_unmodified=True))
    save(dest / 'result.json', result)
    print(json.dumps(dict(key=args.key, modes={k:dict(count=v['count'],changes=len(v['changes'])) for k,v in results.items()})), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--key')
    parser.add_argument('--control-cache',type=Path)
    parser.add_argument('--smooth-neck-clearance',type=float,default=300)
    args = parser.parse_args()
    args.frozen, args.output = args.frozen.resolve(), args.output.resolve()
    if args.key:
        return worker(args)
    args.output.mkdir(parents=True)
    snapshot = args.output / 'code_snapshot/neurd'
    shutil.copytree(args.frozen / 'code_snapshot/neurd', snapshot)
    repo = Path(__file__).resolve().parents[1]
    for name in ('spine_utils.py', 'spine_refinement.py', 'spine_volume.py'):
        shutil.copy2(repo / 'neurd' / name, snapshot / name)
    old = json.loads((args.frozen / 'manifest.json').read_text())['refinement_config']
    configs = dict(control=old, smooth_heads={**old, 'screen_smooth_incomplete_heads': True,
                   'smooth_head_minimum_added_clearance_nm':args.smooth_neck_clearance})
    save(args.output / 'manifest.json', dict(configs=configs, frozen=str(args.frozen),
         code_sha256={p.name:sha(p) for p in snapshot.glob('*.py')},
         scope='Development replay on previously inspected branches; cached original CGAL and length stages.'))
    pinned_runner=args.output/'evaluate_worker.py'
    shutil.copy2(__file__,pinned_runner)
    selection = json.loads((args.frozen / 'selection.json').read_text())['selection']
    def job(row):
        dest = args.output / 'branches' / row['key']
        dest.mkdir(parents=True)
        with (dest / 'run.log').open('w') as log:
            command=[sys.executable, str(pinned_runner), '--frozen', str(args.frozen), '--output', str(args.output), '--key', row['key']]
            if args.control_cache:
                command += ['--control-cache',str(args.control_cache.resolve())]
            proc = subprocess.run(command,
                                  stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        if proc.returncode:
            return dict(status='failed', key=row['key'], returncode=proc.returncode)
        return json.loads((dest / 'result.json').read_text())
    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(job, r) for r in selection]):
            row = future.result()
            results.append(row)
            save(args.output / 'results.json', results)
            print(json.dumps(dict(completed=len(results), key=row['key'], status=row['status'],
                 modes={k:dict(count=v['count'],changes=len(v['changes'])) for k,v in row.get('modes',{}).items()})), flush=True)
    if any(r['status'] != 'passed' for r in results):
        raise RuntimeError('Selected branches failed; inspect logs')


if __name__ == '__main__':
    main()
