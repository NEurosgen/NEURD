"""Build a bounded, offline H01 EM/mesh annotation viewer (no segmentation run).

Example (torch_5060 Python):
  python scripts/build_h01_em_review.py --phase prepare
  python scripts/build_h01_em_review.py --phase download --limit 1
  python scripts/build_h01_em_review.py --phase download
  python scripts/build_h01_em_review.py --phase render

The fixed, targeted sample supports manual diagnosis, not prevalence estimates.
All coordinates are nm; H01 contact coordinates are already converted to nm.
Native XY sections pass through voxel centres. Missing voxels are an error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from PIL import Image

MAIN = Path('/home/eugen/Desktop/CodeWork/Projects/NEURD_modif/NEURD')
DATA = Path('/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/DiffusionNet_spines')
ARCHIVE = Path('/mnt/wwn-0x50014ee26c2ca7b0-part1/Archiv/h01/H01_Seg')
SOURCE = 'precomputed://https://storage.googleapis.com/h01-release/data/20210601/4nm_raw'
ASSETS = Path(__file__).with_name('h01_em_review')
AUDIT_IDS = [5, 19, 26, 30, 51, 77, 94, 106, 147, 169, 192, 209,
             8, 41, 137, 196, 66, 166, 14, 4, 7, 25, 40, 161]
SEED = 20261009
RESOLUTION = np.array([8, 8, 33], dtype=int)
PADDING = np.array([800, 800, 500], dtype=int)
MAX_VOXELS = 30_000_000  # Enforced before any network request, per case.
MAX_TOTAL_VOXELS = 250_000_000


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare(out):
    frame_path = MAIN / 'artifacts/h01_residual_qc_20261009/sampling_frame.json'
    rows = {r['audit_id']: r for r in json.loads(frame_path.read_text())['selected']}
    contacts_path = DATA / 'runs/shape_spectrum_v1/h01_contacts_v20210729/preprocessed/contacts_ready.parquet'
    neurons_path = contacts_path.with_name('neurons_ready.parquet')
    cols = ['annotation_id', 'neuron_id', 'synapse_type', 'synapse_type_label',
            'post_x_nm', 'post_y_nm', 'post_z_nm']
    contacts = pd.read_parquet(contacts_path, columns=cols)
    contacts['neuron_id'] = contacts['neuron_id'].astype(str)
    neurons = pd.read_parquet(neurons_path)
    neurons['neuron_id'] = neurons['neuron_id'].astype(str)
    assert not neurons['neuron_id'].duplicated().any()
    neurons = neurons.set_index('neuron_id')
    order = np.random.default_rng(SEED).permutation(AUDIT_IDS).tolist()
    cases = []
    for n, audit_id in enumerate(order, 1):
        row = rows[audit_id]
        key = row['key']
        _, neuron_id, _, limb, _, branch = key.split('_')
        result_path = MAIN / 'artifacts/spine_shaft_trim_20261009/child_guard_comparison/branches' / key / 'result.json'
        result = json.loads(result_path.read_text())
        parts = [p for p in result['candidate_outputs'] if p['raw_id'] == row['raw_id']]
        if len(parts) != 1:
            raise ValueError(f'{key} raw {row["raw_id"]}: expected one surviving part, found {len(parts)}')
        candidate = Path(parts[0]['mesh'])
        parent = ARCHIVE / f'neuron_{neuron_id}' / f'limb_{limb}' / f'branch_{branch}' / 'branch_mesh.off'
        mesh = trimesh.load(candidate, process=False)
        low = np.floor((mesh.bounds[0] - PADDING) / RESOLUTION).astype(int)
        high = np.ceil((mesh.bounds[1] + PADDING) / RESOLUTION).astype(int)
        shape = high - low
        if np.any(low < 0) or np.prod(shape) > MAX_VOXELS:
            raise ValueError(f'Unbounded crop {key}: {shape}')
        assert neuron_id in neurons.index
        contact_subset = contacts[contacts['neuron_id'] == neuron_id]
        points = contact_subset[['post_x_nm', 'post_y_nm', 'post_z_nm']].to_numpy(float)
        inside = np.all((points >= low * RESOLUTION) & (points < high * RESOLUTION), axis=1)
        local = []
        for _, syn in contact_subset.loc[inside].iterrows():
            local.append(dict(id=str(syn['annotation_id']), kind=int(syn['synapse_type']),
                              kind_label=str(syn['synapse_type_label']),
                              xyz_nm=[float(syn[f'post_{a}_nm']) for a in 'xyz']))
        case = dict(id=f'C{n:02}', audit_id=audit_id, key=key, neuron_id=neuron_id,
                    raw_id=int(row['raw_id']), part_id=int(parts[0]['part_id']),
                    final_index=int(parts[0]['final_index']), target_source=str(candidate),
                    parent_source=str(parent), target_sha256=digest(candidate), parent_sha256=digest(parent),
                    bbox_min_voxels=low.tolist(), bbox_max_voxels=high.tolist(),
                    shape_xyz=shape.tolist(), resolution_nm=RESOLUTION.tolist(),
                    mesh_bounds_nm=mesh.bounds.tolist(), target_faces=len(mesh.faces),
                    source_c3_id=str(neurons.loc[neuron_id, 'c3_rep_strict']),
                    synapses=local, total_listed_neuron_contacts=len(contact_subset),
                    image_source=SOURCE, mip=1, path=f'cases/C{n:02}')
        cases.append(case)
    total = sum(int(np.prod(c['shape_xyz'])) for c in cases)
    assert total <= MAX_TOTAL_VOXELS, total
    identity = dict(cases=cases, source=SOURCE, mip=1, seed=SEED,
                    contacts_sha256=digest(contacts_path), neurons_sha256=digest(neurons_path),
                    sampling_frame_sha256=digest(frame_path))
    identity['manifest_id'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest = dict(schema='h01-em-review/1', **identity, created_date='2026-10-09',
                    sample_kind='Targeted diagnostic set, shuffled; not a prevalence sample',
                    annotation_ground_truth=False, requested_cases=len(cases), total_raw_voxels=total,
                    padding_nm=PADDING.tolist(), max_voxels_per_case=MAX_VOXELS,
                    contacts_source=str(contacts_path),
                    contacts_note='Listed C3 postsynaptic points; incomplete soma-representative subset. '
                                  'IDs map through neurons_ready; absence does not establish absence of a synapse. '
                                  'Points are localization aids, not assignments to the selected spine.',
                    selection_rule='12 residual geometry cases, 4 ambiguous short protrusions, '
                                   '4 controls including two protected true-spine examples, '
                                   '4 incomplete-head/neck examples, based on previous mesh-only review; '
                                   'previous judgments are hidden in the viewer.')
    path = out / 'manifest.json'
    if path.exists() and json.loads(path.read_text())['manifest_id'] != manifest['manifest_id']:
        raise ValueError('The output belongs to a different frozen sample; use a new directory.')
    out.mkdir(parents=True, exist_ok=True)
    write_json(path, manifest)
    for case in cases:
        (out / case['path']).mkdir(parents=True, exist_ok=True)
    print(f'Prepared {len(cases)} fixed cases, {total/1e6:.1f} MB of bounded uint8 crops.', flush=True)
    return manifest


def download(out, manifest, limit):
    from cloudvolume import Bbox, CloudVolume
    # Use a private new cache: never modify the user's existing EM caches.
    volume = CloudVolume(SOURCE, mip=1, progress=False, parallel=1,
                         cache=str(out / 'chunk_cache'), bounded=True, fill_missing=False)
    assert np.array_equal(volume.resolution, RESOLUTION)
    write_json(out / 'image_volume_info.json', volume.info)
    for case in manifest['cases'][:limit]:
        folder = out / case['path']
        path = folder / 'em_volume.npz'
        if path.exists():
            with np.load(path) as saved:
                assert saved['manifest_id'].item() == manifest['manifest_id']
                assert saved['image_xyz'].shape == tuple(case['shape_xyz'])
            print(case['id'], 'cached', flush=True)
            continue
        bbox = Bbox(case['bbox_min_voxels'], case['bbox_max_voxels'])
        assert int(np.prod(case['shape_xyz'])) <= MAX_VOXELS
        assert volume.bounds.contains_bbox(bbox)
        print(case['id'], 'downloading', case['key'], bbox, flush=True)
        image = np.asarray(volume.download(bbox))[..., 0]
        assert image.shape == tuple(case['shape_xyz']) and image.dtype == np.uint8
        assert np.ptp(image) > 0
        temp = folder / 'em_volume.partial.npz'
        np.savez_compressed(temp, image_xyz=image,
                            bbox_min_voxels=case['bbox_min_voxels'],
                            bbox_max_voxels=case['bbox_max_voxels'], resolution_nm=RESOLUTION,
                            manifest_id=manifest['manifest_id'])
        temp.replace(path)
        print(case['id'], 'saved', round(path.stat().st_size/1e6, 1), 'MB', flush=True)


def crop_parent(mesh, low, high):
    triangles = mesh.triangles
    keep = np.all(triangles.max(axis=1) >= low, axis=1) & np.all(triangles.min(axis=1) <= high, axis=1)
    indices = np.flatnonzero(keep)
    assert len(indices) > 0
    return mesh.submesh([indices], append=True, repair=False), indices


def section_lines(mesh, z, origin, resolution):
    section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if section is None:
        return [], 0.0
    lines, area = [], 0.0
    # Path.discrete includes closed paths only. Open entities are essential for
    # precisely the ragged surface patches and incomplete heads under review.
    for entity in section.entities:
        path = entity.discrete(section.vertices)
        if np.linalg.norm(path[0] - path[-1]) < 1:
            xy = path[:, :2] - path[:, :2].mean(axis=0)
            area += abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) -
                        np.dot(xy[:, 1], np.roll(xy[:, 0], 1))) / 2
        pixels = (path[:, :2] - origin[:2]) / resolution[:2]
        lines.append(np.round(pixels, 4).tolist())
    return lines, area


def mesh_data(mesh, origin):
    # Keep exact connectivity and all selected vertices; only translate/change units.
    return dict(vertices=((mesh.vertices - origin) / 1000).tolist(), faces=mesh.faces.tolist())


def render_case(out, manifest, case, force=False):
    folder = out / case['path']
    path = folder / 'em_volume.npz'
    if not path.exists():
        raise FileNotFoundError(f'EM is not downloaded for {case["id"]}: {path}')
    stamp = folder / 'render_provenance.json'
    if not force and stamp.exists():
        old = json.loads(stamp.read_text())
        if old['manifest_id'] == manifest['manifest_id'] and old['builder_sha256'] == digest(__file__):
            return
    assert digest(case['target_source']) == case['target_sha256']
    assert digest(case['parent_source']) == case['parent_sha256']
    target = trimesh.load(case['target_source'], process=False)
    parent = trimesh.load(case['parent_source'], process=False)
    with np.load(path) as saved:
        assert saved['manifest_id'].item() == manifest['manifest_id']
        image = saved['image_xyz']
        low = saved['bbox_min_voxels']
        high = saved['bbox_max_voxels']
        res = saved['resolution_nm']
    assert np.array_equal(low, case['bbox_min_voxels']) and np.array_equal(high, case['bbox_max_voxels'])
    assert image.shape == tuple(high-low) and np.array_equal(res, RESOLUTION)
    origin = low * res
    parent_roi, face_ids = crop_parent(parent, origin, high*res)
    np.save(folder / 'parent_face_indices.npy', face_ids)
    shutil.copyfile(case['target_source'], folder / 'target_mesh.off')
    parent_roi.export(folder / 'parent_roi.off')
    minimum, maximum = np.quantile(image, [0.01, 0.99]).tolist()
    # Raw source intensities in NPZ are preserved. One window across all slices.
    display = np.clip((image.astype(np.float32)-minimum)*255/(maximum-minimum), 0, 255).astype(np.uint8)
    slices = folder / 'slices'
    slices.mkdir(exist_ok=True)
    records = []
    for k in range(image.shape[2]):
        z = (int(low[2])+k+0.5)*res[2]
        target_lines, area = section_lines(target, z, origin, res)
        parent_lines, _ = section_lines(parent_roi, z, origin, res)
        # PNG y grows down; physical y grows up. Save row 0 at maximum y.
        Image.fromarray(display[:, :, k].T[::-1]).save(slices / f'{k:03}.png')
        records.append(dict(z_voxel=int(low[2]+k), z_nm=float(z), target=target_lines,
                            parent=parent_lines, area_nm2=area))
    areas = np.array([r['area_nm2'] for r in records])
    initial = int(areas.argmax()) if areas.max() > 0 else int(np.argmax([sum(len(p) for p in r['target']) for r in records]))
    viewer = dict(**case, origin_nm=origin.tolist(), display_range=[minimum, maximum],
                  initial_slice=initial, sections=records,
                  target_mesh=mesh_data(target, origin), parent_mesh=mesh_data(parent_roi, origin),
                  image_sha256=digest(path))
    (folder / 'data.js').write_text('window.H01_CASES=window.H01_CASES||{};window.H01_CASES['+
                                   json.dumps(case['id'])+']='+json.dumps(viewer, ensure_ascii=False,
                                                                          separators=(',', ':'))+';\n', encoding='utf-8')
    write_json(stamp, dict(manifest_id=manifest['manifest_id'], builder_sha256=digest(__file__),
                           image_sha256=viewer['image_sha256'], target_sha256=case['target_sha256'],
                           parent_sha256=case['parent_sha256'], roi_parent_faces=len(face_ids),
                           target_faces=len(target.faces), initial_slice=initial,
                           initial_slice_rule='Largest closed candidate mesh XY-section area; fallback maximum contour points',
                           display_range=[minimum, maximum], raw_intensities_modified=False,
                           png_row0='maximum physical y; PNG width is x, height is y',
                           section_plane='(absolute_z_voxel + 0.5) * resolution_z_nm',
                           contours='All intersection entities, including open lines; no closed-loop-only filtering',
                           synapse_coordinates='Listed postsynaptic points, already nm; no extra scaling'))
    print(case['id'], 'rendered', len(records), 'real slices,', len(face_ids), 'parent faces', flush=True)


def build_viewer(out, manifest):
    from plotly.offline import get_plotlyjs
    assets = out / 'assets'
    assets.mkdir(exist_ok=True)
    (assets / 'plotly.min.js').write_text(get_plotlyjs(), encoding='utf-8')
    for name in ['viewer.js', 'viewer.css']:
        shutil.copyfile(ASSETS / name, assets / name)
    shutil.copyfile(ASSETS / 'viewer.html', out / 'index.html')
    visible = [{k: c[k] for k in ['id', 'key', 'raw_id', 'part_id', 'final_index', 'audit_id', 'path',
                                  'target_sha256', 'parent_sha256']} for c in manifest['cases']]
    index = dict(schema=manifest['schema'], manifest_id=manifest['manifest_id'], cases=visible,
                 image_source=SOURCE, resolution_nm=RESOLUTION.tolist(), created_date='2026-10-09')
    (assets / 'index.js').write_text('window.REVIEW_INDEX='+json.dumps(index, ensure_ascii=False)+';\n', encoding='utf-8')
    write_json(out / 'builder_provenance.json', dict(script=str(Path(__file__).resolve()),
               builder_sha256=digest(__file__), templates={n:digest(ASSETS/n) for n in ['viewer.html','viewer.js','viewer.css']},
               manifest_id=manifest['manifest_id'], cases_ready=sum((out/c['path']/'data.js').exists() for c in manifest['cases'])))
    print('Viewer:', out / 'index.html', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=MAIN / 'artifacts/h01_em_review_20261009')
    parser.add_argument('--phase', choices=['prepare', 'download', 'render', 'viewer', 'all'], default='all')
    parser.add_argument('--limit', type=int, default=len(AUDIT_IDS))
    parser.add_argument('--force-render', action='store_true')
    args = parser.parse_args()
    assert 1 <= args.limit <= len(AUDIT_IDS)
    out = args.out.resolve()
    manifest = prepare(out) if args.phase in ['prepare', 'all'] else json.loads((out/'manifest.json').read_text())
    if args.phase in ['download', 'all']:
        download(out, manifest, args.limit)
    if args.phase in ['render', 'all']:
        for case in manifest['cases'][:args.limit]:
            render_case(out, manifest, case, args.force_render)
        build_viewer(out, manifest)
    elif args.phase == 'viewer':
        build_viewer(out, manifest)


if __name__ == '__main__':
    main()
