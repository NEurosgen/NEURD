#!/usr/bin/env python3
"""Bounded, read-only frozen-AE diagnostic for reviewed spine refinements.

Scores are descriptive. This script neither changes a segmentation nor uses
distance from Minnie65 as a deletion rule. Outputs live in a new result folder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import torch
import trimesh

MODEL_SHA = '4ed3c7151fcc2f78bf2dbfa69d69b870d70c4e01b87e015b040e3afdf77562b9'
SEED = 20260924
POINTS = 4096
VIEWS = 16


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def masks(path):
    with np.load(path) as z:
        groups = np.split(z['faces'], z['offsets'][1:-1].astype(int))
        result = {(int(raw), int(part)): group.copy()
                  for raw, part, group in zip(z['raw_ids'], z['part_ids'], groups)}
        assert len(result) == len(groups)
        return result


def mesh_arrays(parent, mask):
    faces = np.asarray(parent.faces)[mask]
    used, inverse = np.unique(faces, return_inverse=True)
    return np.asarray(parent.vertices)[used], inverse.reshape(-1, 3)


def normalize(vertices, faces):
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                   triangles[:, 2]-triangles[:, 0]), axis=1)/2
    assert np.isfinite(vertices).all() and areas.sum() > 0
    center = np.average(triangles.mean(axis=1), axis=0, weights=areas)
    scale = np.sqrt(areas.sum())
    normalized = (vertices-center)/scale
    np.testing.assert_allclose(np.average(normalized[faces].mean(1), axis=0,
                                         weights=areas), 0, atol=1e-9)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                    faces[:, [2, 0]]]), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = edges[counts == 1]
    perimeter = np.linalg.norm(vertices[boundary[:, 0]]-vertices[boundary[:, 1]],
                               axis=1).sum()
    _, degrees = np.unique(boundary, return_counts=True)
    geometry = dict(faces=len(faces), area_nm2=float(areas.sum()),
                    area_scale_nm=float(scale), center_nm=center.tolist(),
                    boundary_perimeter_normalized=float(perimeter/scale),
                    branched_boundary_vertices=int(np.count_nonzero(degrees != 2)),
                    nonmanifold_edges=int(np.count_nonzero(counts > 2)))
    return normalized, areas, geometry


def chamfer_kdtree(a, b):
    # Exact Euclidean nearest-neighbor search; same bidirectional squared mean
    # as the model's Chamfer, with no density downsampling or alignment.
    ab = cKDTree(b).query(a)[0]
    ba = cKDTree(a).query(b)[0]
    return float((np.mean(ab**2)+np.mean(ba**2))/2)


def encode(model, points, device):
    codes, frames = [], []
    with torch.inference_mode():
        for start in range(0, len(points), 2):
            z, frame, _, _ = model.encode(torch.from_numpy(points[start:start+2]).to(device))
            codes.append(z.cpu().numpy()); frames.append(frame.cpu().numpy())
    codes, frames = np.concatenate(codes), np.concatenate(frames)
    assert np.isfinite(codes).all() and np.isfinite(frames).all()
    return codes, frames


def neighbor_scores(queries, reference, metadata):
    distances, indices = cKDTree(reference).query(queries, k=min(256, len(reference)))
    scores, neighbors = [], []
    for ds, ids in zip(distances, indices):
        kept, seen = [], set()
        for distance, index in zip(ds, ids):
            row = metadata.iloc[int(index)]
            if row.source_id in seen:
                continue
            seen.add(row.source_id)
            kept.append(dict(spine_key=row.spine_key, neuron_id=row.source_id,
                             mesh_path=row.mesh_path, distance=float(distance)))
            if len(kept) == 5:
                break
        assert len(kept) == 5
        scores.append(float(np.mean([r['distance'] for r in kept])))
        neighbors.append(kept)
    return np.asarray(scores), neighbors


def verify_atlas_rows(root, table, raw_bank, selected):
    """Check saved code rows against object keys, neuron IDs and mesh hashes."""
    sources, checks = {}, []
    for code_source, group in table.loc[selected].groupby('code_source'):
        code_path = root/code_source
        folder = code_path.parent
        selection_path, contract_path = folder/'selection.parquet', folder/'contract.json'
        selection = pd.read_parquet(selection_path)
        assert selection.spine_key.is_unique
        assert json.loads(contract_path.read_text())['model_sha256'] == MODEL_SHA
        ids = group.source_row.to_numpy(dtype=int)
        assert (ids >= 0).all() and (ids < len(selection)).all()
        original = selection.iloc[ids].reset_index(drop=True)
        target = group.reset_index(drop=True)
        np.testing.assert_array_equal(original.spine_key, target.spine_key)
        np.testing.assert_array_equal(original.neuron_id.astype(str), target.source_id.astype(str))
        np.testing.assert_array_equal(original.mesh_sha256, target.mesh_sha256)
        cached = np.load(code_path, mmap_mode='r')
        assert cached.shape == (len(selection), 2, 64)
        np.testing.assert_array_equal(raw_bank[group.index], cached[ids, 0])
        for dependency in [code_path, selection_path, contract_path]:
            sources[str(dependency)] = sha(dependency)
        checks.append(dict(code_source=code_source, rows=len(group),
                           keys_neurons_mesh_hashes_and_codes_match=True))
    return sources, checks


def plot_scores(records, pairs, table, output, threshold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(accepted='#28906b', main_target='#c34b43', tolerated='#bc9436',
                  uncertain='#777777', additional_repair='#497dae')
    names = dict(accepted='Accepted by user (8)', main_target='Main errors (9)',
                 tolerated='Tolerated bumps / collars (6)', uncertain='Uncertain (1)')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for group, label in names.items():
        rows = [r for r in records if r['variant']=='before' and r['group']==group]
        axes[0].scatter([r['minnie_neighbor_distance_mean'] for r in rows],
                        [r['reconstruction_chamfer_mean'] for r in rows],
                        c=colors[group], s=48, label=label)
        for r in rows:
            axes[0].annotate(r['case_id'], (r['minnie_neighbor_distance_mean'],
                                          r['reconstruction_chamfer_mean']),
                             xytext=(4, 4), textcoords='offset points', fontsize=8)
    axes[0].axvline(threshold, color='#999999', ls='--', lw=1,
                   label='Minnie held-out distance q95')
    axes[0].set(xlabel='Distance to Minnie (five distinct reference neurons)',
                ylabel='Reconstruction Chamfer (normalized squared distance)',
                yscale='log', title='Original human-reviewed cases')
    axes[0].legend(fontsize=8, loc='upper left')
    changed = [p for p in pairs if not p['unchanged']]
    for i, p in enumerate(changed):
        label = p['case_id'] if p['case_id'].startswith('C') else 'raw '+str(records[p['before_index']]['raw_id'])
        values = [p['reconstruction_chamfer_mean_before'], p['reconstruction_chamfer_mean_after']]
        axes[1].plot([0, 1], values, '-o', label=label, alpha=.9)
    axes[1].set(xticks=[0, 1], xticklabels=['Before', 'After'], yscale='log',
                ylabel='Reconstruction Chamfer', title='Seven actual parent-surface completions')
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=.15)
    fig.savefig(output/'shape_quality_comparison.png', dpi=170)
    plt.close(fig)
    if all('frozen_shape_umap' in r for r in records):
        fig, ax = plt.subplots(figsize=(9, 7), layout='constrained')
        for dataset, color in [('minnie65', '#61a98b'), ('h01', '#c5ac6b')]:
            rows = table[table.dataset.eq(dataset)]
            ax.scatter(rows.shape_umap1, rows.shape_umap2, s=2, alpha=.16, c=color, label=dataset)
        for p in changed:
            a = np.asarray(records[p['before_index']]['frozen_shape_umap'])
            b = np.asarray(records[p['after_index']]['frozen_shape_umap'])
            ax.annotate('', b, a, arrowprops=dict(arrowstyle='->', color='#466b86', lw=1.4))
            ax.scatter(*a, c='#c34b43', marker='x', s=48)
            ax.scatter(*b, c='#157a9d', s=40)
            label = p['case_id'] if p['case_id'].startswith('C') else 'raw '+str(records[p['before_index']]['raw_id'])
            ax.annotate(label, b, xytext=(4, 3), textcoords='offset points', fontsize=9)
        ax.set(xlabel='Frozen UMAP1', ylabel='Frozen UMAP2',
               title='Existing shape atlas: red before, blue after\nFrozen transform; UMAP movement is descriptive')
        ax.legend()
        fig.savefig(output/'frozen_umap_comparison.png', dpi=170)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['model-root', 'comparison', 'review', 'policy', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--project-umap', action='store_true')
    parser.add_argument('--mesh-root-relocation', type=Path, nargs=2, action='append', default=[],
                        metavar=('OLD_ROOT', 'NEW_ROOT'),
                        help='Resolve a missing historical path only when the relocated bytes match its saved SHA256.')
    args = parser.parse_args()
    root = args.model_root.resolve()
    # External code/data are read-only; all writes use the explicit output.
    sys.path.insert(0, str(root))
    from analysis.vn_folding.prepare import surface_sample
    from spine_diffusion.vn_folding import VNFoldingAutoencoder, chamfer
    model_path = root/'runs/vn_folding/minnie100k_fresh4096_v1/model/best.pt'
    assert sha(model_path) == MODEL_SHA
    ck = torch.load(model_path, map_location='cpu', weights_only=False)
    assert ck['epoch'] == 23
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VNFoldingAutoencoder(**ck['model_options']).to(device).eval()
    model.load_state_dict(ck['model'])
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'objects').mkdir()
    start = time.monotonic()
    small_a = np.random.default_rng(SEED).normal(size=(11, 3)).astype('float32')
    small_b = np.random.default_rng(SEED+1).normal(size=(17, 3)).astype('float32')
    torch_cd = float(chamfer(torch.from_numpy(small_a[None]), torch.from_numpy(small_b[None]))[0][0])
    assert abs(chamfer_kdtree(small_a, small_b)-torch_cd) < 1e-6

    # Replay saved mean8 codes to verify the complete mesh/normalization/sampling
    # path before interpreting distances to the existing reference bank.
    replay, replay_sources = [], {}
    replay_plan = [('minnie65_exc_development', 0), ('minnie65_exc_holdout', 0),
                   ('h01_exc', 0), ('h01_inhibitory', 0)]
    for bank, index in replay_plan:
        folder = root/'runs/shape_spectrum_v1/codes'/bank
        selection = pd.read_parquet(folder/'selection.parquet')
        row = selection.iloc[index]
        mesh_path = Path(row.mesh_path)
        if not mesh_path.exists():
            for old_root, new_root in args.mesh_root_relocation:
                try:
                    alternative = new_root/mesh_path.relative_to(old_root)
                except ValueError:
                    continue
                if alternative.exists():
                    assert sha(alternative) == row.mesh_sha256
                    mesh_path = alternative
                    break
        assert sha(mesh_path) == row.mesh_sha256
        mesh = trimesh.load(mesh_path, process=False, force='mesh')
        replay_sources[str(mesh_path)] = row.mesh_sha256
        for dependency in ['selection.parquet', 'codes.npy', 'contract.json']:
            replay_sources[str(folder/dependency)] = sha(folder/dependency)
        assert json.loads((folder/'contract.json').read_text())['model_sha256'] == MODEL_SHA
        if 'mesh_cache_path' in row and pd.notna(row.mesh_cache_path):
            assert sha(row.mesh_cache_path) == row.mesh_cache_sha256
            replay_sources[str(row.mesh_cache_path)] = row.mesh_cache_sha256
            with np.load(row.mesh_cache_path) as q:
                vertices, faces, areas = q['vertices'], q['faces'], q['areas']
        else:
            assert sha(row.point_path) == row.point_sha256
            replay_sources[str(row.point_path)] = row.point_sha256
            with np.load(row.point_path) as q:
                vertices = (np.asarray(mesh.vertices)-q['center_nm'])/q['area_scale_nm']
            faces, areas = np.asarray(mesh.faces), mesh.area_faces
        seed = int(hashlib.sha256(f'{SEED}:views:{row.spine_key}'.encode()).hexdigest()[:16], 16)
        points = surface_sample(vertices, faces, areas, VIEWS*POINTS,
                                np.random.default_rng(seed)).reshape(VIEWS, POINTS, 3)
        individual, _ = encode(model, points, device)
        codes = individual.reshape(2, 8, 64).mean(1)
        expected = np.load(folder/'codes.npy', mmap_mode='r')[index]
        error = float(np.max(np.abs(codes-expected)))
        assert error < 1e-4, (bank, error)
        replay.append(dict(bank=bank, index=index, spine_key=row.spine_key,
                           max_absolute_error=error, mesh_sha256=row.mesh_sha256,
                           original_mesh_path=row.mesh_path, resolved_mesh_path=str(mesh_path),
                           contract_sha256=sha(folder/'contract.json')))
    dump(out/'reference_replay.json', replay)

    review = json.loads(args.review.read_text())
    policy = json.loads(args.policy.read_text())
    reviewed = {(r['key'], r['raw_id'], r['part_id']): r for r in review['cases']}
    comparison = json.loads((args.comparison/'results.json').read_text())
    records, pairs = [], []
    sources = {str(p.resolve()): sha(p) for p in [args.review, args.policy,
                args.comparison/'results.json', args.comparison/'manifest.json',
                model_path, root/'spine_diffusion/vn_folding.py',
                root/'analysis/vn_folding/prepare.py', Path(__file__)]}
    sources.update(replay_sources)
    independent_repeat_checked = False
    for row in comparison:
        assert row['status'] == 'passed'
        source = Path(row['selection']['source'])
        assert sha(source) == row['source_sha256']
        sources[str(source)] = row['source_sha256']
        folder = args.comparison/'branches'/row['key']
        previous_path, candidate_path = folder/'previous_faces.npz', folder/'candidate_faces.npz'
        sources[str(previous_path.resolve())] = sha(previous_path)
        sources[str(candidate_path.resolve())] = sha(candidate_path)
        previous, candidate = masks(previous_path), masks(candidate_path)
        targets = sorted({(r['raw_id'], r['part_id']) for r in review['cases'] if r['key'] == row['key']} |
                         {(r['raw_id'], part) for r in row['changes']
                          for raw, part in candidate if raw == r['raw_id']})
        if not targets:
            continue
        parent = trimesh.load(source, process=False, force='mesh')
        for raw, part in targets:
            key = f"{row['key']}_raw_{raw:03d}_part_{part:02d}"
            human = reviewed.get((row['key'], raw, part))
            case_id = human['case_id'] if human else key
            group = 'additional_repair'
            if human:
                group = 'uncertain'
                for name, ids in [('main_target', sum(policy['main_targets'].values(), [])+['C02']),
                                  ('accepted', policy['accepted_geometry_controls']),
                                  ('tolerated', policy['lower_priority_tolerated_bumps']+
                                   policy['tolerated_minor_shaft_collars']+policy['accepted_complex_case'])]:
                    if case_id in ids:
                        group = name
                        break
            assert (raw, part) in previous and (raw, part) in candidate
            same = np.array_equal(previous[raw, part], candidate[raw, part])
            before_index = len(records)
            for variant, mask in [('before', previous[raw, part]), ('after', candidate[raw, part])]:
                if variant == 'after' and same:
                    break
                vertices, faces = mesh_arrays(parent, mask)
                normalized, areas, geometry = normalize(vertices, faces)
                # Paired seed is identical for before/after; surface probabilities
                # change with geometry. Independent second bags measure noise.
                seed = int(hashlib.sha256(f'{SEED}:refinement:{key}'.encode()).hexdigest()[:16], 16)
                encode_start = time.monotonic()
                points = surface_sample(normalized, faces, areas, VIEWS*POINTS,
                                        np.random.default_rng(seed)).reshape(VIEWS, POINTS, 3)
                individual, frames = encode(model, points, device)
                sampling_and_encode_seconds = time.monotonic()-encode_start
                codes = individual.reshape(2, 8, 64).mean(1)
                if same and not independent_repeat_checked:
                    repeated, _ = encode(model, points, device)
                    np.testing.assert_array_equal(individual, repeated)
                    independent_repeat_checked = True
                target = surface_sample(normalized, faces, areas, 2*POINTS,
                                        np.random.default_rng(seed+1)).reshape(2, POINTS, 3)
                with torch.inference_mode():
                    decoded = model.decode(torch.from_numpy(codes).to(device),
                                           torch.from_numpy(frames[[0, 8]]).to(device)).cpu().numpy()
                errors = [chamfer_kdtree(decoded[i], target[i]) for i in range(2)]
                cache = out/'objects'/f'{key}_{variant}.npz'
                np.savez_compressed(cache, codes=codes, individual_codes=individual,
                                    input=points[0], reconstruction=decoded[0],
                                    parent_face_indices=mask, vertices_nm=vertices, faces=faces)
                records.append(dict(key=key, case_id=case_id, branch=row['key'],
                                    neuron_id=row['key'].split('_')[1], raw_id=raw, part_id=part,
                                    variant=variant, group=group, unchanged=same,
                                    annotation=human['annotation'] if human else None,
                                    geometry=geometry, codes=codes,
                                    sampling_and_encode_seconds=sampling_and_encode_seconds,
                                    reconstruction_chamfer=errors, object_file=str(cache),
                                    object_sha256=sha(cache)))
                print(f'encoded {case_id} {variant}: {geometry["faces"]} faces', flush=True)
            pairs.append(dict(key=key, case_id=case_id, group=group, unchanged=same,
                              before_index=before_index,
                              after_index=before_index if same else len(records)-1))
    assert len(pairs) == 29 and len(records) == 36
    assert {r['case_id'] for r in pairs if r['case_id'].startswith('C')} == {r['case_id'] for r in review['cases']}

    atlas = root/'runs/joint_dataset_atlas_20261008'
    table = pd.read_parquet(atlas/'atlas_points.parquet')
    with np.load(atlas/'features.npz') as q:
        bank, raw_bank, mean, sd = q['standardized'], q['raw'], q['mean'], q['sd']
    assert len(table) == len(bank) and table.object_id.is_unique
    np.testing.assert_array_equal(bank, (raw_bank-mean)/sd)
    for p in [atlas/'features.npz', atlas/'atlas_points.parquet', atlas/'contract.json']:
        sources[str(p)] = sha(p)
    training = table.dataset.eq('minnie65') & table.encoder_exposure.eq('encoder_train')
    heldout = table.dataset.eq('minnie65') & table.encoder_exposure.eq('held_out')
    bank_sources, row_checks = verify_atlas_rows(root, table, raw_bank, training | heldout)
    sources.update(bank_sources)
    dump(out/'reference_row_verification.json', row_checks)
    assert not set(table.loc[training, 'source_id']) & set(table.loc[heldout, 'source_id'])
    reference = bank[training]
    metadata = table.loc[training].reset_index(drop=True)
    heldout_scores, _ = neighbor_scores(bank[heldout], reference, metadata)
    threshold = float(np.quantile(heldout_scores, .95))
    codes = np.stack([r.pop('codes') for r in records])
    standardized = (codes-mean)/sd
    scores, neighbors = neighbor_scores(standardized.reshape(-1, 64), reference, metadata)
    scores = scores.reshape(len(records), 2)
    for i, record in enumerate(records):
        record['minnie_neighbor_distance_bags'] = scores[i].tolist()
        record['minnie_neighbor_distance_mean'] = float(scores[i].mean())
        record['mean8_resampling_distance'] = float(np.linalg.norm(standardized[i, 0]-standardized[i, 1]))
        record['reconstruction_chamfer_mean'] = float(np.mean(record['reconstruction_chamfer']))
        record['minnie_reference_outlier'] = bool(scores[i].mean() > threshold)
        record['nearest_minnie_distinct_neurons'] = neighbors[2*i]
    for pair in pairs:
        before, after = pair['before_index'], pair['after_index']
        pair['embedding_displacement'] = float(np.linalg.norm(standardized[after].mean(0)-standardized[before].mean(0)))
        pair['resampling_distance_mean'] = float(np.mean([records[i]['mean8_resampling_distance'] for i in [before, after]]))
        pair['displacement_over_resampling'] = pair['embedding_displacement']/max(pair['resampling_distance_mean'], 1e-12)
        for name in ['minnie_neighbor_distance_mean', 'reconstruction_chamfer_mean']:
            pair[name+'_before'] = records[before][name]
            pair[name+'_after'] = records[after][name]
            pair[name+'_relative_change'] = records[after][name]/records[before][name]-1
        for name in ['faces', 'boundary_perimeter_normalized']:
            pair[name+'_before'] = records[before]['geometry'][name]
            pair[name+'_after'] = records[after]['geometry'][name]

    if args.project_umap:
        import joblib
        projection_path = atlas/'umap_shape.joblib'
        projection = joblib.load(projection_path)
        xy = projection.transform(standardized[:, 0].astype('float32'))
        assert np.isfinite(xy).all()
        sources[str(projection_path)] = sha(projection_path)
        for record, point in zip(records, xy):
            record['frozen_shape_umap'] = point.tolist()
        np.save(out/'umap_projection.npy', xy)

    from sklearn.metrics import roc_auc_score
    judged = [r for r in records if r['variant']=='before' and r['group'] in ['accepted', 'main_target']]
    labels = [r['group'] == 'main_target' for r in judged]
    aucs = {name: float(roc_auc_score(labels, [r[name] for r in judged]))
            for name in ['minnie_neighbor_distance_mean', 'reconstruction_chamfer_mean']}
    accepted = [r for r in records if r['variant']=='before' and r['group']=='accepted']
    changed = [r for r in pairs if not r['unchanged']]
    summary = dict(schema='spine-refinement-frozen-ae-probe/1',
                   cases=len(pairs), reviewed_cases=len(review['cases']),
                   unique_surfaces=len(records), repaired_cases=len(changed),
                   minnie_reference_objects=int(training.sum()),
                   minnie_reference_neurons=int(metadata.source_id.nunique()),
                   heldout_objects=int(heldout.sum()),
                   heldout_neurons=int(table.loc[heldout, 'source_id'].nunique()),
                   heldout_q95_distance=threshold,
                   accepted_human_controls=len(accepted),
                   accepted_controls_flagged_as_minnie_outliers=[r['case_id'] for r in accepted if r['minnie_reference_outlier']],
                   targeted_sample_auc=aucs,
                   repaired_cases_with_smaller_minnie_distance=sum(r['minnie_neighbor_distance_mean_relative_change'] < 0 for r in changed),
                   repaired_cases_with_smaller_reconstruction_error=sum(r['reconstruction_chamfer_mean_relative_change'] < 0 for r in changed),
                   all_unchanged_masks_have_identical_codes=all(r['embedding_displacement']==0 for r in pairs if r['unchanged']),
                   independent_identical_input_replay_passed=independent_repeat_checked,
                   reference_rows_verified_by_keys=int((training | heldout).sum()),
                   sampling_and_encode_median_seconds=float(np.median([r['sampling_and_encode_seconds'] for r in records])),
                   device=device, elapsed_seconds=time.monotonic()-start,
                   limitations=['Targeted review sample, not population prevalence or independent classifier evaluation.',
                                'Minnie distance may reflect true H01 shape differences or acquisition/segmentation domain shift.',
                                'Point encoder does not see mesh connectivity, shaft attachment or anatomy.',
                                'Two mean8 bags only; displacement/noise ratio is descriptive, not a confidence interval.',
                                'Decoded clouds are diagnostic; all segmentation repairs use actual parent faces.'],
                   segmentation_changed=False, model_trained=False, full_h01_processing_started=False)
    contract = dict(checkpoint_sha256=MODEL_SHA, checkpoint_epoch=23,
                    normalization='Exact triangle-area surface center; divide by sqrt(original area) before sampling.',
                    inference='Two independent mean8 bags; 4096 area-uniform points per view; eval frozen model.',
                    scoring='Common existing scaler; mean distance to five nearest Minnie development objects from distinct neurons.',
                    reconstruction='Mean8 code decoded with each bag first-view frame; exact bidirectional squared Chamfer to independent 4096-point target.',
                    threshold='Descriptive 95th percentile of frozen Minnie neuron-held-out distances; never applied to segmentation.',
                    unchanged='Identical masks share cached outputs; one actual double inference verifies determinism.',
                    seed=SEED, model_root=str(root), comparison=str(args.comparison.resolve()),
                    review=str(args.review.resolve()), policy=str(args.policy.resolve()),
                    output=str(out), mesh_root_relocations=[[str(x) for x in pair] for pair in args.mesh_root_relocation],
                    project_umap=args.project_umap, sources=sources)
    np.savez_compressed(out/'codes.npz', raw=codes, standardized=standardized, mean=mean, sd=sd)
    dump(out/'records.json', records); dump(out/'pairs.json', pairs)
    pd.DataFrame(pairs).to_csv(out/'paired_metrics.csv', index=False)
    dump(out/'contract.json', contract); dump(out/'summary.json', summary)
    plot_scores(records, pairs, table, out, threshold)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
