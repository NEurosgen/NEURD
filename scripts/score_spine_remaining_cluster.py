#!/usr/bin/env python3
"""Score a bounded mask audit in the replayed historical shape-only island.

The old island is a descriptive ROI, not an anatomical classifier. Record
both independent bags and retain all cases regardless of their membership.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from probe_spine_refinement_embeddings import sha, neighbor_scores, verify_atlas_rows


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--encoder-targets', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.model_root.resolve(), args.audit.resolve()
    started = time.monotonic()
    atlas = root/'runs/joint_dataset_atlas_20261008'
    historical = atlas/'diagnostics/baseline'
    replay = json.loads((out/'historical_projection_replay.json').read_text())
    assert replay['coordinate_replay_passed'] and replay['max_absolute_coordinate_error'] < 1e-5
    model = joblib.load(out/'historical_umap_shape.joblib')
    table = pd.read_parquet(atlas/'atlas_points.parquet')
    old_table = pd.read_parquet(historical/'atlas_points.parquet')
    with np.load(atlas/'features.npz') as q:
        mean, sd, bank, raw_bank = q['mean'], q['sd'], q['standardized'], q['raw']
    with np.load(historical/'features.npz') as q:
        np.testing.assert_array_equal(mean,q['mean'])
        np.testing.assert_array_equal(sd,q['sd'])
    np.testing.assert_array_equal(model.embedding_,old_table[['shape_umap1','shape_umap2']].to_numpy('float32'))
    training = table.dataset.eq('minnie65') & table.encoder_exposure.eq('encoder_train')
    holdout = table.dataset.eq('minnie65') & table.encoder_exposure.eq('held_out')
    sources, row_checks = verify_atlas_rows(root,table,raw_bank,training|holdout)
    save(out/'reference_row_verification.json',row_checks)
    reference, metadata = bank[training],table.loc[training].reset_index(drop=True)
    heldout_scores,_ = neighbor_scores(bank[holdout],reference,metadata)
    cutoff = float(np.quantile(heldout_scores,.95))
    with np.load(out/'codes.npz') as q:codes = q['raw']
    standardized = (codes-mean)/sd
    points = model.transform(standardized.reshape(-1,64).astype('float32')).reshape(len(codes),2,2)
    assert np.isfinite(points).all()
    scores, neighbors = neighbor_scores(standardized.reshape(-1,64),reference,metadata)
    scores = scores.reshape(len(codes),2)
    objects = json.loads((out/'objects.json').read_text())
    references = json.loads((out/'references.json').read_text())
    review = json.loads(args.review.read_text())
    labels = {(r['key'],r['raw_id'],r['part_id']):r for r in review['cases']}
    encoder_targets = json.loads(args.encoder_targets.read_text())
    encoder_labels = {(r['key'],r['source_spine_index'],0):r['case_id'] for r in encoder_targets}
    for i,(record,xy,ds) in enumerate(zip(objects,points,scores)):
        assert record['index']==i
        inside = (xy[:,0]>=21.5) & (xy[:,0]<25.3) & (xy[:,1]<5)
        record.update(historical_shape_umap_bags=xy.tolist(),historical_island_bags=inside.tolist(),
                      historical_island=bool(inside[0]),historical_island_both_bags=bool(inside.all()),
                      island_sampling_agreement=bool(inside[0]==inside[1]),
                      minnie_distance_bags=ds.tolist(),minnie_distance_mean=float(ds.mean()),
                      minnie_distance_above_heldout_q95=bool(ds.mean()>cutoff),
                      nearest_minnie_distinct_neurons=neighbors[2*i])
        human = labels.get((record['branch'],record['raw_id'],record['part_id']))
        record['human_case_id'] = human['case_id'] if human else None
        record['human_annotation'] = human['annotation'] if human else None
        record['encoder_case_id'] = encoder_labels.get((record['branch'],record['raw_id'],record['part_id']))
    summaries = []
    for cohort in sorted({r['cohort'] for r in references}):
        for variant in ['archived','previous','candidate']:
            rows = [r for r in references if r['cohort']==cohort and r['variant']==variant]
            if not rows:continue
            ids = [r['object_index'] for r in rows]
            selected = [objects[i] for i in ids]
            summaries.append(dict(cohort=cohort,variant=variant,objects=len(rows),
                 island_bag0=sum(r['historical_island'] for r in selected),
                 island_bag1=sum(r['historical_island_bags'][1] for r in selected),
                 island_both_bags=sum(r['historical_island_both_bags'] for r in selected),
                 island_either_bag=sum(any(r['historical_island_bags']) for r in selected),
                 native_minnie_outliers=sum(r['minnie_distance_above_heldout_q95'] for r in selected),
                 reconstruction_error_median=float(np.median([r['reconstruction_chamfer_mean'] for r in selected]))))
    candidates = []
    for r in references:
        o = objects[r['object_index']]
        if r['variant']=='candidate' and (any(o['historical_island_bags'])
                or o['human_case_id'] in ['C02','C13','C18']
                or o['encoder_case_id'] in ['E08','E10','E11']):
            candidates.append({**r,**o})
    np.savez_compressed(out/'scored_codes.npz',raw=codes,standardized=standardized,
                        mean=mean,sd=sd,historical_umap=points)
    save(out/'scored_objects.json',objects)
    save(out/'remaining_review_candidates.json',candidates)
    summary = dict(stage='complete_encoding_and_scoring',cohorts=summaries,
        historical_roi='21.5 <= original shape UMAP1 < 25.3 and original UMAP2 < 5',
        original_projection_reproduced_exactly=True,minnie_heldout_q95_distance=cutoff,
        historical_roi_is_not_anatomical_error_label=True,model_trained=False,segmentation_changed=False,
        full_h01_processing_started=False,targeted_branches_not_population_prevalence=True,
        human_cases_remaining=[dict(case_id=r['human_case_id'],island_bags=r['historical_island_bags'],
                                   minnie_distance=r['minnie_distance_mean']) for r in candidates if r['human_case_id']],
        encoder_cases_remaining=[dict(case_id=r['encoder_case_id'],island_bags=r['historical_island_bags'],
                                     minnie_distance=r['minnie_distance_mean']) for r in candidates if r['encoder_case_id']],
        membership_bag0_primary=True,second_bag_is_sampling_sensitivity=True,
        elapsed_seconds=time.monotonic()-started)
    save(out/'cluster_summary.json',summary)
    for p in [atlas/'atlas_points.parquet',atlas/'features.npz',historical/'atlas_points.parquet',
              historical/'features.npz',out/'encoding_contract.json',out/'historical_umap_shape.joblib',
              out/'historical_projection_replay.json',out/'codes.npz',args.review,
              args.encoder_targets,Path(__file__)]:sources[str(p)]=sha(p)
    save(out/'scoring_contract.json',dict(source_hashes=sources,roi=summary['historical_roi'],
         no_dataset_specific_centering=True,no_new_cluster_fitting=True,
         historical_projection_recovered_and_coordinate_replay_verified=True))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
