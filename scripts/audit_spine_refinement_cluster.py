#!/usr/bin/env python3
"""Encode all masks in a bounded refinement comparison with the frozen shape AE.

No segmentation or model training is run. Exact unchanged masks share their
diagnostic output; prior probes are reused only after verifying source hashes,
parent indices and world geometry. Historical island membership is scored in
a separate step using a projection whose original coordinates were replayed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import trimesh

from probe_spine_refinement_embeddings import (
    MODEL_SHA, SEED, POINTS, VIEWS, sha, masks, mesh_arrays,
    normalize, encode, chamfer_kdtree,
)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def read_masks(path):
    # A branch can legitimately lose every candidate. np.split on its empty
    # face array would otherwise manufacture one group without an object ID.
    with np.load(path) as archive:
        if not len(archive['raw_ids']):
            assert len(archive['part_ids']) == len(archive['faces']) == 0
            np.testing.assert_array_equal(archive['offsets'], [0])
            return {}
    return masks(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['model-root', 'full-comparison', 'extra-comparison', 'previous-probe', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--resume', action='store_true',
                        help='Verify and reuse exact completed object caches after an interrupted audit.')
    args = parser.parse_args()
    root, out = args.model_root.resolve(), args.output.resolve()
    sys.path.insert(0, str(root))
    from analysis.vn_folding.prepare import surface_sample
    from spine_diffusion.vn_folding import VNFoldingAutoencoder

    out.mkdir(parents=True, exist_ok=True)
    (out/'objects').mkdir(exist_ok=True)
    model_path = root/'runs/vn_folding/minnie100k_fresh4096_v1/model/best.pt'
    assert sha(model_path) == MODEL_SHA
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == 23
    torch.set_num_threads(4)
    model = VNFoldingAutoencoder(**checkpoint['model_options']).eval()
    model.load_state_dict(checkpoint['model'])
    old_contract = json.loads((args.previous_probe/'contract.json').read_text())
    assert old_contract['checkpoint_sha256'] == MODEL_SHA
    prior = json.loads((args.previous_probe/'records.json').read_text())
    old_by_key = {}
    for record in prior:
        old_by_key.setdefault((record['branch'], record['raw_id'], record['part_id']), []).append(record)
    sources = {str(p.resolve()):sha(p) for p in [model_path, root/'spine_diffusion/vn_folding.py',
               root/'analysis/vn_folding/prepare.py', Path(__file__),
               Path(__file__).with_name('probe_spine_refinement_embeddings.py'),
               args.previous_probe/'contract.json', args.previous_probe/'records.json']}
    metadata, references, codes, geometries = [], [], [], {}
    started = time.monotonic()
    reused_old, calculated, resumed = 0, 0, 0
    cohorts = [('full_pipeline', args.full_comparison, [('previous','previous'),('candidate','candidate')]),
               ('encoder_selected_geometry', args.extra_comparison,
                [('raw','archived'),('previous','previous'),('candidate','candidate')])]
    for cohort, comparison, variants in cohorts:
        rows = json.loads((comparison/'results.json').read_text())
        assert len(rows) == 12 and all(r['status']=='passed' for r in rows)
        for p in [comparison/'results.json', comparison/'manifest.json']:
            sources[str(p.resolve())] = sha(p)
        for row in rows:
            source = Path(row['selection']['source'])
            assert sha(source) == row['source_sha256']
            sources[str(source)] = row['source_sha256']
            parent = trimesh.load(source, process=False, force='mesh')
            branch = row['key']
            folder = comparison/'branches'/branch
            for mask_name, variant in variants:
                mask_path = folder/(mask_name+'_faces.npz')
                sources[str(mask_path.resolve())] = sha(mask_path)
                for (raw,part), mask in read_masks(mask_path).items():
                    assert len(mask)>0
                    token = hashlib.sha256(np.asarray(mask, dtype='<i8').tobytes()).hexdigest()
                    identity = (branch,raw,part,token)
                    if identity not in geometries:
                        vertices, faces = mesh_arrays(parent, mask)
                        normalized, areas, geometry = normalize(vertices, faces)
                        key = f'{branch}_raw_{raw:03d}_part_{part:02d}'
                        seed = int(hashlib.sha256(f'{SEED}:refinement:{key}'.encode()).hexdigest()[:16],16)
                        old = None
                        for item in old_by_key.get((branch,raw,part), []):
                            cache = Path(item['object_file'])
                            assert sha(cache) == item['object_sha256']
                            with np.load(cache) as q:
                                if (np.array_equal(q['parent_face_indices'],mask)
                                        and np.array_equal(q['vertices_nm'],vertices)
                                        and np.array_equal(q['faces'],faces)):
                                    old = dict(item)
                                    value = q['codes'].copy()
                                    break
                        cache_path = out/'objects'/f'{key}_{token[:12]}.npz'
                        if old is not None:
                            assert value.shape == (2,64)
                            errors = old['reconstruction_chamfer']
                            old_source = dict(path=old['object_file'],sha256=old['object_sha256'])
                            reused_old += 1
                        elif args.resume and cache_path.exists():
                            cache_sha = sha(cache_path)
                            with np.load(cache_path) as q:
                                np.testing.assert_array_equal(q['parent_face_indices'], mask)
                                np.testing.assert_array_equal(q['vertices_nm'], vertices)
                                np.testing.assert_array_equal(q['faces'], faces)
                                value = q['codes'].copy()
                                individual = q['individual_codes'].copy()
                                saved_input = q['input'].copy()
                                reconstruction0 = q['reconstruction'].copy()
                            assert value.shape == (2,64) and individual.shape == (16,64)
                            np.testing.assert_array_equal(value, individual.reshape(2,8,64).mean(1))
                            points = surface_sample(normalized,faces,areas,VIEWS*POINTS,
                                                    np.random.default_rng(seed)).reshape(VIEWS,POINTS,3)
                            np.testing.assert_array_equal(saved_input, points[0])
                            # Repeat the original two-view batch containing bag1's
                            # frame. All 16 latent codes are reused unchanged.
                            check_codes, frames = encode(model,points[8:10],'cpu')
                            np.testing.assert_array_equal(check_codes, individual[8:10])
                            with torch.inference_mode():
                                reconstruction1 = model.decode(torch.from_numpy(value[1:2]),
                                                               torch.from_numpy(frames[:1])).numpy()[0]
                            target = surface_sample(normalized,faces,areas,2*POINTS,
                                                    np.random.default_rng(seed+1)).reshape(2,POINTS,3)
                            errors = [chamfer_kdtree(reconstruction0,target[0]),
                                      chamfer_kdtree(reconstruction1,target[1])]
                            assert sha(cache_path) == cache_sha
                            old_source = None
                            resumed += 1
                        else:
                            points = surface_sample(normalized,faces,areas,VIEWS*POINTS,
                                                    np.random.default_rng(seed)).reshape(VIEWS,POINTS,3)
                            individual, frames = encode(model,points,'cpu')
                            value = individual.reshape(2,8,64).mean(1)
                            target = surface_sample(normalized,faces,areas,2*POINTS,
                                                    np.random.default_rng(seed+1)).reshape(2,POINTS,3)
                            with torch.inference_mode():
                                reconstruction = model.decode(torch.from_numpy(value),
                                                              torch.from_numpy(frames[[0,8]])).numpy()
                            errors = [chamfer_kdtree(reconstruction[i],target[i]) for i in range(2)]
                            np.savez_compressed(cache_path,codes=value,individual_codes=individual,
                                                vertices_nm=vertices,faces=faces,parent_face_indices=mask,
                                                input=points[0],reconstruction=reconstruction[0])
                            old_source = None
                            calculated += 1
                        index = len(metadata)
                        geometries[identity] = index
                        codes.append(value)
                        metadata.append(dict(index=index,key=key,branch=branch,raw_id=raw,part_id=part,
                            cohort=cohort,geometry=geometry,mask_sha256=token,
                            reconstruction_chamfer=errors,reconstruction_chamfer_mean=float(np.mean(errors)),
                            seed=seed,object_file=str(cache_path) if old is None else old['object_file'],
                            object_sha256=sha(cache_path) if old is None else old['object_sha256'],
                            reused_prior_probe=old_source))
                        if index % 25 == 0:
                            print(json.dumps(dict(unique_surfaces=len(metadata),new_encoded=calculated,
                                  reused=reused_old,resumed=resumed,
                                  elapsed_seconds=round(time.monotonic()-started,1))),flush=True)
                        if index % 100 == 0:
                            save(out/'progress.json',dict(unique_surfaces=len(metadata),new_encoded=calculated,
                                  reused=reused_old,resumed=resumed,elapsed_seconds=time.monotonic()-started))
                    references.append(dict(cohort=cohort,branch=branch,raw_id=raw,part_id=part,
                                           variant=variant,object_index=geometries[identity]))
            del parent
    counts = {}
    for reference in references:
        key = reference['cohort']+':'+reference['variant']
        counts[key] = counts.get(key,0)+1
    assert counts['full_pipeline:previous']==983 and counts['full_pipeline:candidate']==916
    assert counts['encoder_selected_geometry:archived']==149
    assert counts['encoder_selected_geometry:previous']==105 and counts['encoder_selected_geometry:candidate']==92
    assert np.isfinite(np.asarray(codes)).all()
    np.savez_compressed(out/'codes.npz',raw=np.stack(codes))
    save(out/'objects.json', metadata)
    save(out/'references.json', references)
    save(out/'encoding_contract.json',dict(model_sha256=MODEL_SHA,epoch=23,
         normalization='Exact triangle-area centroid divided by sqrt(original surface area).',
         sampling='16 independent area-uniform 4096-point views; two separate mean8 bags.',
         seed=SEED,device='cpu',unique_surfaces=len(metadata),new_encoded=calculated,reused_prior_probe=reused_old,
         resumed_exact_object_caches=resumed,
         resumed_latents_verified_against_individual_codes_and_deterministic_model_batch=True,
         counts=counts,model_root=str(root),source_hashes=sources,elapsed_seconds=time.monotonic()-started,
         model_trained=False,segmentation_changed=False,full_h01_processing_started=False,
         unchanged_masks_share_exact_codes=True,extra_branches_stage='geometry_only',
         omitted_global_h01_population=True))
    print(json.dumps(dict(status='complete',counts=counts,unique_surfaces=len(metadata),
                          new_encoded=calculated,reused_prior=reused_old)),flush=True)


if __name__ == '__main__':
    main()
