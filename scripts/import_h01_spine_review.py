"""Validate and freeze user answers against the exact H01 viewer sample.

Does not run segmentation or alter meshes, viewer data, or previous judgments.
Anatomy, segmentation quality, confidence, comments, and EM usage remain separate.
The diagnostic sample cannot establish dataset-wide error prevalence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import trimesh

MAIN = Path('/home/eugen/Desktop/CodeWork/Projects/NEURD_modif/NEURD')
ANATOMY = {'', 'one_spine', 'multiple_spines', 'shaft_bump', 'uncertain'}
QUALITY = {'looks_ok', 'false_spine', 'shaft_included', 'spines_merged',
           'head_incomplete', 'neck_incomplete', 'uncertain'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(export, manifest):
    if export.get('schema') != 'h01-em-annotations/1' or export.get('manifest_id') != manifest['manifest_id']:
        raise ValueError('The answers belong to a different viewer sample or schema.')
    if export.get('image_source') != manifest['source'] or export.get('resolution_nm') != [8, 8, 33]:
        raise ValueError('EM source or coordinate scale mismatch.')
    known = {c['id']: c for c in manifest['cases']}
    seen = set()
    for row in export['cases']:
        cid = row['case_id']
        if cid not in known or cid in seen:
            raise ValueError(f'Unknown or duplicate case: {cid}')
        seen.add(cid)
        c = known[cid]
        for field in ['key', 'audit_id', 'raw_id', 'part_id', 'final_index', 'target_sha256', 'parent_sha256']:
            if row.get(field) != c[field]:
                raise ValueError(f'{cid}: mismatch in {field}')
        a = row['annotation']
        if a['anatomy'] not in ANATOMY or a['confidence'] not in {'', 'low', 'medium', 'high'}:
            raise ValueError(f'{cid}: invalid anatomy or confidence')
        if a['em_readability'] not in {'', 'helpful', 'difficult', 'unreadable'}:
            raise ValueError(f'{cid}: invalid EM readability')
        if not isinstance(a['comment'], str) or not isinstance(a['quality'], list) or any(q not in QUALITY for q in a['quality']):
            raise ValueError(f'{cid}: invalid comment or quality')
        depth = c['shape_xyz'][2]
        if not isinstance(a['visited_slices'], list) or any(type(k) is not int or not 0 <= k < depth for k in a['visited_slices']):
            raise ValueError(f'{cid}: slice outside downloaded volume')
        if len(set(a['visited_slices'])) != len(a['visited_slices']):
            raise ValueError(f'{cid}: duplicate visited slice')
        if a['last_slice'] is not None and (type(a['last_slice']) is not int or not 0 <= a['last_slice'] < depth):
            raise ValueError(f'{cid}: last slice outside downloaded volume')
        if not isinstance(a['marks'], list):
            raise ValueError(f'{cid}: invalid marks')
        res = np.array(c['resolution_nm'])
        low, high = np.array(c['bbox_min_voxels'])*res, np.array(c['bbox_max_voxels'])*res
        for mark in a['marks']:
            xyz = np.array(mark['xyz_nm'], dtype=float)
            k = mark['slice_index']
            if mark['kind'] not in {'neck_base', 'missing_head', 'other'} or type(k) is not int or not 0 <= k < depth:
                raise ValueError(f'{cid}: invalid mark type or slice')
            if xyz.shape != (3,) or not np.isfinite(xyz).all() or not ((low <= xyz).all() and (xyz < high).all()):
                raise ValueError(f'{cid}: mark outside physical crop')
            if abs(xyz[2] - (c['bbox_min_voxels'][2]+k+.5)*res[2]) > 1e-6:
                raise ValueError(f'{cid}: mark z does not match voxel-centre plane')
    if seen != set(known):
        raise ValueError('The export does not contain the complete frozen case list.')


def expectations(a):
    anatomy = a['anatomy']
    quality = a['quality']
    return dict(presence='absent' if anatomy == 'shaft_bump' else
                        'present' if anatomy in {'one_spine','multiple_spines'} else 'unknown',
                minimum_spine_count=1 if anatomy == 'one_spine' else 2 if anatomy == 'multiple_spines' else
                                    0 if anatomy == 'shaft_bump' else None,
                exact_spine_count=1 if anatomy == 'one_spine' else 0 if anatomy == 'shaft_bump' else None,
                false_positive=anatomy == 'shaft_bump' or 'false_spine' in quality,
                head_repair_target='head_incomplete' in quality,
                shaft_contamination='shaft_included' in quality,
                accepted_geometry_control='looks_ok' in quality,
                # A partial head on an uncertain object is never a negative
                # example. Confidence/uncertainty must survive later calibration.
                preserve_candidate=anatomy in {'one_spine','multiple_spines','uncertain'} or
                                   'head_incomplete' in quality)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--answers', type=Path, required=True)
    parser.add_argument('--viewer', type=Path, default=MAIN/'artifacts/h01_em_review_20261009')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    viewer, source = args.viewer.resolve(), args.answers.resolve()
    manifest = json.loads((viewer/'manifest.json').read_text())
    export = json.loads(source.read_text())
    validate(export, manifest)
    out = (args.out or viewer/'human_reviews'/source.stem).resolve()
    if out.exists() and (out/'source_answers.json').exists() and sha(out/'source_answers.json') != sha(source):
        raise ValueError('This directory contains a different review; use a new output path.')
    known = {c['id']:c for c in manifest['cases']}
    prior = json.loads((MAIN/'artifacts/h01_residual_qc_20261009/visual_annotations.json').read_text())
    prior = {r['audit_id']:r for r in prior['reviews']}
    records = []
    for row in export['cases']:
        c, a = known[row['case_id']], row['annotation']
        folder = viewer/c['path']
        if sha(folder/'target_mesh.off') != c['target_sha256']:
            raise ValueError(f'{c["id"]}: the displayed mesh changed after review')
        dpath = MAIN/'artifacts/spine_shaft_trim_20261009/child_guard_comparison/branches'/c['key']/'result.json'
        result = json.loads(dpath.read_text())
        if result['source_sha256'] != c['parent_sha256']:
            raise ValueError(f'{c["id"]}: refinement uses another parent mesh')
        decision = next(d for d in result['refinement']['candidates'] if d['raw_input_index']==c['raw_id'])
        mesh = trimesh.load(folder/'target_mesh.off',process=False)
        contacts = []
        if c['synapses']:
            xyz = np.array([s['xyz_nm'] for s in c['synapses']])
            _, distance, _ = trimesh.proximity.closest_point_naive(mesh,xyz)
            contacts = [dict(id=s['id'],type=s['kind'],xyz_nm=s['xyz_nm'],
                             distance_to_selected_mesh_nm=float(dist),assigned_to_spine=False)
                        for s,dist in zip(c['synapses'],distance)]
        records.append(dict(case_id=c['id'],key=c['key'],raw_id=c['raw_id'],part_id=c['part_id'],
                            final_index=c['final_index'],audit_id=c['audit_id'],target_sha256=c['target_sha256'],
                            parent_sha256=c['parent_sha256'],annotation=a,expectations=expectations(a),
                            visited_unique_slices=len(a['visited_slices']),available_slices=c['shape_xyz'][2],
                            prior_assistant_mesh_judgment=prior[c['audit_id']],
                            current_refinement_decision=decision,local_listed_contacts=contacts,
                            shape=dict(area_nm2=float(mesh.area),extent_nm=mesh.extents.tolist())))
    stats = dict(total_cases=len(records),anatomy=dict(Counter(r['annotation']['anatomy'] for r in records)),
                 quality=dict(Counter(q for r in records for q in r['annotation']['quality'])),
                 confidence=dict(Counter(r['annotation']['confidence'] for r in records)),
                 em_readability=dict(Counter(r['annotation']['em_readability'] for r in records)),
                 false_positive_cases=[r['case_id'] for r in records if r['expectations']['false_positive']],
                 accepted_geometry_controls=[r['case_id'] for r in records if r['expectations']['accepted_geometry_control']],
                 multiple_spine_cases=[r['case_id'] for r in records if r['annotation']['anatomy']=='multiple_spines'],
                 head_repair_cases=[r['case_id'] for r in records if r['expectations']['head_repair_target']],
                 unknown_anatomy_cases=[r['case_id'] for r in records if r['annotation']['anatomy']=='uncertain'])
    imported = dict(schema='h01-human-review/1',source_file=str(source),source_sha256=sha(source),
                    manifest_id=manifest['manifest_id'],manifest_sha256=sha(viewer/'manifest.json'),
                    importer_sha256=sha(__file__),statistics=stats,cases=records,
                    review_scope='User manual assessment with 3D mesh and optional EM viewing; '
                                 'visited slices are browser events, not proof of an exhaustive EM review.',
                    sampling='Targeted diagnostic set, not a dataset prevalence estimate.',
                    labels_used_for_algorithm_training=False,segmentation_changed=False)
    out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,out/'source_answers.json')
    (out/'validated_review.json').write_text(json.dumps(imported,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(stats,ensure_ascii=False,indent=2))
    print('Frozen review:',out/'validated_review.json')


if __name__ == '__main__':
    main()
