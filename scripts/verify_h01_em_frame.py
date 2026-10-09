"""Bounded coordinate-frame check against H01 proofread_104, one cube/neuron.

This checks source localization, not anatomical correctness of the spine labels.
Missing segmentation chunks are recorded as unavailable, never filled with zeros.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from cloudvolume import Bbox, CloudVolume
from cloudvolume.exceptions import EmptyVolumeException


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('/home/eugen/Desktop/CodeWork/Projects/NEURD_modif/NEURD/artifacts/h01_em_review_20261009'))
    args = parser.parse_args()
    out = args.out.resolve()
    manifest = json.loads((out/'manifest.json').read_text())
    source = 'precomputed://https://storage.googleapis.com/h01-release/data/20210601/proofread_104'
    volume = CloudVolume(source, mip=0, progress=False, parallel=1, cache=False,
                         bounded=True, fill_missing=False)
    resolution = np.asarray(volume.resolution, float)
    result = dict(source=source, resolution_nm=resolution.tolist(),
                  check='Small cube at a candidate mesh vertex; one per neuron; localization only',
                  fill_missing=False, records=[])
    seen = set()
    for case in manifest['cases']:
        if case['neuron_id'] in seen:
            continue
        seen.add(case['neuron_id'])
        mesh = trimesh.load(case['target_source'], process=False)
        center = mesh.bounds.mean(axis=0)
        point = mesh.vertices[np.argmin(np.linalg.norm(mesh.vertices-center, axis=1))]
        low = np.floor((point-180)/resolution).astype(int)
        high = np.ceil((point+180)/resolution).astype(int)
        assert np.prod(high-low) < 50_000
        record = dict(case_id=case['id'], neuron_id=case['neuron_id'], point_nm=point.tolist(),
                      bbox_min_voxels=low.tolist(), bbox_max_voxels=high.tolist())
        try:
            labels = np.asarray(volume.download(Bbox(low, high)))[..., 0]
            ids, counts = np.unique(labels, return_counts=True)
            present = bool(np.any(labels == int(case['neuron_id'])))
            record.update(status='matched' if present else 'label_mismatch',
                          expected_id_present=present, shape_xyz=list(labels.shape),
                          labels={str(k):int(n) for k,n in zip(ids, counts)})
        except EmptyVolumeException as error:
            record.update(status='unavailable_segmentation_chunk', expected_id_present=None,
                          error=str(error))
        result['records'].append(record)
        (out/'reference_frame_verification.json').write_text(json.dumps(result,indent=2))
        print(case['id'], case['neuron_id'], record['status'], flush=True)
    result['matched'] = sum(r['status']=='matched' for r in result['records'])
    result['unavailable'] = sum(r['status']=='unavailable_segmentation_chunk' for r in result['records'])
    result['mismatched'] = sum(r['status']=='label_mismatch' for r in result['records'])
    (out/'reference_frame_verification.json').write_text(json.dumps(result,indent=2))
    assert result['mismatched'] == 0, 'Some available coordinates do not overlap the expected label.'


if __name__ == '__main__':
    main()
