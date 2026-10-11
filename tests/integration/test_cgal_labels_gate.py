"""Exact native per-face gate for the labels-only spine wrapper.

The committed small oracle and generated disconnected/degenerate fixture run
both wrapper paths with the same NumPy seed and native ray budget. The native
extension itself is unchanged; no developer data mounts are required.
"""
from hashlib import sha256
from importlib.machinery import EXTENSION_SUFFIXES
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

import neurd  # noqa: F401
from neurd import cgal_labels as seam


@pytest.mark.parametrize('fixture', ['committed_oracle', 'disconnected_degenerate'])
def test_labels_only_matches_native_legacy_per_face(tmp_path, monkeypatch, fixture):
    csm = pytest.importorskip('cgal_Segmentation_Module')
    provider = Path(getattr(csm, '__file__', ''))
    if not any(str(provider).endswith(suffix) for suffix in EXTENSION_SUFFIXES):
        pytest.skip('Exact native wrapper gate requires compiled CGAL segmentation')
    if fixture == 'committed_oracle':
        mesh_path = Path(__file__).resolve().parents[1]/'990_mesh.off'
        mesh = trimesh.load(mesh_path, process=False, force='mesh')
        source_hash = sha256(mesh_path.read_bytes()).hexdigest()
    else:
        first = trimesh.creation.icosphere(subdivisions=1, radius=500)
        second = first.copy()
        second.apply_translation([3000, 0, 0])
        mesh = trimesh.util.concatenate([first, second])
        mesh = trimesh.Trimesh(vertices=mesh.vertices,
            faces=np.insert(mesh.faces, [0, 60, 150], [0, 0, 1], axis=0), process=False)
        source_hash = sha256(mesh.vertices.tobytes() + mesh.faces.tobytes()).hexdigest()
    monkeypatch.setenv('NEURD_SDF_RAYS', '25')
    monkeypatch.delenv('NEURD_CGAL_TIMINGS', raising=False)
    previous_rng = np.random.get_state()
    try:
        monkeypatch.setenv('NEURD_LEGACY_CGAL_ASSEMBLY', '1')
        np.random.seed(17)
        legacy = seam.mesh_segmentation_labels(mesh, cgal_folder=tmp_path/'legacy',
                                               clusters=3, smoothness=.2)
        monkeypatch.delenv('NEURD_LEGACY_CGAL_ASSEMBLY')
        np.random.seed(17)
        owned = seam.mesh_segmentation_labels(mesh, cgal_folder=tmp_path/'owned',
                                              clusters=3, smoothness=.2)
    finally:
        np.random.set_state(previous_rng)
    for actual, expected in zip(owned, legacy):
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape == (len(mesh.faces),)
        assert np.array_equal(actual, expected)
    assert np.isfinite(owned[1]).all()
    assert len(np.unique(owned[0])) > 1
    assert not list((tmp_path/'legacy').glob('*'))
    assert not list((tmp_path/'owned').glob('*'))
    receipt = dict(status='passed', fixture=fixture, source_sha256=source_hash,
        parent_faces=len(mesh.faces), seed=17, sdf_rays=25, clusters=3, smoothness=.2,
        compiled_cgal=dict(path=str(provider), sha256=sha256(provider.read_bytes()).hexdigest()),
        code_sha256={str(path.relative_to(Path(neurd.__file__).resolve().parent.parent)):
            sha256(path.read_bytes()).hexdigest() for path in
            (Path(seam.__file__), Path(neurd.__file__).resolve().parent/'submesh_ops.py')},
        exact_per_face_labels=True, exact_per_face_sdf=True,
        labels_sha256=sha256(owned[0].tobytes()).hexdigest(),
        sdf_sha256=sha256(owned[1].tobytes()).hexdigest())
    (tmp_path/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
