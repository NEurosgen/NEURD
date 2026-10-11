"""Validate supplied ordered face indices without inferring geometric ownership."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def file_sha256(path):
    value = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _canonical_triangle_vertices(triangles):
    order = np.lexsort((triangles[..., 2], triangles[..., 1], triangles[..., 0]), axis=1)
    return np.take_along_axis(triangles, order[..., None], axis=1)


def validate_ordered_face_indices(parent, child, indices, *, chunk_size=8192):
    """Check one supplied parent index per child face, retaining row order and IDs.

    Triangle corner permutations are allowed; child face rows are never reordered.
    Duplicate parent geometry cannot resolve ownership: only the provided IDs are
    checked. This establishes consistency of a declared map, not its historical
    algorithmic origin. No lazy trimesh triangle cache or whole triangle array is
    constructed. Missing indices are handled by the export journal, not inferred.
    """
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError('Face-map chunk size must be a positive integer')
    ids = np.asarray(indices)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu':
        raise ValueError('Ordered face map requires one-dimensional integer indices')
    parent_faces = np.asarray(parent.faces)
    child_faces = np.asarray(child.faces)
    if parent_faces.ndim != 2 or child_faces.ndim != 2 or parent_faces.shape[1] != 3 or child_faces.shape[1] != 3:
        raise ValueError('Ordered face maps require triangular parent and child meshes')
    if len(ids) != len(child_faces):
        raise ValueError('Ordered face-map count differs from child face count')
    if (ids < 0).any() or (ids >= len(parent_faces)).any():
        raise ValueError('Ordered face-map index is outside the parent mesh')
    ids = ids.astype(np.int64, copy=False)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Ordered face map contains a repeated parent face index')
    parent_vertices, child_vertices = np.asarray(parent.vertices), np.asarray(child.vertices)
    for start in range(0, len(ids), chunk_size):
        source_faces = parent_faces[ids[start:start + chunk_size]]
        target_faces = child_faces[start:start + chunk_size]
        if (source_faces < 0).any() or (source_faces >= len(parent_vertices)).any():
            raise ValueError('Ordered face-map parent vertex index is outside the parent mesh')
        if (target_faces < 0).any() or (target_faces >= len(child_vertices)).any():
            raise ValueError('Ordered face-map child vertex index is outside the child mesh')
        expected = parent_vertices[source_faces]
        actual = child_vertices[target_faces]
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise ValueError('Ordered face-map triangle geometry is nonfinite')
        if not np.array_equal(_canonical_triangle_vertices(expected), _canonical_triangle_vertices(actual)):
            raise ValueError('Ordered face map differs from the child triangle rows')
    return ids


def write_branch_face_provenance(branch, limb_mesh, directory, *, export_ext='.off',
                                 parent_file_sha256=None):
    """Journal an exported branch's supplied map to its exported limb surface.

    Absent maps are explicitly unavailable. Invalid declared maps fail before any
    map/journal is published. The raw batch runner's enclosing staging transaction
    ensures no incomplete neuron becomes a completed output.
    """
    directory = Path(directory)
    child_path = directory / ('branch_mesh' + export_ext)
    parent_path = directory.parent / ('limb_mesh' + export_ext)
    child_mesh = getattr(branch, 'mesh', None)
    supplied = getattr(branch, 'mesh_face_idx', None)
    record = dict(schema_version=1, status='unavailable',
        index_scope='limb_mesh.face_index', child_mesh=child_path.name,
        parent_mesh='../' + parent_path.name,
        supplied_by='Branch.mesh_face_idx', raw_input_map_available=False,
        geometry_policy='exact_triangle_rows_allow_corner_permutation',
        verification_basis='in_memory_geometry_before_mesh_export_rounding',
        mapping_origin='provided_indices_not_recovered_from_geometry')
    if supplied is None:
        record['reason'] = 'ordered_parent_indices_not_supplied'
    elif child_mesh is None or limb_mesh is None:
        raise ValueError('Declared branch face map requires both branch and limb meshes')
    else:
        ids = validate_ordered_face_indices(limb_mesh, child_mesh, supplied)
        record.update(status='available', child_face_count=len(child_mesh.faces),
            parent_face_count=len(limb_mesh.faces), child_mesh_sha256=file_sha256(child_path),
            parent_mesh_sha256=parent_file_sha256 or file_sha256(parent_path),
            indices_file='parent_face_indices.npy')
        np.save(directory / record['indices_file'], ids, allow_pickle=False)
        record['indices_sha256'] = file_sha256(directory / record['indices_file'])
    temporary = directory / 'face_provenance.json.tmp'
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(directory / 'face_provenance.json')
    return record
