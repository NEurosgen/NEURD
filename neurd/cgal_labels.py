"""The spine detector's labels-only CGAL seam, without discarded mesh assembly.

The installed mesh-tools tuple API remains untouched. Preparation, temporary
filenames, native arguments, CSV parsing and original-face remapping follow its
``mesh_segmentation(..., return_meshes=False)`` path. Plotting and the explicit
``NEURD_LEGACY_CGAL_ASSEMBLY=1`` comparison path use that implementation directly.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

from datasci_tools import numpy_dep as np
from mesh_tools import trimesh_utils as tu

from .submesh_ops import face_groups_by_label


def legacy_assembly_enabled():
    return os.environ.get("NEURD_LEGACY_CGAL_ASSEMBLY") == "1"


@contextmanager
def phase(name, **fields):
    """Emit opt-in, flushed begin/end records, including an unfinished phase."""
    if os.environ.get("NEURD_CGAL_TIMINGS") != "1":
        yield
        return
    started = time.perf_counter()
    record = dict(event="neurd_cgal_phase", phase=name, **fields)
    print(json.dumps(dict(record, state="begin"), sort_keys=True), flush=True)
    status = "complete"
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        print(json.dumps(dict(record, state="end", status=status,
                              seconds=time.perf_counter() - started), sort_keys=True), flush=True)


def mesh_segmentation_labels(mesh, *, clusters=2, smoothness=0.2,
                             cgal_folder=Path("./"), return_sdf=True,
                             delete_temp_files=True, plot_segmentation=False):
    """Return original-parent per-face labels/SDF using the spine call's defaults.

    This deliberately narrow API accepts an in-memory mesh, as the spine caller
    does. Connected-piece significance remains 40 faces with vertex connectivity;
    ignored and degenerate faces retain zero labels/SDF. The return shape and
    float arrays match the installed labels-only tuple path, including sheet
    handling and the per-component label offset.
    """
    legacy = legacy_assembly_enabled() or plot_segmentation
    with phase("labels", parent_faces=len(mesh.faces), legacy=bool(legacy)):
        if legacy:
            return tu.mesh_segmentation(
                mesh, cgal_folder=cgal_folder, clusters=clusters, smoothness=smoothness,
                return_sdf=return_sdf, delete_temp_files=delete_temp_files,
                return_meshes=False, return_ordered_by_size=False,
                plot_segmentation=plot_segmentation)
        with phase("preparation", parent_faces=len(mesh.faces)):
            filtered, nondeg_faces = tu.remove_degenerate_faces(mesh, return_face_idxs=True)
            labels = np.zeros(len(mesh.faces))
            sdf = np.zeros(len(mesh.faces))
            if len(nondeg_faces):
                pieces, piece_faces = tu.split_significant_pieces(
                    filtered, significance_threshold=40, return_face_indices=True,
                    connectivity="vertices")
            else:
                pieces, piece_faces = [], []

        for component, (piece, faces_kept) in enumerate(zip(pieces, piece_faces)):
            with phase("component_preparation", component=component, faces=len(piece.faces)):
                distances = tu.ray_trace_distance(piece)
                sheet = np.sum(distances == 0) / len(distances) == 1
                if sheet:
                    piece_labels = np.zeros(len(piece.faces))
                    piece_sdf = np.zeros(len(piece.faces))
                else:
                    if not cgal_folder.exists():
                        cgal_folder.mkdir(parents=True, exist_ok=False)
                    file_dest = cgal_folder / Path(f"{np.random.randint(10, 1000)}_mesh.off")
                    mesh_filepath = Path(tu.write_neuron_off(piece, file_dest))
                    assert mesh_filepath.exists()
                    stem = mesh_filepath.absolute().parents[0] / mesh_filepath.stem
            if not sheet:
                with phase("native", component=component, faces=len(piece.faces),
                           clusters=clusters, smoothness=smoothness):
                    tu.csm.cgal_segmentation(str(stem), clusters, smoothness)
                with phase("csv", component=component, faces=len(piece.faces)):
                    suffix = "-cgal_" + str(np.round(clusters, 2)) + f"_{smoothness:.2f}"
                    csv_path = Path(str(stem) + suffix + ".csv")
                    sdf_path = Path(str(stem) + suffix + "_sdf.csv")
                    piece_labels = np.genfromtxt(str(csv_path.absolute()), delimiter="\n")
                    piece_sdf = np.genfromtxt(str(sdf_path.absolute()), delimiter="\n")
            labels[nondeg_faces[faces_kept]] = piece_labels + np.max(labels) + 1
            sdf[nondeg_faces[faces_kept]] = piece_sdf
            if not sheet and delete_temp_files:
                csv_path.unlink()
                sdf_path.unlink()
                mesh_filepath.unlink()
        return (labels, sdf) if return_sdf else labels


def assemble_face_groups(mesh, labels):
    """Build the spine caller's label meshes once, preserving exact face order."""
    trace_counts = {"labels": len(np.unique(labels))} if os.environ.get("NEURD_CGAL_TIMINGS") == "1" else {}
    with phase("explicit_face_assembly", parent_faces=len(mesh.faces),
               legacy=legacy_assembly_enabled(), **trace_counts):
        if legacy_assembly_enabled():
            return tu.split_mesh_into_face_groups(
                mesh, labels, return_idx=True, check_connect_comp=False)
        if len(labels) != len(mesh.faces):
            raise Exception("face mapping does not have same length as mesh faces")
        indices = face_groups_by_label(labels)
        meshes = {label: mesh.submesh([faces], append=True, only_watertight=False, repair=False)
                  for label, faces in indices.items()}
        return meshes, indices
