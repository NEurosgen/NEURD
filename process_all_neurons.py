# process_all_neurons.py
# -*- coding: utf-8 -*-
#
# Batch driver for the slim NEURD segmentation pipeline: walks a directory of neuron
# meshes, decomposes each one, and writes the segmentation to disk. Progress is tracked
# in a manifest so an interrupted run resumes where it stopped.
#
# For every limb it records the branch connectivity: alongside branch_nodes.npy /
# branch_edges.npy it writes connectivity.json, holding the directed edges
# (parent_branch_idx -> child_branch_idx) from the soma-attached root branch outward,
# plus a parent map for each branch.
#
# Dataset, spine mode and SDF profile are explicit shared RunConfig settings.

import os
import json
import time
import traceback
import multiprocessing as mp
import signal
import shutil
import tempfile
import importlib.util
import importlib.metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Set, Any, Dict, List, Optional, Tuple

# Keep the numeric backends single-threaded (we parallelize across meshes, not inside them)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from neurd import parameters
from neurd.segmentation_pipeline import segmentation_pipeline
from neurd.mesh_io import load_mesh
from neurd.run_config import RunConfig, add_run_arguments, config_from_args, require_compiled_cgal, runtime_environment
from neurd.h01_batch import sha, digest_json, save, complete_valid, separate_roots, dependency_signature, triangle_keys

# Start a fresh interpreter rather than inheriting numerical-library threads.
mp = mp.get_context("spawn")

# ---------------------- PARAMETERS ----------------------

EXPORT_EXT = ".off"
DATA_TYPE = "microns"  # Legacy default; use --dataset or RunConfig for new callers.


# ---------------------- MEMORY / PROCESS UTILITIES ----------------------

def _try_import_psutil():
    try:
        import psutil
        return psutil
    except Exception:
        return None

_PSUTIL = _try_import_psutil()

def _rss_mb(pid=None):
    """Observed RSS in MiB, including a worker's external subprocesses."""
    if _PSUTIL is None:
        return None
    try:
        process = _PSUTIL.Process(pid or os.getpid())
        processes = [process] + (process.children(recursive=True) if pid else [])
        total = 0
        for item in processes:
            try:
                total += item.memory_info().rss
            except _PSUTIL.NoSuchProcess:
                pass
        return total / (1024 * 1024)
    except (_PSUTIL.NoSuchProcess, _PSUTIL.AccessDenied):
        return None


def _worker_process(off_path_str, out_dir_str, do_decimate, export_ext, conn, *,
                    config=None, no_save=False, decimation_ratio=0.0,
                    run_hash=None, input_record=None, max_somas=None,
                    private_work_dir=None, staging_dir=None):
    """Process one mesh in a private CWD and publish only completed exports."""
    original_cwd = os.getcwd()
    work_dir = staging = None
    try:
        # The process group also contains MeshLab/Xvfb children for limit cleanup.
        if hasattr(os, "setsid"):
            os.setsid()
        off_path = Path(off_path_str).resolve()
        out_dir = Path(out_dir_str).resolve()
        config = config or RunConfig(dataset=DATA_TYPE, calculate_spines=True)
        if input_record is not None and sha(off_path) != input_record["mesh"]["sha256"]:
            raise ValueError("Input mesh changed after batch planning")
        if run_hash is not None and _run_hash(config, decimation_ratio, max_somas) != run_hash:
            raise ValueError("Runtime changed after batch planning")
        work_dir = private_work_dir or tempfile.mkdtemp(prefix="neurd_work_")
        os.chdir(work_dir)
        mesh = load_mesh(off_path)
        if do_decimate and decimation_ratio:
            from neurd import _mesh_ops
            mesh = _mesh_ops.decimate(mesh, decimation_ratio=decimation_ratio)
        neuron_obj = segmentation_pipeline(mesh, config=config, max_somas=max_somas)
        if input_record is not None and sha(off_path) != input_record["mesh"]["sha256"]:
            raise ValueError("Input mesh changed during processing")
        if run_hash is not None and _run_hash(config, decimation_ratio, max_somas) != run_hash:
            raise ValueError("Runtime changed during processing")
        if not no_save:
            destination = out_dir / off_path.stem
            if destination.exists():
                raise ValueError(f"Refusing to overwrite an unverified output: {destination}")
            if staging_dir is None:
                scratch = out_dir / ".work"
                scratch.mkdir(parents=True, exist_ok=True)
                staging = Path(tempfile.mkdtemp(prefix=off_path.stem + "_", dir=scratch))
            else:
                staging = Path(staging_dir)
            save_segmentation(neuron_obj, staging, export_ext=export_ext)
            outputs = {str(path.relative_to(staging)): sha(path)
                       for path in sorted(staging.rglob("*")) if path.is_file()}
            provenance = _run_signature(config, decimation_ratio, max_somas)
            if run_hash is not None and digest_json(provenance) != run_hash:
                raise ValueError("Runtime changed while exporting results")
            save(staging / "result.json", dict(
                schema_version=1, status="complete", run_hash=run_hash,
                inputs=input_record, run_config=config.as_dict(),
                provenance=provenance,
                outputs_sha256=outputs,
            ))
            os.replace(staging, destination)
            staging = None
        conn.send((True, f"ok:{off_path.stem}"))
    except Exception as exc:
        conn.send((False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
    finally:
        os.chdir(original_cwd)
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        conn.close()


# ---------------------- ACCESS TO A LIMB'S CONCEPT NETWORK ----------------------

def _get_limb_concept_network(limb):
    """
    Get a limb's concept_network regardless of how it is stored.
    Returns a networkx-compatible graph, or None.
    """
    cn = getattr(limb, "concept_network", None)
    if cn is not None:
        return cn
    data = getattr(limb, "data", None)
    if data is not None:
        cn = getattr(data, "concept_network", None)
        if cn is not None:
            return cn
    return None


def _get_limb_starting_node(limb) -> Optional[int]:
    """
    Return the index of the branch attached to the soma (the limb's root).
    NEURD keeps it in different attributes depending on the version.
    """
    for name in ("current_starting_node",
                 "starting_node",
                 "current_starting_node_id",
                 "starting_node_id",
                 "root_node"):
        val = getattr(limb, name, None)
        if val is None:
            continue
        try:
            return int(val)
        except Exception:
            pass

    # fallback: through .data
    data = getattr(limb, "data", None)
    if data is not None:
        for name in ("current_starting_node", "starting_node"):
            val = getattr(data, name, None)
            if val is None:
                continue
            try:
                return int(val)
            except Exception:
                pass
    return None


def _to_int_safe(x):
    try:
        return int(x)
    except Exception:
        return x


def _build_parent_map(cn, root: Optional[int]) -> Tuple[List[Tuple[int, int]], Dict[int, Optional[int]]]:
    """
    Build a directed hierarchy from the undirected concept_network, rooted at `root`
    (the branch touching the soma).

    Returns:
       directed_edges: list of (parent, child)
       parent_map:     dict child -> parent (None for the root)
    If root is not given or not found, pick an arbitrary node of minimum degree.
    """
    import networkx as nx

    G = nx.Graph()
    for n in cn.nodes():
        G.add_node(_to_int_safe(n))
    for u, v in cn.edges():
        G.add_edge(_to_int_safe(u), _to_int_safe(v))

    if G.number_of_nodes() == 0:
        return [], {}

    if root is None or root not in G.nodes:
        # heuristic: take a node of minimum degree (often a leaf near the soma)
        root = min(G.nodes, key=lambda n: G.degree(n))

    directed_edges: List[Tuple[int, int]] = []
    parent_map: Dict[int, Optional[int]] = {root: None}

    # BFS over each component, starting from root, then the rest
    visited = set()
    components = list(nx.connected_components(G))
    # move the component containing root to the front
    components.sort(key=lambda c: 0 if root in c else 1)

    for comp in components:
        # starting node of this component
        start = root if root in comp else min(comp, key=lambda n: G.degree(n))
        if start not in parent_map:
            parent_map[start] = None
        queue = [start]
        visited.add(start)
        while queue:
            u = queue.pop(0)
            for v in G.neighbors(u):
                if v in visited:
                    continue
                visited.add(v)
                parent_map[v] = u
                directed_edges.append((u, v))
                queue.append(v)

    return directed_edges, parent_map


def _branch_graph_mapping(branches, cn):
    """Resolve exported branch positions from graph object identity, never counts."""
    graph_records = [(_to_int_safe(node), attrs.get("data"))
                     for node, attrs in cn.nodes(data=True)]
    graph_ids = [node for node, _ in graph_records]
    branch_to_node, resolution = [], []
    used_ids = set()
    for index, branch in enumerate(branches):
        identities = [node for node, data in graph_records if data is branch]
        declared = []
        for name in ("node_id", "branch_id"):
            value = getattr(branch, name, None)
            if value is None:
                continue
            value = _to_int_safe(value)
            if value not in graph_ids:
                raise ValueError(f"Export branch {index}: {name}={value!r} is absent from the graph")
            declared.append(value)
        if declared and any(value != declared[0] for value in declared):
            raise ValueError(f"Export branch {index}: node_id and branch_id disagree")
        if declared and identities and declared[0] not in identities:
            raise ValueError(f"Export branch {index}: declared node ID disagrees with graph object identity")
        if len(identities) == 1:
            node_id = identities[0]
            evidence = dict(status="available", source="graph_object_identity")
        elif len(identities) > 1:
            node_id = None
            evidence = dict(status="unavailable", reason="ambiguous_graph_object_identity")
        elif declared:
            node_id = declared[0]
            graph_data = [data for node, data in graph_records if node == node_id]
            if len(graph_data) != 1 or (graph_data[0] is not None and graph_data[0] is not branch):
                raise ValueError(f"Export branch {index}: declared node ID belongs to different graph data")
            evidence = dict(status="available", source="validated_explicit_node_id")
        else:
            node_id = None
            evidence = dict(status="unavailable", reason="graph_object_identity_missing")
        if node_id is not None:
            if node_id in used_ids:
                raise ValueError(f"Export branch {index}: graph node {node_id!r} is assigned more than once")
            used_ids.add(node_id)
        branch_to_node.append(node_id)
        resolution.append(evidence)
    return branch_to_node, resolution


def _save_limb_connectivity(limb, limb_dir: Path) -> Optional[int]:
    """
    Save a limb's connectivity in several formats:
      branch_nodes.npy            -- every node ID in the graph
      branch_edges.npy            -- undirected edges (u, v)
      connectivity.json           -- the directed hierarchy from the soma:
            {
              "root_branch": <int|null>,
              "directed_edges": [[parent, child], ...],
              "parent_of": { "<child>": <parent or null>, ... },
              "children_of": { "<parent>": [<child>, ...], ... }
            }
      branch_index_map.json       -- branch_index <-> graph node_id correspondence
                                    with explicit availability when a graph exists
    Returns the number of undirected edges (for the summary).
    """
    cn = _get_limb_concept_network(limb)
    if cn is None:
        return None

    branches = list(getattr(limb, "branches", []))
    branch_to_node, resolution = _branch_graph_mapping(branches, cn)
    nodes = [_to_int_safe(n) for n in cn.nodes()]
    edges = [(_to_int_safe(u), _to_int_safe(v)) for (u, v) in cn.edges()]

    # ---- branch_nodes.npy ----
    if all(isinstance(n, int) for n in nodes):
        nodes_arr = np.array(sorted(nodes), dtype=np.int64)
    else:
        nodes_arr = np.array(nodes, dtype=object)
    np.save(limb_dir / "branch_nodes.npy", nodes_arr)

    # ---- branch_edges.npy ----
    if all(isinstance(u, int) and isinstance(v, int) for (u, v) in edges):
        edges_arr = np.asarray(edges, dtype=np.int64).reshape(-1, 2) if edges else np.zeros((0, 2), dtype=np.int64)
    else:
        edges_arr = np.asarray(edges, dtype=object)
    np.save(limb_dir / "branch_edges.npy", edges_arr)

    # ---- directed hierarchy rooted at the soma ----
    root = _get_limb_starting_node(limb)
    directed_edges, parent_map = _build_parent_map(cn, root)

    children_of: Dict[str, List[int]] = {}
    for parent, child in directed_edges:
        children_of.setdefault(str(parent), []).append(child)

    connectivity_payload = {
        "root_branch": root if root is not None else (
            next(iter(parent_map.keys())) if parent_map else None
        ),
        "directed_edges": [[int(p), int(c)] for (p, c) in directed_edges],
        "parent_of": {
            str(child): (int(p) if p is not None else None)
            for child, p in parent_map.items()
        },
        "children_of": children_of,
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
    }
    with open(limb_dir / "connectivity.json", "w", encoding="utf-8") as f:
        json.dump(connectivity_payload, f, ensure_ascii=False, indent=2)

    # Export positions and graph labels have separate scopes. Missing identity
    # remains explicit; graph IDs are never reconstructed from equal counts.
    mapping = {
        "branch_index_to_node_id": branch_to_node,
        "node_id_to_branch_index": {str(node): index for index, node in enumerate(branch_to_node)
                                     if node is not None},
        "branch_index_resolution": resolution,
    }
    with open(limb_dir / "branch_index_map.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    return len(edges)


# ---------------------- NEURD SETUP ----------------------

def ensure_neurd_defaults():
    """
    Apply the NEURD parameter set selected by DATA_TYPE.
    """
    parameters.params.use(DATA_TYPE)


# ---------------------- MANIFEST ----------------------

def read_manifest(manifest_path: Path) -> Set[str]:
    if not manifest_path.exists():
        return set()
    with open(manifest_path, "r", encoding="utf-8") as f:
        return {line.strip().split()[0] for line in f if line.strip()}

def append_manifest_atomic(manifest_path: Path, entry: str) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    old = ""
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            old = f.read()
    tmp = manifest_path.with_suffix(manifest_path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        if old:
            f.write(old)
            if not old.endswith("\n"):
                f.write("\n")
        f.write(entry + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, manifest_path)


# ---------------------- SAVING ----------------------

def _safe_export_mesh(mesh_obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh_obj.export(str(path))

def _safe_save_array(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr)

def _iter_spines(branch):
    def _yield_from(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                yield v
            return
        if callable(obj):
            try:
                ret = obj()
            except Exception:
                return
            yield from _yield_from(ret)
            return
        try:
            it = iter(obj)
        except TypeError:
            return
        else:
            for v in it:
                yield v

    for name in ("spines", "spines_list", "spines_objs",
                 "spines_meshes", "spines_mesh_list", "iter_spines"):
        try:
            cand = getattr(branch, name)
        except Exception:
            continue
        if cand is None:
            continue
        yield from _yield_from(cand)
        return
    return

def _get_soma_mesh(neuron_obj):
    try:
        cn = getattr(neuron_obj, "concept_network", None)
        if cn is not None and 'S0' in cn.nodes:
            data = cn.nodes['S0'].get('data', None)
            mesh = getattr(data, "mesh", None)
            if mesh is not None:
                return mesh
        if cn is not None:
            for n, attrs in cn.nodes(data=True):
                if isinstance(n, str) and n.startswith('S'):
                    data = attrs.get('data', None)
                    mesh = getattr(data, "mesh", None)
                    if mesh is not None:
                        return mesh
    except Exception:
        pass
    return None

def _summarize_neuron(neuron_obj) -> Dict[str, Any]:
    run_config = getattr(neuron_obj, "run_config", None)
    limbs_sum = []
    for limb in getattr(neuron_obj, "limbs", []):
        branches = getattr(limb, "branches", [])
        spines_per_branch = []
        for br in branches:
            cnt = 0
            for _ in _iter_spines(br):
                cnt += 1
            spines_per_branch.append(cnt)

        # short connectivity digest -- handy to keep in the overall summary
        cn = _get_limb_concept_network(limb)
        n_nodes = cn.number_of_nodes() if cn is not None else 0
        n_edges = cn.number_of_edges() if cn is not None else 0
        root = _get_limb_starting_node(limb)

        limbs_sum.append({
            "n_branches": len(branches),
            "spines_per_branch": spines_per_branch,
            "spine_face_provenance_per_branch": [
                "available" if getattr(branch, "spines_face_idx", None) is not None else "unavailable"
                for branch in branches
            ],
            "total_spines": int(sum(spines_per_branch)),
            "graph_n_nodes": int(n_nodes),
            "graph_n_edges": int(n_edges),
            "root_branch": root,
        })
    return {
        "data_type": (run_config or {}).get("dataset", parameters.params.data_type),
        "run_config": run_config,
        "n_limbs": len(limbs_sum),
        "total_spines": int(sum(l["total_spines"] for l in limbs_sum)),
        "limbs": limbs_sum,
        "soma_present": _get_soma_mesh(neuron_obj) is not None,
    }


def _save_spine_masks(branch, spines, path, *, export_ext=EXPORT_EXT):
    """Save supplied parent indices; never infer ownership from duplicate triangles."""
    from types import SimpleNamespace

    masks = getattr(branch, "spines_face_idx", None)
    if masks is None:
        return
    if len(masks) != len(spines):
        raise ValueError("Spine parent masks and exported meshes have different lengths")
    parent = getattr(branch, "mesh", None)
    if parent is None:
        raise ValueError("Spine face provenance requires the parent branch mesh")
    indices = []
    used_surface = set()
    for mask, mesh in zip(masks, spines):
        mask = np.asarray(mask)
        if mask.ndim != 1 or mask.dtype.kind not in "iu":
            raise ValueError("Spine parent masks must be one-dimensional integer indices")
        if (mask < 0).any() or (mask >= len(parent.faces)).any():
            raise ValueError("Spine parent mask index is outside the branch mesh")
        mask = mask.astype(np.int64, copy=False)
        if len(np.unique(mask)) != len(mask):
            raise ValueError("Spine parent mask contains a repeated face index")
        if not hasattr(mesh, "faces") or len(mask) != len(mesh.faces):
            raise ValueError("Spine parent mask does not match the exported face count")
        parent_part = SimpleNamespace(triangles=np.asarray(parent.vertices)[np.asarray(parent.faces)[mask]])
        spine_triangles = triangle_keys(mesh)
        if not np.array_equal(triangle_keys(parent_part), spine_triangles):
            raise ValueError("Spine parent mask does not match the exact exported triangle geometry")
        surface = {key.tobytes() for key in spine_triangles}
        if used_surface & surface:
            raise ValueError("Spine meshes overlap on the same triangle geometry")
        used_surface.update(surface)
        indices.append(mask)
    faces = np.concatenate(indices) if indices else np.empty(0, dtype=np.int64)
    if len(np.unique(faces)) != len(faces):
        raise ValueError("Spine parent masks overlap on the same face index")
    offsets = np.concatenate(([0], np.cumsum([len(mask) for mask in indices], dtype=np.int64)))
    np.savez(path, faces=faces, offsets=offsets, schema_version=np.int64(1),
             scope=np.array("branch_mesh.face_index"), parent_face_count=np.int64(len(parent.faces)),
             spine_files=np.asarray([f"spines/spine_{i:03d}{export_ext}" for i in range(len(spines))], dtype=str))


def save_segmentation(neuron_obj, base_dir: Path, *, export_ext=EXPORT_EXT) -> None:
    """
    Layout:
      base_dir/
        soma/soma_mesh.off
        summary.json
        limb_000/
          limb_mesh.off
          limb_skeleton.npy
          branch_nodes.npy
          branch_edges.npy
          connectivity.json
          branch_index_map.json
          branch_000/
            branch_mesh.off
            branch_skeleton.npy
            parent_face_indices.npy -- supplied ordered limb-local indices, when available
            face_provenance.json    -- map availability, validation scope and file hashes
            spine_faces.npz       -- supplied branch-local masks, when available
          ...
    """
    from neurd.face_provenance import file_sha256, write_branch_face_provenance

    base_dir.mkdir(parents=True, exist_ok=True)

    # Soma
    soma_mesh = _get_soma_mesh(neuron_obj)
    if soma_mesh is not None:
        soma_dir = base_dir / "soma"
        soma_dir.mkdir(parents=True, exist_ok=True)
        _safe_export_mesh(soma_mesh, soma_dir / f"soma_mesh{export_ext}")

    # Summary
    summary = _summarize_neuron(neuron_obj)
    with open(base_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Limbs / Branches / Spines
    for limb_ind, limb in enumerate(getattr(neuron_obj, "limbs", [])):
        limb_dir = base_dir / f"limb_{limb_ind:03d}"
        limb_dir.mkdir(parents=True, exist_ok=True)

        mesh_limb = getattr(limb, "mesh", None)
        if mesh_limb is not None:
            _safe_export_mesh(mesh_limb, limb_dir / f"limb_mesh{export_ext}")

        skeleton_limb = getattr(limb, "skeleton", None)
        if skeleton_limb is not None:
            _safe_save_array(np.asarray(skeleton_limb), limb_dir / "limb_skeleton.npy")

        _save_limb_connectivity(limb, limb_dir)

        limb_file_sha256 = None
        for branch_ind, branch in enumerate(getattr(limb, "branches", [])):
            br_dir = limb_dir / f"branch_{branch_ind:03d}"
            br_dir.mkdir(parents=True, exist_ok=True)

            mesh_branch = getattr(branch, "mesh", None)
            if mesh_branch is not None:
                _safe_export_mesh(mesh_branch, br_dir / f"branch_mesh{export_ext}")

            skeleton_branch = getattr(branch, "skeleton", None)
            if skeleton_branch is not None:
                _safe_save_array(np.asarray(skeleton_branch), br_dir / "branch_skeleton.npy")

            if getattr(branch, "mesh_face_idx", None) is not None and mesh_limb is not None and limb_file_sha256 is None:
                limb_file_sha256 = file_sha256(limb_dir / f"limb_mesh{export_ext}")
            write_branch_face_provenance(branch, mesh_limb, br_dir, export_ext=export_ext,
                                         parent_file_sha256=limb_file_sha256)

            # branch.spines is the list of the branch's trimesh submeshes. Each is saved as
            # spines/spine_<i>.off; branches without spines get no spines/ directory.
            spines = list(_iter_spines(branch))
            _save_spine_masks(branch, spines, br_dir / "spine_faces.npz", export_ext=export_ext)
            if spines:
                spines_dir = br_dir / "spines"
                spines_dir.mkdir(parents=True, exist_ok=True)
                for spine_ind, spine_mesh in enumerate(spines):
                    if spine_mesh is not None and hasattr(spine_mesh, "faces"):
                        _safe_export_mesh(spine_mesh, spines_dir / f"spine_{spine_ind:03d}{export_ext}")


# ---------------------- PROCESSING / SHARED SCHEDULER ----------------------

def process_one_mesh(off_path: Path, out_dir: Path, do_decimate: bool = True, *,
                     config=None, no_save=False, decimation_ratio=0.0,
                     max_somas=None) -> Path:
    """Compatibility in-process entry; use the batch driver for isolated workers."""
    _validate_decimation(decimation_ratio, not do_decimate)
    config = config or RunConfig(dataset=DATA_TYPE, calculate_spines=True)
    mesh = load_mesh(off_path)
    if do_decimate and decimation_ratio:
        from neurd import _mesh_ops
        mesh = _mesh_ops.decimate(mesh, decimation_ratio=decimation_ratio)
    neuron_obj = segmentation_pipeline(mesh, config=config, max_somas=max_somas)
    destination = Path(out_dir) / off_path.stem
    if not no_save:
        save_segmentation(neuron_obj, destination)
    return destination


@dataclass(frozen=True)
class BatchResult:
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    skipped_failed: int = 0

    @property
    def exit_code(self):
        return 1 if self.failed or self.skipped_failed else 0


def _validate_decimation(ratio, no_decimate):
    if not np.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("decimation_ratio must be between 0 and 1 (0 means disabled)")
    if no_decimate and ratio:
        raise ValueError("--no-decimate cannot be combined with a positive decimation ratio")


def _stop_process(process):
    """Terminate the worker and its external helpers, then reap it."""
    owns_group = False
    if hasattr(os, "killpg"):
        try:
            owns_group = os.getpgid(process.pid) == process.pid
            if owns_group:
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
    else:
        process.terminate()
    process.join(5)
    if owns_group:
        # A helper may outlive the terminated Python worker. Reap the whole
        # process group before removing its scratch files.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.is_alive():
        if not owns_group:
            process.kill()
        process.join()


def _cleanup_job(job):
    for key in ("private_work_dir", "staging_dir"):
        if job.get(key) is not None:
            shutil.rmtree(job[key], ignore_errors=True)


def _drain_result(job):
    # Drain while the child is alive: send() of a large traceback can fill a Pipe.
    if "result" not in job:
        try:
            if job["conn"].poll(0):
                job["result"] = job["conn"].recv()
        except (EOFError, OSError):
            pass


def _run_signature(config, decimation_ratio, max_somas):
    root = Path(__file__).resolve().parent
    code = [root / "process_all_neurons.py", root / "process_all_neurons_parallel.py"]
    code.extend(sorted((root / "neurd").glob("*.py")))
    dependencies = dependency_signature()
    versions = dependencies.setdefault("versions", {})
    for name in ("open3d", "numba", "pymeshfix", "pykdtree", "embreex", "pyembree", "rtree"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    # Raw decomposition uses a separate native skeletonizer; H01 branch
    # resegmentation may retain stored skeletons and need only segmentation.
    skeletonizer = importlib.util.find_spec("calcification_param_Module")
    dependencies["skeletonizer"] = (
        dict(path=skeletonizer.origin, sha256=sha(skeletonizer.origin))
        if skeletonizer is not None and skeletonizer.origin else None
    )
    return dict(
        schema_version=1, config=config.as_dict(), decimation_ratio=decimation_ratio,
        max_somas=max_somas, code_sha256={str(path.relative_to(root)): sha(path) for path in code},
        dependencies=dependencies, runtime_environment=runtime_environment(config),
    )


def _run_hash(config, decimation_ratio, max_somas):
    return digest_json(_run_signature(config, decimation_ratio, max_somas))


def _validate_manifest_path(manifest, source, output, files):
    path = Path(manifest)
    path = (path if path.is_absolute() else output / path).resolve()
    if path == source or source in path.parents:
        raise ValueError("Manifest must not be inside the raw input directory")
    if any(path == mesh.resolve() for mesh in files):
        raise ValueError("Manifest must not overwrite a raw input mesh")
    reserved = [output / "failed.txt", output / ".work"]
    reserved.extend(output / mesh.stem for mesh in files)
    if output.is_dir():
        reserved.extend(child for child in output.iterdir() if child.is_dir()
                        and ((child / "result.json").is_file() or (child / "summary.json").is_file()))
    if path == output or any(path == root.resolve() or root.resolve() in path.parents for root in reserved):
        raise ValueError("Manifest conflicts with a reserved output path")
    return path


def process_folder_with_manifest(
    input_dir: str,
    output_dir: str,
    manifest: str = "processed.txt",
    pattern: str = "*.off",
    no_save: bool = False,
    no_decimate: bool = False,
    *,
    config=None,
    retry_failed=False,
    decimation_ratio=0.0,
    max_somas=None,
    max_workers=1,
    input_size_budget_mb=1024,
    poll_interval=0.2,
) -> BatchResult:
    """Run raw-neuron decomposition with verified resume and isolated processes.

    The input-size budget limits concurrent *file bytes*, not RAM. Set
    SEG_RSS_LIMIT_MB for observed per-worker RSS enforcement (requires psutil).
    --no-save computes without creating exports, manifests or failure logs.
    Legacy outputs without result.json must be handled in a new output directory.
    """
    config = config or RunConfig(dataset=DATA_TYPE, calculate_spines=True)
    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    _validate_decimation(decimation_ratio, no_decimate)
    if max_somas is not None and (isinstance(max_somas, bool) or not isinstance(max_somas, int) or max_somas < 1):
        raise ValueError("max_somas must be a positive integer")
    if max_workers is None or max_workers == 0:
        max_workers = os.cpu_count() or 1
    if max_workers < 1 or input_size_budget_mb <= 0 or poll_interval <= 0:
        raise ValueError("Worker count, input-size budget and poll interval must be positive")
    rss_limit_mb = int(os.environ.get("SEG_RSS_LIMIT_MB", "0"))
    timeout = int(os.environ.get("SEG_TIMEOUT_SEC", "0"))
    if rss_limit_mb < 0 or timeout < 0:
        raise ValueError("RSS and timeout limits must be nonnegative")
    if rss_limit_mb and _PSUTIL is None:
        raise RuntimeError("SEG_RSS_LIMIT_MB requires psutil; install it or unset the RSS limit")
    if config.calculate_spines:
        require_compiled_cgal()
    in_dir, out_dir = separate_roots(input_dir, output_dir)
    if not in_dir.is_dir():
        raise ValueError(f"Missing input directory: {in_dir}")
    files = sorted(path for path in in_dir.glob(pattern) if path.is_file())
    if len({path.stem for path in files}) != len(files):
        raise ValueError("Input mesh basenames must be unique")
    if not files:
        raise ValueError(f"No files matched: {in_dir}/{pattern}")
    manifest_path = _validate_manifest_path(manifest, in_dir, out_dir, files)
    if not no_save:
        out_dir.mkdir(parents=True, exist_ok=True)
    failed_log = out_dir / "failed.txt"
    previously_failed = set() if no_save else read_manifest(failed_log)
    run_hash = _run_hash(config, decimation_ratio, max_somas)
    pending = []
    skipped = skipped_failed = 0
    for mesh in files:
        inputs = {"mesh": {"path": str(mesh), "sha256": sha(mesh)}}
        destination = out_dir / mesh.stem
        if not no_save and complete_valid(destination, run_hash, inputs):
            skipped += 1
            continue
        if not no_save and destination.exists():
            raise ValueError(f"Existing output lacks verified completion: {destination}; use a new output directory")
        if mesh.stem in previously_failed and not retry_failed:
            skipped_failed += 1
            print(f"[=] previous failure: {mesh.stem}; use --retry-failed")
            continue
        pending.append(dict(path=mesh, size=mesh.stat().st_size, inputs=inputs))
    pending.sort(key=lambda item: item["size"])
    budget = int(input_size_budget_mb * 1024 * 1024)
    print(f"[i] dataset={config.dataset} spines={config.calculate_spines} "
          f"refinement={config.refinement} sdf_rays={config.sdf_rays}")
    print(f"[i] workers={max_workers}, concurrent input-size budget={input_size_budget_mb} MiB (file bytes)")
    running = []
    processed = failed = 0
    try:
        while pending or running:
            while pending and len(running) < max_workers:
                inflight = sum(job["size"] for job in running)
                pick = next((i for i, item in enumerate(pending)
                             if not running or inflight + item["size"] <= budget), None)
                if pick is None:
                    break
                job = pending.pop(pick)
                parent, child = mp.Pipe(duplex=False)
                process = None
                try:
                    job["private_work_dir"] = tempfile.mkdtemp(prefix="neurd_work_")
                    job["staging_dir"] = None
                    if not no_save:
                        scratch = out_dir / ".work"
                        scratch.mkdir(parents=True, exist_ok=True)
                        job["staging_dir"] = tempfile.mkdtemp(prefix=job["path"].stem + "_", dir=scratch)
                    process = mp.Process(target=_worker_process, args=(
                        str(job["path"]), str(out_dir), not no_decimate, EXPORT_EXT, child,
                    ), kwargs=dict(config=config, no_save=no_save, decimation_ratio=decimation_ratio,
                                   run_hash=run_hash, input_record=job["inputs"], max_somas=max_somas,
                                   private_work_dir=job["private_work_dir"], staging_dir=job["staging_dir"]))
                    process.start()
                except BaseException:
                    if process is not None and process.pid is not None and process.is_alive():
                        _stop_process(process)
                    parent.close()
                    child.close()
                    _cleanup_job(job)
                    raise
                child.close()
                job.update(proc=process, conn=parent, start=time.monotonic())
                running.append(job)
                print(f"[start] {job['path'].name}")
            finished = []
            for job in running:
                process = job["proc"]
                _drain_result(job)
                if process.is_alive():
                    reason = None
                    if timeout and time.monotonic() - job["start"] > timeout:
                        reason = f"Timeout after {timeout}s"
                    if rss_limit_mb and reason is None:
                        observed = _rss_mb(process.pid)
                        if observed is None:
                            if process.is_alive():
                                reason = "Unable to observe RSS while SEG_RSS_LIMIT_MB is required"
                        elif observed > rss_limit_mb:
                            reason = f"RSS limit exceeded: {observed:.0f} MiB > {rss_limit_mb} MiB"
                    if reason:
                        _stop_process(process)
                        job["result"] = False, reason
                if not process.is_alive():
                    finished.append(job)
            for job in finished:
                running.remove(job)
                process = job["proc"]
                process.join()
                _drain_result(job)
                ok, message = job.get("result", (False, f"No worker result; exitcode={process.exitcode}"))
                job["conn"].close()
                _cleanup_job(job)
                ok = bool(ok) and process.exitcode == 0
                if ok:
                    if not no_save:
                        # Completion is independently validated before a manifest entry.
                        try:
                            if not complete_valid(out_dir / job["path"].stem, run_hash, job["inputs"]):
                                ok, message = False, "Worker did not publish a verified result"
                        except (ValueError, OSError) as exc:
                            ok, message = False, f"Worker completion validation failed: {exc}"
                if ok:
                    processed += 1
                    if not no_save:
                        append_manifest_atomic(manifest_path, job["path"].stem)
                else:
                    failed += 1
                    if not no_save:
                        with failed_log.open("a", encoding="utf-8") as stream:
                            stream.write(job["path"].stem + "\t" + json.dumps(message) + "\n")
                    print(f"[x] {job['path'].stem}: {message}")
            if running and not finished:
                time.sleep(poll_interval)
    finally:
        for job in running:
            if job["proc"].is_alive():
                _stop_process(job["proc"])
            job["conn"].close()
            _cleanup_job(job)
    result = BatchResult(processed, skipped, failed, skipped_failed)
    print(f"Done. processed={processed}, skipped={skipped}, failed={failed}, skipped_failed={skipped_failed}")
    return result


# ---------------------- CLI ----------------------

def add_batch_arguments(parser):
    parser.add_argument("input_dir", help="directory of raw neuron meshes")
    parser.add_argument("output_dir", help="separate output directory")
    parser.add_argument("--pattern", default="*.off")
    parser.add_argument("--manifest", default="processed.txt")
    parser.add_argument("--no-save", action="store_true", help="compute without exports, manifests or failure logs")
    parser.add_argument("--no-decimate", action="store_true", help="disable input decimation (already the default)")
    parser.add_argument("--decimation-ratio", type=float, default=0.0,
                        help="optional retained input-face fraction, 0 disables decimation")
    parser.add_argument("--max-somas", type=int, help="positive cap for the soma search")
    parser.add_argument("--retry-failed", action="store_true", help="retry inputs recorded in failed.txt")
    add_run_arguments(parser)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Raw-neuron NEURD segmentation with verified resume.")
    add_batch_arguments(parser)
    args = parser.parse_args(argv)
    try:
        result = process_folder_with_manifest(
            args.input_dir, args.output_dir, manifest=args.manifest, pattern=args.pattern,
            no_save=args.no_save, no_decimate=args.no_decimate, config=config_from_args(args),
            retry_failed=args.retry_failed, decimation_ratio=args.decimation_ratio,
            max_somas=args.max_somas,
        )
        return result.exit_code
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"[x] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
