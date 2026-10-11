"""Export a segmented neuron's geometry and local provenance.

``save_segmentation`` writes directly to the supplied destination and may leave
partial output if an export fails. It does not provide transactional completion,
resume validation, or a ``result.json`` completion record. The batch supervisor
owns staging, completion records, and publication of successful exports.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neurd import parameters
from neurd.batch_utils import save, triangle_keys

__all__ = ["EXPORT_EXT", "save_segmentation"]

EXPORT_EXT = ".off"


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


def _save_branch_origin(branch, branch_dir):
    """Export local observations separately from face-map consistency checks."""
    from neurd.face_provenance import file_sha256

    origin = getattr(branch, "mesh_axis_origin", None)
    reason = "metadata_missing"
    if (isinstance(origin, dict) and type(origin.get("schema_version")) is int
            and origin["schema_version"] == 1
            and origin.get("scope") == "local_map_mp_selection_and_axis_pairing"
            and origin.get("raw_input_map_available") is False):
        try:
            origin = json.loads(json.dumps(origin, allow_nan=False))
            reason = None
        except (TypeError, ValueError):
            reason = "metadata_not_json_safe"
    elif origin is not None:
        reason = "metadata_schema_unavailable"
    payload = dict(schema_version=1, scope="local_map_mp_selection_and_axis_pairing",
                   status="unavailable" if reason else "available", reason=reason,
                   raw_input_map_available=False,
                   axis_fingerprint_scope="observed_before_Branch_normalization_or_later_mutation",
                   observation=origin if reason is None else None)
    skeleton_file = branch_dir / "branch_skeleton.npy"
    payload["exported_skeleton_file_sha256"] = file_sha256(skeleton_file) if skeleton_file.is_file() else None
    save(branch_dir / "branch_origin.json", payload)


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
            branch_origin.json      -- local selection/axis observations; raw origin unavailable
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
            _save_branch_origin(branch, br_dir)

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
