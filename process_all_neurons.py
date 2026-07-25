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
# The dataset parameter set is chosen by DATA_TYPE below -- set it to "h01" for H01
# meshes, "microns" for MICrONS.

import os
import gc
import json
import time
import traceback
import multiprocessing as mp
from pathlib import Path
from typing import Set, Any, Dict, List, Optional, Tuple

# Keep the numeric backends single-threaded (we parallelize across meshes, not inside them)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# CGAL SDF ray count. mesh_segmentation is the largest wall-time lever (~38% on big
# meshes) and its cost is ~linear in the ray count; 12 gives ~2x on the CGAL part while
# leaving the structure intact (soma/branches invariant, spines preserved, SDF corr ~0.99).
# setdefault => an external NEURD_SDF_RAYS wins; 25 is CGAL's own default.
# NOTE: only the compiled cgal_Segmentation_Module reads this. On the pure-Python
# fallback (neurd/_cgal_segmentation.py) it has no effect -- see cgal/README.md.
os.environ.setdefault("NEURD_SDF_RAYS", "12")

import numpy as np

from neurd import neuron
from neurd import parameters
from mesh_tools import trimesh_utils as tu

# ---------------------- PARAMETERS ----------------------

DECIMATION_PARAMETERS = dict(decimation_ratio=0)
EXPORT_EXT = ".off"
DATA_TYPE = "microns"  # <-- the dataset parameter set: "microns" or "h01"


# ---------------------- MEMORY / PROCESS UTILITIES ----------------------

def _try_import_psutil():
    try:
        import psutil
        return psutil
    except Exception:
        return None

_PSUTIL = _try_import_psutil()

def _rss_mb(pid=None):
    if _PSUTIL is None:
        return None
    try:
        p = _PSUTIL.Process(pid or os.getpid())
        return int(p.memory_info().rss / (1024 * 1024))
    except Exception:
        return None


def _worker_process(off_path_str, out_dir_str, do_decimate, export_ext, conn):
    """
    Process one mesh in its own process, so an OOM kill cannot take down the batch.
    """
    import tempfile, shutil
    _orig_cwd = os.getcwd()
    _work_dir = None
    try:
        from pathlib import Path
        from mesh_tools import trimesh_utils as tu
        from neurd import neuron
        import gc

        # the parent process already applied these, but repeat it in the worker because
        # spawned processes do not inherit the parameter state
        ensure_neurd_defaults()

        # Resolve to ABSOLUTE before chdir, then run this worker in a private temp CWD. NEURD's soma
        # extraction creates CWD-relative temp dirs named by segment_id (./{segment_id}, plus
        # ./Poisson_temp, ./temp). Under parallel workers segment_id collides (np.random inherited via
        # fork) so those dirs are shared and workers rmtree each other's -> FileNotFoundError. A
        # private CWD isolates ALL of them per worker (also CGAL temp files); cleaned up in finally.
        off_path = Path(off_path_str).resolve()
        out_dir = Path(out_dir_str).resolve()
        _work_dir = tempfile.mkdtemp(prefix="neurd_work_")
        os.chdir(_work_dir)

        basename = off_path.stem
        neuron_base = out_dir / basename

        mesh = tu.load_mesh_no_processing(str(off_path))
        mesh_proc = mesh  # decimation is off by default (DECIMATION_PARAMETERS ratio=0)

        # Spines are ON: the real CGAL SDF segmenter (cgal_Segmentation_Module, built from
        # cgal/cgal_segmentation/) produces the fine over-segmentation spine detection needs
        # (the KMeans stand-in yielded 0 spines). Costs ~+100 s/neuron. Spine meshes are
        # written to branch_*/spines/spine_*.off by save_segmentation.
        neuron_obj = neuron.Neuron(mesh=mesh_proc, calculate_spines=True)
        save_segmentation(neuron_obj, neuron_base)

        del neuron_obj, mesh_proc, mesh
        gc.collect()
        conn.send((True, f"ok:{basename}"))
    except Exception as e:
        tb = traceback.format_exc()
        conn.send((False, f"{type(e).__name__}: {e}\n{tb}"))
    finally:
        try:
            os.chdir(_orig_cwd)
        except Exception:
            pass
        if _work_dir is not None:
            shutil.rmtree(_work_dir, ignore_errors=True)
        try:
            conn.close()
        except Exception:
            pass


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


def _save_limb_connectivity(limb, limb_dir: Path) -> Optional[int]:
    """
    Save a limb's connectivity in several formats:
      branch_nodes.npy            -- every branch index in the graph
      branch_edges.npy            -- undirected edges (u, v)
      connectivity.json           -- the directed hierarchy from the soma:
            {
              "root_branch": <int|null>,
              "directed_edges": [[parent, child], ...],
              "parent_of": { "<child>": <parent or null>, ... },
              "children_of": { "<parent>": [<child>, ...], ... }
            }
      branch_index_map.json       -- branch_index <-> graph node_id correspondence
                                    (written only when the two differ)
    Returns the number of undirected edges (for the summary).
    """
    cn = _get_limb_concept_network(limb)
    if cn is None:
        return None

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

    # ---- map limb.branches indices to graph node ids ----
    branches = getattr(limb, "branches", [])
    branch_to_node: List[Optional[int]] = []
    for bi, br in enumerate(branches):
        node_id = getattr(br, "node_id", None)
        if node_id is None:
            node_id = getattr(br, "branch_id", None)
        if node_id is None and len(nodes) == len(branches) and all(isinstance(n, int) for n in nodes):
            node_id = bi
        branch_to_node.append(node_id)

    if any(x is not None for x in branch_to_node):
        node_to_branch = {}
        for bi, nid in enumerate(branch_to_node):
            if nid is not None:
                node_to_branch[str(nid)] = bi
        mapping = {
            "branch_index_to_node_id": branch_to_node,
            "node_id_to_branch_index": node_to_branch,
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
            "total_spines": int(sum(spines_per_branch)),
            "graph_n_nodes": int(n_nodes),
            "graph_n_edges": int(n_edges),
            "root_branch": root,
        })
    return {
        "data_type": DATA_TYPE,
        "n_limbs": len(limbs_sum),
        "total_spines": int(sum(l["total_spines"] for l in limbs_sum)),
        "limbs": limbs_sum,
        "soma_present": _get_soma_mesh(neuron_obj) is not None,
    }


def save_segmentation(neuron_obj, base_dir: Path) -> None:
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
          branch_000/
            branch_mesh.off
            branch_skeleton.npy
          ...
    """
    base_dir.mkdir(parents=True, exist_ok=True)

    # Soma
    soma_mesh = _get_soma_mesh(neuron_obj)
    if soma_mesh is not None:
        soma_dir = base_dir / "soma"
        soma_dir.mkdir(parents=True, exist_ok=True)
        _safe_export_mesh(soma_mesh, soma_dir / f"soma_mesh{EXPORT_EXT}")

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
            _safe_export_mesh(mesh_limb, limb_dir / f"limb_mesh{EXPORT_EXT}")

        skeleton_limb = getattr(limb, "skeleton", None)
        if skeleton_limb is not None:
            _safe_save_array(np.asarray(skeleton_limb), limb_dir / "limb_skeleton.npy")

        _save_limb_connectivity(limb, limb_dir)

        for branch_ind, branch in enumerate(getattr(limb, "branches", [])):
            br_dir = limb_dir / f"branch_{branch_ind:03d}"
            br_dir.mkdir(parents=True, exist_ok=True)

            mesh_branch = getattr(branch, "mesh", None)
            if mesh_branch is not None:
                _safe_export_mesh(mesh_branch, br_dir / f"branch_mesh{EXPORT_EXT}")

            skeleton_branch = getattr(branch, "skeleton", None)
            if skeleton_branch is not None:
                _safe_save_array(np.asarray(skeleton_branch), br_dir / "branch_skeleton.npy")

            # branch.spines is the list of the branch's trimesh submeshes. Each is saved as
            # spines/spine_<i>.off; branches without spines get no spines/ directory.
            spines = list(_iter_spines(branch))
            if spines:
                spines_dir = br_dir / "spines"
                spines_dir.mkdir(parents=True, exist_ok=True)
                for spine_ind, spine_mesh in enumerate(spines):
                    if spine_mesh is not None and hasattr(spine_mesh, "faces"):
                        _safe_export_mesh(spine_mesh, spines_dir / f"spine_{spine_ind:03d}{EXPORT_EXT}")


# ---------------------- PROCESS ONE MESH (in-process) ----------------------

def process_one_mesh(off_path: Path, out_dir: Path, do_decimate: bool = True) -> Path:
    basename = off_path.stem
    neuron_base = out_dir / basename

    mesh = tu.load_mesh_no_processing(str(off_path))
    neuron_obj = neuron.Neuron(mesh=mesh, calculate_spines=True)  # spines on (CGAL segmentation)
    save_segmentation(neuron_obj, neuron_base)

    del neuron_obj, mesh
    gc.collect()
    return neuron_base


# ---------------------- PROCESS A DIRECTORY, TRACKED BY MANIFEST ----------------------

def process_folder_with_manifest(
    input_dir: str,
    output_dir: str,
    manifest: str = "processed.txt",
    pattern: str = "*.off",
    no_save: bool = False,
    no_decimate: bool = False,
) -> None:
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    manifest_path = Path(manifest) if os.path.isabs(manifest) else (out_dir / manifest)

    ensure_neurd_defaults()

    files = sorted(in_dir.glob(pattern))
    if not files:
        print(f"[!] no files matched: {in_dir}/{pattern}")
        return

    done = read_manifest(manifest_path)
    done_failed = read_manifest(out_dir / Path("failed.txt"))
    print(f"[i] data_type={DATA_TYPE}")
    print(f"[i] manifest: {manifest_path} (loaded {len(done)} ids)")
    print(f"[i] found {len(files)} mesh(es) in: {in_dir}")

    rss_limit_mb = int(os.environ.get("SEG_RSS_LIMIT_MB", "0"))
    per_item_timeout = int(os.environ.get("SEG_TIMEOUT_SEC", "0"))
    failed_log = out_dir / "failed.txt"

    processed = skipped = failed = 0
    total = len(files)

    for i, mesh_path in enumerate(files, 1):
        spine_id = mesh_path.stem
        if spine_id in done:
            print(f"[{i}/{total}] [=] skip by manifest: {spine_id}")
            skipped += 1
            continue
        if spine_id in done_failed:
            print(f"[{i}/{total}] [=] skip by failed_manifest: {spine_id}")
            skipped += 1
            continue

        print(f"[{i}/{total}] processing: {mesh_path.name} (write={not no_save})")

        parent_conn, child_conn = mp.Pipe(duplex=False)
        p = mp.Process(
            target=_worker_process,
            args=(str(mesh_path), str(out_dir), (not no_decimate), EXPORT_EXT, child_conn),
            daemon=False,
        )
        p.start()
        start = time.time()
        ok = None
        msg = ""

        while p.is_alive():
            if per_item_timeout and (time.time() - start > per_item_timeout):
                p.terminate(); p.join(5)
                ok, msg = False, f"Timeout after {per_item_timeout}s"
                break
            if rss_limit_mb:
                r = _rss_mb(p.pid)
                if r is not None and r > rss_limit_mb:
                    p.terminate(); p.join(5)
                    ok, msg = False, f"RSS limit exceeded: {r} MB > {rss_limit_mb} MB"
                    break
            time.sleep(0.2)

        if ok is None:
            if parent_conn.poll(1.0):
                ok, msg = parent_conn.recv()
            else:
                ok, msg = (p.exitcode == 0), f"exitcode={p.exitcode}"

        parent_conn.close()

        if ok:
            if not no_save:
                append_manifest_atomic(manifest_path, spine_id)
            processed += 1
        else:
            failed += 1
            with open(failed_log, "a", encoding="utf-8") as f:
                f.write(f"{spine_id}\t{msg}\n")
            print(f"    [x] failed: {msg}")

        gc.collect()

    print(f"\nDone. processed={processed}, skipped={skipped}, failed={failed}")
    print(f"[i] manifest now has {len(read_manifest(manifest_path))} ids at {manifest_path}")


# ---------------------- CLI ----------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="NEURD segmentation of neuron meshes.")
    ap.add_argument("input_dir", type=str, help="directory of meshes (.off/.ply/...)")
    ap.add_argument("output_dir", type=str, help="output directory (one subdir per input basename)")
    ap.add_argument("--pattern", type=str, default="*.off", help="glob for the input files")
    ap.add_argument("--manifest", type=str, default="processed.txt",
                    help="manifest path (default: <output_dir>/processed.txt)")
    ap.add_argument("--no-save", action="store_true", help="dry run: do not write to disk")
    ap.add_argument("--no-decimate", action="store_true", help="disable decimation")
    args = ap.parse_args()

    process_folder_with_manifest(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        manifest=args.manifest,
        pattern=args.pattern,
        no_save=args.no_save,
        no_decimate=args.no_decimate,
    )

# Example:
#   python process_all_neurons.py path/to/meshes path/to/output_segmentations
