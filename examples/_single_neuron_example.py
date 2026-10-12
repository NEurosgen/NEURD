"""Measurement and plotting utilities for the single-neuron example notebook."""
from contextlib import contextmanager
import hashlib
from importlib import import_module, metadata
from importlib.machinery import EXTENSION_SUFFIXES
import os
from pathlib import Path
import platform
import resource
import re
import sys
import threading
import time

import psutil


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class RunMeasurements:
    """Stage timers plus sampled RSS; sampling does not track Python allocations."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.process = psutil.Process()
        self.rows, self.samples = [], []
        self.started = time.perf_counter()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _record(self):
        own = self.process.memory_info().rss
        children = 0
        for process in self.process.children(recursive=True):
            try:
                children += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self.samples.append((time.perf_counter() - self.started, own, own + children))

    def _sample(self):
        while not self._stop.is_set():
            self._record()
            self._stop.wait(self.interval)

    @contextmanager
    def stage(self, name):
        first_sample = len(self.samples)
        self._record()
        started = time.perf_counter()
        try:
            yield
        finally:
            finished = time.perf_counter()
            self._record()
            samples = self.samples[first_sample:]
            self.rows.append(dict(stage=name, seconds=finished - started,
                                  sampled_process_peak_bytes=max(row[1] for row in samples),
                                  sampled_process_and_children_peak_bytes=max(row[2] for row in samples)))

    def close(self):
        self._stop.set()
        self._thread.join()

    def report(self):
        factor = 1 if sys.platform == "darwin" else 1024
        return dict(
            stages=self.rows,
            timed_stages_seconds=sum(row["seconds"] for row in self.rows),
            load_segmentation_export_seconds=sum(row["seconds"] for row in self.rows
                                                if row["stage"] != "import"),
            process_lifetime_peak_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * factor),
            sampled_process_peak_bytes=max(row[1] for row in self.samples),
            sampled_process_and_children_peak_bytes=max(row[2] for row in self.samples),
            sampling_interval_seconds=self.interval,
            scope="Kernel RSS includes Jupyter startup. Sampled tree RSS sums live processes; "
                  "shared pages are counted in each process. Short child peaks may be missed.",
        )


def environment_report(neurd_module, config):
    from neurd.run_config import runtime_environment

    package = Path(neurd_module.__file__).parent
    sources = {path.name: sha256(path) for path in sorted(package.glob("*.py"))}
    providers = {}
    for name, entrypoint in (("cgal_Segmentation_Module", "cgal_segmentation"),
                             ("calcification_param_Module", "calcification_param")):
        provider = import_module(name)
        path = Path(provider.__file__)
        if (not any(str(path).endswith(suffix) for suffix in EXTENSION_SUFFIXES)
                or not callable(getattr(provider, entrypoint, None))):
            raise RuntimeError(f"{name} must be a compiled extension with callable {entrypoint}.")
        providers[name] = dict(filename=path.name, sha256=sha256(path))
    return dict(
        python=platform.python_version(), platform=platform.platform(),
        packages=dict(sorted((re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower(), dist.version)
                             for dist in metadata.distributions())),
        neurd_source_sha256=sources,
        neurd_source_digest=hashlib.sha256(
            "\n".join(f"{name}:{digest}" for name, digest in sources.items()).encode()).hexdigest(),
        native_providers=providers,
        thread_environment={name: os.environ.get(name) for name in
                            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                             "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "PYTHONHASHSEED")},
        import_patches=dict(neurd_module.PATCH_STATUS),
        runtime=runtime_environment(config),
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    )


def geometry_metrics(neuron):
    """Geometry counts and lengths, without upstream classifier/synapse fields."""
    import numpy as np

    somas = neuron.get_soma_meshes()
    rows = []
    for limb_id, limb in enumerate(neuron.limbs):
        for branch_id, branch in enumerate(limb.branches):
            spines = [] if branch.spines is None else list(branch.spines)
            skeleton = np.asarray(branch.skeleton, dtype=float).reshape(-1, 2, 3)
            volumes = getattr(branch, "spines_volume", None)
            volumes = np.asarray([] if volumes is None else volumes, dtype=float)
            valid = np.isfinite(volumes) & (volumes > 0)
            rows.append(dict(limb=limb_id, branch=branch_id, mesh_faces=len(branch.mesh.faces),
                             skeleton_length_um=float(np.linalg.norm(
                                 skeleton[:, 1] - skeleton[:, 0], axis=1).sum() / 1000),
                             detected_spines=len(spines),
                             spine_mesh_faces=sum(len(spine.faces) for spine in spines),
                             positive_spine_volumes=int(valid.sum()),
                             spine_volume_um3=float(volumes[valid].sum() / 1e9)))
    summary = dict(n_somas=len(somas), soma_mesh_faces=sum(len(soma.faces) for soma in somas),
                   n_limbs=len(neuron.limbs), n_branches_total=len(rows),
                   branch_mesh_faces_total=sum(row["mesh_faces"] for row in rows),
                   skeleton_length_total_um=sum(row["skeleton_length_um"] for row in rows),
                   n_spines_total=sum(row["detected_spines"] for row in rows),
                   branches_with_spines=sum(row["detected_spines"] > 0 for row in rows),
                   spine_mesh_faces_total=sum(row["spine_mesh_faces"] for row in rows),
                   positive_spine_volumes=sum(row["positive_spine_volumes"] for row in rows))
    return summary, rows


def plot_geometry(neuron, reviewed_spine=None):
    """Full XY skeleton, optionally beside an explicitly supplied reviewed mesh."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np

    somas = neuron.get_soma_meshes()
    origin = np.mean(somas[0].vertices, axis=0) if somas else np.zeros(3)
    figure = plt.figure(figsize=(12, 5) if reviewed_spine is not None else (7, 5),
                        layout="constrained")
    overview = figure.add_subplot(121 if reviewed_spine is not None else 111)
    colors = plt.get_cmap("tab10")
    legend = []
    for limb_id, limb in enumerate(neuron.limbs):
        color = colors(limb_id % 10)
        legend.append(Line2D([], [], color=color, label=f"Limb {limb_id}"))
        for branch in limb.branches:
            segments = (np.asarray(branch.skeleton).reshape(-1, 2, 3) - origin) / 1000
            overview.add_collection(LineCollection(segments[:, :, :2], colors=[color],
                                                     linewidths=0.8, alpha=0.8))
    for soma in somas:
        triangles = (soma.triangles - origin) / 1000
        center = (np.mean(soma.vertices, axis=0) - origin) / 1000
        # The complete skeleton is shown; soma sampling changes only the display.
        triangles = triangles[::max(1, (len(triangles) + 3999) // 4000)]
        overview.add_collection(PolyCollection(triangles[:, :, :2],
                                                facecolors="#333333", edgecolors="none", alpha=0.2))
        overview.scatter(center[0], center[1], color="#333333", s=18, zorder=4)
    if somas:
        legend.append(Line2D([], [], color="#333333", marker="o", linestyle="none", label="Soma"))
    overview.autoscale()
    overview.set_aspect("equal")
    overview.set(title="Reconstruction: XY projection", xlabel="x from soma (µm)",
                 ylabel="y from soma (µm)")
    overview.grid(alpha=0.2)
    if legend:
        overview.legend(handles=legend, loc="upper right", fontsize=8)

    if reviewed_spine is not None:
        context_mesh = reviewed_spine["context_mesh"]
        target_mesh = reviewed_spine["target_mesh"]
        context_mask = np.asarray(reviewed_spine["context_face_mask"])
        if context_mask.dtype != np.bool_ or context_mask.shape != (len(context_mesh.faces),):
            raise ValueError("context_face_mask must be one boolean per context mesh face.")
        provenance = reviewed_spine["provenance"]
        center = np.mean(target_mesh.vertices, axis=0)
        target = (target_mesh.triangles - center) / 1000
        context = (context_mesh.triangles[context_mask] - center) / 1000
        # Render all target and supplied gray-context triangles without mesh repair or sampling.
        normals = np.cross(target[:, 1] - target[:, 0], target[:, 2] - target[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
        light = np.array([0.3, -0.5, 0.8])
        light /= np.linalg.norm(light)
        shades = (0.45 + 0.55 * np.abs(normals @ light))[:, None] * np.array([0.94, 0.24, 0.14])
        detail = figure.add_subplot(122, projection="3d")
        detail.add_collection3d(Poly3DCollection(context, facecolors="#aab4bf",
                                                 edgecolors="none", alpha=0.2))
        detail.add_collection3d(Poly3DCollection(target, facecolors=shades,
                                                 edgecolors="none", alpha=1.0))
        points = np.concatenate((target.reshape(-1, 3), context.reshape(-1, 3)))
        limit = max(0.85, float(np.abs(points).max()) * 1.1)
        detail.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit),
                   xlabel="x (µm)", ylabel="y (µm)", zlabel="z (µm)",
                   title=f"Reviewed H01 spine ({provenance['case_id']})\n"
                         f"Neuron {provenance['neuron_id']} / L{provenance['limb_id']} / B{provenance['branch_id']}")
        detail.set_box_aspect((1, 1, 1))
        detail.set_proj_type("ortho")
        view = provenance.get("view", {})
        detail.view_init(elev=view.get("elev", 20), azim=view.get("azim", 0))
        detail.grid(False)
    return figure
