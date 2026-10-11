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


def plot_geometry(neuron):
    """XY skeleton overview and an actual detected spine with nearby shaft faces."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np

    somas = neuron.get_soma_meshes()
    origin = np.mean(somas[0].vertices, axis=0) if somas else np.zeros(3)
    figure = plt.figure(figsize=(12, 5), layout="constrained")
    overview = figure.add_subplot(121)
    colors = plt.get_cmap("tab10")
    candidates = []
    for limb_id, limb in enumerate(neuron.limbs):
        for branch_id, branch in enumerate(limb.branches):
            segments = (np.asarray(branch.skeleton) - origin) / 1000
            overview.add_collection(LineCollection(segments[:, :, :2], colors=[colors(limb_id % 10)],
                                                    linewidths=0.8, alpha=0.8))
            for spine in [] if branch.spines is None else branch.spines:
                candidates.append((len(spine.faces), limb_id, branch_id, branch, spine))
    for soma in somas:
        triangles = (soma.triangles - origin) / 1000
        # Display sampling only; processing and export use the full meshes.
        overview.add_collection(PolyCollection(triangles[::max(1, len(triangles) // 4000), :, :2],
                                               facecolors="#333333", edgecolors="none", alpha=0.15))
    overview.autoscale()
    overview.set_aspect("equal")
    overview.set(title="Whole neuron: XY projection", xlabel="x from soma (µm)", ylabel="y from soma (µm)")
    overview.grid(alpha=0.2)
    detail = figure.add_subplot(122, projection="3d")
    if candidates:
        _, limb_id, branch_id, branch, spine = max(candidates, key=lambda row: row[0])
        center = np.mean(spine.vertices, axis=0)
        radius = max(1500.0, float(np.ptp(spine.vertices, axis=0).max()))
        nearby = np.all(np.abs(branch.mesh.triangles_center - center) < radius, axis=1)
        shaft = (branch.mesh.triangles[nearby] - center) / 1000
        detail.add_collection3d(Poly3DCollection(shaft[::max(1, len(shaft) // 3000)],
                                                facecolor="#aeb8c2", edgecolor="none", alpha=0.2))
        detail.add_collection3d(Poly3DCollection((spine.triangles - center) / 1000,
                                                facecolor="#db4437", edgecolor="none", alpha=0.9))
        bound = radius / 1000
        detail.set(xlim=(-bound, bound), ylim=(-bound, bound), zlim=(-bound, bound),
                   title=f"Detected spine: limb {limb_id}, branch {branch_id}",
                   xlabel="x (µm)", ylabel="y (µm)", zlabel="z (µm)")
        detail.set_box_aspect((1, 1, 1))
    else:
        detail.text2D(0.2, 0.5, "No spines detected in this run", transform=detail.transAxes)
    return figure
