"""Volume of cut spine surfaces, with explicit closure provenance.

An open input is normal. A nonmanifold output is not a valid volume boundary.
The convex-hull fallback is a geometric approximation, not recovered anatomy.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType

import numpy as np

VOLUME_ALGORITHM = "centered_strict_closure_v1"


def _validate(candidate, bounds, bbox_volume):
    if len(candidate.faces) < 4 or not np.isfinite(candidate.vertices).all():
        return "invalid_geometry"
    if not candidate.is_watertight:
        return "not_watertight"
    if not candidate.is_winding_consistent:
        return "inconsistent_winding"
    span = float(np.max(bounds[1] - bounds[0]))
    tolerance = max(span * 1e-8, 1e-12)
    if np.any(candidate.bounds[0] < bounds[0] - tolerance) or np.any(candidate.bounds[1] > bounds[1] + tolerance):
        return "closure_outside_input_bounds"
    volume = float(candidate.volume)
    if not np.isfinite(volume) or volume <= 0:
        return "nonpositive_volume"
    if volume > bbox_volume * (1 + 1e-6):
        return "volume_exceeds_bbox"
    shifted = candidate.copy()
    shifted.apply_translation(np.array([11.25, -7.5, 3.25]) * span)
    if not np.isclose(shifted.volume, volume, rtol=1e-6, atol=bbox_volume * 1e-12):
        return "translation_dependent_volume"
    return None


def estimate_mesh_volume(mesh, *, close_holes=None):
    """Return a JSON-safe quality record; invalid volumes are ``None``.

    All work uses a centered copy. ``close_holes`` is an optional closure
    callable, chiefly useful for testing the production mesh_tools routine in
    isolation. Input geometry and metadata are never modified.
    """
    report = {"algorithm": VOLUME_ALGORITHM, "method": "invalid", "valid": False,
              "volume_nm3": None, "input_watertight": False, "fallback_reason": None}
    if mesh is None or len(mesh.vertices) < 4 or len(mesh.faces) < 2:
        report["fallback_reason"] = "insufficient_geometry"
        return report
    if not np.isfinite(mesh.vertices).all():
        report["fallback_reason"] = "nonfinite_vertices"
        return report
    source = mesh.copy()
    report["input_watertight"] = bool(source.is_watertight)
    source.apply_translation(-source.bounds.mean(axis=0))
    bounds = np.asarray(source.bounds).copy()
    bbox_volume = float(np.prod(source.extents))
    report["bbox_volume_nm3"] = bbox_volume
    if not np.isfinite(bbox_volume) or bbox_volume <= 0:
        report["fallback_reason"] = "degenerate_bbox"
        return report

    if source.is_watertight:
        candidate, method = source, "input_closed"
        reason = _validate(candidate, bounds, bbox_volume)
    else:
        if close_holes is None:
            # Do not import the heavy NEURD stack for closed-mesh validation.
            from mesh_tools.trimesh_utils import fill_mesh_holes_with_fan
            close_holes = fill_mesh_holes_with_fan
        try:
            candidate = close_holes(source.copy())
            method = "fan_closed"
            reason = _validate(candidate, bounds, bbox_volume)
        except (ImportError, ModuleNotFoundError):
            raise  # A missing dependency must not silently change the method.
        except Exception as error:
            reason = f"closure_failed:{type(error).__name__}"

    if reason is not None:
        report["fallback_reason"] = reason
        try:
            candidate = source.convex_hull
            method = "convex_hull"
            hull_reason = _validate(candidate, bounds, bbox_volume)
        except Exception as error:
            hull_reason = f"hull_failed:{type(error).__name__}"
        if hull_reason is not None:
            report["invalid_reason"] = hull_reason
            return report

    report.update(valid=True, method=method, volume_nm3=float(candidate.volume))
    return report


def mesh_volume(mesh):
    """Scalar adapter for whole-spine and compartment descriptor code."""
    report = estimate_mesh_volume(mesh)
    return report["volume_nm3"] if report["valid"] else float("nan")


@dataclass(frozen=True)
class VolumeQualityRecord:
    """Process-local handoff for the same ordered in-memory accepted geometry.

    The retained mesh reference proves identity; the digest detects mutation.
    This record cannot replace an independent validation of rounded OFF data.
    """
    mesh: object
    geometry_sha256: str
    quality: object
    scope: str = "in_memory_before_off_rounding"


def mesh_geometry_digest(mesh):
    """Hash ordered vertices/faces with dtype/shape using bounded scratch."""
    digest = hashlib.sha256()
    if mesh is None:
        digest.update(b"missing_mesh")
        return digest.hexdigest()
    for name in ("vertices", "faces"):
        array = np.asarray(getattr(mesh, name))
        if array.dtype.hasobject:
            raise ValueError("Geometry digest requires numerical arrays")
        digest.update(json.dumps(dict(name=name, dtype=array.dtype.str,
                                      shape=list(array.shape)), sort_keys=True).encode())
        if array.size == 0:
            continue
        if array.flags.c_contiguous:
            view = memoryview(array).cast("B")
            for offset in range(0, len(view), 1024 * 1024):
                digest.update(view[offset:offset + 1024 * 1024])
        else:
            # Trimesh arrays are normally contiguous. This fallback preserves
            # the same C-order bytes, copying at most 8192 rows per chunk.
            for offset in range(0, len(array), 8192):
                digest.update(memoryview(np.ascontiguousarray(array[offset:offset + 8192])).cast("B"))
    return digest.hexdigest()


def mesh_volume_with_quality(mesh):
    """Run the identical estimator once and retain its exact scalar/report."""
    report = estimate_mesh_volume(mesh)
    scalar = report["volume_nm3"] if report["valid"] else float("nan")
    record = VolumeQualityRecord(mesh, mesh_geometry_digest(mesh), MappingProxyType(dict(report)))
    return scalar, record


def reuse_volume_quality(record, mesh):
    """Reject unavailable, different-object or stale-geometry handoffs."""
    if not isinstance(record, VolumeQualityRecord) or record.mesh is not mesh:
        raise ValueError("Volume quality record does not reference the accepted mesh")
    if mesh_geometry_digest(mesh) != record.geometry_sha256:
        raise ValueError("Accepted mesh geometry changed since volume validation")
    if record.scope != "in_memory_before_off_rounding":
        raise ValueError("Volume quality record does not describe the accepted in-memory geometry")
    if record.quality.get("algorithm") != VOLUME_ALGORITHM:
        raise ValueError("Volume quality algorithm differs from the current estimator")
    return dict(record.quality)
