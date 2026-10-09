"""Experimental, conservative refinement of a CGAL spine partition.

All lengths are nanometres. Only original parent faces may be added; no mesh
repair, smoothing or synthetic caps are performed. A shaft-surface estimate is
a geometric proxy, not anatomical ground truth. Uncertain repairs retain their
original seed. This module uses numpy, scipy, trimesh and matplotlib.path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from time import perf_counter

import numpy as np
import trimesh
from matplotlib.path import Path as PolygonPath
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class RefinementConfig:
    """Exploratory thresholds; the branch API leaves this stage disabled."""

    minimum_faces: int = 15
    screen_gap_nm: float = 200.0
    sheet_protrusion_nm: float = 100.0
    sheet_boundary_ratio: float = 6.0
    # A branched cut rim prevents safe growth, but does not prevent measuring
    # whether the candidate is just a patch on the parent shaft surface.
    filter_ragged_surface_patches: bool = True
    maximum_section_surface_spread_nm: float = 700.0
    incomplete_gap_nm: float = 200.0
    growth_clearance_nm: float = 50.0
    growth_margin_nm: float = 2500.0
    maximum_area_ratio: float = 4.0
    maximum_added_faces: int = 5000
    maximum_extent_nm: float = 5000.0
    section_offsets_nm: tuple = (-400.0, -200.0, 0.0, 200.0, 400.0)
    section_radius_nm: float = 3000.0
    minimum_sections: int = 3
    maximum_section_area_mad: float = 0.5
    maximum_section_center_spread_nm: float = 700.0
    skeleton_step_nm: float = 100.0
    tangent_radius_nm: float = 700.0
    growth_profile_step_nm: float = 400.0
    maximum_growth_profiles: int = 32
    maximum_repairs_per_branch: int = 8
    # Complete perforated head masks on the existing parent surface. A separate
    # bounded budget preserves repairs that ordinary cap growth already made.
    grow_ragged_heads: bool = False
    maximum_ragged_repairs_per_branch: int = 8
    ragged_minimum_added_clearance_nm: float = 300.0
    # Optional subtraction/splitting of shaft pollution. This is separate from
    # seed-preserving head growth, and remains disabled pending H01 validation.
    trim_shaft_connections: bool = False
    trim_clearance_nm: float = 50.0
    trim_collar_nm: float = 100.0
    trim_minimum_area_nm2: float = 100_000.0
    trim_minimum_area_fraction: float = 0.15
    trim_minimum_protrusion_nm: float = 300.0
    trim_maximum_surface_spread_nm: float = 300.0
    trim_maximum_axis_distance_nm: float = 3000.0
    trim_use_non_candidate_sections: bool = True
    trim_allow_mesh_direction: bool = True
    trim_maximum_section_bridge_nm: float = 700.0
    # A leaning neck can also run close to the estimated shaft surface. Keep
    # single protrusions intact unless this exploratory mode is requested.
    trim_single_protrusions: bool = False
    # A large ragged cut with little body above its own rim is a surface
    # fragment, even when holes prevent closed shaft sections. Run after head
    # completion, so a recovered protrusion is measured in its completed form.
    filter_low_relief_sheets: bool = False
    sheet_minimum_boundary_ratio: float = 4.5
    sheet_maximum_relative_relief_nm: float = 300.0
    sheet_mesh_radii_nm: tuple = (1000.0, 1600.0, 2200.0, 3200.0)
    # Retry unchanged, uncertain candidates only when the stored axis misses a
    # substantial part of the shaft. Primary repairs retain their exact masks.
    retry_with_mesh_axis: bool = False
    axis_coverage_distance_nm: float = 10_000.0
    axis_coverage_maximum_missing_fraction: float = 0.2
    axis_coverage_sample_faces: int = 4096
    mesh_axis_maximum_area_ratio: float = 8.0
    mesh_axis_maximum_growth_profiles: int = 64
    require_distal_growth: bool = False

    def __post_init__(self):
        positive = (self.section_radius_nm, self.skeleton_step_nm,
                    self.tangent_radius_nm, self.growth_profile_step_nm,
                    self.maximum_extent_nm, self.maximum_area_ratio)
        if any(not np.isfinite(x) or x <= 0 for x in positive):
            raise ValueError("Refinement lengths and area ratio must be positive")
        if self.minimum_sections < 1 or self.minimum_sections > len(self.section_offsets_nm):
            raise ValueError("Invalid minimum_sections")
        if self.growth_clearance_nm < 0 or self.growth_margin_nm < 0:
            raise ValueError("Growth clearance and margin must be nonnegative")
        if not np.isfinite(self.ragged_minimum_added_clearance_nm) or self.ragged_minimum_added_clearance_nm < 0:
            raise ValueError("Added head clearance must be finite and nonnegative")
        trim_values = (self.trim_clearance_nm, self.trim_collar_nm,
                       self.trim_minimum_area_nm2, self.trim_minimum_protrusion_nm,
                       self.trim_maximum_surface_spread_nm, self.trim_maximum_axis_distance_nm,
                       self.trim_maximum_section_bridge_nm)
        if any(not np.isfinite(v) or v < 0 for v in trim_values):
            raise ValueError("Shaft trimming thresholds must be finite and nonnegative")
        if not 0 <= self.trim_minimum_area_fraction <= 1:
            raise ValueError("Invalid shaft trimming area fraction")
        if (not np.isfinite(self.sheet_minimum_boundary_ratio) or self.sheet_minimum_boundary_ratio <= 0
                or not np.isfinite(self.sheet_maximum_relative_relief_nm) or self.sheet_maximum_relative_relief_nm < 0
                or len(self.sheet_mesh_radii_nm) < 2
                or any(not np.isfinite(r) or r <= 0 for r in self.sheet_mesh_radii_nm)):
            raise ValueError("Invalid low-relief sheet thresholds")
        if (not np.isfinite(self.axis_coverage_distance_nm) or self.axis_coverage_distance_nm <= 0
                or not 0 < self.axis_coverage_maximum_missing_fraction < 1
                or not isinstance(self.axis_coverage_sample_faces, int)
                or self.axis_coverage_sample_faces < 16):
            raise ValueError("Invalid axis coverage thresholds")
        if (not np.isfinite(self.mesh_axis_maximum_area_ratio) or self.mesh_axis_maximum_area_ratio < 1
                or not isinstance(self.mesh_axis_maximum_growth_profiles, int)
                or self.mesh_axis_maximum_growth_profiles < 1):
            raise ValueError("Invalid auxiliary growth limits")


def _quantile(values, weights, quantiles):
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    total = weights.sum()
    if not len(values) or total <= 0 or not np.isfinite(total):
        raise ValueError("empty_or_degenerate_surface")
    positions = (np.cumsum(weights) - 0.5 * weights) / total
    return np.interp(quantiles, positions, values, left=values[0], right=values[-1])


def _frame(tangent):
    tangent = np.array(tangent, dtype=float, copy=True)
    tangent /= np.linalg.norm(tangent)
    reference = np.eye(3)[np.argmin(abs(tangent))]
    u = np.cross(tangent, reference)
    u /= np.linalg.norm(u)
    return np.column_stack((u, np.cross(tangent, u), tangent))


class _Contour:
    def __init__(self, polygon):
        self.vertices = np.asarray(polygon, dtype=float)
        self.edge = np.roll(self.vertices, -1, axis=0) - self.vertices
        self.norm = np.maximum(np.einsum("ij,ij->i", self.edge, self.edge), 1e-15)
        self.tree = self.ids = None
        self.path = PolygonPath(np.vstack((self.vertices, self.vertices[0])))

    def search_index(self):
        if self.tree is not None:
            return
        samples, ids = [], []
        for i, (a, edge, size) in enumerate(zip(self.vertices, self.edge, np.sqrt(self.norm))):
            fractions = np.linspace(0, 1, max(2, int(np.ceil(size / 50.0)) + 1))
            samples.append(a + fractions[:, None] * edge)
            ids.extend([i] * len(fractions))
        self.tree, self.ids = cKDTree(np.concatenate(samples)), np.asarray(ids)

    def distance(self, points):
        """Exact segment distances using sampled points only as a search index.

        Samples include all endpoints and have spacing <=50 nm. For a closest
        interior point at distance d, a sample on that same segment lies within
        sqrt(d*d + 25*25). Thus the query below includes the true closest edge;
        it does not replace the contour with a sampled approximation.
        """
        points = np.asarray(points, dtype=float)
        if not len(points):
            return np.empty(0)
        # Small batches (especially choosing a contour from one axis point)
        # are cheaper directly; avoid building a spatial index for them.
        if len(points) * len(self.vertices) <= 100_000:
            delta = points[:, None] - self.vertices
            fraction = np.clip(np.einsum("nkj,kj->nk", delta, self.edge) / self.norm, 0, 1)
            residual = delta - fraction[:, :, None] * self.edge
            result = np.sqrt(np.einsum("nkj,nkj->nk", residual, residual).min(axis=1))
            result[self.path.contains_points(points)] *= -1
            return result
        self.search_index()
        _, nearest = self.tree.query(points)
        edges = self.ids[nearest]
        delta = points - self.vertices[edges]
        fraction = np.clip(np.einsum("ij,ij->i", delta, self.edge[edges]) / self.norm[edges], 0, 1)
        upper = np.linalg.norm(delta - fraction[:, None] * self.edge[edges], axis=1)
        neighborhoods = self.tree.query_ball_point(points, np.hypot(upper, 25.0) + 1e-6)
        counts = np.fromiter((len(ids) for ids in neighborhoods), dtype=int, count=len(points))
        rows = np.repeat(np.arange(len(points)), counts)
        edges = self.ids[np.concatenate(neighborhoods)]
        delta = points[rows] - self.vertices[edges]
        fraction = np.clip(np.einsum("ij,ij->i", delta, self.edge[edges]) / self.norm[edges], 0, 1)
        distance = np.linalg.norm(delta - fraction[:, None] * self.edge[edges], axis=1)
        result = np.full(len(points), np.inf)
        np.minimum.at(result, rows, distance)
        result[self.path.contains_points(points)] *= -1
        return result


def _polygon_distance(points, polygon):
    contour = polygon if isinstance(polygon, _Contour) else _Contour(polygon)
    return contour.distance(points)


class _Axis:
    def __init__(self, skeleton, config):
        segments = np.asarray(skeleton, dtype=float).reshape(-1, 2, 3)
        length = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
        valid = np.isfinite(segments).all(axis=(1, 2)) & (length > 1e-5)
        self.segments, self.length = segments[valid], length[valid]
        if not len(self.segments):
            raise ValueError("empty_skeleton")
        self.direction = (self.segments[:, 1] - self.segments[:, 0]) / self.length[:, None]
        samples, ids, profiles = [], [], []
        for i, (segment, size) in enumerate(zip(self.segments, self.length)):
            t = np.linspace(0, 1, int(np.ceil(size / config.skeleton_step_nm)) + 1)
            samples.append(segment[0] + t[:, None] * (segment[1] - segment[0]))
            ids.extend([i] * len(t))
            t = np.linspace(0, 1, int(np.ceil(size / config.growth_profile_step_nm)) + 1)
            profiles.append(segment[0] + t[:, None] * (segment[1] - segment[0]))
        self.ids = np.asarray(ids)
        self.tree = cKDTree(np.concatenate(samples))
        # Skeletons often contain short segments: spacing is enforced on the
        # whole sample set rather than generating a separate profile per edge.
        points = np.concatenate(profiles)
        cells = np.floor(points / config.growth_profile_step_nm).astype(np.int64)
        _, indices = np.unique(cells, axis=0, return_index=True)
        self.profile_tree = cKDTree(points[np.sort(indices)])
        self.config = config

    def project(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        _, near = self.tree.query(points, k=min(8, len(self.ids)))
        if near.ndim == 1:
            near = near[:, None]
        ids = self.ids[near]
        start, direction = self.segments[ids, 0], self.direction[ids]
        along = np.einsum("nkj,nkj->nk", points[:, None, :] - start, direction)
        along = np.clip(along, 0, self.length[ids])
        feet = start + along[:, :, None] * direction
        distance = np.linalg.norm(points[:, None, :] - feet, axis=2)
        best = np.argmin(distance, axis=1)
        rows = np.arange(len(points))
        return feet[rows, best], distance[rows, best]

    def tangent(self, anchor):
        ids = self.tree.query_ball_point(anchor, self.config.tangent_radius_nm)
        points = self.tree.data[ids]
        if len(points) < 6:
            raise ValueError("insufficient_local_axis_support")
        delta = points - points.mean(axis=0)
        _, basis = np.linalg.eigh(delta.T @ delta)
        return basis[:, -1]


def _boundary(parent, indices):
    faces = parent.faces[indices]
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("nonmanifold_candidate")
    edges = edges[counts == 1]
    if not len(edges):
        raise ValueError("no_open_boundary")
    if np.any(np.unique(edges, return_counts=True)[1] != 2):
        raise ValueError("branched_open_boundary")
    vertices = parent.vertices[edges]
    length = np.linalg.norm(vertices[:, 1] - vertices[:, 0], axis=1)
    fractions = np.array([1 / 6, 0.5, 5 / 6])
    samples = (vertices[:, 0, None] * (1 - fractions)[None, :, None]
               + vertices[:, 1, None] * fractions[None, :, None]).reshape(-1, 3)
    weights = np.repeat(length / 3, 3)
    area = parent.area_faces[indices].sum()
    if area <= 0 or weights.sum() <= 0:
        raise ValueError("empty_or_degenerate_surface")
    return samples, weights, float(weights.sum() / np.sqrt(area))


def _surface_boundary(parent, indices):
    """Cut-edge samples for rejection, without assuming a simple rim loop.

    CGAL can join surface fragments through one vertex or split a patch around
    small holes. Those cuts have branched boundaries even on a valid parent.
    Rejecting a surface scrap only needs its cut-edge lengths and protrusion;
    it does not need an ordered closed rim. Duplicate triangles are ignored
    locally for this measurement. The original mesh and seed remain intact.
    This descriptor must never authorize growth across an invalid boundary.
    """
    faces = parent.faces[indices]
    _, unique = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    unique_indices = indices[np.sort(unique)]
    # Degenerate triangles do not describe a surface or a physical cut edge.
    unique_indices = unique_indices[parent.area_faces[unique_indices] > 1e-12]
    faces = parent.faces[unique_indices]
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    edges = edges[counts == 1]
    if not len(edges):
        raise ValueError("no_open_boundary")
    vertices = parent.vertices[edges]
    length = np.linalg.norm(vertices[:, 1] - vertices[:, 0], axis=1)
    fractions = np.array([1 / 6, 0.5, 5 / 6])
    samples = (vertices[:, 0, None] * (1 - fractions)[None, :, None]
               + vertices[:, 1, None] * fractions[None, :, None]).reshape(-1, 3)
    weights = np.repeat(length / 3, 3)
    area = parent.area_faces[unique_indices].sum()
    if area <= 0 or weights.sum() <= 0:
        raise ValueError("empty_or_degenerate_surface")
    return (samples, weights, float(weights.sum() / np.sqrt(area))), unique_indices


def _section_polygons(vertices, faces, anchor, basis, offset, support_faces=None):
    # Triangle/plane intersections follow the diagnostic attachment estimator.
    # Duplicate parent triangles are ignored locally without editing the input.
    d = (vertices - anchor) @ basis[:, 2] - offset - 1e-5
    dt = d[faces]
    intersect = (dt.min(axis=1) < 0) & (dt.max(axis=1) > 0)
    faces = faces[intersect]
    if support_faces is not None:
        support_faces = np.asarray(support_faces)[intersect]
    if not len(faces):
        return []
    edges = np.stack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=1)
    cross = (d[edges[:, :, 0]] > 0) != (d[edges[:, :, 1]] > 0)
    crossed = np.sort(edges[cross], axis=1)
    unique, inverse = np.unique(crossed, axis=0, return_inverse=True)
    support_nodes = None
    if support_faces is not None:
        support_nodes = np.zeros(len(unique), dtype=bool)
        np.maximum.at(support_nodes, inverse, np.repeat(support_faces, 2))
    a, b = vertices[unique[:, 0]], vertices[unique[:, 1]]
    da, db = d[unique[:, 0]], d[unique[:, 1]]
    points = a + (da / (da - db))[:, None] * (b - a)
    adjacency = {}
    for i, j in inverse.reshape(-1, 2):
        adjacency.setdefault(int(i), []).append(int(j))
        adjacency.setdefault(int(j), []).append(int(i))
    visited, polygons = set(), []
    for start in adjacency:
        if start in visited:
            continue
        stack, component = [start], set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency[current])
        visited.update(component)
        if len(component) < 6 or any(len(adjacency[i]) != 2 for i in component):
            continue
        ordered, previous, current = [start], -1, start
        while True:
            nxt = next(i for i in adjacency[current] if i != previous)
            if nxt == start:
                break
            ordered.append(nxt)
            previous, current = current, nxt
            if len(ordered) > len(component):
                raise ValueError("invalid_section_loop")
        polygon = (points[ordered] - anchor) @ basis[:, :2]
        area = abs(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1)
                          - np.roll(polygon[:, 0], -1) * polygon[:, 1])) / 2
        if area > np.pi * 30 ** 2:
            if support_nodes is None:
                polygons.append((polygon, float(area)))
            else:
                polygons.append((polygon, float(area), support_nodes[ordered]))
    return polygons


class _Context:
    def __init__(self, parent, skeleton, groups, config):
        self.parent, self.config = parent, config
        self.axis = _Axis(skeleton, config)
        self.centers, self.areas = parent.triangles_center, parent.area_faces
        self.face_tree = cKDTree(self.centers)
        # Lazy: most candidates only need cheap boundary descriptors.
        self.canonical = self.representatives = self.adjacency = None
        self.groups = groups
        self.excluded_support = np.zeros(len(parent.faces), dtype=bool)
        for group in groups:
            self.excluded_support[group] = True
        self.profiles = {}
        self.growth_scores = self.growth_scored = None
        self.statistics = dict(section_models=0, screened=0, assessed=0, repairs_attempted=0,
                               growth_faces_scored=0, trims_attempted=0)
        self.max_face_radius = float(np.max(np.linalg.norm(parent.triangles - self.centers[:, None], axis=2)))

    def unique_faces(self):
        if self.canonical is None:
            _, representatives, inverse = np.unique(np.sort(self.parent.faces, axis=1), axis=0,
                                                     return_index=True, return_inverse=True)
            self.representatives = np.sort(representatives)
            self.canonical = representatives[inverse]
            if hasattr(self, "excluded_support"):
                physical = self.canonical[np.flatnonzero(self.excluded_support)]
                self.excluded_support |= np.isin(self.canonical,physical)
        return self.representatives

    def model(self, anchor, *, basis=None, shaft_support_only=False):
        basis = _frame(self.axis.tangent(anchor)) if basis is None else basis
        radius = np.hypot(self.config.section_radius_nm, max(abs(x) for x in self.config.section_offsets_nm))
        ids = np.asarray(self.face_tree.query_ball_point(anchor, radius + self.max_face_radius), dtype=int)
        # Only transform nearby vertices, not the full branch at every section.
        self.unique_faces()
        ids = ids[self.canonical[ids] == ids]
        faces = self.parent.faces[ids]
        vertices, inverse = np.unique(faces, return_inverse=True)
        faces = inverse.reshape(-1, 3)
        vertices = self.parent.vertices[vertices]
        radial = np.linalg.norm((vertices - anchor) @ basis[:, :2], axis=1)
        selected = (radial[faces] < self.config.section_radius_nm).any(axis=1)
        faces = faces[selected]
        support = ~self.excluded_support[ids[selected]] if shaft_support_only else None
        polygons, areas = [], []
        for offset in self.config.section_offsets_nm:
            candidates = _section_polygons(vertices, faces, anchor, basis, offset, support)
            if shaft_support_only:
                candidates = [result for item in candidates
                              if (result := _shaft_contour(item, self.config.trim_maximum_section_bridge_nm)) is not None]
            if not candidates:
                continue
            polygon, area = min(candidates, key=lambda item:
                                max(0, _polygon_distance(np.zeros((1, 2)), item[0])[0])
                                + 0.001 * np.linalg.norm(item[0].mean(axis=0)))
            polygons.append(polygon)
            areas.append(area)
        if len(polygons) < self.config.minimum_sections:
            raise ValueError("insufficient_closed_shaft_sections")
        areas = np.asarray(areas)
        centers = np.array([p.mean(axis=0) for p in polygons])
        mad = np.median(abs(areas - np.median(areas))) / np.median(areas)
        spread = np.max(np.linalg.norm(centers - np.median(centers, axis=0), axis=1))
        if mad > self.config.maximum_section_area_mad or spread > self.config.maximum_section_center_spread_nm:
            raise ValueError("unstable_shaft_sections")
        self.statistics["section_models"] += 1
        return anchor, basis, [_Contour(polygon) for polygon in polygons]

    @staticmethod
    def score(points, model):
        anchor, basis, polygons = model
        xy = (points - anchor) @ basis[:, :2]
        return np.quantile(np.stack([_polygon_distance(xy, polygon) for polygon in polygons]), 0.25, axis=0)

    def assess(self, indices, boundary=None, *, measure_surface_spread=False, model=None):
        rim, weights, ratio = _boundary(self.parent, indices) if boundary is None else boundary
        anchor = self.axis.project([np.average(rim, axis=0, weights=weights)])[0][0]
        model = self.model(anchor) if model is None else model
        gap = float(_quantile(self.score(rim, model), weights, 0.1))
        if measure_surface_spread:
            anchor, basis, polygons = model
            xy = (self.centers[indices] - anchor) @ basis[:, :2]
            distances = np.stack([_polygon_distance(xy, polygon) for polygon in polygons])
            lower, upper = np.quantile(distances, (0.25, 0.75), axis=0)
            spread = float(_quantile(upper - lower, self.areas[indices], 0.95))
        else:
            lower = self.score(self.centers[indices], model)
        protrusion = float(_quantile(lower, self.areas[indices], 0.95))
        self.statistics["assessed"] += 1
        geometry = dict(gap_q10_nm=gap, surface_q95_nm=protrusion,
                        boundary_perimeter_normalized=ratio)
        if measure_surface_spread:
            geometry["section_surface_spread_q95_nm"] = spread
        return geometry, model

    def suspicious(self, boundary):
        rim, weights, ratio = boundary
        if ratio >= self.config.sheet_boundary_ratio:
            return True
        if self.config.maximum_repairs_per_branch == 0:
            return False
        anchor = self.axis.project([np.average(rim, axis=0, weights=weights)])[0][0]
        tangent = self.axis.tangent(anchor)
        ids = np.asarray(self.face_tree.query_ball_point(anchor, self.config.section_radius_nm), dtype=int)
        v = self.centers[ids] - anchor
        along = v @ tangent
        ids, v, along = ids[abs(along) < 400], v[abs(along) < 400], along[abs(along) < 400]
        if len(ids) < 6:
            return True
        radius = _quantile(np.linalg.norm(v - along[:, None] * tangent, axis=1), self.areas[ids], 0.25)
        gap = _quantile(self.axis.project(rim)[1], weights, 0.1) - radius
        return gap > self.config.screen_gap_nm

    def curved_scores(self, face_indices):
        if self.growth_scores is None:
            self.growth_scores = np.full(len(self.parent.faces), np.nan)
            self.growth_scored = np.zeros(len(self.parent.faces), dtype=bool)
        uncached = face_indices[~self.growth_scored[face_indices]]
        points = self.centers[uncached]
        if not len(points):
            return self.growth_scores[face_indices]
        feet, _ = self.axis.project(points)
        _, ids = self.axis.profile_tree.query(feet)
        if len(np.unique(ids)) > self.config.maximum_growth_profiles:
            raise ValueError("growth_profile_budget_exhausted")
        score = np.full(len(points), np.nan)
        for key in np.unique(ids):
            key = int(key)
            if key not in self.profiles:
                try:
                    self.profiles[key] = self.model(self.axis.profile_tree.data[key])
                except ValueError:
                    self.profiles[key] = None
            model = self.profiles[key]
            if model is not None:
                selected = np.flatnonzero(ids == key)
                score[selected] = self.score(points[selected], model)
        self.growth_scores[uncached] = score
        self.growth_scored[uncached] = True
        self.statistics["growth_faces_scored"] += len(uncached)
        return self.growth_scores[face_indices]

    def graph(self):
        if self.adjacency is None:
            unique = self.unique_faces()
            # Edge connectivity prevents growth through a single shared vertex.
            adjacency = trimesh.graph.face_adjacency(self.parent.faces[unique])
            self.adjacency = unique[adjacency]
        return self.adjacency

    def grow(self, indices, input_index, model, claimed):
        anchor, basis, _ = model
        seed_points = self.centers[indices]
        axial = (seed_points - anchor) @ basis[:, 2]
        radial = np.linalg.norm((seed_points - anchor) @ basis[:, :2], axis=1).max() + 1000
        extent = max(abs(axial.min()), abs(axial.max())) + self.config.growth_margin_nm
        local = np.asarray(self.face_tree.query_ball_point(anchor, np.hypot(radial, extent)), dtype=int)
        v = self.centers[local] - anchor
        along = v @ basis[:, 2]
        local = local[(along >= axial.min() - self.config.growth_margin_nm)
                      & (along <= axial.max() + self.config.growth_margin_nm)]
        score = self.curved_scores(local)
        eligible = local[np.isfinite(score) & (score > self.config.growth_clearance_nm)]
        owners = np.full(len(self.parent.faces), -1, dtype=int)
        for i, group in enumerate(self.groups):
            core = np.unique(self.canonical[group])
            conflicts = (owners[core] != -1) & (owners[core] != i)
            owners[core[~conflicts]] = i
            owners[core[conflicts]] = -2
        if np.any(owners[self.canonical[indices]] == -2):
            raise ValueError("overlapping_seed_geometry")
        # Already accepted additions are protected too, so sequential repairs
        # cannot reuse triangles. Competition uses all original candidates.
        owners[claimed >= 0] = claimed[claimed >= 0]
        eligible = eligible[owners[self.canonical[eligible]] != -2]
        local_seeds = local[owners[self.canonical[local]] >= 0]
        nodes = np.unique(self.canonical[np.r_[eligible, local_seeds, indices]])
        pairs = self.graph()
        pairs = pairs[np.isin(pairs[:, 0], nodes) & np.isin(pairs[:, 1], nodes)]
        mapped = np.searchsorted(nodes, pairs)
        distance = np.maximum(np.linalg.norm(self.centers[pairs[:, 0]] - self.centers[pairs[:, 1]], axis=1), 1e-6)
        graph = coo_matrix((distance, (mapped[:, 0], mapped[:, 1])), shape=(len(nodes), len(nodes))).tocsr()
        sources = np.flatnonzero(owners[nodes] >= 0)
        _, _, nearest = dijkstra(graph, directed=False, indices=sources, min_only=True, return_predecessors=True)
        assignment = np.full(len(nodes), -1, dtype=int)
        reached = nearest >= 0
        assignment[reached] = owners[nodes[nearest[reached]]]
        grown = np.union1d(indices, nodes[(assignment == input_index) & (owners[nodes] < 0)])
        added = np.setdiff1d(grown, indices)
        # Protected raw seed triangles are never taken from a neighbour.
        if np.any(owners[added] >= 0):
            raise ValueError("neighbor_seed_conflict")
        if not len(added):
            raise ValueError("no_safe_growth")
        if len(added) > self.config.maximum_added_faces:
            raise ValueError("growth_face_limit")
        if self.areas[grown].sum() > self.config.maximum_area_ratio * self.areas[indices].sum():
            raise ValueError("growth_area_limit")
        vertices = self.parent.vertices[np.unique(self.parent.faces[grown])]
        if np.max(np.ptp(vertices, axis=0)) > self.config.maximum_extent_nm:
            raise ValueError("growth_extent_limit")
        return grown

    def trim(self, indices):
        """Subtract a well-supported shaft patch and recover a small collar.

        Connectivity alone never authorizes splitting. Parts must remain
        distinct at three nearby surface thresholds, and substantial distal
        surface cannot be lost. No faces outside the input are taken here.
        """
        cfg = self.config
        boundary, measured = _surface_boundary(self.parent, indices)
        rim, weights, _ = boundary
        origin = np.average(rim, axis=0, weights=weights)
        axis_distance = _quantile(self.axis.project(rim)[1], weights, .9)
        covered = axis_distance <= cfg.trim_maximum_axis_distance_nm
        failures, unchanged = [], None
        try:
            model, direction = None, "skeleton"
            if covered:
                anchor = self.axis.project([origin])[0][0]
                try:
                    model = self.model(anchor, shaft_support_only=cfg.trim_use_non_candidate_sections)
                except ValueError:
                    if not cfg.trim_allow_mesh_direction:
                        raise
            if model is None:
                if not cfg.trim_allow_mesh_direction:
                    raise ValueError("shaft_trim_axis_does_not_cover_attachment")
                anchor, basis = self.mesh_frame(origin)
                model = self.model(anchor, basis=basis, shaft_support_only=True)
                direction = "mesh"
            parts, detail = self._trim_proposal(indices, measured, boundary, model, allow_remove=True)
            detail["model_direction"] = direction
            detail["shaft_support_only"] = cfg.trim_use_non_candidate_sections or direction=="mesh"
            if detail["action"] != "screened_keep":
                return parts, detail
            unchanged = parts, detail
        except ValueError as exc:
            failures.append(str(exc))
        # Candidate arcs can hide the shaft curvature when omitted. A full
        # contour may establish *separate* attached protrusions, but never
        # authorizes a new surface rejection or trimming a single leaning neck.
        if covered and cfg.trim_use_non_candidate_sections:
            try:
                anchor = self.axis.project([origin])[0][0]
                model = self.model(anchor)
                parts, detail = self._trim_proposal(indices, measured, boundary, model, allow_remove=False)
                if len(parts)>1:
                    detail.update(model_direction="skeleton",shaft_support_only=False)
                    return parts, detail
            except ValueError as exc:
                failures.append(str(exc))
        if unchanged is not None:
            return unchanged
        raise ValueError(failures[0] if failures else "shaft_trim_no_reliable_surface_model")

    def grow_ragged_head(self, indices, input_index, boundary, geometry, model, claimed):
        """A long cut rim can come from holes within a real, incomplete head.

        No synthetic filling is performed: competing seeds and the existing
        shaft-clearance mask constrain additions. A local axis must cover the
        seed; distant/global skeleton projections cannot authorize this growth.
        For masks already touching the shaft, require a substantial reduction
        of the physical cut perimeter relative to area, rather than mistaking
        their low rim quantile for a complete head.
        """
        cfg = self.config
        rim, weights, ratio = boundary
        spread = geometry.get("section_surface_spread_q95_nm", 0.0)
        if geometry["surface_q95_nm"] <= max(cfg.trim_minimum_protrusion_nm, spread+cfg.growth_clearance_nm):
            raise ValueError("ragged_head_no_distinct_protrusion")
        if geometry["gap_q10_nm"] <= cfg.incomplete_gap_nm and ratio < cfg.sheet_boundary_ratio:
            raise ValueError("ragged_head_no_incomplete_boundary")
        distance = self.axis.project(rim)[1]
        if _quantile(distance, weights, .9) > cfg.trim_maximum_axis_distance_nm:
            raise ValueError("ragged_head_axis_does_not_cover_seed")
        attempts = self.statistics.get("ragged_repairs_attempted", 0)
        if attempts >= cfg.maximum_ragged_repairs_per_branch:
            raise ValueError("ragged_head_repair_budget_exhausted")
        self.statistics["ragged_repairs_attempted"] = attempts+1
        proposed = self.grow(indices, input_index, model, claimed)
        after_boundary, measured = _surface_boundary(self.parent, proposed)
        after, _ = self.assess(measured, after_boundary, measure_surface_spread=True, model=model)
        gap_improved = (geometry["gap_q10_nm"] > cfg.incomplete_gap_nm
                        and after["gap_q10_nm"] <= cfg.incomplete_gap_nm
                        and after["gap_q10_nm"] < geometry["gap_q10_nm"])
        rim_improved = (ratio >= cfg.sheet_boundary_ratio
                        and after["boundary_perimeter_normalized"] <= .7*ratio
                        and after["gap_q10_nm"] <= cfg.incomplete_gap_nm)
        if not (gap_improved or rim_improved):
            raise ValueError("ragged_head_boundary_did_not_improve")
        # Reserve the examined footprint even if its additions are rejected.
        # A neighbour must not inherit a dubious strip of shaft just because
        # this candidate retained its original head. Ordinary caps already own
        # their completed surface at this point.
        claimed[self.canonical[proposed]] = input_index
        added = np.setdiff1d(proposed, indices)
        added_clearance = float(_quantile(self.curved_scores(added), self.areas[added], .25))
        after["added_surface_q25_nm"] = added_clearance
        if not np.isfinite(added_clearance):
            raise ValueError("ragged_growth_unmeasured_additions")
        if added_clearance <= cfg.ragged_minimum_added_clearance_nm:
            raise ValueError("ragged_growth_mostly_near_shaft")
        return proposed, after

    def _trim_proposal(self, indices, measured, boundary, model, *, allow_remove):
        cfg = self.config
        geometry, model = self.assess(measured, boundary, measure_surface_spread=True, model=model)
        if geometry["section_surface_spread_q95_nm"] > cfg.trim_maximum_surface_spread_nm:
            raise ValueError("shaft_trim_sections_disagree")
        score = self.score(self.centers[measured], model)
        near = score <= cfg.trim_clearance_nm
        total = self.areas[measured].sum()
        low_area = self.areas[measured[near]].sum()
        if low_area < cfg.trim_minimum_area_nm2 or low_area / total < cfg.trim_minimum_area_fraction:
            return [indices], dict(action="screened_keep", geometry=geometry)
        if (allow_remove and geometry["surface_q95_nm"] <= cfg.sheet_protrusion_nm
                and geometry["boundary_perimeter_normalized"] >= cfg.sheet_boundary_ratio):
            return [], dict(action="remove_surface_patch", geometry=geometry)
        if geometry["surface_q95_nm"] < cfg.trim_minimum_protrusion_nm:
            raise ValueError("shaft_trim_short_protrusion_uncertain")
        self.statistics["trims_attempted"] += 1
        pairs = _edge_pairs(self.parent.faces[measured])
        edge_length = np.maximum(np.linalg.norm(self.centers[measured[pairs[:, 0]]]
                                  - self.centers[measured[pairs[:, 1]]], axis=1), 1e-6)
        graph = coo_matrix((edge_length, (pairs[:, 0], pairs[:, 1])),
                           shape=(len(measured), len(measured))).tocsr()

        def cores(threshold):
            active = score > threshold
            links = pairs[active[pairs].all(axis=1)]
            _, labels = connected_components(coo_matrix((np.ones(len(links)), (links[:, 0], links[:, 1])),
                                               shape=graph.shape).tocsr(), directed=False)
            groups = [np.flatnonzero(active & (labels == label)) for label in np.unique(labels[active])]
            main, small = [], []
            for group in groups:
                if (len(group) >= cfg.minimum_faces and self.areas[measured[group]].sum() >= cfg.trim_minimum_area_nm2
                        and _quantile(score[group], self.areas[measured[group]], .9) >= cfg.trim_minimum_protrusion_nm):
                    main.append(group)
                else:
                    small.append(group)
            if not main:
                raise ValueError("shaft_trim_no_reliable_protrusion_core")
            lost = np.concatenate(small) if small else np.empty(0, dtype=int)
            if self.areas[measured[lost]].sum() > .03 * total:
                raise ValueError("shaft_trim_would_lose_distal_surface")
            return main

        main = cores(cfg.trim_clearance_nm)
        if len(main)==1 and (not cfg.trim_single_protrusions or not allow_remove):
            return [indices], dict(action="screened_keep",geometry=geometry,
                                   reason="shaft_trim_single_neck_uncertain")
        for threshold in (max(0, cfg.trim_clearance_nm - 25), cfg.trim_clearance_nm + 25):
            check = cores(threshold)
            if len(check) != len(main):
                raise ValueError("shaft_trim_unstable_protrusion_count")
            overlap = np.array([[self.areas[measured[np.intersect1d(a,b)]].sum()
                                 / min(self.areas[measured[a]].sum(), self.areas[measured[b]].sum())
                                 for b in check] for a in main])
            if (overlap.max(axis=1) < .85).any() or len(np.unique(overlap.argmax(axis=1))) != len(main):
                raise ValueError("shaft_trim_unstable_protrusion_membership")
        sources = np.concatenate(main)
        owner = np.full(len(measured), -1, dtype=int)
        for part, group in enumerate(main):
            owner[group] = part
        distance, _, nearest = dijkstra(graph, directed=False, indices=sources, min_only=True,
                                        return_predecessors=True, limit=cfg.trim_collar_nm)
        assignment = np.full(len(measured), -1, dtype=int)
        reached = np.isfinite(distance) & (nearest >= 0)
        assignment[reached] = owner[nearest[reached]]
        # A collar that rejoins two cores indicates a shared base or ambiguous
        # local estimate. Keep a possibly branched true spine intact.
        collar_links = pairs[reached[pairs].all(axis=1)]
        if (assignment[collar_links[:, 0]] != assignment[collar_links[:, 1]]).any():
            raise ValueError("shaft_trim_protrusions_share_a_base")
        self.unique_faces()
        canonical = self.canonical[indices]
        output = [indices[np.isin(canonical, measured[assignment == part])] for part in range(len(main))]
        retained = np.concatenate(output)
        removed = np.setdiff1d(indices, retained)
        removed_area = self.areas[removed].sum()
        if removed_area < cfg.trim_minimum_area_nm2 or removed_area / self.areas[indices].sum() < cfg.trim_minimum_area_fraction:
            return [indices], dict(action="screened_keep", geometry=geometry)
        # Everything above the shaft collar belongs to one retained part;
        # otherwise a ragged/fragmented head could silently disappear.
        lost_high = measured[(assignment < 0) & (score > cfg.trim_minimum_protrusion_nm)]
        if self.areas[lost_high].sum() > .01 * total:
            raise ValueError("shaft_trim_would_lose_head_surface")
        part_geometry = [self._validate_trim_part(part, model) for part in output]
        detail = dict(action="split" if len(output)>1 else "trim", geometry=geometry,
                      removed_faces=len(removed), removed_area_fraction=float(removed_area/self.areas[indices].sum()),
                      protrusion_parts=len(output), part_geometry=part_geometry)
        return output, detail

    def _validate_trim_part(self, indices, model):
        """A tall tail cannot make a broad shaft patch a reliable new spine.

        Validate the majority of each child's area against both the section
        uncertainty and its own cut rim. A locally underestimated shaft can
        elevate the whole patch, including its rim, without forming a neck.
        Failure rejects the entire split and preserves the intact candidate.
        """
        boundary, measured = _surface_boundary(self.parent, indices)
        rim, weights, _ = boundary
        geometry, _ = self.assess(measured, boundary, measure_surface_spread=True, model=model)
        median = float(_quantile(self.score(self.centers[measured], model), self.areas[measured], .5))
        rim_median = float(_quantile(self.score(rim, model), weights, .5))
        geometry.update(surface_q50_nm=median, rim_q50_nm=rim_median)
        if median <= geometry["section_surface_spread_q95_nm"] + self.config.trim_clearance_nm:
            raise ValueError("shaft_trim_part_within_section_uncertainty")
        if median - rim_median < self.config.trim_minimum_protrusion_nm:
            raise ValueError("shaft_trim_part_has_no_distinct_protrusion")
        return geometry

    def _mesh_frame_at(self, origin, radius):
        """Local shaft support supplies orientation when stored axes fail.

        Two independent direction estimates must agree. This is a local frame
        for sectioning the existing mesh, not a reconstructed skeleton.
        """
        ids = np.asarray(self.face_tree.query_ball_point(origin, radius), dtype=int)
        self.unique_faces()
        ids = ids[(self.canonical[ids] == ids) & ~self.excluded_support[ids] & (self.areas[ids] > 1e-8)]
        if len(ids) < 80:
            raise ValueError("shaft_trim_insufficient_local_shaft_support")
        weights = self.areas[ids]
        normals = self.parent.face_normals[ids]
        ne, nv = np.linalg.eigh((normals.T * weights) @ normals / weights.sum())
        anchor = np.average(self.centers[ids], axis=0, weights=weights)
        delta = self.centers[ids] - anchor
        pe, pv = np.linalg.eigh((delta.T * weights) @ delta / weights.sum())
        if ne[0] > .28 or ne[1] < .1 or pe[-2] / pe[-1] > .7 or abs(nv[:,0] @ pv[:,-1]) < .85:
            raise ValueError("shaft_trim_ambiguous_mesh_direction")
        return anchor, _frame(nv[:,0])

    def mesh_frame(self, origin):
        # Preserve the previous estimator when the new filter is disabled.
        return self._mesh_frame_at(origin, 2200.0)

    def low_relief_sheet(self, indices):
        """Measure body height relative to a large cut rim at two local scales.

        This does not fit/cap an open shaft section. Shaft-supported normals and
        position covariance independently establish a direction at each scale;
        two scales must agree. A complete spine has a compact neck rim or a
        body clearly above it. Partial caps/ribbons can be removed after an
        unsuccessful completion, without assuming they are complete spines.
        """
        cfg = self.config
        boundary, measured = _surface_boundary(self.parent, indices)
        rim, weights, ratio = boundary
        if ratio < cfg.sheet_minimum_boundary_ratio:
            return dict(action="screened_keep", boundary_perimeter_normalized=ratio)
        self.statistics["sheet_checks"] = self.statistics.get("sheet_checks", 0)+1
        origin = np.average(rim, axis=0, weights=weights)
        frames = []
        for radius in sorted(set(cfg.sheet_mesh_radii_nm)):
            try:
                anchor, basis = self._mesh_frame_at(origin, radius)
            except ValueError:
                continue
            frames.append((radius, anchor, basis))
        pair = next(((a, b) for i, a in enumerate(frames) for b in frames[i+1:]
                     if abs(a[2][:, 2] @ b[2][:, 2]) >= .9), None)
        if pair is None:
            raise ValueError("sheet_no_consistent_local_shaft_directions")
        estimates = []
        for radius, anchor, basis in pair:
            surface = np.linalg.norm((self.centers[measured]-anchor) @ basis[:, :2], axis=1)
            cut = np.linalg.norm((rim-anchor) @ basis[:, :2], axis=1)
            body = float(_quantile(surface, self.areas[measured], .95))
            edge = float(_quantile(cut, weights, .75))
            estimates.append(dict(support_radius_nm=radius, surface_radius_q95_nm=body,
                                  rim_radius_q75_nm=edge, relative_relief_nm=body-edge))
        relief = max(item["relative_relief_nm"] for item in estimates)
        if not np.isfinite(relief):
            raise ValueError("sheet_nonfinite_relative_relief")
        return dict(action="remove" if relief <= cfg.sheet_maximum_relative_relief_nm else "keep",
                    boundary_perimeter_normalized=ratio, maximum_relative_relief_nm=relief,
                    direction_agreement=float(abs(pair[0][2][:, 2] @ pair[1][2][:, 2])),
                    local_estimates=estimates)


def _edge_pairs(faces):
    """All shared-edge neighbours, including locally nonmanifold edges.

    Input faces are unique. A fan with more than two incident triangles is
    connected through its first face; duplicated parent triangles are handled
    by the caller's canonical mapping instead of splitting a physical surface.
    """
    edges = np.sort(np.concatenate((faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]])), axis=1)
    owners = np.tile(np.arange(len(faces)), 3)
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    offsets = np.r_[0,np.cumsum(counts)]
    chunks = [np.column_stack((np.full(counts[i]-1,owners[order[offsets[i]]]), owners[order[offsets[i]+1:offsets[i+1]]]))
              for i in np.flatnonzero(counts>1)]
    return np.concatenate(chunks) if chunks else np.empty((0,2),dtype=int)


def _shaft_contour(item, maximum_bridge_nm):
    """Remove candidate arcs, closing only short gaps in the shaft contour."""
    polygon, _, supported = item
    ids = np.flatnonzero(supported)
    if len(ids) < 6:
        return None
    next_ids = np.roll(ids, -1)
    gaps = (next_ids - ids) % len(polygon) > 1
    if np.linalg.norm(polygon[next_ids[gaps]] - polygon[ids[gaps]],axis=1).max(initial=0) > maximum_bridge_nm:
        return None
    contour = polygon[ids]
    # Closing a missing arc must not cross an existing shaft edge. Such a
    # shortcut would make the signed surface distance depend on a self-crossing
    # polygon rather than a well-defined shaft section.
    start, end = contour, np.roll(contour,-1,axis=0)
    edges = end-start
    for i in np.flatnonzero(gaps):
        direction = edges[i]
        delta = start-start[i]
        denominator = direction[0]*edges[:,1]-direction[1]*edges[:,0]
        valid = abs(denominator)>1e-10
        numerator_t = delta[:,0]*edges[:,1]-delta[:,1]*edges[:,0]
        numerator_s = delta[:,0]*direction[1]-delta[:,1]*direction[0]
        t = np.divide(numerator_t,denominator,out=np.zeros(len(edges)),where=valid)
        s = np.divide(numerator_s,denominator,out=np.zeros(len(edges)),where=valid)
        if (valid & (t>1e-8) & (t<1-1e-8) & (s>=-1e-8) & (s<=1+1e-8)).any():
            return None
    area = abs(np.sum(contour[:,0]*np.roll(contour[:,1],-1)
                      -np.roll(contour[:,0],-1)*contour[:,1])) / 2
    if area < np.pi*30**2:
        return None
    return contour, float(area)


def _build_mesh_axis(parent, excluded):
    """Auxiliary shaft axis in the original mesh coordinates; no export edits.

    Import the existing NEURD backend lazily. Normal branches and default
    configurations never load or run this extra skeletonization.
    """
    from mesh_tools import meshparty_skeletonize

    retained = np.flatnonzero(~excluded)
    if not len(retained):
        raise ValueError("no_shaft_support")
    shaft = parent.submesh([retained], append=True, repair=False)
    obj = meshparty_skeletonize.skeletonize_mesh_largest_component(
        shaft, filter_mesh=False, invalidation_d=12_000, smooth_neighborhood=1)
    return np.asarray(obj.vertices)[np.asarray(obj.edges)]


def _axis_coverage(axis, centers, weights, config):
    distance = axis.project(centers)[1]
    return dict(sampled_faces=len(centers),
                missing_area_fraction=float(weights[distance > config.axis_coverage_distance_nm].sum()
                                            / weights.sum()),
                distance_q95_nm=float(_quantile(distance, weights, 0.95)))


def _retry_failed_with_mesh_axis(parent, context, groups, protection,
                                 refined, records, config, pre_trim, pre_growth):
    """A bounded second pass that cannot replace a successful primary repair."""
    eligible = []
    for record, seed in zip(records, groups):
        outputs = record["output_indices"]
        if (record["action"] in ("unknown", "unknown_keep") and len(outputs) == 1
                and np.array_equal(refined[outputs[0]], seed)
                and len(seed) >= config.minimum_faces):
            eligible.append(record["input_index"])
    detail = dict(status="not_needed", eligible_candidates=len(eligible))
    if not eligible:
        return refined, detail

    excluded = np.zeros(len(parent.faces), dtype=bool)
    for group in groups + protection:
        excluded[group] = True
    support = np.flatnonzero(~excluded & (parent.area_faces > 0))
    if len(support) < 16:
        detail.update(status="skipped", reason="insufficient_shaft_support")
        return refined, detail
    support = support[np.linspace(0, len(support)-1,
                                  min(len(support), config.axis_coverage_sample_faces), dtype=int)]
    centers, weights = parent.triangles_center[support], parent.area_faces[support]
    axis = context.axis if context is not None else None
    before = None if axis is None else _axis_coverage(axis, centers, weights, config)
    detail["coverage_before"] = before
    if before is not None and before["missing_area_fraction"] <= config.axis_coverage_maximum_missing_fraction:
        detail.update(status="covered")
        return refined, detail

    started = perf_counter()
    try:
        skeleton = _build_mesh_axis(parent, excluded)
        detail["build_seconds"] = perf_counter() - started
        auxiliary = _Axis(skeleton, config)
        after = _axis_coverage(auxiliary, centers, weights, config)
        detail.update(axis_segments=len(auxiliary.segments), coverage_after=after)
    except Exception as exc:
        # Optional backend failure must not discard the primary segmentation.
        detail.update(status="build_failed", reason=f"{type(exc).__name__}: {str(exc)[:300]}",
                      build_seconds=perf_counter() - started)
        return refined, detail
    if (after["missing_area_fraction"] > config.axis_coverage_maximum_missing_fraction
            or (before is not None and
                (after["missing_area_fraction"] > 0.5 * before["missing_area_fraction"]
                 or after["distance_q95_nm"] > 0.5 * before["distance_q95_nm"]))):
        detail.update(status="rejected", reason="auxiliary_axis_coverage_did_not_improve")
        return refined, detail

    # All raw seeds (including upstream rejects and removed fragments) and all
    # primary additions remain reserved. Recursive protection subtracts only
    # the retry seeds themselves. Accepted retries then compete with each other.
    # Merge overlapping reservations: the raw mask and its primary result
    # describe the same neighbour, not two conflicting owners. All reserved
    # faces are equivalent competing seeds when only the retry owner matters.
    protected = [np.unique(np.concatenate(groups + protection + refined))]
    retry_trim, retry_growth = {}, {}
    started = perf_counter()
    retry_groups, retry_diagnostics = refine_spine_face_indices(
        parent, skeleton, [groups[i] for i in eligible],
        replace(config, retry_with_mesh_axis=False, require_distal_growth=True,
                maximum_area_ratio=max(config.maximum_area_ratio, config.mesh_axis_maximum_area_ratio),
                maximum_growth_profiles=max(config.maximum_growth_profiles, config.mesh_axis_maximum_growth_profiles)),
        protected_face_groups=protected,
        pre_trim_face_groups=retry_trim, pre_growth_face_groups=retry_growth)
    detail.update(status="retried", retry_seconds=perf_counter()-started,
                  statistics=retry_diagnostics["statistics"], changed_candidates=0)
    replacements = {}
    for j, i in enumerate(eligible):
        retry = retry_diagnostics["candidates"][j]
        parts = [retry_groups[k] for k in retry["output_indices"]]
        geometry = retry.get("geometry_before", {})
        if (not parts and retry["action"] == "remove_surface_patch"
                and geometry.get("section_surface_spread_q95_nm", 0) >
                config.trim_maximum_surface_spread_nm):
            # A fresh axis does not make conflicting shaft contours reliable.
            # Keep small protrusions when the sections may include their head.
            records[i]["mesh_axis_retry"] = dict(action="unknown_keep",
                reason="auxiliary_axis_surface_estimate_uncertain", geometry=geometry)
            continue
        if len(parts) == 1 and np.array_equal(parts[0], groups[i]):
            records[i]["mesh_axis_retry"] = dict(action=retry["action"], reason=retry.get("reason"),
                                                 shaft_trim=retry.get("shaft_trim"))
            continue
        primary = records[i].copy()
        records[i].clear()
        records[i].update(retry, input_index=i, before_mesh_axis_retry=primary,
                          mesh_axis_retry=dict(applied=True))
        replacements[i] = parts
        if pre_trim is not None and j in retry_trim:
            pre_trim[i] = retry_trim[j]
        if pre_growth is not None and j in retry_growth:
            pre_growth[i] = retry_growth[j]
        # Ordinary growth also needs a rollback seed when enabled by this
        # retry. The branch API replays all its final filters on that seed.
        if retry.get("added_faces", 0) and pre_growth is not None:
            pre_growth[i] = groups[i]
            records[i].setdefault("head_growth", dict(action="grow", input_faces=len(groups[i])))
        detail["changed_candidates"] += 1
    ordered = []
    for i, record in enumerate(records):
        parts = replacements[i] if i in replacements else [refined[k] for k in record["output_indices"]]
        outputs = list(range(len(ordered), len(ordered)+len(parts)))
        record.update(output_index=outputs[0] if outputs else None, output_indices=outputs)
        ordered.extend(parts)
    return ordered, detail


def refine_spine_face_indices(parent, skeleton, face_groups, config=None, *, protected_face_groups=(),
                              pre_trim_face_groups=None, pre_growth_face_groups=None):
    """Return ``(face_groups, diagnostics)`` without mutating input geometry.

    Diagnostics give an input/output mapping and explain skipped or rejected
    changes. Surface scraps are removed only when protrusion is weak and the
    cut rim is extensive. Optional low-relief filtering measures body height
    relative to that rim using shaft directions at two scales, without closed
    sections. Short necks and open bases alone are allowed.
    Growth preserves every seed and competes with all neighbouring seeds.
    Optional trim_shaft_connections can subtract shaft surface or split a
    candidate. output_indices maps all its parts; output_index names the first.
    pre_trim_face_groups, when supplied, collects the pre-cut masks so the
    branch API can restore an entire candidate if final filters reject a part.
    protected_face_groups additionally protects candidates rejected by upstream
    filters, without spending time assessing those candidates.
    Optional retry_with_mesh_axis creates a temporary shaft axis only for gross
    coverage failures, and retries unchanged uncertain candidates. All primary
    changed masks and their newly acquired surface remain protected.
    """
    started = perf_counter()
    if pre_trim_face_groups is not None:
        pre_trim_face_groups.clear()
    if pre_growth_face_groups is not None:
        pre_growth_face_groups.clear()
    config = RefinementConfig() if config is None else config
    if isinstance(config, dict):
        config = RefinementConfig(**config)
    groups = [np.unique(np.asarray(ids, dtype=np.int64)) for ids in face_groups]
    for group in groups:
        if len(group) and (group[0] < 0 or group[-1] >= len(parent.faces)):
            raise ValueError("Spine face index outside parent mesh")
    records = [dict(input_index=i, input_faces=len(ids), action="keep") for i, ids in enumerate(groups)]
    diagnostics = dict(config=asdict(config), candidates=records)
    if not groups:
        diagnostics.update(seconds=perf_counter() - started, counts={}, statistics={})
        return [], diagnostics
    selected = np.zeros(len(parent.faces), dtype=bool)
    for group in groups:
        selected[group] = True
    protection = []
    for group in protected_face_groups:
        group = np.unique(np.asarray(group, dtype=np.int64))
        if len(group) and (group[0] < 0 or group[-1] >= len(parent.faces)):
            raise ValueError("Protected face index outside parent mesh")
        remaining = group[~selected[group]]
        if len(remaining):
            protection.append(remaining)
    try:
        context = _Context(parent, skeleton, groups + protection, config)
    except ValueError as exc:
        for record in records:
            record.update(action="unknown", reason=str(exc))
        context = None
    refined, claimed = [], np.full(len(parent.faces), -1, dtype=int)
    order = list(range(len(groups)))
    boundaries = {}
    if context is not None and config.grow_ragged_heads:
        ragged = set()
        for i, indices in enumerate(groups):
            if len(indices) < config.minimum_faces:
                continue
            try:
                boundaries[i] = (_boundary(parent, indices), None)
            except ValueError as exc:
                boundaries[i] = (None, str(exc))
                if str(exc) in ("nonmanifold_candidate", "branched_open_boundary"):
                    ragged.add(i)
        # Ordinary cap repairs keep their established priority and territory.
        # New ragged-head growth can only use the surface left unclaimed by
        # those repairs. Public output order is restored below.
        order.sort(key=lambda i: i in ragged)
    for i in order:
        indices, record = groups[i], records[i]
        result = indices
        if context is not None and len(indices) >= config.minimum_faces:
            try:
                boundary_error = None
                measurement_indices = indices
                try:
                    if i in boundaries:
                        boundary, error = boundaries[i]
                        if error is not None:
                            raise ValueError(error)
                    else:
                        boundary = _boundary(parent, indices)
                except ValueError as exc:
                    boundary_error = str(exc)
                    if (not config.filter_ragged_surface_patches or boundary_error not in
                            ("nonmanifold_candidate", "branched_open_boundary")):
                        raise
                    boundary, measurement_indices = _surface_boundary(parent, indices)
                    record["surface_boundary_status"] = boundary_error
                if not context.suspicious(boundary):
                    if boundary_error is not None:
                        raise ValueError(boundary_error)
                    record["action"] = "screened_keep"
                    context.statistics["screened"] += 1
                else:
                    geometry, model = context.assess(
                        measurement_indices, boundary,
                        measure_surface_spread=boundary_error is not None)
                    record["geometry_before"] = geometry
                    if (geometry["surface_q95_nm"] <= config.sheet_protrusion_nm
                            and geometry["boundary_perimeter_normalized"] >= config.sheet_boundary_ratio):
                        # Some sections include a lateral head in the same
                        # contour as the shaft. A large disagreement with
                        # the other sections makes rejection unreliable.
                        if (boundary_error is not None and
                                geometry["section_surface_spread_q95_nm"] >
                                config.maximum_section_surface_spread_nm):
                            raise ValueError("unstable_shaft_surface_estimate")
                        record["action"] = "remove_surface_patch"
                        result = None
                    elif boundary_error is not None and config.grow_ragged_heads:
                        proposed, after = context.grow_ragged_head(
                            indices, i, boundary, geometry, model, claimed)
                        record.update(action="grow_ragged_head", geometry_proposed=after,
                                      head_growth=dict(action="grow", input_faces=len(indices)))
                        if pre_growth_face_groups is not None:
                            pre_growth_face_groups[i] = indices
                        result = proposed
                        claimed[context.canonical[proposed]] = i
                    elif boundary_error is not None:
                        # A positive protrusion is preserved. An invalid rim
                        # still cannot establish a safe growth boundary.
                        raise ValueError(boundary_error)
                    elif geometry["gap_q10_nm"] > config.incomplete_gap_nm:
                        if context.statistics["repairs_attempted"] >= config.maximum_repairs_per_branch:
                            raise ValueError("branch_repair_budget_exhausted")
                        context.statistics["repairs_attempted"] += 1
                        proposed = context.grow(indices, i, model, claimed)
                        after, _ = context.assess(proposed)
                        record["geometry_proposed"] = after
                        if after["gap_q10_nm"] > config.incomplete_gap_nm:
                            raise ValueError("repair_still_incomplete")
                        if after["gap_q10_nm"] >= geometry["gap_q10_nm"]:
                            raise ValueError("repair_did_not_improve_attachment")
                        if config.require_distal_growth:
                            added = np.setdiff1d(proposed, indices)
                            clearance = float(_quantile(context.curved_scores(added), context.areas[added], .25))
                            after["added_surface_q25_nm"] = clearance
                            if not np.isfinite(clearance) or clearance <= config.ragged_minimum_added_clearance_nm:
                                raise ValueError("cap_growth_mostly_near_shaft")
                        record["action"], result = "grow", proposed
                        claimed[context.canonical[proposed]] = i
            except ValueError as exc:
                record.update(action="unknown_keep", reason=str(exc))
        elif context is not None:
            record["action"] = "small_keep"
        if context is not None and result is not None and config.filter_low_relief_sheets:
            try:
                detail = context.low_relief_sheet(result)
                record["surface_shape"] = detail
                if detail["action"] == "remove":
                    record["action_before_surface_shape"] = record["action"]
                    if "reason" in record:
                        record["reason_before_surface_shape"] = record.pop("reason")
                    record["action"] = "remove_low_relief_sheet"
                    result = None
            except ValueError as exc:
                record["surface_shape"] = dict(action="unknown_keep", reason=str(exc))
        if result is not None:
            parts = [result]
            if context is not None and config.trim_shaft_connections:
                try:
                    parts, trim = context.trim(result)
                    record["shaft_trim"] = trim
                    if trim["action"] in ("trim", "split", "remove_surface_patch"):
                        if pre_trim_face_groups is not None:
                            pre_trim_face_groups[i] = result
                        record["action_before_trim"] = record["action"]
                        if "reason" in record:
                            record["reason_before_trim"] = record.pop("reason")
                        record["action"] = ("remove_surface_patch" if not parts else
                                             "split_shaft_connection" if len(parts)>1 else "trim_shaft_surface")
                except ValueError as exc:
                    record["shaft_trim"] = dict(action="unknown_keep", reason=str(exc))
            output_indices = list(range(len(refined),len(refined)+len(parts)))
            union = np.unique(np.concatenate(parts)) if parts else np.empty(0,dtype=int)
            record.update(output_index=output_indices[0] if parts else None, output_indices=output_indices,
                          output_faces=sum(len(p) for p in parts),
                          added_faces=len(np.setdiff1d(union,indices)), removed_faces=len(np.setdiff1d(indices,union)))
            refined.extend(parts)
        else:
            record.update(output_index=None, output_indices=[], output_faces=0, added_faces=0, removed_faces=len(indices))
    if config.grow_ragged_heads:
        ordered = []
        for record in records:
            parts = [refined[j] for j in record["output_indices"]]
            indices = list(range(len(ordered), len(ordered)+len(parts)))
            record.update(output_index=indices[0] if indices else None, output_indices=indices)
            ordered.extend(parts)
        refined = ordered
    if config.retry_with_mesh_axis:
        retry_started = perf_counter()
        diagnostics["primary_seconds"] = retry_started - started
        refined, diagnostics["auxiliary_axis"] = _retry_failed_with_mesh_axis(
            parent, context, groups, protection, refined, records, config,
            pre_trim_face_groups, pre_growth_face_groups)
        diagnostics["auxiliary_axis"]["seconds"] = perf_counter() - retry_started
    counts = {}
    for record in records:
        counts[record["action"]] = counts.get(record["action"], 0) + 1
    diagnostics.update(seconds=perf_counter() - started, counts=counts,
                       statistics={} if context is None else context.statistics)
    return refined, diagnostics
