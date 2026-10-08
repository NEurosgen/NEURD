"""Experimental, conservative refinement of a CGAL spine partition.

All lengths are nanometres. Only original parent faces may be added; no mesh
repair, smoothing or synthetic caps are performed. A shaft-surface estimate is
a geometric proxy, not anatomical ground truth. Uncertain repairs retain their
original seed. This module uses numpy, scipy, trimesh and matplotlib.path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import trimesh
from matplotlib.path import Path as PolygonPath
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
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


def _section_polygons(vertices, faces, anchor, basis, offset):
    # Triangle/plane intersections follow the diagnostic attachment estimator.
    # Duplicate parent triangles are ignored locally without editing the input.
    d = (vertices - anchor) @ basis[:, 2] - offset - 1e-5
    dt = d[faces]
    faces = faces[(dt.min(axis=1) < 0) & (dt.max(axis=1) > 0)]
    if not len(faces):
        return []
    edges = np.stack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=1)
    cross = (d[edges[:, :, 0]] > 0) != (d[edges[:, :, 1]] > 0)
    crossed = np.sort(edges[cross], axis=1)
    unique, inverse = np.unique(crossed, axis=0, return_inverse=True)
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
            polygons.append((polygon, float(area)))
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
        self.profiles = {}
        self.growth_scores = self.growth_scored = None
        self.statistics = dict(section_models=0, screened=0, assessed=0, repairs_attempted=0,
                               growth_faces_scored=0)
        self.max_face_radius = float(np.max(np.linalg.norm(parent.triangles - self.centers[:, None], axis=2)))

    def unique_faces(self):
        if self.canonical is None:
            _, representatives, inverse = np.unique(np.sort(self.parent.faces, axis=1), axis=0,
                                                     return_index=True, return_inverse=True)
            self.representatives = np.sort(representatives)
            self.canonical = representatives[inverse]
        return self.representatives

    def model(self, anchor):
        basis = _frame(self.axis.tangent(anchor))
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
        faces = faces[(radial[faces] < self.config.section_radius_nm).any(axis=1)]
        polygons, areas = [], []
        for offset in self.config.section_offsets_nm:
            candidates = _section_polygons(vertices, faces, anchor, basis, offset)
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

    def assess(self, indices, boundary=None, *, measure_surface_spread=False):
        rim, weights, ratio = _boundary(self.parent, indices) if boundary is None else boundary
        anchor = self.axis.project([np.average(rim, axis=0, weights=weights)])[0][0]
        model = self.model(anchor)
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


def refine_spine_face_indices(parent, skeleton, face_groups, config=None, *, protected_face_groups=()):
    """Return ``(face_groups, diagnostics)`` without mutating input geometry.

    Diagnostics give an input/output mapping and explain skipped or rejected
    changes. Surface scraps are removed only when BOTH protrusion is weak and
    the cut rim is extensive. Short necks and open bases alone are allowed.
    Growth preserves every seed and competes with all neighbouring seeds.
    protected_face_groups additionally protects candidates rejected by upstream
    filters, without spending time assessing those candidates.
    """
    started = perf_counter()
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
    for i, (indices, record) in enumerate(zip(groups, records)):
        result = indices
        if context is not None and len(indices) >= config.minimum_faces:
            try:
                boundary_error = None
                measurement_indices = indices
                try:
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
                        record["action"], result = "grow", proposed
                        claimed[context.canonical[proposed]] = i
            except ValueError as exc:
                record.update(action="unknown_keep", reason=str(exc))
        elif context is not None:
            record["action"] = "small_keep"
        if result is not None:
            record.update(output_index=len(refined), output_faces=len(result), added_faces=len(result) - len(indices))
            refined.append(result)
        else:
            record.update(output_index=None, output_faces=0, added_faces=0)
    counts = {}
    for record in records:
        counts[record["action"]] = counts.get(record["action"], 0) + 1
    diagnostics.update(seconds=perf_counter() - started, counts=counts,
                       statistics={} if context is None else context.statistics)
    return refined, diagnostics
