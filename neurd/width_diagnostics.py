"""Bounded support observations for failed H01 width eligibility checks.

These samples never establish eligibility. The caller retains the exact
``calculate_new_width`` result separately; this module diagnoses at most 16
uniformly selected resized edges using the owned distance kernel.
"""

import numpy as np


def sample_width_support(mesh, skeleton, *, max_edges=16, max_faces=2_000_000):
    """Report slice support and unfilled diagnostic distances on sampled edges.

    The active calculation may use the legacy backend. Distances here come from
    a separate owned-kernel rerun with its same width parameters; neither the
    samples nor their positive count describe all original width slices.
    """
    if max_edges < 1 or max_faces < 1:
        raise ValueError('Diagnostic budgets must be positive')
    record = dict(schema_version=1, kind='uniform_width_support_sampler',
                  diagnostic_only=True, units='nm', max_sampled_edges=max_edges,
                  maximum_parent_faces=max_faces, parent_faces=len(mesh.faces),
                  sampling='uniform_resized_edge_indices',
                  raw_distances_source='separate_owned_kernel_rerun',
                  active_raw_distances_observed=False,
                  parameters=dict(skeleton_segment_size_nm=1000, buffer_nm=0.01,
                                  distance_threshold_nm=3000, distance_by_mesh_center=True,
                                  connectivity='edges'))
    if len(mesh.faces) > max_faces:
        record.update(status='not_sampled', reason='diagnostic_face_budget_exceeded')
        return record
    from neurd import width_utils, skeletal_distance_ops as sdo

    edges = np.asarray(width_utils.skeleton_resized_ordered(
        skeleton, skeleton_segment_size=1000), dtype=float)
    if edges.ndim != 3 or edges.shape[1:] != (2, 3) or not np.isfinite(edges).all():
        raise ValueError('Invalid resized width axis')
    count = len(edges)
    indices = np.unique(np.linspace(0, count - 1, min(max_edges, count), dtype=int))
    record.update(total_resized_edges=count, sampled_edges=len(indices),
                  full_edge_coverage=len(indices) == count)
    centers = np.asarray(mesh.triangles_center)
    observations, supported_edges, supported_positions = [], [], []
    for index in indices:
        edge = edges[index]
        vector = edge[1] - edge[0]
        observation = dict(resized_edge_index=int(index), edge_nm=edge.tolist())
        observations.append(observation)
        if np.sum(np.abs(vector)) < 0.001:
            observation.update(status='degenerate_edge_skipped', raw_distance_nm=None)
            continue
        basis = sdo.sk.change_basis_matrix(vector)
        transformed = basis @ centers.T
        endpoints = basis @ edge.T
        limits = np.sort(endpoints[2]) + np.asarray([-0.01, 0.01])
        axial = (transformed[2] > limits[0]) & (transformed[2] < limits[1])
        midpoint = np.mean(endpoints.T, axis=0)
        near = np.linalg.norm(transformed.T[:, :2] - midpoint[:2], axis=1) < 3000
        supported = int(np.count_nonzero(axial & near))
        observation.update(axial_candidate_faces=int(np.count_nonzero(axial)),
                           within_distance_candidate_faces=supported)
        if supported:
            observation.update(status='supported_slice_distance_pending', raw_distance_nm=None)
            supported_positions.append(len(observations) - 1)
            supported_edges.append(edge)
        else:
            observation.update(status='empty_slice', raw_distance_nm=0.0)
    # Empty sampled slices have exactly zero distance. Avoid constructing a
    # full face-adjacency graph when none of these sampled slices has support.
    if supported_edges:
        try:
            distances, _, _, _ = sdo.skeletal_distance(
                mesh, np.asarray(supported_edges), buffer=0.01,
                distance_threshold=3000, distance_by_mesh_center=True,
                connectivity='edges', keep_empty_placeholder=True)
            if len(distances) != len(supported_positions):
                raise ValueError('Diagnostic distance/edge alignment differs')
            for position, value in zip(supported_positions, distances):
                value = float(value)
                observations[position].update(
                    status=('nonfinite_distance' if not np.isfinite(value) else
                            'positive_distance' if value > 0 else 'nonpositive_distance'),
                    raw_distance_nm=value if np.isfinite(value) else None)
        except Exception as exc:
            record['raw_distance_error'] = f'{type(exc).__name__}: {str(exc)[:300]}'
    record.update(status='partial' if 'raw_distance_error' in record else 'sampled', slices=observations,
                  sampled_supported_slice_count=len(supported_edges),
                  sampled_empty_slice_count=sum(o['status'] == 'empty_slice' for o in observations),
                  sampled_positive_raw_distance_count=sum(o['status'] == 'positive_distance' for o in observations),
                  sampled_nonfinite_raw_distance_count=sum(o['status'] == 'nonfinite_distance' for o in observations))
    return record
