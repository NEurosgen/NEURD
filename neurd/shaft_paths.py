"""Ordered bidirectional BFS for a serial batch of immutable shaft-graph queries.

The spine detector lazily imports this optional Numba helper only for sufficiently
large batches under the verified NetworkX 3.6.1 unweighted tie policy. CSR rows
preserve Graph.adj insertion order and the compiled search preserves full paths.
Other backends and small batches retain NetworkX in the caller.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from numba import njit


@njit(cache=True)
def _ordered_bidirectional_path(
    indptr, neighbors, source, target, pred, succ, forward_seen, reverse_seen,
    forward_queue, reverse_queue, epoch,
):
    if source == target:
        result = np.empty(1, dtype=np.int64)
        result[0] = source
        return result

    # Each queue stores successive levels contiguously. The [start:end] slice
    # is precisely NetworkX's current fringe; appends form the next fringe.
    forward_seen[source] = epoch
    reverse_seen[target] = epoch
    pred[source] = -1
    succ[target] = -1
    forward_queue[0] = source
    reverse_queue[0] = target
    forward_start = 0
    forward_end = 1
    forward_count = 1
    reverse_start = 0
    reverse_end = 1
    reverse_count = 1
    meeting = -1

    while forward_start < forward_end and reverse_start < reverse_end:
        if forward_end - forward_start <= reverse_end - reverse_start:
            for queue_idx in range(forward_start, forward_end):
                vertex = forward_queue[queue_idx]
                for adj_idx in range(indptr[vertex], indptr[vertex + 1]):
                    neighbor = neighbors[adj_idx]
                    if forward_seen[neighbor] != epoch:
                        forward_seen[neighbor] = epoch
                        pred[neighbor] = vertex
                        forward_queue[forward_count] = neighbor
                        forward_count += 1
                    if reverse_seen[neighbor] == epoch:
                        meeting = neighbor
                        break
                if meeting != -1:
                    break
            forward_start = forward_end
            forward_end = forward_count
        else:
            for queue_idx in range(reverse_start, reverse_end):
                vertex = reverse_queue[queue_idx]
                for adj_idx in range(indptr[vertex], indptr[vertex + 1]):
                    neighbor = neighbors[adj_idx]
                    if reverse_seen[neighbor] != epoch:
                        reverse_seen[neighbor] = epoch
                        succ[neighbor] = vertex
                        reverse_queue[reverse_count] = neighbor
                        reverse_count += 1
                    if forward_seen[neighbor] == epoch:
                        meeting = neighbor
                        break
                if meeting != -1:
                    break
            reverse_start = reverse_end
            reverse_end = reverse_count
        if meeting != -1:
            break

    if meeting == -1:
        return np.empty(0, dtype=np.int64)

    left_count = 0
    vertex = meeting
    while vertex != -1:
        left_count += 1
        vertex = pred[vertex]
    right_count = 0
    vertex = succ[meeting]
    while vertex != -1:
        right_count += 1
        vertex = succ[vertex]

    result = np.empty(left_count + right_count, dtype=np.int64)
    vertex = meeting
    for idx in range(left_count - 1, -1, -1):
        result[idx] = vertex
        vertex = pred[vertex]
    vertex = succ[meeting]
    for idx in range(left_count, left_count + right_count):
        result[idx] = vertex
        vertex = succ[vertex]
    return result


class ExactBidirectionalPaths:
    """Snapshot CSR and serial scratch space preserving ordered unweighted paths."""

    def __init__(self, graph):
        if not isinstance(graph, nx.Graph):
            raise TypeError("Expected a NetworkX Graph")
        if graph.is_directed() or graph.is_multigraph():
            raise TypeError("Only undirected simple Graphs are supported")
        self.nodes = tuple(graph.nodes)
        self.node_to_id = {node: idx for idx, node in enumerate(self.nodes)}
        self.indptr = np.empty(len(self.nodes) + 1, dtype=np.int64)
        self.indptr[0] = 0
        neighbor_ids = []
        for idx, node in enumerate(self.nodes):
            neighbor_ids.extend(self.node_to_id[neighbor] for neighbor in graph.adj[node])
            self.indptr[idx + 1] = len(neighbor_ids)
        self.neighbors = np.asarray(neighbor_ids, dtype=np.int64)
        self.indptr.flags.writeable = False
        self.neighbors.flags.writeable = False
        count = len(self.nodes)
        self._pred = np.empty(count, dtype=np.int64)
        self._succ = np.empty(count, dtype=np.int64)
        self._forward_seen = np.zeros(count, dtype=np.int64)
        self._reverse_seen = np.zeros(count, dtype=np.int64)
        self._forward_queue = np.empty(count, dtype=np.int64)
        self._reverse_queue = np.empty(count, dtype=np.int64)
        self._epoch = 0

    def _lookup(self, node, role):
        try:
            return self.node_to_id[node]
        except (KeyError, TypeError):
            raise nx.NodeNotFound(f"{role} {node} is not in G") from None

    def shortest_path(self, source, target, *, weight=None):
        if weight is not None:
            raise TypeError("Only weight=None is supported")
        source_id = self._lookup(source, "Source")
        target_id = self._lookup(target, "Target")
        if self._epoch == np.iinfo(np.int64).max:
            self._forward_seen.fill(0)
            self._reverse_seen.fill(0)
            self._epoch = 0
        self._epoch += 1
        result = _ordered_bidirectional_path(
            self.indptr, self.neighbors, source_id, target_id, self._pred, self._succ,
            self._forward_seen, self._reverse_seen, self._forward_queue,
            self._reverse_queue, self._epoch,
        )
        if len(result) == 0:
            raise nx.NetworkXNoPath(f"No path between {source} and {target}.")
        return [self.nodes[int(idx)] for idx in result]

    @property
    def owned_array_bytes(self):
        return sum(array.nbytes for array in (
            self.indptr, self.neighbors, self._pred, self._succ,
            self._forward_seen, self._reverse_seen, self._forward_queue,
            self._reverse_queue,
        ))
