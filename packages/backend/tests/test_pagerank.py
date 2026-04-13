# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""Comprehensive tests for the PyTorch PageRank implementation.

Covers edge cases, dtype boundaries, convergence behavior, and API contracts.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from embedding_atlas.pagerank import (
    compute_pagerank_column,
    knn_to_edges,
    pagerank,
)


class TestEdgeCases:
    def test_empty_edges(self):
        """Empty edge list with n=0 should return an empty array."""
        scores = pagerank([], n=0)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 0

    def test_empty_edges_with_n(self):
        """Empty edge list with explicit n should return uniform scores."""
        scores = pagerank([], n=3)
        assert len(scores) == 3
        assert np.isclose(scores.sum(), 1.0)
        assert np.allclose(scores, 1 / 3)

    def test_single_node_self_loop(self):
        """A single node with a self-loop should get all the rank."""
        scores = pagerank([(0, 0, 1.0)], n=1)
        assert len(scores) == 1
        assert np.isclose(scores[0], 1.0)

    def test_single_edge(self):
        """Two nodes, one directed edge: target gets more rank."""
        scores = pagerank([(0, 1, 1.0)], n=2)
        assert len(scores) == 2
        assert np.isclose(scores.sum(), 1.0)
        # Node 1 receives a link from node 0; node 0 is a dangling node
        # whose rank gets redistributed. Node 1 should have higher rank.
        assert scores[1] > scores[0]

    def test_fully_disconnected_graph(self):
        """Two nodes with self-loops only: equal rank (no cross-edges)."""
        scores = pagerank([(0, 0, 1.0), (1, 1, 1.0)], n=2)
        assert len(scores) == 2
        assert np.isclose(scores.sum(), 1.0)
        assert np.allclose(scores[0], scores[1])

    def test_self_loops_do_not_break_convergence(self):
        """Graph with self-loops mixed with real edges should converge."""
        edges = [
            (0, 0, 1.0),  # self-loop
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 0, 1.0),
        ]
        scores = pagerank(edges, n=3)
        assert len(scores) == 3
        assert np.isclose(scores.sum(), 1.0)
        assert all(s > 0 for s in scores)

    def test_dangling_node(self):
        """A node with no outgoing edges (dangling) should still get rank."""
        # 0 -> 1, 0 -> 2; nodes 1 and 2 are dangling
        edges = [(0, 1, 1.0), (0, 2, 1.0)]
        scores = pagerank(edges, n=3)
        assert len(scores) == 3
        assert np.isclose(scores.sum(), 1.0)
        # Dangling nodes redistribute their rank uniformly,
        # so all nodes should have positive scores.
        assert all(s > 0 for s in scores)

    def test_gap_in_node_ids(self):
        """Nodes with gaps in IDs: phantom nodes in the gap are dangling."""
        # Nodes 0 and 5 — IDs 1-4 don't appear in edges but exist in the array
        edges = [(0, 5, 1.0), (5, 0, 1.0)]
        scores = pagerank(edges, n=6)
        assert len(scores) == 6
        assert np.isclose(scores.sum(), 1.0)
        # Nodes 0 and 5 should have more rank than phantom nodes 1-4
        assert scores[0] > scores[1]
        assert scores[5] > scores[1]

    def test_trailing_nodes_with_n(self):
        """Explicit n includes trailing nodes that have no edges."""
        edges = [(0, 1, 1.0), (1, 0, 1.0)]
        scores = pagerank(edges, n=5)
        assert len(scores) == 5
        assert np.isclose(scores.sum(), 1.0)
        # Trailing nodes 2-4 are dangling and get some rank
        assert all(s > 0 for s in scores)

    def test_n_smaller_than_edges_raises(self):
        """If n is too small for the edge node IDs, raise ValueError."""
        edges = [(0, 5, 1.0), (5, 0, 1.0)]
        with pytest.raises(ValueError, match="n=3 but edges contain node ID 5"):
            pagerank(edges, n=3)

    def test_duplicate_edges_are_summed(self):
        """Duplicate edges should be coalesced (weights summed) by sparse tensor."""
        edges_single = [(0, 1, 2.0), (1, 0, 1.0)]
        edges_dup = [(0, 1, 1.0), (0, 1, 1.0), (1, 0, 1.0)]
        scores_single = pagerank(edges_single, n=2)
        scores_dup = pagerank(edges_dup, n=2)
        assert np.allclose(scores_single, scores_dup, atol=1e-6)

    def test_unweighted_edges(self):
        """Unweighted (2-tuple) edges should work."""
        edges = [(0, 1), (1, 2), (2, 0)]
        scores = pagerank(edges, n=3)
        assert len(scores) == 3
        assert np.isclose(scores.sum(), 1.0)
        # Cycle => all equal
        assert np.allclose(scores, scores[0])


class TestDtypeBoundaries:
    def test_integer_edge_weights(self):
        """Integer weights should be accepted and produce valid results."""
        edges = [(0, 1, 2), (1, 2, 3), (2, 0, 1)]
        scores = pagerank(edges, n=3)
        assert len(scores) == 3
        assert np.isclose(scores.sum(), 1.0)

    def test_very_small_weights(self):
        """Very small weights should not cause division-by-zero."""
        edges = [(0, 1, 1e-30), (1, 0, 1e-30)]
        scores = pagerank(edges, n=2)
        assert len(scores) == 2
        assert np.isclose(scores.sum(), 1.0)
        assert not np.any(np.isnan(scores))
        assert not np.any(np.isinf(scores))

    def test_very_large_weights(self):
        """Very large weights should not overflow."""
        edges = [(0, 1, 1e15), (1, 0, 1e15)]
        scores = pagerank(edges, n=2)
        assert len(scores) == 2
        assert np.isclose(scores.sum(), 1.0)
        assert not np.any(np.isnan(scores))

    def test_mixed_weight_magnitudes(self):
        """Edges with vastly different weight magnitudes.

        In a 2-node graph each node has one outgoing edge, so after
        column-normalization both transition probabilities become 1.0
        regardless of the raw weight. Need 3+ nodes to see weight effects.
        """
        # Node 0 splits outgoing: heavy to 1, light to 2
        # Node 1 -> 0, Node 2 -> 0
        edges = [
            (0, 1, 1e10),
            (0, 2, 1e-10),
            (1, 0, 1.0),
            (2, 0, 1.0),
        ]
        scores = pagerank(edges, n=3)
        assert len(scores) == 3
        assert np.isclose(scores.sum(), 1.0)
        # Node 1 should get much more rank than node 2
        assert scores[1] > scores[2]

    def test_output_dtype_is_float64(self):
        """Output should be float64 (from PyTorch float64 computation)."""
        edges = [(0, 1, 1.0), (1, 0, 1.0)]
        scores = pagerank(edges, n=2)
        assert scores.dtype == np.float64


class TestConvergence:
    def test_damping_zero(self):
        """damping=0 means pure teleportation: uniform distribution."""
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]
        scores = pagerank(edges, n=3, damping=0.0)
        assert np.isclose(scores.sum(), 1.0)
        expected = np.array([1 / 3, 1 / 3, 1 / 3])
        assert np.allclose(scores, expected, atol=1e-6)

    def test_damping_one(self):
        """damping=1.0 means no teleportation: pure link-following."""
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]
        scores = pagerank(edges, n=3, damping=1.0)
        assert np.isclose(scores.sum(), 1.0)
        # Cycle with damping=1 should still be uniform
        assert np.allclose(scores, 1 / 3, atol=1e-6)

    def test_damping_one_asymmetric(self):
        """damping=1.0 on asymmetric graph follows link structure only."""
        # 0 -> 2, 1 -> 2: two sources feeding into node 2
        edges = [(0, 2, 1.0), (1, 2, 1.0)]
        scores = pagerank(edges, n=3, damping=1.0)
        assert np.isclose(scores.sum(), 1.0)
        # (nodes 0 and 1 are dangling, their rank gets redistributed uniformly,
        # then flows to 2 again). In steady state: node 2 gets 60% of rank.
        assert scores[2] > scores[0]
        assert scores[2] > scores[1]

    def test_max_iterations_respected(self):
        """With max_iterations=1, result should not fully converge on asymmetric graph."""
        # Asymmetric graph: 0->1, 0->2, 1->0 (node 2 is dangling)
        edges = [(0, 1, 1.0), (0, 2, 1.0), (1, 0, 1.0)]
        scores_1 = pagerank(edges, n=3, max_iterations=1, tolerance=0.0)
        scores_100 = pagerank(edges, n=3, max_iterations=100)
        assert np.isclose(scores_1.sum(), 1.0)
        assert np.isclose(scores_100.sum(), 1.0)
        # 1 iteration should not match the converged result
        assert not np.allclose(scores_1, scores_100, atol=1e-6)

    def test_tight_tolerance_converges(self):
        """Very tight tolerance should still produce valid results."""
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]
        scores = pagerank(edges, n=3, tolerance=1e-15, max_iterations=1000)
        assert np.isclose(scores.sum(), 1.0)

    def test_large_chain_graph(self):
        """A long chain requires many iterations; verify it converges."""
        n = 100
        edges = [(i, i + 1, 1.0) for i in range(n - 1)]
        edges.append((n - 1, 0, 1.0))  # close the cycle
        scores = pagerank(edges, n=n, max_iterations=500)
        assert len(scores) == n
        assert np.isclose(scores.sum(), 1.0)
        # Cycle => uniform
        assert np.allclose(scores, 1 / n, atol=1e-4)

    def test_bipartite_structure(self):
        """Bipartite graph: group A links to group B, B links back to A."""
        edges = [
            (0, 2, 1.0),
            (0, 3, 1.0),
            (1, 2, 1.0),
            (1, 3, 1.0),
            (2, 0, 1.0),
            (2, 1, 1.0),
            (3, 0, 1.0),
            (3, 1, 1.0),
        ]
        scores = pagerank(edges, n=4)
        assert np.isclose(scores.sum(), 1.0)
        # All nodes are symmetric => uniform
        assert np.allclose(scores, 0.25, atol=1e-6)


class TestAPIContract:
    def test_negative_weight_accepted(self):
        """Negative weights are not rejected (no validation), but result should
        still be a numpy array. This documents current behavior."""
        edges = [(0, 1, -1.0), (1, 0, 1.0)]
        scores = pagerank(edges, n=2)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 2

    def test_zero_weight_edges(self):
        """Zero-weight edges should not cause errors."""
        edges = [(0, 1, 0.0), (1, 0, 1.0)]
        scores = pagerank(edges, n=2)
        assert len(scores) == 2
        assert np.isclose(scores.sum(), 1.0)

    def test_returns_numpy_array(self):
        """Return type must be np.ndarray, not a torch tensor."""
        edges = [(0, 1, 1.0), (1, 0, 1.0)]
        scores = pagerank(edges, n=2)
        assert isinstance(scores, np.ndarray)
        assert not isinstance(scores, torch.Tensor)

    def test_single_node_no_self_loop(self):
        """A dangling target node (only incoming edges) should still get rank."""
        edges = [(0, 1, 1.0)]
        scores = pagerank(edges, n=2)
        # Node 1 is dangling, but should have rank
        assert len(scores) == 2
        assert np.isclose(scores.sum(), 1.0)

    def test_large_node_ids(self):
        """Large node IDs should work (array is sized max_id + 1)."""
        edges = [(0, 999, 1.0), (999, 0, 1.0)]
        scores = pagerank(edges, n=1000)
        assert len(scores) == 1000
        assert np.isclose(scores.sum(), 1.0)
        assert scores[0] > 0
        assert scores[999] > 0


class TestAnalyticalResults:
    def test_two_node_cycle(self):
        """Two nodes in a cycle: equal rank of 0.5 each."""
        edges = [(0, 1, 1.0), (1, 0, 1.0)]
        scores = pagerank(edges, n=2)
        assert np.allclose(scores, [0.5, 0.5])

    def test_three_node_cycle(self):
        """Three nodes in a cycle: equal rank of 1/3 each."""
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]
        scores = pagerank(edges, n=3)
        assert np.allclose(scores, [1 / 3, 1 / 3, 1 / 3])

    def test_star_incoming(self):
        """All nodes point to hub — hub gets most rank."""
        # 1->0, 2->0, 3->0, 4->0
        edges = [(i, 0, 1.0) for i in range(1, 5)]
        scores = pagerank(edges, n=5)
        assert np.isclose(scores.sum(), 1.0)
        assert scores[0] == max(scores)

    def test_star_outgoing(self):
        """Hub points to all — leaves get more rank than hub."""
        # 0->1, 0->2, 0->3, 0->4
        edges = [(0, i, 1.0) for i in range(1, 5)]
        scores = pagerank(edges, n=5)
        assert np.isclose(scores.sum(), 1.0)
        # All leaves are equivalent
        leaf_scores = scores[1:]
        assert np.allclose(leaf_scores, leaf_scores[0], atol=1e-6)
        # Leaves get more rank than hub (hub distributes all its rank out)
        assert scores[1] > scores[0]

    def test_weight_sensitivity(self):
        """Higher weight edge should direct more rank to its target."""
        edges_equal = [(0, 1, 1.0), (0, 2, 1.0), (1, 0, 1.0), (2, 0, 1.0)]
        edges_biased = [(0, 1, 10.0), (0, 2, 1.0), (1, 0, 1.0), (2, 0, 1.0)]

        scores_equal = pagerank(edges_equal, n=3)
        scores_biased = pagerank(edges_biased, n=3)

        # With equal weights, nodes 1 and 2 should have equal rank
        assert np.isclose(scores_equal[1], scores_equal[2], atol=1e-6)
        # With biased weights, node 1 should have more rank than node 2
        assert scores_biased[1] > scores_biased[2]


class TestKnnToEdges:
    def test_basic_conversion(self):
        """Should produce correct number of edges."""
        indices = np.array([[1, 2], [0, 2], [0, 1]])
        distances = np.array([[0.1, 0.2], [0.1, 0.3], [0.2, 0.3]])
        edges = knn_to_edges(indices, distances)
        assert len(edges) == 6  # 3 nodes × 2 neighbors

    def test_skips_self_loops(self):
        """Should skip edges where source == target."""
        indices = np.array([[0, 1], [1, 0]])  # Node 0 points to itself
        distances = np.array([[0.0, 0.5], [0.0, 0.5]])
        edges = knn_to_edges(indices, distances)
        # Should have 2 edges (0->1 and 1->0), not 4
        assert len(edges) == 2
        assert all(src != tgt for src, tgt, _ in edges)

    def test_edge_weights_are_positive(self):
        """UMAP-style weights should be positive."""
        indices = np.array([[1, 2], [0, 2], [0, 1]])
        distances = np.array([[0.1, 0.2], [0.1, 0.3], [0.2, 0.3]])
        edges = knn_to_edges(indices, distances)
        assert all(w > 0 for _, _, w in edges)


class TestComputePagerankColumn:
    def test_compute_pagerank_column(self):
        """Should add a 'pagerank' column to the DataFrame."""
        indices = np.array([[1, 2], [0, 2], [0, 1]])
        distances = np.array([[0.1, 0.2], [0.1, 0.3], [0.2, 0.3]])
        df = pd.DataFrame(
            {
                "__neighbors": [
                    {"ids": indices[i], "distances": distances[i]}
                    for i in range(len(indices))
                ]
            }
        )

        df["pagerank"] = compute_pagerank_column(df)

        assert len(df["pagerank"]) == 3
        assert np.isclose(df["pagerank"].sum(), 1.0)
        assert all(s > 0 for s in df["pagerank"])


class TestIntegration:
    def test_knn_to_pagerank_pipeline(self):
        """Full pipeline: KNN arrays -> edges -> PageRank scores."""
        # Simple 3-node graph
        indices = np.array([[1, 2], [0, 2], [0, 1]])
        distances = np.array([[0.1, 0.2], [0.1, 0.3], [0.2, 0.3]])

        edges = knn_to_edges(indices, distances)
        scores = pagerank(edges, n=3)

        assert len(scores) == 3
        assert np.isclose(scores.sum(), 1.0)
        assert all(s > 0 for s in scores)


class TestUmapWeightCompatibility:
    """Verify that our edge weight computation matches UMAP's implementation."""

    def test_edge_weights_match_umap(self):
        """Our edge weights should match UMAP's compute_membership_strengths."""
        from umap.umap_ import compute_membership_strengths, smooth_knn_dist

        # Generate random KNN data
        np.random.seed(123)
        n_samples = 50
        n_neighbors = 10

        # Random neighbor indices (ensuring no self-loops for simplicity)
        knn_indices = np.zeros((n_samples, n_neighbors), dtype=np.int32)
        for i in range(n_samples):
            candidates = [j for j in range(n_samples) if j != i]
            knn_indices[i] = np.random.choice(candidates, n_neighbors, replace=False)

        # Random distances (sorted) - use float32 for UMAP compatibility
        knn_distances = np.sort(
            np.random.rand(n_samples, n_neighbors).astype(np.float32), axis=1
        )

        # Compute sigmas and rhos using UMAP
        umap_sigmas, umap_rhos = smooth_knn_dist(
            knn_distances,
            k=n_neighbors,
            local_connectivity=1.0,
            bandwidth=1.0,
        )
        # Ensure float32 for numba compatibility
        umap_sigmas = umap_sigmas.astype(np.float32)
        umap_rhos = umap_rhos.astype(np.float32)

        # Compute membership strengths using UMAP
        result = compute_membership_strengths(
            knn_indices, knn_distances, umap_sigmas, umap_rhos, return_dists=False
        )
        rows, cols, vals = result[0], result[1], result[2]

        # Build a dict of UMAP weights: (source, target) -> weight
        umap_weights = {}
        for r, c, v in zip(rows, cols, vals):
            umap_weights[(int(r), int(c))] = float(v)

        # Compute using our implementation
        our_edges = knn_to_edges(knn_indices, knn_distances)
        our_weights = {(src, tgt): w for src, tgt, w in our_edges}

        # Compare weights for each edge
        mismatches = []
        for (src, tgt), our_w in our_weights.items():
            umap_w = umap_weights.get((src, tgt))
            if umap_w is None:
                mismatches.append(f"Edge ({src}, {tgt}) not in UMAP output")
            elif not np.isclose(our_w, umap_w, rtol=1e-7):
                mismatches.append(
                    f"Edge ({src}, {tgt}): ours={our_w:.8f}, umap={umap_w:.8f}"
                )

        assert len(mismatches) == 0, "Weight mismatches:\n" + "\n".join(mismatches[:10])
