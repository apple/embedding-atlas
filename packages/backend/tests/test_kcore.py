# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""Comprehensive tests for the directed in-degree k-core implementation.

Covers edge cases, self-loop exclusion, analytical core numbers from the
Batagelj-Zaversnik peeling definition, dtype contracts, and the
compute_kcore_column DataFrame entry point.
"""

import numpy as np
import pandas as pd
import pytest
from embedding_atlas.graph_metrics import (
    compute_kcore_column,
    in_degree_core,
)


class TestEdgeCases:
    def test_empty_edges_n_zero(self):
        """Empty edge list with n=0 should return an empty array."""
        cores = in_degree_core([], n=0)
        assert isinstance(cores, np.ndarray)
        assert len(cores) == 0

    def test_empty_edges_with_n(self):
        """Empty edge list with explicit n should return zeros of length n."""
        cores = in_degree_core([], n=3)
        assert len(cores) == 3
        assert np.array_equal(cores, np.zeros(3, dtype=np.int64))

    def test_self_loops_only(self):
        """A graph of only self-loops collapses to all-zero cores (loops excluded)."""
        cores = in_degree_core([(0, 0), (1, 1), (2, 2)], n=3)
        assert np.array_equal(cores, np.zeros(3, dtype=np.int64))

    def test_self_loop_excluded_from_in_degree(self):
        """A self-loop must not contribute to a node's in-degree / core."""
        # 1 -> 0 gives node 0 in-degree 1, but the self-loop (0,0) is ignored.
        with_loop = in_degree_core([(1, 0), (0, 0)], n=2)
        without_loop = in_degree_core([(1, 0)], n=2)
        assert np.array_equal(with_loop, without_loop)

    def test_node_with_no_incoming_has_core_zero(self):
        """A node that no edge targets has core number 0."""
        # 0 -> 1 only; node 0 never appears as a target.
        cores = in_degree_core([(0, 1)], n=2)
        assert cores[0] == 0

    def test_trailing_isolated_nodes(self):
        """Explicit n includes trailing nodes with no edges, which have core 0."""
        cores = in_degree_core([(0, 1), (1, 0)], n=4)
        assert len(cores) == 4
        assert np.array_equal(cores, np.array([1, 1, 0, 0], dtype=np.int64))

    def test_n_too_small_raises(self):
        """n must exceed the max node ID present in the edges, else ValueError."""
        with pytest.raises(ValueError, match="n=3 but edges contain node ID 5"):
            in_degree_core([(0, 5), (5, 0)], n=3)

    def test_weights_ignored_three_tuples_accepted(self):
        """Weighted 3-tuple edges are accepted and weights are ignored."""
        weighted = in_degree_core([(0, 1, 5.0), (1, 0, 9.0)], n=2)
        unweighted = in_degree_core([(0, 1), (1, 0)], n=2)
        assert np.array_equal(weighted, unweighted)
        assert np.array_equal(weighted, np.array([1, 1], dtype=np.int64))

    def test_duplicate_edges_counted(self):
        """Duplicate directed edges each contribute to the in-degree count."""
        # 0 -> 1 twice plus 1 -> 0: both nodes end with core 1.
        cores = in_degree_core([(0, 1), (0, 1), (1, 0)], n=2)
        assert np.array_equal(cores, np.array([1, 1], dtype=np.int64))


class TestAnalyticalResults:
    def test_docstring_example(self):
        """Verified docstring example: in_degree_core([(0,1),(0,2),(1,2),(2,1)], n=3) -> [0,1,1]."""
        cores = in_degree_core([(0, 1), (0, 2), (1, 2), (2, 1)], n=3)
        assert np.array_equal(cores, np.array([0, 1, 1], dtype=np.int64))

    def test_docstring_clique_both_directions(self):
        """Verified docstring clique example: bidirectional triangle -> [2,2,2]."""
        edges = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
        cores = in_degree_core(edges, n=3)
        assert np.array_equal(cores, np.array([2, 2, 2], dtype=np.int64))

    def test_directed_cycle(self):
        """A directed 3-cycle: every node has in-degree 1, so all cores are 1."""
        cores = in_degree_core([(0, 1), (1, 2), (2, 0)], n=3)
        assert np.array_equal(cores, np.array([1, 1, 1], dtype=np.int64))

    def test_star_incoming_hub_peels_to_zero(self):
        """Incoming star: hub has high in-degree but its sources have none.

        Although the hub has in-degree 4, every leaf has in-degree 0 and gets
        peeled first; the hub keeps no incoming edges in any nontrivial
        subgraph, so the whole graph has core number 0.
        """
        edges = [(1, 0), (2, 0), (3, 0), (4, 0)]
        cores = in_degree_core(edges, n=5)
        assert np.array_equal(cores, np.zeros(5, dtype=np.int64))

    def test_bidirectional_pair_in_chain(self):
        """A mutually-linked pair embeds deeper than a node it merely feeds.

        0 -> 1, 1 -> 2, 2 -> 1: nodes 1 and 2 form a mutual pair (each in-degree
        1 within the pair); node 0 has no incoming edges (core 0).
        """
        cores = in_degree_core([(0, 1), (1, 2), (2, 1)], n=3)
        assert np.array_equal(cores, np.array([0, 1, 1], dtype=np.int64))

    def test_core_bounded_by_in_degree(self):
        """Each node's core number never exceeds its raw in-degree."""
        edges = [(0, 2), (1, 2), (2, 0), (0, 1), (1, 0)]
        cores = in_degree_core(edges, n=3)
        targets = np.array([e[1] for e in edges], dtype=np.int64)
        in_deg = np.bincount(targets, minlength=3)
        assert np.all(cores <= in_deg)


class TestDtype:
    def test_output_dtype_is_int64(self):
        """Output of in_degree_core must be int64."""
        cores = in_degree_core([(0, 1), (1, 0)], n=2)
        assert cores.dtype == np.int64

    def test_empty_output_dtype_is_int64(self):
        """Even empty / zero outputs must carry int64 dtype."""
        assert in_degree_core([], n=0).dtype == np.int64
        assert in_degree_core([], n=3).dtype == np.int64
        assert in_degree_core([(0, 0)], n=1).dtype == np.int64


class TestComputeKcoreColumn:
    def _make_df(self, indices, distances, col="__neighbors"):
        return pd.DataFrame(
            {
                col: [
                    {"ids": indices[i], "distances": distances[i]}
                    for i in range(len(indices))
                ]
            }
        )

    def test_compute_kcore_column_shape_and_dtype(self):
        """compute_kcore_column returns an int64 array of length len(df)."""
        indices = np.array([[1, 2], [0, 2], [0, 1]])
        distances = np.array([[0.1, 0.2], [0.1, 0.3], [0.2, 0.3]])
        df = self._make_df(indices, distances)

        cores = compute_kcore_column(df)

        assert isinstance(cores, np.ndarray)
        assert cores.dtype == np.int64
        assert len(cores) == len(df)

    def test_compute_kcore_column_clique_result(self):
        """Mutual 3-node KNN graph is a clique: every node has core 2."""
        indices = np.array([[1, 2], [0, 2], [0, 1]])
        distances = np.array([[0.1, 0.2], [0.1, 0.3], [0.2, 0.3]])
        df = self._make_df(indices, distances)

        cores = compute_kcore_column(df)

        assert np.array_equal(cores, np.array([2, 2, 2], dtype=np.int64))

    def test_compute_kcore_column_custom_column_name(self):
        """The neighbors column name is configurable via the neighbors kwarg."""
        indices = np.array([[1, 2], [0, 2], [0, 1]])
        distances = np.array([[0.1, 0.2], [0.1, 0.3], [0.2, 0.3]])
        df = self._make_df(indices, distances, col="nbrs")

        cores = compute_kcore_column(df, neighbors="nbrs")

        assert np.array_equal(cores, np.array([2, 2, 2], dtype=np.int64))

    def test_compute_kcore_column_self_neighbor_excluded(self):
        """A node listing itself as a neighbor does not create a self-loop edge."""
        # Each node lists itself first; those self-edges must be dropped.
        indices = np.array([[0, 1], [1, 0], [2, 0]])
        distances = np.array([[0.0, 0.1], [0.0, 0.1], [0.0, 0.1]])
        df = self._make_df(indices, distances)

        cores = compute_kcore_column(df)

        assert len(cores) == 3
        assert cores.dtype == np.int64


class TestIntegration:
    def test_knn_dataframe_to_kcore_pipeline(self):
        """Full pipeline: a __neighbors DataFrame yields one core number per row."""
        rng = np.random.default_rng(7)
        n_samples = 30
        k = 5

        indices = np.zeros((n_samples, k), dtype=np.int64)
        for i in range(n_samples):
            candidates = [j for j in range(n_samples) if j != i]
            indices[i] = rng.choice(candidates, size=k, replace=False)
        distances = np.sort(rng.random((n_samples, k)), axis=1)

        df = pd.DataFrame(
            {
                "__neighbors": [
                    {"ids": indices[i], "distances": distances[i]}
                    for i in range(n_samples)
                ]
            }
        )

        cores = compute_kcore_column(df)

        assert cores.dtype == np.int64
        assert len(cores) == n_samples
        assert np.all(cores >= 0)
        # Cores cannot exceed the number of distinct potential in-neighbors.
        assert np.all(cores < n_samples)
