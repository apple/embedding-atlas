# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""Comprehensive tests for the degree (in/out/undirected) implementation.

Covers edge cases, self-loop exclusion, multigraph counting, weight
handling, dtype contracts, and the compute_degree_columns entry point.
"""

import numpy as np
import pandas as pd
import pytest
from embedding_atlas.graph_metrics import (
    compute_degree_columns,
    degree,
)


class TestEdgeCases:
    def test_empty_edges_n_zero(self):
        """Empty edge list with n=0 returns three empty int64 arrays."""
        in_deg, out_deg, deg = degree([], n=0)
        for arr in (in_deg, out_deg, deg):
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (0,)
            assert arr.dtype == np.int64

    def test_empty_edges_with_n(self):
        """Empty edge list with n>0 returns three zero arrays of length n."""
        in_deg, out_deg, deg = degree([], n=3)
        assert np.array_equal(in_deg, np.zeros(3, dtype=np.int64))
        assert np.array_equal(out_deg, np.zeros(3, dtype=np.int64))
        assert np.array_equal(deg, np.zeros(3, dtype=np.int64))

    def test_only_self_loops_yields_zeros(self):
        """If every edge is a self-loop, all degrees are zero."""
        in_deg, out_deg, deg = degree([(0, 0), (1, 1)], n=2)
        assert np.array_equal(in_deg, [0, 0])
        assert np.array_equal(out_deg, [0, 0])
        assert np.array_equal(deg, [0, 0])

    def test_self_loop_excluded(self):
        """The (0,0) self-loop is ignored; only the (0,1) edge counts."""
        in_deg, out_deg, deg = degree([(0, 0), (0, 1)], n=2)
        # (0,0) dropped: node 0 has one outgoing, node 1 has one incoming.
        assert np.array_equal(in_deg, [0, 1])
        assert np.array_equal(out_deg, [1, 0])
        assert np.array_equal(deg, [1, 1])

    def test_trailing_nodes_with_n(self):
        """Explicit n includes trailing nodes that have no incident edges."""
        in_deg, out_deg, deg = degree([(0, 1), (1, 0)], n=5)
        assert len(in_deg) == 5
        assert np.array_equal(in_deg, [1, 1, 0, 0, 0])
        assert np.array_equal(out_deg, [1, 1, 0, 0, 0])
        assert np.array_equal(deg, [2, 2, 0, 0, 0])

    def test_n_smaller_than_edges_raises(self):
        """If n <= max node ID, raise ValueError with the offending node ID."""
        with pytest.raises(ValueError, match="n=3 but edges contain node ID 5"):
            degree([(0, 5)], n=3)

    def test_n_equals_max_id_raises(self):
        """n must be strictly greater than the max node ID, not equal."""
        with pytest.raises(ValueError, match="n=2 but edges contain node ID 2"):
            degree([(0, 2)], n=2)

    def test_large_node_ids(self):
        """Large node IDs work; arrays are sized to n."""
        in_deg, out_deg, deg = degree([(0, 999), (999, 0)], n=1000)
        assert len(deg) == 1000
        assert in_deg[0] == 1 and in_deg[999] == 1
        assert out_deg[0] == 1 and out_deg[999] == 1
        assert deg[0] == 2 and deg[999] == 2


class TestMultigraphAndWeights:
    def test_duplicate_edges_count_twice(self):
        """A repeated edge (0,1),(0,1) counts twice (multigraph semantics)."""
        in_deg, out_deg, deg = degree([(0, 1), (0, 1)], n=2)
        assert out_deg[0] == 2
        assert in_deg[1] == 2
        assert np.array_equal(deg, [2, 2])

    def test_weights_ignored(self):
        """Edge weights are ignored: weighted edges match unweighted ones."""
        weighted = degree([(0, 1, 0.5), (1, 0, 9.0)], n=2)
        unweighted = degree([(0, 1), (1, 0)], n=2)
        for w, u in zip(weighted, unweighted):
            assert np.array_equal(w, u)

    def test_zero_weight_edges_still_count(self):
        """A zero-weight edge is still a present edge and is counted."""
        in_deg, out_deg, deg = degree([(0, 1, 0.0)], n=2)
        assert out_deg[0] == 1
        assert in_deg[1] == 1
        assert np.array_equal(deg, [1, 1])


class TestAnalyticalResults:
    def test_docstring_example(self):
        """The verified docstring example produces exactly the documented arrays."""
        edges = [(0, 1), (0, 2), (1, 2), (2, 0)]
        in_deg, out_deg, deg = degree(edges, n=3)
        assert np.array_equal(in_deg, [1, 1, 2])
        assert np.array_equal(out_deg, [2, 1, 1])
        assert np.array_equal(deg, [3, 2, 3])

    def test_degree_is_in_plus_out(self):
        """Invariant: degree == in_degree + out_degree on a random-ish graph."""
        rng = np.random.default_rng(42)
        n = 40
        # Random directed edges, including duplicates; self-loops allowed
        # (they are dropped internally and must not break the invariant).
        srcs = rng.integers(0, n, size=300)
        tgts = rng.integers(0, n, size=300)
        edges = list(zip(srcs.tolist(), tgts.tolist()))
        in_deg, out_deg, deg = degree(edges, n=n)
        assert np.array_equal(deg, in_deg + out_deg)

    def test_star_incoming(self):
        """All leaves point to the hub: hub has high in-degree, zero out-degree."""
        edges = [(i, 0) for i in range(1, 5)]
        in_deg, out_deg, deg = degree(edges, n=5)
        assert in_deg[0] == 4
        assert out_deg[0] == 0
        assert np.array_equal(out_deg[1:], [1, 1, 1, 1])
        assert np.array_equal(in_deg[1:], [0, 0, 0, 0])
        assert deg[0] == 4

    def test_star_outgoing(self):
        """Hub points to all leaves: hub has high out-degree, zero in-degree."""
        edges = [(0, i) for i in range(1, 5)]
        in_deg, out_deg, deg = degree(edges, n=5)
        assert out_deg[0] == 4
        assert in_deg[0] == 0
        assert np.array_equal(in_deg[1:], [1, 1, 1, 1])
        assert deg[0] == 4


class TestDtype:
    def test_output_dtype_is_int64(self):
        """All three outputs must be int64."""
        in_deg, out_deg, deg = degree([(0, 1), (1, 2), (2, 0)], n=3)
        assert in_deg.dtype == np.int64
        assert out_deg.dtype == np.int64
        assert deg.dtype == np.int64

    def test_empty_output_dtype_is_int64(self):
        """Empty results also carry the int64 dtype."""
        for arr in degree([], n=0):
            assert arr.dtype == np.int64
        for arr in degree([], n=3):
            assert arr.dtype == np.int64

    def test_self_loop_only_output_dtype_is_int64(self):
        """When all edges are self-loops, the zeroed output is still int64."""
        for arr in degree([(0, 0)], n=2):
            assert arr.dtype == np.int64


class TestComputeDegreeColumns:
    def _make_df(self):
        indices = np.array([[1, 2], [0, 2], [0, 1]])
        distances = np.array([[0.1, 0.2], [0.1, 0.3], [0.2, 0.3]])
        return pd.DataFrame(
            {
                "__neighbors": [
                    {"ids": indices[i], "distances": distances[i]}
                    for i in range(len(indices))
                ]
            }
        )

    def test_returns_three_arrays_of_correct_length(self):
        """compute_degree_columns returns three arrays each of length len(df)."""
        df = self._make_df()
        in_deg, out_deg, deg = compute_degree_columns(df)
        assert len(in_deg) == len(df)
        assert len(out_deg) == len(df)
        assert len(deg) == len(df)

    def test_output_dtype_is_int64(self):
        """The three columns produced are int64."""
        df = self._make_df()
        in_deg, out_deg, deg = compute_degree_columns(df)
        assert in_deg.dtype == np.int64
        assert out_deg.dtype == np.int64
        assert deg.dtype == np.int64

    def test_invariant_holds_on_columns(self):
        """degree == in_degree + out_degree must hold for computed columns."""
        df = self._make_df()
        in_deg, out_deg, deg = compute_degree_columns(df)
        assert np.array_equal(deg, in_deg + out_deg)

    def test_columns_assignable_to_dataframe(self):
        """Results plug straight into DataFrame columns of the right length."""
        df = self._make_df()
        in_deg, out_deg, deg = compute_degree_columns(df)
        df["in_degree"] = in_deg
        df["out_degree"] = out_deg
        df["degree"] = deg
        assert len(df["degree"]) == 3
        # Fully connected 3-node KNN graph: every node is incident to edges.
        assert all(d > 0 for d in df["degree"])


class TestIntegration:
    def test_knn_to_degree_pipeline(self):
        """Full pipeline: KNN arrays -> edges -> degree counts."""
        from embedding_atlas.graph_metrics import knn_to_edges

        indices = np.array([[1, 2], [0, 2], [0, 1]])
        distances = np.array([[0.1, 0.2], [0.1, 0.3], [0.2, 0.3]])
        edges = knn_to_edges(indices, distances)
        in_deg, out_deg, deg = degree(edges, n=3)
        assert len(deg) == 3
        assert np.array_equal(deg, in_deg + out_deg)
        assert all(d > 0 for d in deg)
