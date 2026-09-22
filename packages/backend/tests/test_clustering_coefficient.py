# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""Comprehensive tests for the local clustering coefficient implementation.

Covers edge cases, analytical results, dtype contracts, and the
compute_clustering_column entry point.
"""

import numpy as np
import pandas as pd
from embedding_atlas.graph_metrics import (
    clustering_coefficient,
    compute_clustering_column,
)


class TestEdgeCases:
    def test_empty_edges_n_zero(self):
        """Empty edge list with n=0 should return an empty array."""
        cc = clustering_coefficient([], n=0)
        assert isinstance(cc, np.ndarray)
        assert len(cc) == 0

    def test_empty_edges_with_n(self):
        """Empty edge list with n>0 should return zeros of length n."""
        cc = clustering_coefficient([], n=3)
        assert len(cc) == 3
        assert np.array_equal(cc, np.zeros(3))

    def test_self_loops_excluded(self):
        """Self-loops must be excluded from the adjacency, so they do not count."""
        # Without self-loops node 0 has out-neighbors {1, 2} and 1->2 exists.
        with_self_loops = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (2, 2)]
        without = [(0, 1), (0, 2), (1, 2)]
        cc_with = clustering_coefficient(with_self_loops, n=3)
        cc_without = clustering_coefficient(without, n=3)
        assert np.allclose(cc_with, cc_without)
        # And the self-loop on 0 does not inflate node 0's coefficient.
        assert np.isclose(cc_with[0], 0.5)

    def test_self_loop_among_neighbors_ignored(self):
        """A self-loop on an out-neighbor is not counted as an edge among neighbors."""
        # Node 0 -> {1, 2}; the only edge among neighbors candidates is 1->1
        # (a self-loop, excluded) so coefficient must be 0, not from 1->1.
        edges = [(0, 1), (0, 2), (1, 1)]
        cc = clustering_coefficient(edges, n=3)
        assert np.isclose(cc[0], 0.0)

    def test_weights_ignored(self):
        """Edge weights must not affect the result (3-tuples vs 2-tuples)."""
        weighted = [(0, 1, 5.0), (0, 2, 0.001), (1, 2, 99.0)]
        unweighted = [(0, 1), (0, 2), (1, 2)]
        cc_w = clustering_coefficient(weighted, n=3)
        cc_u = clustering_coefficient(unweighted, n=3)
        assert np.allclose(cc_w, cc_u)

    def test_fewer_than_two_out_neighbors(self):
        """Nodes with 0 or 1 out-neighbors have coefficient 0."""
        # Node 0 -> {1} (one neighbor), node 1 -> {} (none).
        edges = [(0, 1)]
        cc = clustering_coefficient(edges, n=2)
        assert np.array_equal(cc, np.zeros(2))

    def test_trailing_nodes_with_no_edges(self):
        """Explicit n larger than referenced nodes yields zeros for the rest."""
        edges = [(0, 1), (0, 2), (1, 2)]
        cc = clustering_coefficient(edges, n=6)
        assert len(cc) == 6
        assert np.isclose(cc[0], 0.5)
        assert np.array_equal(cc[3:], np.zeros(3))

    def test_duplicate_edges_do_not_double_count(self):
        """Duplicate edges are coalesced into sets, so they do not inflate counts."""
        with_dups = [(0, 1), (0, 1), (0, 2), (0, 2), (1, 2), (1, 2)]
        without = [(0, 1), (0, 2), (1, 2)]
        cc_dup = clustering_coefficient(with_dups, n=3)
        cc_plain = clustering_coefficient(without, n=3)
        assert np.allclose(cc_dup, cc_plain)


class TestAnalyticalResults:
    def test_docstring_example(self):
        """The verified docstring example: triangle 0->1, 0->2, 1->2 -> [0.5, 0, 0]."""
        edges = [(0, 1), (0, 2), (1, 2)]
        cc = clustering_coefficient(edges, n=3)
        assert np.allclose(cc, [0.5, 0.0, 0.0])

    def test_fully_mutual_neighbors_gives_one(self):
        """A fully mutually-connected out-neighbor set gives coefficient 1.0."""
        # Node 0 -> {1, 2}, and both directed edges 1->2 and 2->1 exist.
        edges = [(0, 1), (0, 2), (1, 2), (2, 1)]
        cc = clustering_coefficient(edges, n=3)
        assert np.isclose(cc[0], 1.0)

    def test_complete_directed_triangle(self):
        """A complete directed triangle: every node sees its neighbors fully linked."""
        # All ordered pairs among {0,1,2}.
        edges = [
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 2),
            (2, 0),
            (2, 1),
        ]
        cc = clustering_coefficient(edges, n=3)
        # Each node has 2 out-neighbors that are mutually connected => 1.0.
        assert np.allclose(cc, [1.0, 1.0, 1.0])

    def test_partial_neighbor_connectivity(self):
        """Node 0 with 3 out-neighbors and a known fraction of internal edges."""
        # Node 0 -> {1, 2, 3}. d=3, max possible directed edges = 3*2 = 6.
        # Among neighbors: 1->2 and 2->3 exist => count=2 => 2/6 = 1/3.
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]
        cc = clustering_coefficient(edges, n=4)
        assert np.isclose(cc[0], 2.0 / 6.0)

    def test_all_values_in_unit_interval(self):
        """Across a mixed graph all coefficients must lie within [0, 1]."""
        edges = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (1, 0),
            (2, 3),
            (2, 0),
            (3, 0),
        ]
        cc = clustering_coefficient(edges, n=4)
        assert np.all(cc >= 0.0)
        assert np.all(cc <= 1.0)


class TestDtype:
    def test_output_dtype_is_float64(self):
        """Output array dtype must be float64."""
        edges = [(0, 1), (0, 2), (1, 2)]
        cc = clustering_coefficient(edges, n=3)
        assert cc.dtype == np.float64

    def test_empty_outputs_are_float64(self):
        """Both empty-result paths (n=0 and empty edges) return float64."""
        assert clustering_coefficient([], n=0).dtype == np.float64
        assert clustering_coefficient([], n=3).dtype == np.float64

    def test_returns_numpy_array(self):
        """Return type must be a numpy ndarray."""
        cc = clustering_coefficient([(0, 1), (0, 2), (1, 2)], n=3)
        assert isinstance(cc, np.ndarray)


class TestComputeClusteringColumn:
    def test_basic_neighbors_dataframe(self):
        """compute_clustering_column returns a float64 array of length len(df)."""
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

        result = compute_clustering_column(df)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert len(result) == len(df)

    def test_values_in_unit_interval(self):
        """All coefficients from a neighbors DataFrame stay within [0, 1]."""
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

        result = compute_clustering_column(df)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_fully_connected_triangle_neighbors(self):
        """A mutually-connected 3-node neighborhood yields coefficient 1.0 each."""
        # Every node lists the other two as neighbors => complete directed triangle.
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

        result = compute_clustering_column(df)
        assert np.allclose(result, [1.0, 1.0, 1.0])

    def test_custom_neighbors_column_name(self):
        """The neighbors column name is configurable via the keyword argument."""
        indices = np.array([[1, 2], [0, 2], [0, 1]])
        distances = np.array([[0.1, 0.2], [0.1, 0.3], [0.2, 0.3]])
        df = pd.DataFrame(
            {
                "knn": [
                    {"ids": indices[i], "distances": distances[i]}
                    for i in range(len(indices))
                ]
            }
        )

        result = compute_clustering_column(df, neighbors="knn")
        assert len(result) == 3
        assert np.allclose(result, [1.0, 1.0, 1.0])

    def test_self_referencing_neighbors_excluded(self):
        """Self-references in the neighbor lists are excluded as self-loops."""
        # Node 0 lists itself plus 1 and 2; the self-edge must be dropped.
        # Node 1 -> {0, 2}; node 2 -> {0} only (so 2->1 does NOT exist).
        indices = np.array([[0, 1, 2], [0, 0, 2], [0, 0, 0]])
        distances = np.array([[0.0, 0.1, 0.2], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
        df = pd.DataFrame(
            {
                "__neighbors": [
                    {"ids": indices[i], "distances": distances[i]}
                    for i in range(len(indices))
                ]
            }
        )

        result = compute_clustering_column(df)
        # Node 0 out-neighbors {1, 2} (self dropped); 1->2 exists, 2->1 does not
        # => 1 of 2 possible directed edges => 0.5.
        assert np.isclose(result[0], 0.5)


class TestIntegration:
    def test_neighbors_to_clustering_pipeline(self):
        """Full pipeline: KNN neighbor dicts -> clustering coefficients."""
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

        result = compute_clustering_column(df)
        assert len(result) == 3
        assert np.all((result >= 0.0) & (result <= 1.0))
