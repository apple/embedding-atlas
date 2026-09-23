# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""Per-point graph metrics computed from the kNN neighbor graph.

This module groups graph metrics that all operate on the same ``__neighbors``
column format (one dict per row with parallel ``ids`` and ``distances`` arrays,
as produced by projection.py):

- PageRank: :func:`pagerank`, :func:`compute_pagerank_column`
- Degree centrality: :func:`degree`, :func:`compute_degree_columns`
- In-degree k-core: :func:`in_degree_core`, :func:`compute_kcore_column`
- Local clustering coefficient: :func:`clustering_coefficient`,
  :func:`compute_clustering_column`

Run as a script to add a metric as column(s) to a parquet file::

    python -m embedding_atlas.graph_metrics degree --in data.parquet --out out.parquet
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# PageRank
# ----------------------------------------------------------------------


def pagerank(
    edges: Sequence[tuple[int, int] | tuple[int, int, float]],
    *,
    n: int,
    damping: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """
    Compute PageRank scores from a list of edges of a graph using PyTorch
    sparse matrix power iteration. The graph can be either unweighted (each
    edge consists of source node ID and target node ID), or weighted (each
    edge has an additional third element: edge weight).

    Args:
        edges: List of tuples representing edges. Can be:
               - Unweighted: [(source1, target1), (source2, target2), ...]
               - Weighted: [(source1, target1, weight1), (source2, target2, weight2), ...]
               Weighted vs unweighted is auto-detected based on tuple length.
        n: Number of nodes in the graph. The returned array will have this length.
        damping: PageRank damping factor (default: 0.85).
        max_iterations: Maximum number of iterations (default: 100).
        tolerance: Convergence tolerance (default: 1e-9).

    Returns:
        np.ndarray of shape (n,) containing PageRank scores.
        Scores are ordered by node index (scores[i] is the score for node i).

    Example:
        >>> edges = [(0, 1, 0.5), (0, 2, 1.0), (1, 2, 0.8), (2, 0, 1.0)]
        >>> scores = pagerank(edges, n=3)
        >>> scores  # scores[i] is the PageRank score for node i
        array([0.32..., 0.21..., 0.46...])

        # With KNN arrays:
        >>> edges = knn_to_edges(knn_indices, knn_distances)
        >>> scores = pagerank(edges, n=len(knn_indices))
    """

    import torch

    if len(edges) == 0:
        if n > 0:
            return np.full(n, 1.0 / n)
        return np.array([])

    # Parse edges into source, target, weight arrays
    sources = []
    targets = []
    weights = []
    for edge in edges:
        sources.append(edge[0])
        targets.append(edge[1])
        weights.append(float(edge[2]) if len(edge) == 3 else 1.0)

    # Validate n covers all node IDs in the edge list
    max_node_id = max(max(sources), max(targets))
    if n <= max_node_id:
        raise ValueError(
            f"n={n} but edges contain node ID {max_node_id} (n must be > max node ID)"
        )

    # Build sparse transition matrix M where M[j, i] = weight(i -> j) / out_degree(i)
    # This means column i represents outgoing edges from node i.
    # We need to normalize each column by its sum.
    src = torch.tensor(sources, dtype=torch.long)
    tgt = torch.tensor(targets, dtype=torch.long)
    w = torch.tensor(weights, dtype=torch.float64)

    # Compute column sums (out-weight per source node)
    col_sums = torch.zeros(n, dtype=torch.float64)
    col_sums.scatter_add_(0, src, w)

    # Normalize weights by column sum to get transition probabilities
    # Avoid division by zero for dangling nodes (handled separately)
    safe_col_sums = col_sums[src]
    safe_col_sums[safe_col_sums == 0] = 1.0
    normalized_w = w / safe_col_sums

    # Build sparse matrix: M[tgt, src] = normalized_w
    # This is the column-stochastic transition matrix
    indices = torch.stack([tgt, src])
    M = torch.sparse_coo_tensor(indices, normalized_w, size=(n, n), dtype=torch.float64)
    M = M.coalesce()

    # Identify dangling nodes (no outgoing edges)
    is_dangling = col_sums == 0

    # Initialize rank vector uniformly
    r = torch.full((n,), 1.0 / n, dtype=torch.float64)

    teleport = (1.0 - damping) / n

    for i in range(max_iterations):
        # Dangling node contribution: their rank is distributed uniformly
        dangling_sum = r[is_dangling].sum().item()

        # Power iteration step
        r_new = damping * torch.mv(M, r) + teleport + damping * dangling_sum / n

        # Check convergence (L1 norm)
        diff = torch.abs(r_new - r).sum().item()
        r = r_new

        if diff < tolerance:
            break

    return r.numpy()


def knn_to_edges(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    local_connectivity: float = 1.0,
) -> list[tuple[int, int, float]]:
    """
    Convert raw UMAP k-nearest-neighbor (KNN) arrays into a weighted edge
    list, which can then be passed into pagerank().

    Raw KNN distances are not directly usable as edge weights because higher
    distance means weaker connection, and distances are not normalized across
    points with varying local density. This method transforms raw distances
    into UMAP-style membership strengths in [0, 1], where higher values
    indicate stronger connections. The transformation is density-adaptive:
    each point's distances are normalized relative to its local neighborhood
    via per-point sigma and rho parameters.

    The raw arrays come from Projection.knn_indices and
    Projection.knn_distances (see projection.py), which store raw distances
    from umap.umap_.nearest_neighbors(). During UMAP's fit_transform(),
    these raw distances are internally converted to membership strengths
    via smooth_knn_dist() and compute_membership_strengths(), but those
    intermediate results are not exposed. Since Projection only stores the
    raw distances, this method re-derives the membership weights by calling
    the same UMAP functions:

    1. smooth_knn_dist() computes per-point sigma (bandwidth) and rho
       (distance to nearest neighbor) values. rho ensures every point has
       at least one neighbor with membership strength ~1.0. sigma controls
       how fast the strength decays for farther neighbors.

    2. compute_membership_strengths() transforms each raw distance into a
       membership weight via exp(-(distance - rho) / sigma), producing
       values in [0, 1]. Distances <= rho are clamped to weight 1.0.

    Args:
        knn_indices: Array of shape (N, k) where knn_indices[i] contains
                     the 0-indexed row IDs of the k nearest neighbors of
                     row i (may include i itself).
        knn_distances: Array of shape (N, k) where knn_distances[i] contains
                       the raw distances to the k nearest neighbors of row i,
                       aligned with knn_indices (distances[j] corresponds to
                       indices[j]).
        local_connectivity: UMAP local_connectivity parameter (default: 1.0).
                            The default of 1.0 matches UMAP's own default, so this
                            does not need to be provided unless local_connectivity
                            was explicitly set to a non-default value in umap_args
                            when computing the projection (see projection.py). In
                            that case, the same value must be passed here to ensure
                            the membership weights are consistent with the projection.

    Returns:
        List of (source, target, weight) tuples, with self-loops excluded.

    Example:
        >>> indices = np.array([[1, 2], [0, 2], [0, 1]])
        >>> distances = np.array([[0.1, 0.2], [0.1, 0.3], [0.2, 0.3]])
        >>> edges = knn_to_edges(indices, distances)
    """
    from umap.umap_ import compute_membership_strengths, smooth_knn_dist

    n_neighbors = knn_distances.shape[1]

    # Compute sigmas and rhos
    sigmas, rhos = smooth_knn_dist(
        knn_distances,
        k=n_neighbors,
        local_connectivity=local_connectivity,
    )

    # Compute membership strengths (edge weights)
    result = compute_membership_strengths(
        knn_indices.astype(np.int32),
        knn_distances.astype(np.float32),
        sigmas.astype(np.float32),
        rhos.astype(np.float32),
        return_dists=False,
    )
    rows, cols, vals = result[0], result[1], result[2]

    # Convert to edge list, filtering out self-loops
    edges = [(int(r), int(c), float(v)) for r, c, v in zip(rows, cols, vals) if r != c]

    return edges


def compute_pagerank_column(
    dataframe: pd.DataFrame,
    *,
    neighbors: str = "__neighbors",
    local_connectivity: float = 1.0,
    damping: float = 0.85,
):
    """
    Compute PageRank scores from a DataFrame that contains a neighbors column.

    The neighbors column contains one dict per row with two parallel arrays:
      - 'ids': 0-indexed row IDs of the k nearest neighbors (int[])
      - 'distances': raw distances to those neighbors (float[])

    The arrays are aligned: ids[j] is the neighbor and distances[j] is its
    distance. A row's own ID typically appears in its own ids array (often
    at position 0 with distance 0.0), but it is not guaranteed to be first
    because other neighbors can also have distance 0.0. For example:

      Row 0: ids=[0, 110431, 61815, ...], distances=[0.0, 0.07, 0.11, ...]
      Row 4: ids=[113494, 75640, 4, ...], distances=[0.0, 0.0, 0.0, ...]

    This is the format produced by compute_text_projection,
    compute_vector_projection, and compute_image_projection in projection.py.

    Args:
        dataframe: pandas DataFrame containing the neighbor data.
        neighbors: Column name containing the neighbors dicts.
        local_connectivity: UMAP local_connectivity parameter (default: 1.0).
                            See knn_to_edges() for when this needs to be changed.
        damping: PageRank damping factor (default: 0.85).

    Returns:
        np.ndarray of shape (len(dataframe),) containing PageRank scores.
    """
    neighbors_col = dataframe[neighbors]
    knn_indices = np.stack([np.array(row["ids"]) for row in neighbors_col])
    knn_distances = np.stack([np.array(row["distances"]) for row in neighbors_col])

    edges = knn_to_edges(
        knn_indices, knn_distances, local_connectivity=local_connectivity
    )
    scores = pagerank(edges, n=len(dataframe), damping=damping)

    return scores


# ----------------------------------------------------------------------
# Degree centrality
# ----------------------------------------------------------------------


def degree(
    edges: Sequence[tuple[int, int] | tuple[int, int, float]],
    *,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute in-degree, out-degree, and (undirected) degree for each node
    from a list of edges of a directed multigraph. Edge weights, if
    present, are ignored — only edge presence is counted.

    The graph is treated as a multigraph: a→b and b→a are two separate
    edges. The undirected degree counts every incident edge regardless
    of direction, so degree = in_degree + out_degree.

    Args:
        edges: List of tuples representing directed edges. Can be:
               - Unweighted: [(source1, target1), (source2, target2), ...]
               - Weighted: [(source1, target1, weight1), ...] (weights ignored)
               Self-loops are excluded from all counts.
        n: Number of nodes in the graph. The returned arrays will have this length.

    Returns:
        Tuple of three np.ndarray of shape (n,) and dtype int64:
        - in_degree: in_degree[i] is the number of edges pointing to node i.
        - out_degree: out_degree[i] is the number of edges leaving node i.
        - degree: degree[i] is the number of edges incident to node i
                  when edge direction is ignored (= in_degree + out_degree).

    Example:
        >>> edges = [(0, 1), (0, 2), (1, 2), (2, 0)]
        >>> in_deg, out_deg, deg = degree(edges, n=3)
        >>> in_deg   # node 2 has two incoming edges
        array([1, 1, 2])
        >>> out_deg  # node 0 has two outgoing edges
        array([2, 1, 1])
        >>> deg      # degree = in + out for each node
        array([3, 2, 3])

        # With KNN arrays (weights are ignored):
        >>> edges = knn_to_edges(knn_indices, knn_distances)
        >>> in_deg, out_deg, deg = degree(edges, n=len(knn_indices))
    """
    if len(edges) == 0:
        z = np.zeros(n, dtype=np.int64)
        return z.copy(), z.copy(), z.copy()

    sources = np.array([e[0] for e in edges], dtype=np.int64)
    targets = np.array([e[1] for e in edges], dtype=np.int64)

    # Exclude self-loops
    mask = sources != targets
    sources = sources[mask]
    targets = targets[mask]

    if len(sources) == 0:
        z = np.zeros(n, dtype=np.int64)
        return z.copy(), z.copy(), z.copy()

    # Validate n covers all node IDs in the edge list
    max_node_id = max(int(sources.max()), int(targets.max()))
    if n <= max_node_id:
        raise ValueError(
            f"n={n} but edges contain node ID {max_node_id} (n must be > max node ID)"
        )

    # In-degree: count edges pointing to each node
    in_deg = np.bincount(targets, minlength=n).astype(np.int64)

    # Out-degree: count edges leaving each node
    out_deg = np.bincount(sources, minlength=n).astype(np.int64)

    # Undirected degree: in a multigraph, degree = in_degree + out_degree
    deg = in_deg + out_deg

    return in_deg, out_deg, deg


def compute_degree_columns(
    dataframe: pd.DataFrame,
    *,
    neighbors: str = "__neighbors",
    local_connectivity: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute in-degree, out-degree, and degree from a DataFrame that contains
    a neighbors column.

    The neighbors column format is the same as for compute_pagerank_column()
    here: one dict per row with 'ids' and 'distances' arrays.
    See compute_pagerank_column() for full format documentation.

    Args:
        dataframe: pandas DataFrame containing the neighbor data.
        neighbors: Column name containing the neighbors dicts.
        local_connectivity: UMAP local_connectivity parameter (default: 1.0).
                            See knn_to_edges() for when this needs to
                            be changed.

    Returns:
        Tuple of three np.ndarray of shape (len(dataframe),) and dtype int64:
        - in_degree, out_degree, degree (see degree() for definitions).
    """

    neighbors_col = dataframe[neighbors]
    knn_indices = np.stack([np.array(row["ids"]) for row in neighbors_col])
    knn_distances = np.stack([np.array(row["distances"]) for row in neighbors_col])

    edges = knn_to_edges(
        knn_indices, knn_distances, local_connectivity=local_connectivity
    )
    return degree(edges, n=len(dataframe))


# ----------------------------------------------------------------------
# In-degree k-core
# ----------------------------------------------------------------------


def in_degree_core(
    edges: Sequence[tuple[int, int] | tuple[int, int, float]],
    *,
    n: int,
) -> np.ndarray:
    """
    Compute directed k-core decomposition based on in-degree, returning
    the core number for each node. The in-degree core number of a node is
    the largest k such that the node belongs to a subgraph where every node
    has in-degree >= k (within that subgraph).

    This is the directed analogue of k-core that uses in-degree for the
    peeling criterion. It is more informative than undirected k-core on
    KNN graphs because out-degree is fixed at k (each node's k nearest
    neighbors) while in-degree varies — nodes that many others consider a
    nearest neighbor (hubs) get higher core numbers.

    Uses a Batagelj-Zaversnik-style peeling algorithm on in-degree, running
    in O(m) time where m is the number of edges.

    Args:
        edges: List of tuples representing directed edges. Can be:
               - Unweighted: [(source1, target1), (source2, target2), ...]
               - Weighted: [(source1, target1, weight1), ...] (weights ignored)
               Self-loops are excluded.
        n: Number of nodes in the graph. The returned array will have this length.

    Returns:
        np.ndarray of shape (n,) and dtype int64 where result[i] is the
        in-degree core number of node i. Nodes with no incoming edges have
        core number 0.

    Example:
        >>> # 0 points to 1 and 2; 1 and 2 point to each other
        >>> edges = [(0, 1), (0, 2), (1, 2), (2, 1)]
        >>> in_degree_core(edges, n=3)
        array([0, 1, 1])

        >>> # Clique: every pair has edges in both directions
        >>> edges = [(0,1),(1,0),(0,2),(2,0),(1,2),(2,1)]
        >>> in_degree_core(edges, n=3)
        array([2, 2, 2])
    """
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    if len(edges) == 0:
        return np.zeros(n, dtype=np.int64)

    sources = np.array([e[0] for e in edges], dtype=np.int64)
    targets = np.array([e[1] for e in edges], dtype=np.int64)

    # Exclude self-loops
    mask = sources != targets
    sources = sources[mask]
    targets = targets[mask]

    if len(sources) == 0:
        return np.zeros(n, dtype=np.int64)

    # Validate n covers all node IDs in the edge list
    max_node_id = max(int(sources.max()), int(targets.max()))
    if n <= max_node_id:
        raise ValueError(
            f"n={n} but edges contain node ID {max_node_id} (n must be > max node ID)"
        )

    # forward_adj[v] = nodes that v points to (v's out-neighbors).
    # When we peel v, each out-neighbor loses one in-edge from v.
    forward_adj: list[list[int]] = [[] for _ in range(n)]
    for s, t in zip(sources, targets):
        forward_adj[int(s)].append(int(t))

    # Compute initial in-degree
    in_deg = np.bincount(targets, minlength=n).astype(np.int64)

    # Batagelj-Zaversnik peeling on in-degree
    max_deg = int(in_deg.max()) if n > 0 else 0

    # bin[d] = start index in vert[] for nodes of in-degree d
    bin_start = np.zeros(max_deg + 1, dtype=np.int64)
    for v in range(n):
        bin_start[in_deg[v]] += 1

    # Cumulative sum to get start positions
    start = 0
    for d in range(max_deg + 1):
        count = bin_start[d]
        bin_start[d] = start
        start += count

    # pos[v] = position of node v in vert[]
    pos = np.zeros(n, dtype=np.int64)
    vert = np.zeros(n, dtype=np.int64)
    for v in range(n):
        pos[v] = bin_start[in_deg[v]]
        vert[pos[v]] = v
        bin_start[in_deg[v]] += 1

    # Restore bin_start (shift right by 1)
    for d in range(max_deg, 0, -1):
        bin_start[d] = bin_start[d - 1]
    bin_start[0] = 0

    # Core decomposition: process nodes in order of increasing in-degree.
    # When we peel node v, every node u that v points to (forward_adj[v])
    # loses one in-edge from the remaining subgraph.
    core = in_deg.copy()
    for i in range(n):
        v = vert[i]
        for u in forward_adj[v]:
            if core[u] > core[v]:
                du = core[u]
                pu = pos[u]
                pw = bin_start[du]
                w = vert[pw]
                if u != w:
                    pos[u] = pw
                    pos[w] = pu
                    vert[pu] = w
                    vert[pw] = u
                bin_start[du] += 1
                core[u] -= 1

    return core


def compute_kcore_column(
    dataframe: pd.DataFrame,
    *,
    neighbors: str = "__neighbors",
) -> np.ndarray:
    """
    Compute k-core decomposition from a DataFrame that contains a neighbors
    column.

    The neighbors column format is the same as for compute_pagerank_column()
    here: one dict per row with 'ids' and 'distances' arrays.
    See compute_pagerank_column() for full format documentation.

    Args:
        dataframe: pandas DataFrame containing the neighbor data.
        neighbors: Column name containing the neighbors dicts.

    Returns:
        np.ndarray of shape (len(dataframe),) and dtype int64 where result[i]
        is the core number of node i.
    """
    neighbors_col = dataframe[neighbors]
    knn_indices = np.stack([np.array(row["ids"]) for row in neighbors_col])

    # Build edge list directly from KNN indices (k-core ignores weights)
    n, k = knn_indices.shape
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(k):
            t = int(knn_indices[i, j])
            # Skip self-loops and missing neighbors (marked as -1)
            if t >= 0 and i != t:
                edges.append((i, t))

    return in_degree_core(edges, n=n)


# ----------------------------------------------------------------------
# Local clustering coefficient
# ----------------------------------------------------------------------


def clustering_coefficient(
    edges: Sequence[tuple[int, int] | tuple[int, int, float]],
    *,
    n: int,
) -> np.ndarray:
    """
    Compute the local clustering coefficient for each node in a directed graph.

    The clustering coefficient of node v measures what fraction of possible
    directed edges between v's out-neighbors actually exist. For a node with
    d out-neighbors, the maximum possible edges among them is d * (d - 1)
    (each ordered pair). The coefficient is the number of actual edges divided
    by this maximum.

    Edge weights, if present, are ignored.

    Args:
        edges: List of tuples representing directed edges. Can be:
               - Unweighted: [(source1, target1), (source2, target2), ...]
               - Weighted: [(source1, target1, weight1), ...] (weights ignored)
               Self-loops are excluded.
        n: Number of nodes in the graph. The returned array will have this length.

    Returns:
        np.ndarray of shape (n,) and dtype float64 where result[i] is the
        clustering coefficient of node i, in [0, 1]. Nodes with fewer than
        2 out-neighbors have coefficient 0.

    Example:
        >>> # Triangle: 0→1, 0→2, 1→2
        >>> edges = [(0, 1), (0, 2), (1, 2)]
        >>> clustering_coefficient(edges, n=3)
        array([0.5, 0., 0.])
    """
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    if len(edges) == 0:
        return np.zeros(n, dtype=np.float64)

    # Build directed adjacency sets, excluding self-loops
    adj: list[set[int]] = [set() for _ in range(n)]
    for edge in edges:
        s, t = int(edge[0]), int(edge[1])
        if s != t:
            adj[s].add(t)

    cc = np.zeros(n, dtype=np.float64)

    for v in range(n):
        neighbors = adj[v]
        d = len(neighbors)
        if d < 2:
            continue

        # Count directed edges among v's out-neighbors
        count = 0
        for u in neighbors:
            for w in neighbors:
                if u != w and w in adj[u]:
                    count += 1

        cc[v] = count / (d * (d - 1))

    return cc


def compute_clustering_column(
    dataframe: pd.DataFrame,
    *,
    neighbors: str = "__neighbors",
) -> np.ndarray:
    """
    Compute local clustering coefficient from a DataFrame that contains a
    neighbors column.

    The neighbors column format is the same as for compute_pagerank_column()
    here: one dict per row with 'ids' and 'distances' arrays.
    See compute_pagerank_column() for full format documentation.

    Args:
        dataframe: pandas DataFrame containing the neighbor data.
        neighbors: Column name containing the neighbors dicts.

    Returns:
        np.ndarray of shape (len(dataframe),) and dtype float64 where result[i]
        is the clustering coefficient of node i.
    """
    neighbors_col = dataframe[neighbors]
    knn_indices = np.stack([np.array(row["ids"]) for row in neighbors_col])

    # Build edge list directly from KNN indices (clustering ignores weights)
    n, n_neighbors = knn_indices.shape
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(n_neighbors):
            t = int(knn_indices[i, j])
            # Skip self-loops and missing neighbors (marked as -1)
            if t >= 0 and i != t:
                edges.append((i, t))

    return clustering_coefficient(edges, n=n)
