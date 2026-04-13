from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch


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


if __name__ == "__main__":
    import argparse
    import time

    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(
        description="""\
            Compute PageRank scores from a parquet file containing KNN neighbor data.

            Input parquet file must contain a '__neighbors' column where each row is a
            dict with two parallel arrays:
            - 'ids': 0-indexed row IDs of the k nearest neighbors (int[])
                e.g. [0, 110431, 61815, ...] or [113494, 75640, 4, ...]
            - 'distances': raw distances to those neighbors (float[])
                e.g. [0.0, 0.07, 0.11, ...] or [0.0, 0.0, 0.0, ...]

            The arrays are aligned: ids[j] is the neighbor and distances[j] is its
            distance. A row's own ID typically appears in its own ids array (often at
            position 0 with distance 0.0), but it is not guaranteed to be first because
            other neighbors can also have distance 0.0.

            This is the format produced by projection.py (see compute_text_projection,
            compute_vector_projection, compute_image_projection).

            Output parquet file contains all original columns plus a 'pagerank' column
            with float scores that sum to 1.0, e.g. 0.000312 (higher = more central
            in the KNN graph).
            """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--in",
        dest="input_file",
        type=str,
        required=True,
        help="Input parquet file with __neighbors column (see above for format)",
    )
    parser.add_argument(
        "--out",
        dest="output_file",
        type=str,
        required=True,
        help="Output parquet file: same as input with a 'pagerank' column added",
    )
    args = parser.parse_args()

    # Load parquet and extract KNN arrays from the __neighbors column
    print(f"Loading {args.input_file} ...")
    df = pq.read_table(args.input_file).to_pandas()
    print(f"Loaded {len(df)} rows")

    # Compute PageRank and add as a column
    print("Computing PageRank...")
    start_time = time.time()
    df["pagerank"] = compute_pagerank_column(df)
    print(f"PageRank completed in {time.time() - start_time:.4f} seconds")

    # Summary
    scores = df["pagerank"].values
    top_indices = np.argsort(scores)[::-1][:10]
    print("\nTop 10 nodes by PageRank score:")
    for idx in top_indices:
        print(f"  node {idx:>6d}   score {scores[idx]:.10f}")

    print(
        f"\nScore statistics: min={scores.min():.10f}, max={scores.max():.10f}, mean={scores.mean():.10f}"
    )

    # Write output
    df.to_parquet(args.output_file, index=False)
    print(f"\nSaved to {args.output_file} with 'pagerank' column added")
