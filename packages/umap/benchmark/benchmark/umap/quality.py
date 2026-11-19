"""Embedding quality metrics: trustworthiness and kNN recall against input space."""

import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

N_NEIGHBORS = 15
MAX_POINTS = 5_000
SEED = 42


def _neighbor_indices(points, metric, k):
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="brute")
    nn.fit(points)
    return nn.kneighbors(points, return_distance=False)[:, 1:]


class QualityEvaluator:
    """Scores embeddings of one dataset on a fixed subsample of at most MAX_POINTS.

    The exact input-space kNN of the subsample is computed once and reused for
    every embedding of that dataset.
    """

    def __init__(self, data, metric):
        n = data.shape[0]
        rng = np.random.RandomState(SEED)
        self.indices = np.sort(rng.choice(n, size=min(n, MAX_POINTS), replace=False))
        self.data = data[self.indices]
        self.metric = metric
        self.input_neighbors = _neighbor_indices(self.data, metric, N_NEIGHBORS)

    def evaluate(self, embedding):
        """Return {"trustworthiness": float, "knn_recall": float}."""
        if not np.isfinite(embedding).all():
            return {"trustworthiness": "", "knn_recall": ""}
        embedding = embedding[self.indices]
        embedding_neighbors = _neighbor_indices(embedding, "euclidean", N_NEIGHBORS)
        hits = sum(
            len(np.intersect1d(expected, found))
            for expected, found in zip(self.input_neighbors, embedding_neighbors)
        )
        return {
            "trustworthiness": round(
                trustworthiness(
                    self.data,
                    embedding,
                    n_neighbors=N_NEIGHBORS,
                    metric=self.metric,
                ),
                4,
            ),
            "knn_recall": round(hits / self.input_neighbors.size, 4),
        }
