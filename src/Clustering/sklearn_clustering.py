from typing import Any
from sklearn.cluster import KMeans


def generate_sklearn_clustering_model(X: Any, seed: int = 42):
    kmeans = KMeans(
        init="random", n_clusters=3, n_init=10, max_iter=300, random_state=seed
    )
    return kmeans.fit(X)
