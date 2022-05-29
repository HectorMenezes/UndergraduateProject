from typing import Any
from sklearn.cluster import KMeans

def untrained_model(seed: int = 42):
    return KMeans(
        init="random", n_clusters=3, n_init=10, max_iter=300, random_state=seed
    )

def generate_sklearn_clustering_model(X: Any, Y: Any = None, seed: int = 42):
    return untrained_model(seed=seed).fit(X)


