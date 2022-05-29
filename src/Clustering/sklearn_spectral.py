from sklearn.cluster import SpectralClustering
from typing import Any

def untrained_model(seed: int = 42):
    return SpectralClustering(
        init="random", n_clusters=2, random_state=seed
    )

def generate_sklearn_spectral_model(X: Any, Y: Any = None, seed: int = 42):
    return untrained_model(seed=seed).fit(X)
