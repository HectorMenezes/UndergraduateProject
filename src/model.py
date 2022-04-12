import time
from typing import Callable
from sklearn.model_selection import cross_val_score

from src.utils import ModelNature, Tech


class Model:
    def __init__(
        self,
        nature: ModelNature,
        algorithm: str,
        tech: Tech,
        data: str,
        seed: int = 42,
        local: bool = True
    ) -> None:
        self.nature = nature
        self.algorithm = algorithm
        self.tech = tech
        self.data = data
        self.seed = seed,
        self.local = True

    def __repr__(self) -> str:
        return (
            f"<Model(nature={self.nature},"
            f"algorithm={self.algorithm},"
            f"tech={self.tech},"
            f"data={self.data},"
            f"seed={self.seed})>"
        )

    def train_or_load_model(self, local :bool, generating_function: Callable, arguments):
        self.model = generating_function(**{x: arguments[x] for x in arguments if x not in ["seed"]})


    def evaluate_model_cv(self, X, Y, cv, estimator):
        start = time.time()
        mean = cross_val_score(estimator, X, Y, cv=cv).mean()
        end = time.time()
        return end - start, mean