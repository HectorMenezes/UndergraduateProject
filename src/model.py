import time
from typing import Callable
from sklearn.model_selection import cross_val_score

from src.utils import ModelNature, Tech


class Model:
    def __init__(
        self,
        name: str,
        nature: ModelNature,
        algorithm: str,
        tech: Tech,
        data: str,
        seed: int = 42,
        local: bool = True
    ) -> None:
        self.name = name
        self.nature = nature
        self.algorithm = algorithm
        self.tech = tech
        self.data = data
        self.seed = seed,
        self.local = True

    def __repr__(self) -> str:
        return (
            f"<Model(name={self.name},"
            f"nature={self.nature},"
            f"algorithm={self.algorithm},"
            f"tech={self.tech},"
            f"data={self.data},"
            f"seed={self.seed})>"
        )

    def train_or_load_model(self, local :bool, generating_function: Callable, arguments):
        self.model = generating_function(**arguments)


    def evaluate_model_cv(self, X, Y, cv, estimator):
        start = time.time()
        print(cross_val_score(estimator(seed=self.seed), X, Y, cv=5).mean())
        end = time.time()
        print(end - start)