from re import A
import time
from typing import Callable

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import cross_val_score
from qiskit_machine_learning.datasets import ad_hoc_data


from src.utils import ModelNature, Tech


def folds(data, cv):
    eo = int(len(data) / cv)
    training_sets = []
    testing_sets = []
    for epoch in range(cv):
        if epoch == 0:
            training_sets.append(data[-eo * (cv - 1) :])
            testing_sets.append(data[:eo])
        elif epoch == cv - 1:
            training_sets.append(data[: eo * (cv - 1)])
            testing_sets.append(data[-eo:])
        else:

            try:
                training_sets.append(
                    np.vstack([data[: (eo * epoch)], data[-eo * (cv - epoch - 1) :]])
                )
            except:
                training_sets.append(
                    np.append(data[: (eo * epoch)], data[-eo * (cv - epoch - 1) :])
                )
            testing_sets.append(data[eo * epoch : -eo * (cv - epoch - 1) :])

    return training_sets, testing_sets


def create_samples(X, Y, cv):
    X_train, X_test = folds(X, cv)
    Y_train, Y_test = folds(Y, cv)
    return X_train, X_test, Y_train, Y_test


def acc(classifier, X, Y):
    return 1 - np.count_nonzero(classifier.predict(X) - Y) / len(Y)


class Model:
    def __init__(
        self,
        nature: ModelNature,
        algorithm: str,
        tech: Tech,
        data: str,
        supervised: bool,
        seed: int = 42,
        local: bool = True,
    ) -> None:
        self.nature = nature
        self.algorithm = algorithm
        self.tech = tech
        self.data = data
        self.seed = seed
        self.supervised = supervised
        self.local = True

    def __repr__(self) -> str:
        return (
            f"<Model(nature={self.nature},"
            f"algorithm={self.algorithm},"
            f"tech={self.tech},"
            f"data={self.data},"
            f"seed={self.seed})>"
        )

    def train_or_load_model(
        self, local: bool, generating_function: Callable, arguments
    ):
        self.model = generating_function(
            **{x: arguments[x] for x in arguments if x not in ["seed"]}
        )

    def evaluate_model_cv(self, X, Y, cv, estimator):
        if self.algorithm == "clustering" and self.tech.name == Tech.QISKIT.name:
            cluster_labels = SpectralClustering(2, affinity="precomputed").fit_predict(
                estimator.evaluate(x_vec=X)
            )

        if self.supervised:
            start = time.time()
            mean = cross_val_score(estimator, X, Y, cv=cv).mean()
            end = time.time()
            return end - start, mean

        X_train, X_test, Y_train, Y_test = create_samples(X, Y, cv)
        start = time.time()

        sum = 0
        for index in range(cv):

            classificator = estimator.fit(X_train[index], Y_train[index])

            sum += acc(classificator, X_test[index], Y_test[index])

        end = time.time()
        return end - start, sum / cv

    def ami(self, max_points, estimator):
        (
            train_features,
            train_labels,
            test_features,
            test_labels,
            adhoc_total,
        ) = ad_hoc_data(
            training_size=max_points,
            test_size=0,
            n=2,
            gap=0.6,
            plot_data=False,
            one_hot=False,
            include_sample_total=True,
        )
        start = time.time()
        if self.nature.name == ModelNature.QUANTUM.name:
            kernel = estimator
            matrix = kernel.evaluate(x_vec=train_features)
            score = adjusted_mutual_info_score(SpectralClustering(2, affinity="precomputed").fit_predict(matrix), train_labels)
        else:
            score = adjusted_mutual_info_score(estimator.fit_predict(train_features), train_labels)
        
        
        end = time.time()
        return end - start, score


