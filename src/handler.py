import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

from src.utils import ModelNature, Tech
from src.model import Model
from definitions import ROOT_DIR
from src.SVMs.sklearn_svm import (
    generate_sklearn_svm_model,
    untrained_model as untrained_sklearn_svm_model,
)
from src.SVMs.pennylane_svm import (
    generate as generate_pennylane_svm_model,
    untrained_model as untrained_pennylane_svm_model,
)


limit = 50

available_training_functions = {
    "classical": {"svm": {"sklearn": generate_sklearn_svm_model}},
    "quantum": {"svm": {"pennylane": generate_pennylane_svm_model}},
}

available_untrained_functions = {
    "classical": {"svm": {"sklearn": untrained_sklearn_svm_model}},
    "quantum": {"svm": {"pennylane": untrained_pennylane_svm_model}},
}


def accuracy(classifier, X, Y_target):
    return 1 - np.count_nonzero(classifier.predict(X) - Y_target) / len(Y_target)


def test_data(file_data: str, supervised: bool):
    data = pd.read_csv(ROOT_DIR + f"/data/{file_data}", header=None)
    X = data.iloc[:, :-1].to_numpy()
    Y = data.iloc[:, -1:].values.ravel() if supervised else None

    X = X[limit:]
    Y = Y[limit:]
    # print(X)
    # print(Y)
    return X, Y


def load_partial_data(file_data: str, supervised: bool):
    """
    Load data from the data dir, according to parameters
    """
    data = pd.read_csv(ROOT_DIR + f"/data/{file_data}", header=None)
    X = data.iloc[:, :-1].to_numpy()
    Y = data.iloc[:, -1:].values.ravel() if supervised else None

    X = X[:limit]
    Y = Y[:limit]
    # print(X)
    # print(Y)
    return X, Y


def load_full_data(file_data: str, supervised: bool):
    """
    Load data from the data dir, according to parameters
    """
    data = pd.read_csv(ROOT_DIR + f"/data/{file_data}", header=None)
    X = data.iloc[:, :-1].to_numpy()
    Y = data.iloc[:, -1:].values.ravel() if supervised else None

    return X, Y


def create_model(
    nature: ModelNature,
    algorithm: str,
    data: str,
    tech: Tech,
    supervised: bool,
    seed: int,
    local: bool,
):
    """
    Function to create models, load the data
    """

    model = Model(
        name="Super Model",
        nature=nature,
        algorithm=algorithm,
        data=data,
        tech=tech,
        seed=seed,
    )

    X, Y = load_partial_data(data, supervised)
    # x_testing, y_testing = test_data(data, supervised)

    model.train_or_load_model(
        local=local,
        generating_function=available_training_functions.get(nature.value)
        .get(algorithm)
        .get(tech.value),
        arguments={"seed": seed, "X": X, "Y": Y},
    )

    # print(accuracy(model.model, x_testing, y_testing))

    return model


def evaluate_estimators(
    nature: ModelNature,
    algorithm: str,
    data: str,
    tech: Tech,
    supervised: bool,
    seed: int,
    local: bool,
):
    estimator = (
        available_untrained_functions.get(nature.value).get(algorithm).get(tech.value)
    )

    X, Y = load_full_data(data, supervised)

    print(cross_val_score(estimator(seed=seed), X, Y, cv=5).mean())
