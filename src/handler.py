import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score

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
from src.Clustering.sklearn_clustering import(
    untrained_model as untrained_sklearn_clustering_model,
    generate_sklearn_clustering_model as generate_sklearn_clustering_model
)
from src.Clustering.sklearn_spectral import (
    untrained_model as untrained_sklearn_spectral_model,
    generate_sklearn_spectral_model 
)
from src.Clustering.qiskit_clustering_spectral import(
    untrained_model as untrained_qiskit_clustering_model
)

limit = 50

available_training_functions = {
    "classical": {
        "svm": {"sklearn": generate_sklearn_svm_model},
        "clustering": {"sklearn": generate_sklearn_clustering_model},
        "spectral": {"skllearn": generate_sklearn_spectral_model}},
    "quantum": {"svm": {"pennylane": generate_pennylane_svm_model}},
}

available_untrained_functions = {
    "classical": {
        "svm": {"sklearn": untrained_sklearn_svm_model},
        "clustering": {"sklearn": untrained_sklearn_clustering_model},
        "spectral": {"skelarn": untrained_sklearn_spectral_model}},
    "quantum": {
        "svm": {"pennylane": untrained_pennylane_svm_model},
        "clustering": {"qiskit": untrained_qiskit_clustering_model}},
}


def load_partial_data(file_data: str, supervised: bool, limit: int):
    """
    Load data from the data dir, according to parameters
    """
    data = pd.read_csv(ROOT_DIR + f"/data/{file_data}", header=None)
    X = data.iloc[:, :-1].to_numpy()[:limit]
    Y = data.iloc[:, -1:].values.ravel()[:limit] if supervised else None

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
    model: Model,
    max_points: int
):
    """
    Function to create models, load the data
    """

    X, Y = load_partial_data(model.data, True, max_points)
    model.train_or_load_model(
        local=model.local,
        generating_function=available_training_functions.get(model.nature.value)
        .get(model.algorithm)
        .get(model.tech.value),
        arguments={"seed": model.seed, "X": X, "Y": Y},
    )

    return model


def perform_cross_validation(model: Model, k_folds: int, max_points: int):
    X, Y = load_partial_data(model.data, supervised=True, limit=max_points)
    estimator = (
        available_untrained_functions.get(model.nature.value)
        .get(model.algorithm)
        .get(model.tech.value)
    )
    time_elapsed, mean =  model.evaluate_model_cv(X, Y, k_folds, estimator())
    return time_elapsed, mean

def perform_MIS(model: Model, max_points: int):
    estimator = (
        available_untrained_functions.get(model.nature.value)
        .get(model.algorithm)
        .get(model.tech.value)
    )
    time_elapsed, score = model.ami(max_points, estimator())
    return time_elapsed, score

