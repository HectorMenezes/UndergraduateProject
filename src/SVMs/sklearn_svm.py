"""
Module to generate SVM model after the provided data and then
persists it into a pickle file.
# """
from typing import Any
from sklearn.svm import SVC

def untrained_model(seed: int = 42):
    return SVC(random_state=seed)

def generate_sklearn_svm_model(X: Any, Y: Any, seed: int = 42):
    model = untrained_model(seed)
    return model.fit(X, Y)
