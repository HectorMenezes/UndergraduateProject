from os.path import exists
import time
from src import Technology, Type

import numpy as np
import dill as pickle


class Manager:
    def __init__(
        self, model_file: str, generate_model, model_type: Type, technology: Technology
    ):
        self.model_type = model_type
        self.technology = technology

        start = time.time()
        if exists(model_file):
            self._load_model(model_file)
        else:
            self.model = generate_model()
        self.time_elapsed = time.time() - start

    def _load_model(self, filename: str):
        with open(filename, "rb") as f:
            self.model = pickle.load(f)
            return self.model

    def save_model(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self.model, f)

    def accuracy(self, X, Y):
        return 1 - np.count_nonzero(self.model.predict(X) - Y) / len(Y)

    def make_summary(self, filename: str, X, Y):
        header = f"#Report of **{self.model_type}** model made with {str(self.technology.value)}\n\n"
        speed = (
            "##Time elapsed to generate/load the model: \n"
            + str(self.time_elapsed)
            + " seconds\n"
        )
        acc = "##Accuracy:\n " + str(self.accuracy(X, Y))

        with open(filename, "wb") as f:
            f.write((header + speed + acc).encode())
