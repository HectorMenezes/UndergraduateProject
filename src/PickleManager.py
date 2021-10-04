from os.path import exists

import numpy as np
import pickle


class Manager:
    def __init__(self, model_file, generate_model):
        if exists(model_file):
            self.load_model(model_file)
        else:
            self.model = generate_model()

    def load_model(self, filename: str):
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)
            return self.model

    def save_model(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL)

    def accuracy(self, X, Y):
        return 1 - np.count_nonzero(self.predict(X) - Y) / len(Y)
