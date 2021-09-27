import numpy as np
import pickle


class Manager:
    def __init__(self, model_file=None, generate_model=None):
        if model_file:
            self.model = self.load_model(model_file)
        elif not model_file and generate_model:
            self.model = generate_model
        else:
            raise ValueError('You need to provide the model or the generation function')

    def load_model(self, filename: str):
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)
            return self.model

    def save_model(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL)

    def accuracy(self, X, Y):
        return 1 - np.count_nonzero(self.predict(X) - Y) / len(Y)