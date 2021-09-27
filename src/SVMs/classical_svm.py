"""
Module to generate SVM model after the provided data and then
persists it into a pickle file.
"""
from os.path import exists

from sklearn.svm import SVC

from src.SVMs import X, Y, save_decision_boundary_image
from src.PickleManager import Manager
from definitions import SEED, ROOT_DIR

classical_SVM_manager = None
path_to_model = ROOT_DIR + '/src/SVMs/classical_svm_model.pickle'


if exists(path_to_model):
    classical_SVM_manager = Manager(model_file=path_to_model)
else:
    def generate():
        model = SVC(random_state=SEED)
        model.fit(X, Y)
        return model

    classical_SVM_manager = Manager(generate_model=generate())


classical_SVM_manager.save_model(path_to_model)


save_decision_boundary_image(X, Y, ROOT_DIR + '/figures/Decision.png', classical_SVM_manager.model)
