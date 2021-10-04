"""
Module to generate SVM model after the provided data and then
persists it into a pickle file.
"""
from os.path import exists
import time

from sklearn.svm import SVC

from src.SVMs import X, Y, save_decision_boundary_image
from src.PickleManager import Manager
from definitions import SEED, ROOT_DIR

path_to_model = ROOT_DIR + '/src/SVMs/classical_svm_model.pickle'

start = time.time()


def generate():
    model = SVC(random_state=SEED)
    model.fit(X, Y)
    return model


classical_SVM_manager = Manager(model_file=path_to_model, generate_model=generate)

classical_SVM_manager.save_model(path_to_model)

save_decision_boundary_image(X, Y, ROOT_DIR + '/figures/ClassicalDecisionBoundary.png', classical_SVM_manager.model)

end = time.time()
print(classical_SVM_manager.model.predict([[1, 0]]))
print('Time alapse:', end - start)
