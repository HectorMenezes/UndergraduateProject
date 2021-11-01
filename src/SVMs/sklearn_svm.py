"""
Module to generate SVM model after the provided data and then
persists it into a pickle file.
"""
from sklearn.svm import SVC

from src.SVMs import X, Y
from src.PickleManager import Manager
from definitions import SEED, ROOT_DIR
from src import Technology, Type


def generate():
    model = SVC(random_state=SEED)
    model.fit(X, Y)
    return model


path_to_model = ROOT_DIR + '/src/SVMs/sklearn_svm.pickle'
classical_SVM_manager = Manager(model_file=path_to_model,
                                generate_model=generate,
                                model_type=Type.CLASSICAL,
                                technology=Technology.SKLEARN)

classical_SVM_manager.save_model(path_to_model)
print(classical_SVM_manager.model.predict([[5.3586,3.7557,-1.7345,1.0789]]))

# classical_SVM_manager.make_summary('SUMMARY1.md', X, Y)
