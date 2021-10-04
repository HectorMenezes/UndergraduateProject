from src.SVMs.classical_svm import classical_SVM_manager
from src.SVMs.quantum_svm import quantum_SVM_manager
print(classical_SVM_manager.model.predict([[1, 0]]))
print(quantum_SVM_manager.model.predict([[1, 0]]))
