"""
This module implements the QSVM from [pennylane's documentation]
(https://pennylane.ai/qml/demos/tutorial_kernels_module.html).
"""
from os.path import exists

import pennylane as qml
from pennylane import numpy as np
from sklearn.svm import SVC

from src.SVMs import X, Y, save_decision_boundary_image
from definitions import SEED, ROOT_DIR
from src.PickleManager import Manager

path_to_model = ROOT_DIR + '/src/SVMs/quantum_svm_model.pickle'


def generate():
    return 'hey'


# quantum_SVM_manager = Manager(model_file=path_to_model, generate_model=generate)


def layer(x, params, wires, i0=0, inc=1):
    """Building block of the embedding ansatz"""
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
        qml.RY(params[0, j], wires=[wire])

    qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[1])


def random_params(num_wires, num_layers):
    """Generate random variational parameters in the shape for the ansatz."""
    return np.random.uniform(0, 2 * np.pi, (num_layers, 2, num_wires))


def ansatz(x, params, wires):
    """The embedding ansatz"""
    for j, layer_params in enumerate(params):
        layer(x, layer_params, wires, i0=j * len(wires))


adjoint_ansatz = qml.adjoint(ansatz)

dev = qml.device("default.qubit", wires=5, shots=None)
wires = dev.wires.tolist()


@qml.qnode(dev)
def kernel_circuit(x1, x2, params):
    ansatz(x1, params, wires=wires)
    adjoint_ansatz(x2, params, wires=wires)
    return qml.probs(wires=wires)


def kernel(x1, x2, params):
    return kernel_circuit(x1, x2, params)[0]


init_params = random_params(num_wires=5, num_layers=6)
init_kernel = lambda x1, x2: kernel(x1, x2, init_params)
svm = SVC(random_state=SEED, kernel=lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, init_kernel)).fit(X, Y)

print("1")
print(svm.predict([[1, 0]]))
print("1")
# quantum_SVM_manager.save_model(path_to_model)
save_decision_boundary_image(X, Y, ROOT_DIR + '/figures/QuantumDecisionBoundary.png', svm)

print("1")