"""
This module implements the QSVM from [pennylane's documentation]
(https://pennylane.ai/pennylane/demos/tutorial_kernels_module.html).
"""
from typing import Any

import pennylane
from pennylane import numpy as np
from sklearn.svm import SVC

from src.SVMs import X, Y, Xm, Ym
from definitions import SEED, ROOT_DIR
from src.PickleManager import Manager
from src import Type, Technology


def layer(x, params, wires, i0=0, inc=1):
    """Building block of the embedding ansatz"""
    i = i0
    for j, wire in enumerate(wires):
        pennylane.Hadamard(wires=[wire])
        pennylane.RZ(x[i % len(x)], wires=[wire])
        i += inc
        pennylane.RY(params[0, j], wires=[wire])

    pennylane.broadcast(unitary=pennylane.CRZ, pattern="ring", wires=wires, parameters=params[1])


def random_params(num_wires, num_layers):
    """Generate random variational parameters in the shape for the ansatz."""
    return np.random.uniform(0, 2 * np.pi, (num_layers, 2, num_wires))


def ansatz(x, params, wires):
    """The embedding ansatz"""
    for j, layer_params in enumerate(params):
        layer(x, layer_params, wires, i0=j * len(wires))


adjoint_ansatz = pennylane.adjoint(ansatz)

dev = pennylane.device("default.qubit", wires=5, shots=None)
wires = dev.wires.tolist()


@pennylane.qnode(dev)
def kernel_circuit(x1, x2, params):
    ansatz(x1, params, wires=wires)
    adjoint_ansatz(x2, params, wires=wires)
    return pennylane.probs(wires=wires)


def kernel(x1, x2, params):
    return kernel_circuit(x1, x2, params)[0]


def untrained_model(seed: int = 42):
    init_params = random_params(num_wires=5, num_layers=6)
    init_kernel = lambda x1, x2: kernel(x1, x2, init_params)
    return SVC(random_state=seed, kernel=lambda X1, X2: pennylane.kernels.kernel_matrix(X1, X2, init_kernel)) 

def generate(X :Any, Y: Any, seed: int = 42):
    return untrained_model(seed).fit(X, Y)

