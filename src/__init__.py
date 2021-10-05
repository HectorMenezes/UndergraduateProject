from enum import Enum


class Type(Enum):
    QUANTUM = 'Quantum'
    CLASSICAL = 'Classical'


class Technology(Enum):
    PENNYLANE = 'Pennylane'
    QISKIT = 'Qiskit'
    TENSORFLOW = 'Tensorflow'
    SKLEARN = 'sklearn'
