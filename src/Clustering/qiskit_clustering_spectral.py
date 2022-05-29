from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import QuantumInstance
from qiskit import BasicAer
from qiskit_machine_learning.kernels import QuantumKernel


backend = QuantumInstance(
    BasicAer.get_backend("qasm_simulator"), shots=1024, seed_simulator=42, seed_transpiler=42
)

feature_map = ZZFeatureMap(feature_dimension=4, reps=2, entanglement="linear")

kernel = QuantumKernel(feature_map=feature_map, quantum_instance=backend)

def untrained_model():
    return kernel
