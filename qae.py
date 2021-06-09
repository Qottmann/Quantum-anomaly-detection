"""A quantum auto-encoder (QAE)."""

from typing import Union, Optional, List, Tuple, Callable, Any
import numpy as np

from qiskit import *
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.circuit.library.standard_gates import RYGate, CZGate
from qiskit.circuit.gate import Gate
from qiskit.algorithms.optimizers import Optimizer, SPSA
from qiskit.utils import algorithm_globals
from qiskit.providers import BaseBackend, Backend


class QAEAnsatz(TwoLocal):
    def __init__(
        self,
        num_qubits: int,
        num_trash_qubits: int,
        trash_qubits_idxs: Union[np.ndarray, List] = [1, 2],  # TODO
        measure_trash: bool = False,
        rotation_blocks: Gate = RYGate,
        entanglement_blocks: Gate = CZGate,
        parameter_prefix: str = 'θ',
        insert_barriers: bool = False,
        initial_state: Optional[Any] = None,
    ) -> None:
        """Create a new QAE circuit.

        Args:
            num_qubits: The number of qubits of the QAE circuit.
            num_trash_qubits: The number of trash qubits that should be measured in the end.
            trash_qubits_idxs: The explicit indices of the trash qubits, i.e., where the trash
                qubits should be placed.
            measure_trash: If True, the trash qubits will be measured at the end. If False, no
                measurement takes place.
            rotation_blocks: The blocks used in the rotation layers. If multiple are passed,
                these will be applied one after another (like new sub-layers).
            entanglement_blocks: The blocks used in the entanglement layers. If multiple are passed,
                these will be applied one after another.
            parameter_prefix: The prefix used if default parameters are generated.
            insert_barriers: If True, barriers are inserted in between each layer. If False,
                no barriers are inserted.
            initial_state: A `QuantumCircuit` object which can be used to describe an initial state
                prepended to the NLocal circuit.
        """

        assert num_trash_qubits < num_qubits

        self.num_trash_qubits = num_trash_qubits
        self.trash_qubits_idxs = trash_qubits_idxs
        self.measure_trash = measure_trash
        entanglement = [QAEAnsatz._generate_entangler_map(
            num_qubits, num_trash_qubits, i, trash_qubits_idxs) for i in range(num_trash_qubits)]

        super().__init__(num_qubits=num_qubits,
                         rotation_blocks=rotation_blocks,
                         entanglement_blocks=entanglement_blocks,
                         entanglement=entanglement,
                         reps=num_trash_qubits,
                         skip_final_rotation_layer=True,
                         parameter_prefix=parameter_prefix,
                         insert_barriers=insert_barriers,
                         initial_state=initial_state)
        self.add_register(ClassicalRegister(self.num_trash_qubits))

    @staticmethod
    def _generate_entangler_map(num_qubits: int, num_trash_qubits: int, i_permut: int = 1, trash_qubits_idxs: Union[np.ndarray, List] = [1, 2]) -> List[Tuple[int, int]]:
        """Generates entanglement map for QAE circuit

        Entangling gates are only added between trash and non-trash-qubits.

        Args:
            num_qubits: The number of qubits of the QAE circuit.
            num_trash_qubits: The number of trash qubits that should be measured in the end.
            i_permut: Permutation index; increases for every layer of the circuit
            trash_qubits_idxs: The explicit indices of the trash qubits, i.e., where the trash
                qubits should be placed.

        Returns:
            entanglement map: List of pairs of qubit indices that should be entangled
        """
        result = []
        nums_compressed = list(range(num_qubits))
        for trashqubit in trash_qubits_idxs:
            nums_compressed.remove(trashqubit)
        if trash_qubits_idxs == None:
            nums_compressed = list(range(num_qubits))[:num_qubits-num_trash_qubits]
            trash_qubits_idxs = list(range(num_qubits))[-num_trash_qubits:]

        # combine all trash qubits with themselves
        for i,trash_q in enumerate(trash_qubits_idxs[:-1]):
            result.append((trash_qubits_idxs[i+1], trash_qubits_idxs[i]))
        # combine each of the trash qubits with every n-th
        # repeat the list of trash indices cyclicly
        repeated = list(trash_qubits_idxs) * (num_qubits-num_trash_qubits)
        for i in range(num_qubits-num_trash_qubits):
            result.append((repeated[i_permut + i], nums_compressed[i]))
        return result

    def _build(self) -> None:
        """Build the circuit."""
        if self._data:
            return

        _ = self._check_configuration()

        self._data = []

        if self.num_qubits == 0:
            return

        # use the initial state circuit if it is not None
        if self._initial_state:
            circuit = self._initial_state.construct_circuit('circuit', register=self.qregs[0])
            self.compose(circuit, inplace=True)

        param_iter = iter(self.ordered_parameters)

        # build the prepended layers
        self._build_additional_layers('prepended')

        # main loop to build the entanglement and rotation layers
        for i in range(self.reps):
            # insert barrier if specified and there is a preceding layer
            if self._insert_barriers and (i > 0 or len(self._prepended_blocks) > 0):
                self.barrier()

            # build the rotation layer
            self._build_rotation_layer(param_iter, i)

            # barrier in between rotation and entanglement layer
            if self._insert_barriers and len(self._rotation_blocks) > 0:
                self.barrier()

            # build the entanglement layer
            self._build_entanglement_layer(param_iter, i)

        # add the final rotation layer
        if self.insert_barriers and self.reps > 0:
            self.barrier()

        for j, block in enumerate(self.rotation_blocks):

            # create a new layer
            layer = QuantumCircuit(*self.qregs)

            block_indices = [[i] for i in self.trash_qubits_idxs]

            # apply the operations in the layer
            for indices in block_indices:
                parameterized_block = self._parameterize_block(block, param_iter, i, j, indices)
                layer.compose(parameterized_block, indices, inplace=True)

            # add the layer to the circuit
            self.compose(layer, inplace=True)

        # add the appended layers
        self._build_additional_layers('appended')

        # measure trash qubits if set
        if self.measure_trash:
            for i, j in enumerate(self.trash_qubits_idxs):
                self.measure(self.qregs[0][j], self.cregs[0][i])

    @property
    def num_parameters_settable(self) -> int:
        """The number of total parameters that can be set to distinct values.

        Returns:
            The number of parameters originally available in the circuit.
        """
        return super().num_parameters_settable + self.num_trash_qubits


def hamming_distance(out) -> int:
    """Computes the Hamming distance of a measurement outcome to the
    all zero state. For example: A single measurement outcome 101 would
    have a Hamming distance of 2.

    Args:
        out: The measurement outcomes; a dictionary containing all possible measurement strings
            as keys and their occurences as values.

    Returns:
        Hamming distance
    """
    return sum(key.count('1') * value for key, value in out.items())


class QAE:
    def __init__(
        self,
        num_qubits: int,
        num_trash_qubits: int,
        ansatz: Optional[QuantumCircuit] = None,
        initial_params: Optional[Union[np.ndarray, List]] = None,
        optimizer: Optional[Optimizer] = None,
        shots: int = 1000,
        num_epochs: int = 100,
        save_training_curve: Optional[bool] = False,
        seed: int = 123,
        backend: Union[BaseBackend, Backend] = Aer.get_backend('qasm_simulator')
    ) -> None:
        """Quantum auto-encoder.

        Args:
            num_qubits: The number of qubits of the QAE circuit.
            num_trash_qubits: The number of trash qubits that should be measured in the end.
            ansatz: A parameterized quantum circuit ansatz to be optimized.
            initial_params: The initial list of parameters for the circuit ansatz
            optimizer: The optimizer used for training (default is SPSA)
            shots: The number of measurement shots when training and evaluating the QAE.
            num_epochs: The number of training iterations/epochs.
            save_training_curve: If True, the cost after each optimizer step is computed and stored.
            seed: Random number seed.
            backend: The backend on which the QAE is performed.
        """
        algorithm_globals.random_seed = seed
        np.random.seed(seed)

        self.costs = []
        if save_training_curve:
            callback = self._store_intermediate_result
        else:
            callback = None

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = SPSA(num_epochs, callback=callback)

        self.backend = backend

        if ansatz:
            self.ansatz = ansatz
        else:
            self.ansatz = QAEAnsatz(num_qubits, num_trash_qubits, measure_trash=True)

        if initial_params:
            self.initial_params = initial_params
        else:
            self.initial_params = np.random.uniform(0, 2*np.pi, self.ansatz.num_parameters_settable)

        self.shots = shots
        self.save_training_curve = save_training_curve

    def run(self, input_state: Optional[Any] = None, params: Optional[Union[np.ndarray, List]] = None):
        """Execute ansatz circuit and measure trash qubits

        Args:
            input_state: If provided, circuit is initialized accordingly
            params: If provided, list of optimization parameters for circuit

        Returns:
            measurement outcomes
        """
        if params is None:
            params = self.initial_params

        if input_state is not None:
            if type(input_state) == QuantumCircuit:
                circ = input_state
            elif type(input_state) == list or type(input_state) == np.ndarray:
                circ = QuantumCircuit(self.ansatz.num_qubits, self.ansatz.num_trash_qubits)
                circ.initialize(input_state)
            else:
                raise TypeError("input_state has to be an array or a QuantumCircuit.")
            circ = circ.compose(self.ansatz)
        else:
            circ = self.ansatz

        circ = circ.assign_parameters(params)

        job_sim = execute(circ, self.backend, shots=self.shots)
        return job_sim.result().get_counts(circ)

    def cost(self, input_state: Optional[Any] = None, params: Optional[Union[np.ndarray, List]] = None) -> float:
        """ Cost function

        Average Hamming distance of measurement outcomes to zero state.

        Args:
            input_state: If provided, circuit is initialized accordingly
            params: If provided, list of optimization parameters for circuit

        Returns:
            Cost
        """
        out = self.run(input_state, params)
        cost = hamming_distance(out)
        return cost/self.shots

    def _store_intermediate_result(self, eval_count, parameters, mean, std, ac):
        """Callback function to save intermediate costs during training."""
        self.costs.append(mean)

    def train(self, input_state: Optional[Any] = None):
        """ Trains the QAE using optimizer (default SPSA)

        Args:
            input_state: If provided, circuit is initialized accordingly

        Returns:
            Result of optimization: optimized parameters, cost, iterations
            Training curve: Cost function evaluated after each iteration
        """

        result = self.optimizer.optimize(
            num_vars=len(self.initial_params),
            objective_function=lambda params: self.cost(input_state, params),
            initial_point=self.initial_params
        )
        self.initial_params = result[0]

        return result, self.costs

    def reset(self):
        """Resets parameters to random values"""
        self.costs = []
        self.initial_params = np.random.uniform(0, 2*np.pi, self.ansatz.num_parameters_settable)


if __name__ == '__main__':
    num_qubits = 5
    num_trash_qubits = 2
    qae = QAE(num_qubits, num_trash_qubits, save_training_curve=True)

    # for demonstration purposes QAE is trained on a random state
    input_state = np.random.uniform(size=2**num_qubits)
    input_state /= np.linalg.norm(input_state)

    result, cost = qae.train(input_state)
