"""A quantum auto-encoder (QAE)."""

from typing import Union, Optional, List, Tuple, Callable, Any
import numpy as np

from qiskit import *

from qiskit.circuit.library import TwoLocal
from qiskit.algorithms.optimizers import *
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates import RYGate, CZGate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.utils import algorithm_globals
from qiskit.providers import BaseBackend, Backend

class QAEAnsatz(TwoLocal):
    def __init__(
        self,
        num_qubits: int,
        num_trash_qubits: int,
        nums_trash:Union[np.ndarray, List]=[1,2],
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
            nums_trash: The explicit indices of the trash qubits
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
        self.nums_trash = nums_trash
        self.measure_trash = measure_trash
        entanglement = [QAEAnsatz._generate_entangler_map(num_qubits, num_trash_qubits, i, nums_trash) for i in range(num_trash_qubits)]

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
    def _generate_entangler_map(num_qubits:int, num_trash_qubits:int, i_permut:int=1, nums_trash:Union[np.ndarray, List]=[1,2]) -> List[Tuple[int, int]]:
        """Generates entanglement map for QAE circuit

        Entangling gates are only added between trash and non-trash-qubits.

        Args:
            num_qubits: The number of qubits of the QAE circuit.
            num_trash_qubits: The number of trash qubits that should be measured in the end.
            i_permut: Permutation index; increases for every layer of the circuit
            nums_trash: which qubits should be the trash qubits

        Returns:
            entanglement map: List of pairs of qubit indices that should be entangled
        """
        result = []
        nums_compressed = list(range(num_qubits))
        for trashqubit in nums_trash:
            nums_compressed.remove(trashqubit)
        if nums_trash == None: #old way  
            nums_compressed = list(range(num_qubits))[:L-num_trash]
            nums_trash = list(range(num_qubits))[-num_trash:]

        # combine all trash qubits with themselves
        for trash_q in nums_trash[:-1]:
            result.append((trash_q+1,trash_q))
        # combine each of the trash qubits with every n-th
        repeated = list(nums_trash) * (num_qubits-num_trash_qubits) # repeat the list of trash indices cyclicly
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

            block_indices = [[i] for i in self.nums_trash]

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
            for i,j in enumerate(self.nums_trash):
                self.measure(self.qregs[0][j], self.cregs[0][i])

    @property
    def num_parameters_settable(self) -> int:
        """The number of total parameters that can be set to distinct values.

        Returns:
            The number of parameters originally available in the circuit.
        """
        return super().num_parameters_settable + self.num_trash_qubits


class QAE:
    def __init__(
        self,
        num_qubits: int,
        num_trash_qubits: int,
        ansatz: Optional[QuantumCircuit] = None,
        initial_params: Optional[Union[np.ndarray, List]] = None,
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
            initial_params: The initial list of parameters for the circuit ansatz.
            shots: The number of measurement shots when training and evaluating the QAE.
            num_epochs: The number of training iterations/epochs.
            save_training_curve: If True, the cost after each optimizer step is computed and stored.
            seed: Random number seed.
            backend: The backend on which the QAE is performed.
        """
        algorithm_globals.random_seed = seed
        np.random.seed(seed)

        self.backend = backend

        if ansatz:
            self.ansatz = ansatz
        else:
            self.ansatz = QAEAnsatz(num_qubits, num_trash_qubits, measure_trash=False)

        if initial_params:
            self.initial_params = initial_params
        else:
            self.initial_params = np.random.uniform(0, 2*np.pi, self.ansatz.num_parameters_settable)

        self.shots = shots
        self.num_epochs = num_epochs
        self.save_training_curve = save_training_curve


    def run(self, input_state: Optional[Any] = None, params: Optional[Union[np.ndarray, List]]=None):
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
            self.ansatz.initialize(input_state, self.ansatz.qregs[0])

        circ = self.ansatz.assign_parameters(params)

        for i in range(circ.num_trash_qubits):
                circ.measure(circ.qregs[0][circ.num_qubits-i-1], circ.cregs[0][i])

        job_sim = execute(circ, self.backend, shots=self.shots)
        return job_sim.result().get_counts(circ)


    def cost(self, input_state: Optional[Any] = None, params: Optional[Union[np.ndarray, List]]=None) -> float:
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

    def train(self, input_state: Optional[Any] = None):
        """ Trains the QAE using SPSA

        Args:
            input_state: If provided, circuit is initialized accordingly

        Returns:
            Result of optimization: optimized parameters, cost, iterations
            Training curve: Cost function evaluated after each iteration
        """
        costs = []
        def store_intermediate_result(eval_count, parameters, mean, std, ac):
            costs.append(mean)

        if self.save_training_curve:
            callback = store_intermediate_result
        else:
            callback = None
            costs = None


        optimizer = SPSA(maxiter=self.num_epochs,
                         blocking=False,
                         callback=callback,
                         learning_rate=None,
                         perturbation=None
                         )

        result = optimizer.optimize(
                                num_vars=len(self.initial_params),
                                objective_function = lambda params: self.cost(input_state, params),
                                initial_point=self.initial_params
                                )
        self.initial_params = result[0]
        return result, costs
