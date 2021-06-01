import time
import datetime
import numpy as np


import qiskit
from qiskit import *
from qiskit.opflow import X,Z,I
from qiskit.opflow.state_fns import StateFn, CircuitStateFn
from qiskit.providers.aer import StatevectorSimulator, AerSimulator
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA, SLSQP, SPSA
from qiskit.opflow import CircuitSampler


from qiskit.ignis.mitigation.measurement import CompleteMeasFitter # you will need to pip install qiskit-ignis
from qiskit.ignis.mitigation.measurement import complete_meas_cal

from scipy import sparse
import scipy.sparse.linalg.eigen.arpack as arp
from modules.utils import *

IBMQ.load_account() # this then automatically loads your saved account
provider = IBMQ.get_provider(hub='ibm-q-research')
device = provider.backend.ibmq_rome
print(device)
#backend = device
backend = qiskit.providers.aer.AerSimulator.from_backend(device)
coupling_map = device.configuration().coupling_map
noise_model = qiskit.providers.aer.noise.NoiseModel.from_backend(device)
basis_gates = noise_model.basis_gates
#aqua_globals.random_seed = seed
qi = qiskit.utils.QuantumInstance(backend=backend, # , seed_simulator=seed, seed_transpiler=seed
                         coupling_map=coupling_map, noise_model=noise_model,
                         measurement_error_mitigation_cls= CompleteMeasFitter, 
                         cals_matrix_refresh_period=30  #How often to refresh the calibration matrix in measurement mitigation. in minutes
                                 )

L = 5
num_trash = 2
anti = -1
logspace_size = 20
maxiter = 500
filename = f'data/noisy_VQE_maxiter-{maxiter}_Ising_L{L:.0f}_anti_{anti:.0f}_{logspace_size}.npz'
print(filename)
gx_vals = np.logspace(-2,2,logspace_size)
gz_vals = [0.]
gz = 0.



rotation_blocks = "ry"
entanglement_blocks = "cz"
entanglement = "sca"
reps = 1

ansatz_config = dict(rotation_blocks = rotation_blocks, entanglement_blocks = entanglement_blocks, reps = reps, entanglement = entanglement)

ansatz = qiskit.circuit.library.TwoLocal(L,rotation_blocks="ry", entanglement_blocks=entanglement_blocks, entanglement=entanglement, reps=reps)
#ansatz.draw("mpl")
ansatz = qiskit.transpile(ansatz, backend)
#ansatz.draw("mpl")


counts = []
values = []

optimizer = SPSA(maxiter=maxiter)

def store_intermediate_result(eval_count, parameters, mean, std):
    counts.append(eval_count)
    values.append(mean)

vqe = VQE(ansatz=ansatz, optimizer=optimizer, callback=store_intermediate_result, quantum_instance=qi)

opt_params = []
gx_list = []
gz_list = []
countss, valuess = [], []
Qmag, Qen, Smag, Sen = [None] * len(gx_vals),[None] * len(gx_vals),[None] * len(gx_vals),[None] * len(gx_vals)

for j,gx in enumerate(gx_vals):
    t0 = datetime.datetime.now()
    counts = []
    values = []

    H = QHIsing(L,anti,np.float32(gx),np.float32(gz))
    result = vqe.compute_minimum_eigenvalue(H, aux_operators = [QMag(L,anti)]) #ED with Qiskit VQE
    Qen[j] = result.eigenvalue
    Qmag[j] = result.aux_operator_eigenvalues[0][0]
    countss.append(counts)
    valuess.append(values)

    ED_state, ED_E, ham = ising_groundstate(L, anti, np.float32(gx), np.float32(gz))
    Sen[j] = ED_E
    Smag[j] = ED_state.T.conj()@Mag(L,anti)@ED_state
    print(Qen[j])
    print(Qmag[j])
    
    print(f"ED energy: {Sen[j]} ;; VQE energy: {Qen[j]} ;; diff {Sen[j] - Qen[j]}")
    print(f"ED mag: {Smag[j]} ;; VQE mag: {Qmag[j]} ;; diff {Smag[j] - Qmag[j]}")


    gx_list.append(gx)
    gz_list.append(gz)
    opt_params.append(sort_params(result.optimal_parameters))
    print(f"{j+1} / {len(opt_params)}, gx = {gx:.2f}, gz = {gz:.2f}, time : {(datetime.datetime.now() - t0)}")

np.savez(filename, gx_list=gx_list, gz_list=gz_list, opt_params=opt_params, Qmag=Qmag, Qen=Qen, Sen=Sen, Smag=Smag, ansatz_config=ansatz_config)