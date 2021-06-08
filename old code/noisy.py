# -*- coding: utf-8 -*-
"""
Created on Mon May 31 22:57:39 2021

@author: nbaldelli
"""

import time
import itertools
import numpy as np
from scipy.optimize import minimize, basinhopping
from qiskit import *
from qiskit.quantum_info import Statevector
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, SPSA, SLSQP
from qiskit.algorithms import VQE
from qiskit.opflow.state_fns import StateFn, CircuitStateFn
from qiskit.opflow import CircuitSampler
from qiskit.providers.aer import StatevectorSimulator, AerSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.utils import QuantumInstance
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter,complete_meas_cal
from modules.utils import *
import qiskit
from qiskit import IBMQ
#%%

IBMQ.load_account() # Load account from disk
IBMQ.providers()    # List all available providers
provider = IBMQ.get_provider(hub='ibm-q')
print(provider.backends)
real_backend = provider.backends(simulator=False, operational=True)[6]


L=5
mag = QMag(L,-1) #magnetization operator (Qiskit)
~StateFn(mag)
gx_vals = np.logspace(-2,2,10)[::-1]
opt_params_noiseless=[]
opt_params_noisy=[]
Qmags=np.zeros(len(gx_vals))
load=False

    
##############################################################################
#NOISY SIMULATION
backend = qiskit.providers.aer.AerSimulator.from_backend(real_backend)
coupling_map = backend.configuration().coupling_map
noise_model = NoiseModel.from_backend(backend)
optimizer = SPSA(maxiter=1000)
reps=1
ansatz = qiskit.circuit.library.EfficientSU2(L, reps=reps)
# ansatz = qiskit.circuit.library.TwoLocal(L,rotation_blocks="ry", entanglement_blocks=entanglement_blocks, entanglement=entanglement, reps=reps)
ansatz.draw("mpl")
q_instance = QuantumInstance(backend, coupling_map=coupling_map, noise_model=noise_model, measurement_error_mitigation_cls=CompleteMeasFitter, cals_matrix_refresh_period=30 )
gz=1e-3
filename = f'data/params_VQE_Ising_L{L:.0f}_anti_N_reps{reps}_gz{gz}.npz'

##%%
if load==False: 
    for i,gx in enumerate(gx_vals):

        print('gx: %.2f' %(gx))

        if i != 0:
            vqe = VQE(ansatz=ansatz, initial_point = opt_params_noisy[-1], optimizer=optimizer, quantum_instance=q_instance)
        else:
            vqe = VQE(ansatz=ansatz, optimizer=optimizer, quantum_instance=q_instance)
        
        H = QHIsing(L,-1,np.float32(gx),np.float32(gz))
        result = vqe.compute_minimum_eigenvalue(H) #ED with Qiskit VQE

        opt_params_noisy.append(sort_params(result.optimal_parameters))

    np.savez(filename, opt_params=opt_params_noisy)

