import time
import numpy as np
import qiskit
from qiskit.opflow import X,Z,I
from qiskit.opflow.state_fns import StateFn, CircuitStateFn
from qiskit.providers.aer import StatevectorSimulator, AerSimulator
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA, SLSQP, SPSA
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse.linalg.eigen.arpack as arp
from modules.utils import *

%matplotlib inline

for anti in [1., -1.]:

    L = 6

    gx_vals = np.logspace(-2,2,10)
    gz_vals = np.logspace(-2,2,10)

    backend = StatevectorSimulator()
    optimizer = SLSQP(maxiter=1000)
    ansatz = qiskit.circuit.library.EfficientSU2(L, reps=3)

    #backend= AerSimulator() 
    #optimizer = SPSA(maxiter=1000)
    #ansatz = qiskit.circuit.library.EfficientSU2(L, reps=3)

    vqe = VQE(ansatz, optimizer, quantum_instance=backend) 

    opt_params = []
    gx_list = []
    gz_list = []

    for i,gx in enumerate(gx_vals):
        for j,gz in enumerate(gz_vals):

            print('gx: %.2f, gz: %.2f' %(gx,gz))

            H = QHIsing(L,anti,np.float32(gx),np.float32(gz))
            result = vqe.compute_minimum_eigenvalue(H) #ED with Qiskit VQE

            gx_list.append(gx)
            gz_list.append(gz)
            opt_params.append(sort_params(result.optimal_parameters))

    np.savez(f'data/params_VQE_Ising_L%.0f_anti_%.0f.npz' %(L,anti), gx_list=gx_list, gz_list=gz_list, opt_params=opt_params)