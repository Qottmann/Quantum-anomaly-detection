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

from tenpy.networks.mps import MPS
from tenpy.models.hubbard import BoseHubbardChain
from tenpy.algorithms import dmrg
from tenpy.linalg import np_conserved

def DMRG_EBH(L, V, t_list, chi_max=30, bc_MPS='infinite'):

    model_params = dict(n_max=1, filling=0.5, bc_MPS=bc_MPS, t=t_list,
                        L=L, V=V, mu=0, conserve='N', verbose=0)
    M = BoseHubbardChain(model_params)
        
    vector=[]
    for i in range(M.lat.N_sites):
        if i%2:
            vector.append(1)
        else:
            vector.append(0)

    psi = MPS.from_product_state(M.lat.mps_sites(), vector, bc=M.lat.bc_MPS)    
        
    dmrg_params = {                                                                                             
        'mixer': True,                                                                                          
        'trunc_params': {                                                                                       
        'chi_max': chi_max,                                                                                                                                                                    
        },                                                                                                      
        'max_E_err': 1.e-16,                                                                                    
        'verbose': 0
    }
    
    info = dmrg.run(psi, M, dmrg_params)
    
    return info['E'], psi

chi = 50
V_list = np.logspace(-2,2,50)
deltat_list = np.linspace(-0.95,0.95,50)
L = 12
mu = 0
site = 0

psi_array = []

for deltat in deltat_list:
    
    print('deltat', deltat)
    t_list = np.ones(L-1)
    for i in range(len(t_list)):
        t_list[i] -= deltat*(-1)**i
    
    psi_0 = []
    for V in V_list:

        print('V', V)
        E0, psi0 = DMRG_EBH(L, V, t_list, chi_max=chi, bc_MPS='finite')
        psi_0 = np.append(psi_0, psi0)

    psi_array.append(psi_0)
    
    np.save('data/BH_MPS_L%.0f_logV_50x50.npy' %L, psi_array)
    
wf_array = []
V_array = []
deltat_array = []

for i, deltat in enumerate(deltat_list):
    for j, V in enumerate(V_list):
        
        psi = psi_array[i][j]
        wf = psi.get_theta(0,L).to_ndarray().reshape(-1)
        wf_array.append(wf)
        V_array.append(V)
        deltat_array.append(deltat)

np.savez(f'data/wf_BH_L%.0f_logV_50x50.npz' %(L), deltat_array=deltat_array, V_array=V_array, wf_array = wf_array)