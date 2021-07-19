import time
import itertools
import numpy as np

import qiskit
from qiskit import *
from qiskit.quantum_info import Statevector
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, SPSA
from qiskit.opflow.state_fns import StateFn, CircuitStateFn
from qiskit.providers.aer import StatevectorSimulator, AerSimulator

from qiskit.opflow import CircuitSampler

from qiskit.ignis.mitigation.measurement import CompleteMeasFitter # you will need to pip install qiskit-ignis
from qiskit.ignis.mitigation.measurement import complete_meas_cal

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
cmap = plt.get_cmap("plasma") #'viridis'

from modules.utils import *

from qae import *

import datetime

import tenpy
from tenpy.networks.mps import MPS
from tenpy.models.hubbard import BoseHubbardChain
from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms import dmrg
from tenpy.linalg import np_conserved

def DMRG_EBH(L, V, t_list, chi_max=30, bc_MPS='infinite'):

    model_params = dict(n_max=1, filling=0.5, bc_MPS=bc_MPS, t=t_list,
                        L=L, V=V, mu=0, conserve='N')
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
        #'verbose': 0
    }
    
    info = dmrg.run(psi, M, dmrg_params)
    
    return info['E'], psi

def DMRG_Ising(L, J, g, chi_max=30, bc_MPS='finite'):

    model_params = dict(bc_MPS=bc_MPS, bc_x="open",
                        L=L, J=J, g=g, conserve="best")
    M = TFIChain(model_params)

    product_state = ["up"] * M.lat.N_sites
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
        
    dmrg_params = {                                                                                             
        'mixer': True,                                                                                          
        'trunc_params': {                                                                                       
        'chi_max': chi_max,                                                                                                                                                                    
        },                                                                                                      
        'max_E_err': 1e-16,                                                                                    
        #'verbose': 0
    }
    
    info = dmrg.run(psi, M, dmrg_params)
    
    return info['E'], psi

### Preliminaries
qiskit_chi = 100
L = 8
num_trash = int(np.log(L)/np.log(2))
anti = 1 # 1 for ferromagnetic Ising model, -1 for antiferromagnet
g = 0.05
J=anti
filename = "data/QAEAnsatz_scaling_MPS_script"

backend = qiskit.providers.aer.AerSimulator(method="matrix_product_state",
                                            precision="single",
                                            matrix_product_state_max_bond_dimension = qiskit_chi,
                                            matrix_product_state_truncation_threshold = 1e-10,
                                            #mps_sample_measure_algorithm = "mps_apply_measure", #alt: "mps_probabilities" 
                                           )

from qiskit.utils import algorithm_globals
algorithm_globals.random_seed = 333

def qiskit_state(psi0):
    # G is only the local tensor (not multiplied by any singular values) - see https://tenpy.readthedocs.io/en/latest/reference/tenpy.networks.mps.html
    A_list = [psi0.get_B(i, form="G").to_ndarray().transpose([1,0,2]) for i in range(L)]
    for i,A in enumerate(A_list):
        A_list[i] = (A[0], A[1])
    S_list = [psi0.get_SR(i) for i in range(L-1)] # skip trivial last bond; hast to be of size L-1
    return (A_list, S_list)

def prepare_circuit(thetas, L=6, num_trash=2, init_state=None, measurement=True,vqe=False):
    # QAE ansatz
    QAE_circ = QAEAnsatz(num_qubits = L, num_trash_qubits= num_trash, trash_qubits_idxs = list(range(L//2-1,L//2-1+num_trash)), measure_trash=measurement).assign_parameters(thetas)
    # initialize state vector
    initcirc = QuantumCircuit(QuantumRegister(L,"q"),ClassicalRegister(num_trash, 'c'))
    if init_state != None:
        initcirc.set_matrix_product_state(qiskit_state(init_state))
    # compose circuits
    fullcirc = initcirc.compose(QAE_circ)
    return fullcirc

### Execute circuit
count = 0
def run_circuit(thetas, L, num_trash, init_state, vqe=False, shots=100, meas_fitter = None):
    #global count
    #count += 1
    #print(count, "thetas: ", thetas)
    #print(L, num_trash)
    circ = prepare_circuit(thetas, L, num_trash, init_state, vqe=vqe)
    #circ.draw("mpl")
    #tcirc = qiskit.transpile(circ, backend)
    # Execute the circuit 
    job_sim = backend.run(circ, shots=shots, seed_simulator=333, seed_transpiler=444) #fix seed to make it reproducible
    result = job_sim.result()
    # Results without mitigation
    counts = result.get_counts()
    if meas_fitter != None:
        # Get the filter object
        meas_filter = meas_fitter.filter

        # Results with mitigation
        mitigated_results = meas_filter.apply(result)
        counts = mitigated_results.get_counts(0)
    return counts

def count_ones(string):
    return np.sum([int(_) for _ in string])

### Optimize circuit
def cost_function_single(thetas, L, num_trash, init_state, shots=1000, vqe=True, param_encoding=False, x=0, meas_fitter=None):
    """ Optimizes circuit """
    if param_encoding: thetas = feature_encoding(thetas, x) 
    out = run_circuit(thetas, L, num_trash, init_state, vqe=vqe, shots=shots, meas_fitter=meas_fitter)
    cost = np.sum([out[_]*count_ones(_) for _ in out if _ != "0" * num_trash]) # all measurement results except "000"
    return cost/shots

def cost_function(thetas, L, num_trash, init_states, shots=1000, vqe=True, param_encoding=False, x=0, meas_fitter=None):
    """ Optimizes circuit """
    cost = 0.
    for init_state in init_states:
        cost += cost_function_single(thetas, L, num_trash, init_state, shots, vqe, param_encoding, meas_fitter=meas_fitter)
    return cost/len(init_states)

def optimize(init_states, L=6, num_trash=2, thetas=None, shots=1000, max_iter=400, vqe=True, param_encoding=False, x=0, pick_optimizer = None,
            meas_fitter=None):
    if thetas is None:
        n_params = (num_trash*L+num_trasg)*2 if param_encoding else (num_trash*L+num_trash)
        thetas = np.random.uniform(0, 2*np.pi, n_params) # initial parameters without feature encoding
        
    #print("Initial cost: {:.3f}".format(cost_function(thetas, L, num_trash, init_states, shots, vqe, param_encoding, x)))
    
    counts, values, accepted = [], [], []
    def store_intermediate_result(eval_count, parameters, mean, std, ac):
        # counts.append(eval_count)
        values.append(mean)
        accepted.append(ac)

    # Initialize optimizer
    if pick_optimizer == "cobyla":
        optimizer = COBYLA(maxiter=max_iter, tol=0.0001)
    if pick_optimizer == "adam" or pick_optimizer == "ADAM":
        optimizer = qiskit.algorithms.optimizers.ADAM(maxiter=max_iter)
    # optimizer = L_BFGS_B(maxfun=300, maxiter=max_iter)#, factr=10, iprint=- 1, epsilon=1e-08)
    if pick_optimizer == "spsa" or pick_optimizer == None:
        optimizer = SPSA(maxiter=max_iter,
                         #blocking=True,
                         callback=store_intermediate_result,
                         #learning_rate=0.3,
                         #perturbation=0.1
                         ) # recommended from qiskit (first iteraction takes quite long)
                           # to reduce time figure out optimal learning rate and perturbation in advance
    start_time = time.time()
    ret = optimizer.optimize(
                            num_vars=len(thetas),
                            objective_function=(lambda thetas: cost_function(thetas, L, num_trash, init_states, shots, vqe, param_encoding, x, meas_fitter=meas_fitter)),
                            initial_point=thetas
                            )
    print("Time: {:.5f} sec".format(time.time()-start_time))
    print(ret)
    return ret[0], values, accepted

Ls = [12,12,12,12] # [3,4,8,10,12,14,16,20, 32, 32, 32]
max_iter = [400] * len(Ls)
num_trashs = np.log(Ls)/np.log(2)
num_trashs = np.array(num_trashs, dtype="int")

losses = [None] * len(Ls); accepted = [None] * len(Ls); thetas_opt= [None] * len(Ls)
times = [None] * len(Ls)

for j,L in enumerate(Ls):
    L = Ls[j]
    num_trash = num_trashs[j]
    
    # BH
    V = 1
    deltat=-1
    chi = 100
    
    g=0.05 # ordered phase to make things a bit more interesting
    print(f"bond dimension {chi}, max_iter {max_iter[j]}, L {L}, num_trash {num_trash}")

    t_list = np.ones(L-1)
    for i in range(len(t_list)):
        t_list[i] -= deltat*(-1)**i
    #E0, psi0 = DMRG_EBH(L, V, t_list, chi_max=chi, bc_MPS='finite')
    E0, psi0 = DMRG_Ising(L, J, g, chi_max=chi, bc_MPS='finite')
    SZ = psi0.expectation_value("Sigmaz")
    print(f"Psi magnetization: {SZ}") # note that in tenpy Sz and Sx roles are reversed, i.e. in tenpy its -SxSx - g Sz
    t0 = time.time()
    thetas_opt[j], losses[j], accepted[j] = optimize([psi0], max_iter=max_iter[j], L=L, num_trash=num_trash, meas_fitter=None) #, pick_optimizer="adam")
    times[j] = time.time() - t0
    loss_final = cost_function_single(thetas_opt[j], L, num_trash, psi0, shots=1000)
    print(f"opt loss = {loss_final}, min loss = {np.min(losses[j])}, computation time = {times[j]} sec")
    print(f"losses vs steps: ", losses[j]) # to save computation time, dont save intermediate results and just compute the loss in the end

np.savez(filename + "_results", thetas_opt=thetas_opt, losses=losses, accepted=accepted,Ls=Ls, max_iter=max_iter, times=times)