import time
import numpy as np
from scipy.optimize import minimize, basinhopping
from qiskit import *
from qiskit.quantum_info import Statevector
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, SPSA
import matplotlib.pyplot as plt
from modules.utils import *
#%matplotlib inline

### Preliminaries
L = 6

filename = f'params_VQE_ising_N{L}.npy'   # name of the data file that is used
name = f"000_L-{L}_data-set-{filename}"    # name of the data produced by this notebook

##############################################################################
### I - data  ################################################################
##############################################################################
# We load the data from VQE and compare it to exact diagonalization results to
# make sure what we are doing makes sense

VQE_vals = np.load(filename, allow_pickle=True).item()
lambdas = np.array([_ for _ in VQE_vals]) # list of lambda values (the items in the dictionary)
mag = QMag(L) #magnetization operator (Qiskit)
Smag = Mag(L) #magnetization operator (numpy)

Qen=np.zeros(len(lambdas)); Sen=np.zeros(len(lambdas)) #energies
Qmags=np.zeros(len(lambdas)); Smags=np.zeros(len(lambdas)) #magnetizations

for j,lambda0 in enumerate(lambdas):
    print(lambda0)
    H = QHIsing(L,float(lambda0),1e-4) # build Hamiltonian Op
    state = init_vqe(VQE_vals[lambda0])
    StateFn(state)   
    meas_outcome = ~StateFn(mag) @ StateFn(state)
    Qmags[j]=meas_outcome.eval().real
    e_outcome = ~StateFn(H) @ StateFn(state)
    Qen[j]=e_outcome.eval().real
    
    init_state, E, ham = ising_groundstate(L, 1., float(lambda0),1e-4)
    Sen[j]=E
    Smags[j]=np.real(init_state.T.conj()@Smag@init_state) #Magnetization with Numpy results


fig, axs = plt.subplots(ncols=2,figsize=(10,5))
ax = axs[0]
ax.plot(lambdas,Sen,".--",label="sparse ED")
ax.plot(lambdas,Qen,"x--", label="Qiskit")
ax.set_xscale("log")
ax.set_ylabel("GS energy")
ax.set_xlabel("Mag. field")
ax.legend()

ax = axs[1]
ax.plot(lambdas,Smags,".--",label="sparse ED")
ax.plot(lambdas,Qmags,"x--", label="Qiskit")
ax.set_xscale("log")
ax.set_ylabel("Magnetization")
ax.set_xlabel("Mag. field")
ax.legend()
plt.grid()
plt.savefig("plots/VQE-comparison/" + name + ".png")
plt.close()
print(Sen,Qen)




##############################################################################
### II - Training  ###########################################################
##############################################################################


# Use Aer's qasm_simulator
backend_sim = Aer.get_backend('qasm_simulator')


thetas = np.random.uniform(0, 2*np.pi, 2*L+2) # initial parameters without feature encoding
# thetas = np.random.uniform(0, 2*np.pi, (2*L+2, 2)) # initial parameters with feature encoding

def prepare_circuit(init_state=None, measurement=True):
    qreg = QuantumRegister(L, 'q')
    creg = ClassicalRegister(2, 'c')
    circ = QuantumCircuit(qreg, creg)
    entangler_map1 = [(5, 4), (5, 3), (5, 1), (4, 2), (4, 0)]
    entangler_map2 = [(5, 4), (5, 2), (4, 3), (5, 0), (4, 1)]
    circ += circuit.library.TwoLocal(L, 'ry', 'cz', entanglement = [entangler_map1, entangler_map2], reps=2, insert_barriers=True, skip_final_rotation_layer=True)
    circ.ry(circuit.Parameter('θ1'), 4)
    circ.ry(circuit.Parameter('θ2'), 5)
    if measurement:
        circ.measure(qreg[4], creg[0])
        circ.measure(qreg[5], creg[1])
    if init_state is not None: 
        circ = init_vqe(init_state) + circ
    
    return circ

def prepare_circuit2(thetas, init_state=None, measurement=True):
    # need this function for the explicit theta dependence
    qreg = QuantumRegister(L, 'q')
    creg = ClassicalRegister(2, 'c')
    circ = QuantumCircuit(qreg, creg)
    #if init_state is not None: circ.initialize(init_state, qreg)
    for i,t in enumerate(thetas[:L]):
        circ.ry(t, i)
    circ.cz(5,4)
    circ.cz(5,3)
    circ.cz(5,1)
    circ.cz(4,2)
    circ.cz(4,0)
    for i,t in enumerate(thetas[L:2*L]):
        circ.ry(t, i)
    circ.cz(5,4)
    circ.cz(5,2)
    circ.cz(4,3)
    circ.cz(5,0)
    circ.cz(4,1)
    circ.ry(thetas[2*L], 4)
    circ.ry(thetas[2*L+1], 5)
    if measurement:
        circ.measure(qreg[4], creg[0])
        circ.measure(qreg[5], creg[1])
    if init_state is not None: 
        circ = init_vqe(init_state) + circ
    return circ

def feature_encoding(thetas, x):
    """ thetas: parameters to be optimized, x: Ising model parameter (eg. field) """
    new_thetas = []
    thetas = thetas.reshape((-1,2))
    for theta in thetas:
        new_thetas.append(theta[0] * x + theta[1])
    return new_thetas

circ = prepare_circuit()
t_qc = transpile(circ, backend=backend_sim)


### Execute circuit
# Circuit is executed on simulator and measurement outcomes on the trash qubits are stored
J, g = 1., np.float32(lambdas[3]) # Ising parameters for which ground state should be compressed
init_state = VQE_vals[lambdas[3]] # train on smallest lambda ;; this may grammatically be confusing, init_state = dictionary with unsorted parameters
print(lambdas[3])


def run_circuit(thetas, init_state, shots=100):
    circ = prepare_circuit2(thetas, init_state)

    # Execute the circuit on the qasm simulator.
    job_sim = execute(circ, backend_sim, shots=shots)#, memory=True)

    # Grab the results from the job.
    result_sim = job_sim.result()

    counts = result_sim.get_counts(circ)
#     print(counts)
    
#     mems = result_sim.get_memory(circ)
#     print(mems)
    return counts


### Optimize circuit
#Define cost function (averaged hamming distance of measurement outcomes) and minimze it using either scipy or qiskit optimizer modules (the latter is also based on scipy though).

circ = prepare_circuit(init_state)
t_qc = transpile(circ, backend=backend_sim)

def cost_function_single(thetas):
    """ Optimizes circuit for single input state """
    shots = 1000 # Number of measurements for single training example
    cost = 0.
    out = run_circuit(thetas, init_state, shots=shots)
    cost = out.get('11', 0)*2 + out.get('01', 0) + out.get('10', 0)
    return cost/shots

thetas = np.random.uniform(0, 2*np.pi, 2*L+2) # initial parameters without feature encoding
# thetas = np.random.uniform(0, 2*np.pi, ((2*L+2)*2)) # initial parameters with feature encoding

# counts, values = [], []
# def store_intermediate_result(eval_count, parameters, mean, std):
#     counts.append(eval_count)
#     values.append(mean)

# Initialize optimizer
optimizer = COBYLA(maxiter=500, tol=0.0001)
# optimizer = L_BFGS_B(maxfun=300, maxiter=500)#, factr=10, iprint=- 1, epsilon=1e-08)
# optimizer = SPSA(maxiter=1)#, callback=store_intermediate_result) # recommended from qiskit but terribly slow

start_time = time.time()
# ret = optimizer.optimize(num_vars=thetas.shape[0], objective_function=(lambda thetas: cost_function_single(thetas, t_qc)), initial_point=thetas)
ret = optimizer.optimize(num_vars=thetas.shape[0], objective_function=cost_function_single, initial_point=thetas)
print("{optimizer.__name__} Optimization Time: ", time.time()-start_time)

thetas_opt = ret[0]


##############################################################################
### III - Inference  #########################################################
##############################################################################

### Test circuit
#Test optimized circuit on different Ising parameter values and (hopefully) observe phase transition

cost = []
shots = 10000
for g in lambdas:
    out = run_circuit(thetas_opt, VQE_vals[g], shots=shots)
    cost.append((out.get('11', 0)*2 + out.get('01', 0) + out.get('10', 0))/shots)


plt.plot(np.array(lambdas,dtype=np.float32), cost,"x--")
plt.xlabel(r"$g$")
plt.ylabel("Cost")
plt.xscale("log")
plt.savefig("plots/" + name + "_result.png")

#lambdas