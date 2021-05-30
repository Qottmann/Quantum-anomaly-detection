import time
import itertools
import numpy as np
from scipy.optimize import minimize, basinhopping
from qiskit import *
from qiskit.quantum_info import Statevector
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, SPSA
from qiskit.utils import algorithm_globals


num_trashs = [2,4][:]
seeds = np.arange(20) + 10
idxs = [(2,5),(10,10),(48,5),(25,47)][3:]
max_iter = 3000 #!!!!!

a = [num_trashs, idxs, seeds] #10
b = list(itertools.product(*a))
print(len(b))


if len(sys.argv) > 1:
    job_id = int(sys.argv[1])
else:
    job_id = 0
c = b[job_id]
print(job_id)
print(c)


backend_sim = Aer.get_backend('qasm_simulator')

seed = c[2]
L = 12
num_trash = c[0]
N = 50
d_idx = c[1][0]
v_idx = c[1][1]
shots = 1000
vqe = False
#filename = f'data/wf_BH_L{L}.npz'
#filename = '../wf_BH_L20.npz'
#filename = 'data/wf_BH_L10_logV.npz'
filename = f'data/wf_BH_L{L}_logV_50x50.npz'

if num_trash == 2:
    max_iter = 3000
elif num_trash == 4:
    max_iter = 5000


VQE_vals = np.load(filename, allow_pickle=True)
deltas = VQE_vals['deltat_array']
Vs = VQE_vals['V_array']
init_states = VQE_vals['wf_array']
# Vs = np.linspace(2,8,10)
# Vs = np.linspace(0,1,10)
Vs = np.logspace(-2,2,N)
# deltas = np.linspace(-1,1,10)
# deltas = np.linspace(-2,2,10)
deltas = np.linspace(-0.95,0.95,N)
init_states = init_states.reshape(N,N,-1) #(delta, V)


algorithm_globals.random_seed = seed
np.random.seed(seed)
init_statess = [init_states[d_idx,v_idx]]

def init_vqe(vals, L):
    #return qiskit.circuit.library.EfficientSU2(L, reps=3).assign_parameters(sort_vals(vals))
    return qiskit.circuit.library.EfficientSU2(L, reps=3).assign_parameters(vals)


# linear entangler (as in scales linearly with trash qubits)
def get_entangler_map(L, num_trash, i_permut=1):
    result = []
    nums = list(range(L)) # here was the problem, it doesnt like when list elements are taken from numpy
    nums_compressed = nums.copy()[:L-num_trash]
    nums_trash = nums.copy()[-num_trash:]
    #print(nums, nums_compressed, nums_trash)
    # combine all trash qubits with themselves
    for trash_q in nums_trash[:-1]:
        result.append((trash_q+1,trash_q))
    # combine each of the trash qubits with every n-th
    repeated = list(nums_trash) * (L-num_trash) # repeat the list of trash indices cyclicly
    for i in range(L-num_trash):
        result.append((repeated[i_permut + i], nums_compressed[i]))
    return result

def QAE_Ansatz(thetas, L, num_trash, insert_barriers=False, parametrized_gate = "ry", entangling_gate = "cz"):
    entanglement = [get_entangler_map(L,num_trash,i_permut) for i_permut in range(num_trash)]
    circ = qiskit.circuit.library.TwoLocal(L,
                                           parametrized_gate,
                                           entangling_gate,
                                           entanglement,
                                           reps=num_trash,
                                           insert_barriers=insert_barriers,
                                           skip_final_rotation_layer=True
                                          ).assign_parameters(thetas[:-num_trash])
    if insert_barriers: circ.barrier()
    for i in range(num_trash):
        circ.ry(thetas[L-i-1], L-i-1)
        #circ.ry(circuit.Parameter(f'θ{i}'), L-i-1)
    return circ

def prepare_circuit3(thetas, L=6, num_trash=2, init_state=None, measurement=True, vqe=True):
    qreg = QuantumRegister(L, 'q')
    creg = ClassicalRegister(num_trash, 'c')
    circ = QuantumCircuit(qreg, creg)
    if init_state is not None:
        if not vqe:
            circ.initialize(init_state, qreg)
    circ += QAE_Ansatz(thetas, L, num_trash, insert_barriers=True)#.assign_parameters(thetas) # difference to bind?
    if measurement:
        for i in range(num_trash):
            circ.measure(qreg[L-i-1], creg[i])
    if init_state is not None:
        if vqe:
            circ = init_vqe(init_state, L) + circ
    return circ

def run_circuit(thetas, L, num_trash, init_state, vqe=True, shots=100):
    circ = prepare_circuit3(thetas, L, num_trash, init_state, vqe=vqe)

    # Execute the circuit on the qasm simulator.
    job_sim = execute(circ, backend_sim, shots=shots)#, seed_simulator=123, seed_transpiler=234)

    # Grab the results from the job.
    result_sim = job_sim.result()
    return result_sim.get_counts(circ)

def hamming_distance(out):
    return sum(key.count('1') * value for key, value in out.items())

def cost_function_single(thetas, L, num_trash, p, shots=1000, vqe=True, param_encoding=False, x=0, model="bh"):
    """ Optimizes circuit """
    if vqe:
        init_state = phis[p]
    else:
        if model=="ising":
            J, gx, gz = p
            init_state, _ = ising_groundstate(L, J, gx, gz)
        elif model=="bh":
            init_state = p#init_states[p]
    if param_encoding: thetas = feature_encoding(thetas, x)
    out = run_circuit(thetas, L, num_trash, init_state, vqe=vqe, shots=shots)
    cost = hamming_distance(out)
    return cost/shots

def cost_function(thetas, L, num_trash, ising_params, shots=1000, vqe=True, param_encoding=False, x=0):
    """ Optimizes circuit """
    cost = 0.
    n_samples = len(ising_params)
    for i, p in enumerate(ising_params):
        if param_encoding:
            cost += cost_function_single(thetas, L, num_trash, p, shots, vqe, param_encoding, x[i])
        else:
            cost += cost_function_single(thetas, L, num_trash, p, shots, vqe, param_encoding)
    return cost/n_samples

def optimize(ising_params, L=6, num_trash=2, thetas=None, shots=1000, max_iter=500, vqe=True, param_encoding=False, x=0):
    if thetas is None:
        n_params = (num_trash*L+num_trash)*2 if param_encoding else (num_trash*L+num_trash)
        thetas = np.random.uniform(0, 2*np.pi, n_params) # initial parameters without feature encoding

    print("Initial cost: {:.3f}".format(cost_function(thetas, L, num_trash, ising_params, shots, vqe, param_encoding, x)))

    counts, values, accepted = [], [], []
    def store_intermediate_result(eval_count, parameters, mean, std, ac):
        # counts.append(eval_count)
        values.append(mean)
        accepted.append(ac)

    # Initialize optimizer
    # optimizer = COBYLA(maxiter=max_iter, tol=0.0001)
    # optimizer = L_BFGS_B(maxfun=300, maxiter=max_iter)#, factr=10, iprint=- 1, epsilon=1e-08)
    optimizer = SPSA(maxiter=max_iter,
                     #blocking=True,
                     callback=store_intermediate_result,
                     #learning_rate=1e-1,
                     #perturbation=0.4
                     ) # recommended from qiskit (first iteraction takes quite long)
                       # to reduce time figure out optimal learning rate and perturbation in advance

    start_time = time.time()
    ret = optimizer.optimize(
                            num_vars=len(thetas),
                            objective_function=(lambda thetas: cost_function(thetas, L, num_trash, ising_params, shots, vqe, param_encoding, x)),
                            initial_point=thetas
                            )
    print("Time: {:.5f} sec".format(time.time()-start_time))
    print(ret)
    return ret[0], values, accepted

thetas, loss, accepted = optimize(init_statess, L, num_trash, max_iter=max_iter, vqe=vqe)

save_name = f'_bh_L{L}_trash{num_trash}_d{d_idx}_v{v_idx}_seed{seed}.npy'
np.save(f'data_rike/thetas' + save_name, thetas)
np.save(f'data_rike/loss' + save_name, loss)

# thetas = np.load(f'data_rike/thetas_bh_L{L}_trash{num_trash}_d{d_idx}_v{v_idx}_seed{seed}.npy', allow_pickle=True)


cost = np.zeros((N,N))
for i,d in enumerate(deltas):
    for j,v in enumerate(Vs):
        cost[i,j] = cost_function_single(thetas, L, num_trash, init_states[i,j], shots=shots, vqe=vqe)
np.save(f'data_rike/cost' + save_name, cost)
