import scipy.sparse as sparse
import scipy.sparse.linalg.eigen.arpack as arp
import warnings
import numpy as np

import qiskit
from qiskit.opflow import X,Z,I
from qiskit.opflow.state_fns import StateFn, CircuitStateFn


# from: https://tenpy.readthedocs.io/en/latest/toycodes/tfi_exact.html

def ising_groundstate(L, Jz, gx, gz=0): # gx is transverse field, gz the longitudinal
    """For comparison: obtain ground state energy from exact diagonalization.

    Exponentially expensive in L, only works for small enough `L` <~ 20.
    """
    if L >= 20:
        warnings.warn("Large L: Exact diagonalization might take a long time!")
    # get single site operaors
    sx = sparse.csr_matrix(np.array([[0., 1.], [1., 0.]]))
    sz = sparse.csr_matrix(np.array([[1., 0.], [0., -1.]]))
    id = sparse.csr_matrix(np.eye(2))
    sx_list = []  # sx_list[i] = kron([id, id, ..., id, sx, id, .... id])
    sz_list = []
    for i_site in range(L):
        x_ops = [id] * L
        z_ops = [id] * L
        x_ops[i_site] = sx
        z_ops[i_site] = sz
        X = x_ops[0]
        Z = z_ops[0]
        for j in range(1, L):
            X = sparse.kron(X, x_ops[j], 'csr')
            Z = sparse.kron(Z, z_ops[j], 'csr')
        sx_list.append(X)
        sz_list.append(Z)
    H_zz = sparse.csr_matrix((2**L, 2**L))
    H_z = sparse.csr_matrix((2**L, 2**L))
    H_x = sparse.csr_matrix((2**L, 2**L))
    for i in range(L - 1):
        H_zz = H_zz + sz_list[i] * sz_list[(i + 1) % L]
    for i in range(L):
        H_z = H_z + sz_list[i]
        H_x = H_x + sx_list[i]
    H = -Jz * H_zz - gx * H_x - gz * H_z 
    E, V = arp.eigsh(H, k=1, which='SA', return_eigenvectors=True, ncv=20)
    return V[:,0], E[0], H


# For handling parameters from VQE in qiskit
def sort_vals(vals):
    """ vals is (unsorted) dictionary of parameters from VQE ansatz circuit, this returns sorted values as list """
    indices = np.array([_.index for _ in vals])           # unordered list of indices from the ParameterVectorElement(Theta(INDEX))
    vals_sorted = np.array([vals[_] for _ in vals])       # unordered list of values (but same ordering as indices)
    return vals_sorted[np.argsort(indices)]

def init_vqe(vals,L=6):
    # obviously this wont work anymore once we change the ansatz
    return qiskit.circuit.library.EfficientSU2(L, reps=3).assign_parameters(sort_vals(vals))


### Niccolo stuff
def QNKron(N,op1,op2,pos): 
    '''
    Tensor product operator (Qiskit Pauli operators)
    returns tensor product of op1,op2 on sites pos,pos+1 and identity on remaining sites

    N:number of sites
    op1,op2: Pauli operators on neighboring sites
    pos: site to insert op1
    '''
    temp=np.array([I]*(N))
    temp[pos]=op1 
    if pos!=(N-1):
        temp[pos+1]=op2
    mat=1
    for j in range(N):
        mat=mat^temp[j]
    return mat
def QHIsing(N,lam,p):
    '''
    Quantum Ising Hamiltonian (1D) with transverse field (Qiskit Pauli operators)
    
    N:number of sites 
    lam: transverse field)
    '''

    H=-QNKron(N,Z,Z,0)-lam*QNKron(N,X,I,0)-p*QNKron(N,Z,I,0)
    for i in range(1,N-1):
        H=H-QNKron(N,Z,Z,i)-lam*QNKron(N,X,I,i)-p*QNKron(N,Z,I,i)
    H=H-lam*QNKron(N,X,I,N-1)-p*QNKron(N,Z,I,N-1)
    return H

def NKron(N,op1,op2,pos): 
    '''
    Tensor product operator 
    returns tensor product of op1,op2 on sites pos,pos+1 and identity on remaining sites

    N:number of sites
    op1,op2: Pauli operators on neighboring sites
    pos: site to insert op1
    
    '''
    ide=np.eye(2)
    temp=np.array([ide]*(N),dtype=np.complex128)
    temp[pos,:,:]=op1 
    # if pos!=(N-1):
    temp[(pos+1)%N,:,:]=op2
    mat=1
    for j in range(N):
        mat=np.kron(mat,temp[j])
    return mat

def Mag(N): #magnetization operator (numpy array)
    sz=np.array([[1,0],[0,-1]])
    M=np.zeros((2**N,2**N))
    for i in range(N):
        M=M+NKron(N,sz,np.eye(2),i)
    return M/N

def QMag(N): #magnetization operator (Qiskit operator)
    M=QNKron(N,Z,I,0)
    for i in range(1,N):
        M=M+QNKron(N,Z,I,i)
    return M/N