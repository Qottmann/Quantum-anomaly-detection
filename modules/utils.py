import scipy.sparse as sparse
import scipy.sparse.linalg.eigen.arpack as arp
import warnings
import numpy as np

import qiskit
from qiskit.opflow import X,Z,I
from qiskit.opflow.state_fns import StateFn, CircuitStateFn


# from: https://tenpy.readthedocs.io/en/latest/toycodes/tfi_exact.html

def ising_groundstate(L, J, gx, gz):
    """For comparison: obtain ground state energy from exact diagonalization.
    Exponentially expensive in L, only works for small enough `L` <~ 20.
    
    L:number of sites 
    J: hopping
    gx: transverse field
    gz: longitudinal field
    '''
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
    H_x = sparse.csr_matrix((2**L, 2**L))
    for i in range(L - 1):
        H_zz = H_zz + J*sz_list[i] * sz_list[(i + 1) % L]
    for i in range(L):
        H_x = H_x + gx*sx_list[i] +gz*sz_list[i]
    H = - H_zz - H_x - 1e-4*sz_list[0] #the last term breaks spin inversion symmetry in the ordered phase of AF Ising
    E, V = arp.eigsh(H, k=1, which='SA', return_eigenvectors=True, ncv=20)
    return V[:,0], E[0], H

def sort_params(vals):
    """ vals is (unsorted) dictionary of parameters from VQE ansatz circuit, this returns sorted values as list """
    indices = np.array([_.index for _ in vals])           # unordered list of indices from the ParameterVectorElement(Theta(INDEX))
    vals_sorted = np.array([vals[_] for _ in vals])       # unordered list of values (but same ordering as indices)
    return vals_sorted[np.argsort(indices)]

def init_vqe(vals, reps=3, L=6):
    # obviously this wont work anymore once we change the ansatz
    return qiskit.circuit.library.EfficientSU2(L, reps=reps).assign_parameters(vals)

def QHIsing(L,J,gx,gz):
    '''
    Quantum Ising Hamiltonian (1D) with transverse & longitudinal field (Qiskit Pauli operators)
    
    L:number of sites 
    J: hopping
    gx: transverse field
    gz: longitudinal field
    '''

    H=-J*QNKron(L,Z,Z,0)-gx*QNKron(L,X,I,0)-(gz+1e-4)*QNKron(L,Z,I,0) #the last term breaks spin inversion symmetry in the ordered phase of AF Ising
    for i in range(1,L-1):
        H=H-J*QNKron(L,Z,Z,i)-gx*QNKron(L,X,I,i)-gz*QNKron(L,Z,I,i)
    H=H-gx*QNKron(L,X,I,L-1)-gz*QNKron(L,Z,I,L-1)
    return H

def QNKron(L,op1,op2,pos): 
    '''
    Tensor product operator (Qiskit Pauli operators)
    returns tensor product of op1,op2 on sites pos,pos+1 and identity on remaining sites

    L:number of sites
    op1,op2: Pauli operators on neighboring sites
    pos: site to insert op1
    '''
    temp=np.array([I]*(L))
    temp[pos]=op1 
    if pos!=(L-1):
        temp[pos+1]=op2
    mat=1
    for j in range(L):
        mat=mat^temp[j]
    return mat

def NKron(L,op1,op2,pos): 
    '''
    Tensor product operator 
    returns tensor product of op1,op2 on sites pos,pos+1 and identity on remaining sites

    L:number of sites
    op1,op2: Pauli operators on neighboring sites
    pos: site to insert op1
    
    '''
    ide=np.eye(2)
    temp=np.array([ide]*(L),dtype=np.complex128)
    temp[pos,:,:]=op1 
    # if pos!=(N-1):
    temp[(pos+1)%L,:,:]=op2
    mat=1
    for j in range(L):
        mat=np.kron(mat,temp[j])
    return mat

def Mag(L,anti): 
    '''
     Magnetization operator (Numpy array)
    '''
    sz=np.array([[1,0],[0,-1]])
    M=np.zeros((2**L,2**L))
    for i in range(L):
        M=M+(anti**i)*NKron(L,sz,np.eye(2),i)
    return M/L

def QMag(L,anti): 
    '''
     Magnetization operator (Qiskit operator)
    '''
    M=QNKron(L,Z,I,0)
    for i in range(1,L):
        M=M+(anti**i)*QNKron(L,Z,I,i)
    return M/L










