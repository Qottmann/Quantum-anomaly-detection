# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:01:42 2021

@author: nbaldelli
"""

import numpy as np
import numpy.linalg as alg
import matplotlib.pyplot as plt
from qiskit.opflow import X,Z,I
from qiskit.providers.aer import QasmSimulator  
from qiskit.providers.aer import StatevectorSimulator  
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA, SLSQP
from qiskit.circuit.library import EfficientSU2



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
    if pos!=(N-1):
        temp[pos+1,:,:]=op2
    mat=1
    for j in range(N):
        mat=np.kron(mat,temp[j])
    return mat

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

def HIsing(N,lam):
    '''
    Quantum Ising Hamiltonian (1D) with transverse field
    
    N:number of sites 
    lam: transverse field)
    '''
    sx=np.array([[0,1],[1,0]])
    sz=np.array([[1,0],[0,-1]])
    ide=np.eye(2)
    H=np.zeros((2**N,2**N))
    for i in range(N-1):
        print(i)
        H=H-NKron(N,sz,sz,i)-lam*NKron(N,sx,ide,i)
    H=H-lam*NKron(N,sx,ide,N-1)
    return H

def QHIsing(N,lam):
    '''
    Quantum Ising Hamiltonian (1D) with transverse field (Qiskit Pauli operators)
    
    N:number of sites 
    lam: transverse field)
    '''

    H=-QNKron(N,Z,Z,0)-lam*QNKron(N,X,I,0)
    for i in range(1,N-1):
        H=H-QNKron(N,Z,Z,i)-lam*QNKron(N,X,I,i)
    H=H-lam*QNKron(N,X,I,N-1)  
    return H

def Mag(N):
    sz=np.array([[1,0],[0,-1]])
    M=np.zeros((2**N,2**N))
    for i in range(N):
        M=M+NKron(N,sz,np.eye(2),i)
    return M/N

#PARAMETERS INITIALIZATION
lams=np.logspace(-2,2,10) #array of magnetic fields
mags=np.zeros(len(lams)); Qmags=np.zeros(len(lams)) #arrays of magnetizations

backend= StatevectorSimulator()
optimizer = SLSQP(maxiter=1000)
ansatz = EfficientSU2(N, reps=3)
vqe = VQE(ansatz, optimizer, quantum_instance=backend) 

for j in range(len(lams)):
    N=5 #number of sites
    lam=lams[j] #magnetic field
    
    ham=HIsing(N,lam) #Numpy Hamiltonian
    Qham=QHIsing(N,lam) #Qiskit Hamiltonian
    
    vals,vecs=alg.eigh(ham) #ED with Numpy
    result = vqe.compute_minimum_eigenvalue(Qham) #ED with Qiskit VQE
    
    mags[j]=vecs[:,0].T.conj()@Mag(N)@vecs[:,0] #Magnetization with Numpy results
    Qmags[j]=result.eigenstate.T.conj()@Mag(N)@result.eigenstate #Magnetization with Qiskit results
    
plt.figure(1,dpi=220)
plt.scatter(lams,mags)
plt.scatter(lams,Qmags)
plt.legend(["Numpy","Qiskit"])
plt.xscale("log")
plt.xlabel("Magnetic field")
plt.ylabe("Magnetization")


