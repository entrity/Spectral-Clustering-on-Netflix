#!/usr/bin/python3
import numpy as np

# np.set_printoptions(formatter={'float_kind': lambda x: ("%.4f" if np.abs(x) > 1e-6 else "%.1f   ") % x})
np.set_printoptions(precision=4)

A = np.array([
    [0,1,1,0,0,0,0,0],
    [1,0,1,0,0,0,0,0],
    [1,1,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,1,1],
    [0,0,0,0,0,1,0,1],
    [0,0,0,0,0,1,1,0],])

d = np.sum(A, axis=0)
D = np.diag(d)
# print(D)
#
# Unnormalized L
print('Unnorm')
Lu = D - A
# print('Lu', Lu)
valsU,vecsU = np.linalg.eig(Lu)
idx = np.argsort(valsU)
print(valsU[idx])
print(vecsU[:,idx])

# CuttingElephants
print('Eleph')
eleD = np.diag(d**(-1/2))
Le = np.eye(A.shape[0]) - np.matmul(np.matmul(eleD, A), eleD)
valsE,vecsE = np.linalg.eig(Le)
idx = np.argsort(valsE)
print(valsE[idx])
print(vecsE[:,idx])

# Normalized
print('Norma')
normD = np.diag(d**(-1/2))
Ln = np.matmul(np.matmul(normD,Lu), normD)
valsN,vecsN = np.linalg.eig(Ln)
idx = np.argsort(valsN)
print(valsN[idx])
print(vecsN[:,idx])

# Random walk
print('Rand Walk')
normR = np.diag(1/d)
Lr = np.matmul(normR,Lu)
valsR,vecsR = np.linalg.eig(Lr)
idx = np.argsort(valsR)
print(valsR[idx])
print(vecsR[:,idx])
