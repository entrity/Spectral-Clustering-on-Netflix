#!/usr/bin/env python3

import numpy as np
np.set_printoptions(precision=3)

A = [
    [0, 1, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0],
];
A = np.array(A)
d = np.sum(A, axis=0)
D = np.diag(d)

# Unnormalized L
Lu = D - A;

# ala CuttingElephants
eleD = np.diag(d**(-1/2));
Le = np.eye(8) - np.matmul(np.matmul(eleD,A),eleD)
val_e,vec_e = np.linalg.eig(Le);
I = np.argsort(val_e);
print(' >>> like CuttingElephants <<<')
print(val_e[I]);
print(vec_e[:,I]);

# Normalized L
normD = np.diag(d**(-1/2));
Ln = np.matmul(np.matmul(normD,Lu),normD)
val_n,vec_n = np.linalg.eig(Ln);
I = np.argsort(val_n);
print(' >>> Normalized <<<')
print(val_n[I])
print(vec_n[:,I])

# Random walk L
randD = np.diag(1/d);
Lr = randD * Lu;
val_r,vec_r = np.linalg.eig(Ln);
I = np.argsort(val_r);
print(' >>> Random walk <<<')
print(val_r[I])
print(vec_r[:,I])
