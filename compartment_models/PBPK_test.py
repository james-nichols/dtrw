#! /usr/bin/env python

# Libraries are in parent directory
import sys
sys.path.append('../')

import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import pdb

from dtrw import *

T = 100.0
dT = 0.01
ts = np.arange(0., T, dT)

initial = [0., 0., 0., 0., 0., 0.]

mu    = 0.5 # Kidney removal rate 
V_max = 2.69
K_m = 0.59
# [P, R, F, K, L, A]
Vs = [28.6, 6.90, 15.10, 0.267, 1.508, 1.570]
Qs = [1.46, 1.43,  0.29,  1.14,  1.52]
Rs = [0.69, 0.79,  0.39,  0.80,  0.78]
alpha = 1.0

g = 1.0
g_T = 1.0

dtrw = DTRW_PBPK(initial, T, dT, Vs, Qs, Rs, mu, V_max, K_m, g, g_T)
dtrw_anom = DTRW_PBPK_anom(initial, T, dT, Vs, Qs, Rs, mu, V_max, K_m, g, g_T, alpha)

dtrw.solve_all_steps()
dtrw_anom.solve_all_steps()
pdb.set_trace()
max_level = max([dtrw.Xs[0,:].max(), dtrw.Xs[1,:].max(), dtrw.Xs[2,:].max(), dtrw.Xs[3,:].max(), dtrw.Xs[4,:].max(), dtrw.Xs[5,:].max()])

fig = plt.figure(figsize=(8,8))
plt.xlim(0,T)
plt.ylim(0,1.1 * max_level)
plt.xlabel('Time')

P, = plt.plot(ts, dtrw.Xs[0,:])
R, = plt.plot(ts, dtrw.Xs[1,:])
F, = plt.plot(ts, dtrw.Xs[2,:])
K, = plt.plot(ts, dtrw.Xs[3,:])
L, = plt.plot(ts, dtrw.Xs[4,:])
A, = plt.plot(ts, dtrw.Xs[5,:])
plt.legend([P, R, F, K, L, A], ["Poorly perfused", "Richly perfused", "Fatty tissue", "Kidneys", "Liver", "Arterial blood"])

Pa, = plt.plot(ts, dtrw_anom.Xs[0,:],'b:')
Ra, = plt.plot(ts, dtrw_anom.Xs[1,:],'g:')
Fa, = plt.plot(ts, dtrw_anom.Xs[2,:],'r:')
Ka, = plt.plot(ts, dtrw_anom.Xs[3,:],'c:')
La, = plt.plot(ts, dtrw_anom.Xs[4,:],'m:')
Aa, = plt.plot(ts, dtrw_anom.Xs[5,:],'y:')

plt.show()

