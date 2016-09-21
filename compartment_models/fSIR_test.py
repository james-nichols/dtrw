#!/usr/local/bin/python3

# Libraries are in parent directory
import sys
sys.path.append('../')

import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import pdb

from dtrw import *

T = 20.0
dT = 0.01
ts = np.arange(0., T, dT)

initial = [0.995, 0.005, 0.]

lam =   0. 
omega = 5. # Infection rate
gamma = 0. # Death rate, applied to all compartments
mu    = 2. # Recovery rate
alpha = 0.75

dtrw_sir = DTRW_SIR(initial, T, dT, lam, omega, gamma, mu, 1.0)
dtrw_sir_anom = DTRW_SIR(initial, T, dT, lam, omega, gamma, mu, alpha)

dtrw_sir.solve_all_steps()
dtrw_sir_anom.solve_all_steps()

fig = plt.figure(figsize=(8,8))
plt.xlim(0,T)
plt.ylim(0,1.0)
plt.xlabel('Time')

plt.plot(ts, dtrw_sir.Xs[0,:])
plt.plot(ts, dtrw_sir.Xs[1,:])
plt.plot(ts, dtrw_sir.Xs[2,:])

plt.plot(ts, dtrw_sir_anom.Xs[0,:], 'b:')
plt.plot(ts, dtrw_sir_anom.Xs[1,:], 'g:')
plt.plot(ts, dtrw_sir_anom.Xs[2,:], 'r:')

plt.show()

