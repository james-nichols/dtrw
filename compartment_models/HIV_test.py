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
 
class DTRW_HIV(DTRW_compartment):

    def __init__(self, X_inits, T, dT, \
                 s, r, k, T_max, delta_T, delta_I, delta_V, burst, alpha):

        if len(X_inits) != 3:
            # Error!
            print("Need three initial points")
            raise SystemExit 

        super(DTRW_HIV, self).__init__(X_inits, T, dT)
        
        self.s = s
        self.r = r
        self.k = k
        self.T_max = T_max
        self.delta_T = delta_T
        self.delta_I = delta_I
        self.delta_V = delta_V
        self.burst = burst 

        self.alpha = alpha

        self.Ks[0] = calc_sibuya_kernel(self.N+1, self.alpha)
        self.Ks[1] = calc_sibuya_kernel(self.N+1, self.alpha)
    
    def creation_flux(self, n):
        return np.array([self.s * self.dT + self.r * self.Xs[0, n] * self.dT,
                         self.removal_flux_markovian(n)[0, 1],
                         self.removal_flux_anomalous(n)[1] * self.burst * self.dT])

    def anomalous_rates(self, n):
        return np.array([pow(self.delta_T, self.alpha), \
                         pow(self.delta_I, self.alpha), 0.])

    def removal_rates(self, n):
        return np.array([[r * (self.Xs[0,n]+self.Xs[1,n]) / self.T_max, \
                          self.k * self.Xs[2,n]], \
                         [0., 0.], \
                         [self.delta_V, 0.]])

T = 10.0
dT = 0.01
ts = np.arange(0., T, dT)

initial = [1000, 0., 100.]

s = 10.
r = 0.03
k = 2.4e-3
T_max = 1500.
delta_T = 0.02
delta_I = 0.7
delta_V = 2.4
burst = 1000. # This "Varies" according to Perelson '93 

alpha = 0.8 

dtrw_hiv = DTRW_HIV(initial, T, dT, s, r, k, T_max, delta_T, delta_I, delta_V, burst, 1.0)
dtrw_hiv_anom = DTRW_HIV(initial, T, dT, s, r, k, T_max, delta_T, delta_I, delta_V, burst, alpha)

dtrw_hiv.solve_all_steps()
dtrw_hiv_anom.solve_all_steps()
fig = plt.figure(figsize=(8,8))
plt.xlim(0,T)
plt.ylim(0,T_max)
plt.xlabel('Time')

plt.plot(ts, dtrw_hiv.Xs[0,:])
plt.plot(ts, dtrw_hiv.Xs[1,:])
plt.plot(ts, dtrw_hiv.Xs[2,:])

plt.plot(ts, dtrw_hiv_anom.Xs[0,:], 'b:')
plt.plot(ts, dtrw_hiv_anom.Xs[1,:], 'g:')
plt.plot(ts, dtrw_hiv_anom.Xs[2,:], 'r:')

plt.show()

