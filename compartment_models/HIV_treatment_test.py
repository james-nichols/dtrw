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
                 k, T_cells, eff, delta_I, delta_V, burst, alpha):

        if len(X_inits) != 2:
            # Error!
            print("Need two initial points")
            raise SystemExit 

        super(DTRW_HIV, self).__init__(X_inits, T, dT)
        
        self.T_cells = T_cells
        self.k = k
        self.eff = eff

        self.delta_I = delta_I
        self.delta_V = delta_V
        self.burst = burst 
        self.alpha = alpha

        self.Ks[0] = calc_sibuya_kernel(self.N+1, self.alpha)
    
    def creation_flux(self, n):
        return np.array([(1.0 - self.eff) * self.k * self.T_cells * self.Xs[1,n] * self.dT, 
                         self.Xs[0,n] * self.burst * self.dT])
                         #self.removal_flux_anomalous(n)[0] * self.burst * self.dT])

    def anomalous_rates(self, n):
        return np.array([pow(self.delta_I, self.alpha), 0.])

    def removal_rates(self, n):
        return np.array([[0.], \
                         [self.delta_V]])

T = 10.0
dT = 0.01
ts = np.arange(0., T, dT)

initial = [1., 1.]

k = 2.4e-5
T_cells = 1000.
eff = 1.0
delta_I = 0.7
delta_V = 5.4
burst = 10. # This "Varies" according to Perelson '93 

alpha = 0.8 

dtrw_hiv = DTRW_HIV(initial, T, dT, k, T_cells, eff, delta_I, delta_V, burst, 1.0)
dtrw_hiv_anom = DTRW_HIV(initial, T, dT, k, T_cells, eff, delta_I, delta_V, burst, alpha)

dtrw_hiv.solve_all_steps()
dtrw_hiv_anom.solve_all_steps()

fig = plt.figure(figsize=(8,8))
plt.xlim(0,T)
plt.ylim(0,2.)
plt.xlabel('Time')

I_line, = plt.plot(ts, dtrw_hiv.Xs[0,:])
V_line, = plt.plot(ts, dtrw_hiv.Xs[1,:])

I_line_anom, = plt.plot(ts, dtrw_hiv_anom.Xs[0,:], 'b--')
V_line_anom, = plt.plot(ts, dtrw_hiv_anom.Xs[1,:], 'g--')

plt.legend([I_line, V_line, I_line_anom, V_line_anom], ['Infected, Markovian model (alpha=1.0)', 'Virus, Markovian model (alpha=1.0)', 'Infected, anomalous model (alpha=0.8)', 'Virus, anomalous model (alpha=0.8)'])
plt.title('HIV viral clearance model, anomalous vs Markovian')
plt.xlabel('Time')
plt.ylabel('Normalised cell concentration')
plt.show()

