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

class DTRW_PBPK(DTRW_compartment):

    def __init__(self, X_inits, T, dT, V, Q, R, mu, Vmax, Km, g, g_T):
        
        if len(X_inits) != 6:
            # Error!
            print "Need six initial points"
            raise SystemExit

        super(DTRW_PBPK, self).__init__(X_inits, T, dT)
         
        self.Vs = np.array(V)
        self.Qs = np.array(Q)
        self.Rs = np.array(R)
        self.mu = mu
        self.Vmax = Vmax
        self.Km = Km
        self.g = g
        self.g_T = g_T

    def creation_flux(self, n):
        g_N = 0.
        if (n * self.dT < self.g_T):
            g_N = self.g * self.dT
        
        creation = np.zeros(self.n_species)
        creation[:-1] = self.removal_flux_markovian(n)[5,:]
        creation[-1] = (self.removal_flux_markovian(n)[:5, 0]).sum() + g_N
        
        return creation
        """return np.array([(1. - np.exp(-self.dT * self.Qs[0] / self.Vs[5])) * self.Xs[5,n], \
                         (1. - np.exp(-self.dT * self.Qs[1] / self.Vs[5])) * self.Xs[5,n], \
                         (1. - np.exp(-self.dT * self.Qs[2] / self.Vs[5])) * self.Xs[5,n], \
                         (1. - np.exp(-self.dT * self.Qs[3] / self.Vs[5])) * self.Xs[5,n], \
                         (1. - np.exp(-self.dT * self.Qs[4] / self.Vs[5])) * self.Xs[5,n], \
                         (1. - np.exp(-self.dT * self.Qs[0] / (self.Vs[0] * self.Rs[0]))) * self.Xs[0,n] + \
                         (1. - np.exp(-self.dT * self.Qs[1] / (self.Vs[1] * self.Rs[1]))) * self.Xs[1,n] + \
                         (1. - np.exp(-self.dT * self.Qs[2] / (self.Vs[2] * self.Rs[2]))) * self.Xs[2,n] + \
                         (1. - np.exp(-self.dT * self.Qs[3] / (self.Vs[3] * self.Rs[3]))) * self.Xs[3,n] + \
                         (1. - np.exp(-self.dT * self.Qs[4] / (self.Vs[4] * self.Rs[4]))) * self.Xs[4,n] + \
                         g_N ])"""
 
    def removal_rates(self, n):
        rates = np.zeros([self.n_species, 5])
        rates[:-1, 0] = self.Qs / (self.Vs[:-1] * self.Rs)
        rates[3, 1] = self.mu / self.Vs[3]
        rates[4, 1] = self.Vmax / (self.Vs[4] * self.Km + self.Xs[4,n])
        rates[5,:] = self.Qs / self.Vs[-1]

        return rates

class DTRW_PBPK_anom(DTRW_compartment):

    def __init__(self, X_inits, T, dT, V, Q, R, mu, Vmax, Km, g, g_T, alpha):
        
        if len(X_inits) != 6:
            # Error!
            print "Need six initial points"
            raise SystemExit

        super(DTRW_PBPK_anom, self).__init__(X_inits, T, dT)
          
        self.Vs = np.array(V)
        self.Qs = np.array(Q)
        self.Rs = np.array(R)
        self.mu = mu
        self.Vmax = Vmax
        self.Km = Km
        self.g = g
        self.g_T = g_T

        self.alpha = alpha
        self.Ks[2] = calc_sibuya_kernel(self.N+1, self.alpha)
        self.Ks[5] = calc_sibuya_kernel(self.N+1, self.alpha)
        self.anom_rates[2] = self.Qs[2] / (self.Vs[2] * self.Rs[2])
        self.anom_rates[5] = self.Qs[2] / (self.Vs[-1])
        
    def creation_flux(self, n):
        g_N = 0.
        if (n * self.dT < self.g_T):
            g_N = self.g * self.dT
        
        creation = np.zeros(self.n_species)
        creation[:-1] = self.removal_flux_markovian(n)[5,:]
        creation[2] = self.removal_flux_anomalous(n)[5]
        creation[-1] = (self.removal_flux_markovian(n)[:5, 0]).sum() + self.removal_flux_anomalous(n)[2] + g_N
        
        return creation
 
    def removal_rates(self, n):
        rates = np.zeros([self.n_species, 5])
        rates[:-1, 0] = self.Qs / (self.Vs[:-1] * self.Rs)
        rates[2,0] = 0.
        rates[3, 1] = self.mu / self.Vs[3]
        rates[4, 1] = self.Vmax / (self.Vs[4] * self.Km + self.Xs[4,n])
        rates[5,:] = self.Qs / self.Vs[-1]
        rates[5,2] = 0.
        
        return rates

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
alpha = 0.8

g = 1.0
g_T = 1.0

dtrw = DTRW_PBPK(initial, T, dT, Vs, Qs, Rs, mu, V_max, K_m, g, g_T)
dtrw_anom = DTRW_PBPK_anom(initial, T, dT, Vs, Qs, Rs, mu, V_max, K_m, g, g_T, alpha)

dtrw.solve_all_steps()
dtrw_anom.solve_all_steps()

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

T, = plt.plot(ts, dtrw.Xs.sum(0), 'k')
Ta, = plt.plot(ts, dtrw_anom.Xs.sum(0), 'k:')

plt.show()
