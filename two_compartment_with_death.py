#!/usr/local/bin/python3

import numpy as np
import time
from dtrw import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm

import pdb


class DTRW_anomalous_transition(DTRW_subdiffusive):
    
    def __init__(self, X_inits, N, alpha, k_1, k_2, clearance_rate, infection_rate, history_length, beta = 0., potential = np.array([]), is_periodic=False):
        
        self.k_1 = k_1 
        self.k_2 = k_2
        # For now NO DEATH PROCESS - FOR TESTING CONSERVATION OF PARTICLES
        self.clearance_rate = 0. #clearance_rate 
        self.infection_rate = infection_rate

        super(DTRW_subdiffusive_with_transition, self).__init__(X_inits, N, alpha, history_length, beta, potential, is_periodic)

    def calc_omega(self):
        """ Probability of death between n and n+1"""
        if self.omegas == None:
            self.omegas = [np.zeros((X.shape[0], X.shape[1])) for X in self.Xs]
            
        # We assume that the calculation has been made for all n up to now, so we simply update the n-th point
        # Virions, layer 1
        self.omegas[0][:,:] = 1. - np.exp(-self.k_1) 
        
        # Virions, layer 2 
        self.omegas[1][:,:] = 1. - np.exp(-self.k_2 - self.clearance_rate)

        # Target CD4+ cells, layer 2
        self.omegas[2][:,:] = 1. - np.exp(-self.infection_rate * self.Xs[1][:,:,self.n])
        
        # Infected CD4+ cells, layer 2
        # For now NO DEATH PROCESS - FOR TESTING CONSERVATION OF PARTICLES

    def calc_nu(self):
        """Likelihood of birth happening at i"""
        if self.nus == None:
            self.nus = [np.zeros((X.shape[0], X.shape[1])) for X in self.Xs]
    
        # Here we get birth rates that reflect death rates, that is, everything balances out in the end.
        self.nus[0][:,:] = (1. - np.exp(-self.k_2)) * self.Xs[1][:,:,self.n]
        self.nus[1][:,:] = (1. - np.exp(-self.k_1)) * self.Xs[0][:,:,self.n]
        # No birth proces for target cells
        #self.nus[2][:,:,self.n]
        self.nus[3][:,:] = (1. - np.exp(-self.infection_rate * self.Xs[1][:,:,self.n])) * self.Xs[2][:,:,self.n]


T = 50.
n_points = 1000

V0 = 1.
k_1 = 1.0
delta = 0.1

xs = np.linspace(0., T, n_points, endpoint=False)
ys1 = V0 * (1. - np.exp(-0.1 * xs)) * np.exp(-delta * xs)
ys2 = V0 * (1. - np.exp(-0.2 * xs)) * np.exp(-delta * xs)
ys3 = V0 * (1. - np.exp(-0.5 * xs)) * np.exp(-delta * xs)
ys4 = V0 * (1. - np.exp(-1.0 * xs)) * np.exp(-delta * xs)

fig = plt.figure(figsize=(8,8))
line1, = plt.plot(xs,ys1, 'r-')
line1, = plt.plot(xs,ys2, 'g-')
line1, = plt.plot(xs,ys3, 'b-')
line1, = plt.plot(xs,ys4, 'y-')
#plt.legend([line1, line2, line3], ["Normal diffusion", "Sub-diffusion, alpha=3/4", "analytic diffusion"])
plt.show()
