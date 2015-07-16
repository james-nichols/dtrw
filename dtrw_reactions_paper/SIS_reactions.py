#! /usr/bin/env python

# Libraries are in parent directory
import sys
sys.path.append('../')

import numpy as np
import time
from dtrw import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm

class DTRW_diffusive_SIS(DTRW_diffusive):
    
    def __init__(self, X_inits, N, alpha, beta, mu, gamma, r=1., history_length=0, boltz_beta=0., potential = np.array([]), boundary_condition=BC()):
        
        self.beta = beta
        self.mu = mu
        self.gamma = gamma
            
        super(DTRW_diffusive_SIS, self).__init__(X_inits, N, alpha, r, history_length, boltz_beta, potential, boundary_condition)
   
    def calc_omega(self):
        """ Probability of death between n and n+1"""
        if self.omegas == None:
            self.omegas = [np.zeros((X.shape[0], X.shape[1])) for X in self.Xs]
            
        # We assume that the calculation has been made for all n up to now, so we simply update the n-th point
        # S 
        self.omegas[0][:,:] = 1. - np.exp(- self.beta * self.Xs[1][:,:,self.n] - mu) 
        #self.omegas[0][:,:] = self.beta * self.Xs[1][:,:,self.n] + self.mu
        # I 
        self.omegas[1][:,:] = 1. - np.exp(-(self.mu + self.gamma))
        #self.omegas[1][:,:] = self.mu + self.gamma

    def calc_nu(self):
        """Likelihood of birth happening at i"""
        if self.nus == None:
            self.nus = [np.zeros((X.shape[0], X.shape[1])) for X in self.Xs]
    
        self.nus[0][:,:] = (1. - np.exp(-(self.mu + self.gamma))) * self.Xs[1][:,:,self.n]
        #self.nus[0][:,:] = self.gamma * self.Xs[1][:,:,self.n] + self.mu
        self.nus[1][:,:] = (1. - np.exp(-self.beta * self.Xs[1][:,:,self.n])) * self.Xs[0][:,:,self.n]
        #self.nus[1][:,:] = self.beta * self.Xs[1][:,:,self.n] * self.Xs[0][:,:,self.n]

class DTRW_subdiffusive_SIS(DTRW_subdiffusive):
    
    def __init__(self, X_inits, N, alpha, beta, mu, gamma, r=1., history_length=0, boltz_beta=0., potential = np.array([]), boundary_condition=BC()):
        
        self.beta = beta
        self.mu = mu
        self.gamma = gamma
            
        super(DTRW_subdiffusive_SIS, self).__init__(X_inits, N, alpha, r, history_length, boltz_beta, potential, boundary_condition)
   
    def calc_omega(self):
        """ Probability of death between n and n+1"""
        if self.omegas == None:
            self.omegas = [np.zeros((X.shape[0], X.shape[1])) for X in self.Xs]
            
        # We assume that the calculation has been made for all n up to now, so we simply update the n-th point
        # S 
        self.omegas[0][:,:] = 1. - np.exp(-self.beta * self.Xs[1][:,:,self.n] - mu) 
        #self.omegas[0][:,:] = self.beta * self.Xs[1][:,:,self.n] + self.mu
        # I 
        self.omegas[1][:,:] = 1. - np.exp(-(self.mu + self.gamma))
        #self.omegas[1][:,:] = self.mu + self.gamma
    
    def calc_nu(self):
        """Likelihood of birth happening at i"""
        if self.nus == None:
            self.nus = [np.zeros((X.shape[0], X.shape[1])) for X in self.Xs]
    
        self.nus[0][:,:] = (1. - np.exp(-(self.mu + self.gamma))) * self.Xs[1][:,:,self.n]
        #self.nus[0][:,:] = self.gamma * self.Xs[1][:,:,self.n] + self.mu
        self.nus[1][:,:] = (1. - np.exp(-self.beta * self.Xs[1][:,:,self.n])) * self.Xs[0][:,:,self.n]
        #self.nus[1][:,:] = self.beta * self.Xs[1][:,:,self.n] * self.Xs[0][:,:,self.n]

n_points = 100
L = 1.0
dX = L / n_points
xs = np.linspace(0., 1., n_points+1, endpoint=True)
#xs = np.arange(-dX, 1+2.*dX, dX)

S_init = np.ones(len(xs))

#I_init = np.exp(- 25. * (xs - 0.5) * (xs - 0.5))
#I_init = np.zeros(len(xs))
#I_init[n_points/4:n_points/2] = 4. * xs[n_points/4:n_points/2] - 1.
#I_init[n_points/2:3*n_points/4] = - 4. * xs[n_points/4:n_points/2] + 2.
I_init = 0.5 - 0.5 * np.cos(2. * math.pi * xs)

alpha_1 = 0.8
alpha_2 = 0.6
D_alpha = 0.005

T = 20.

dT_1 = pow((dX * dX / (2.0 * D_alpha)), 1./alpha_1)
dT_2 = pow((dX * dX / (2.0 * D_alpha)), 1./alpha_2)
dT = 0.001

N = int(math.floor(T / dT))+1
N_1 = int(math.floor(T / dT_1))+1
N_2 = int(math.floor(T / dT_2))+1

ts = np.array(np.arange(N) * dT)
ts_1 = np.array(np.arange(N_1) * dT_1)
ts_2 = np.array(np.arange(N_2) * dT_2)

beta = 0.2
mu = 0.0
gamma = 0.1

# Calculate r for diffusive case so as to get the *same* dT as the subdiffusive case
r = dT / (dX * dX / (2.0 * D_alpha))

print "DTRW for dX =", dX, "dT =", dT, "r =", r
print "DTRW subdiff for dX =", dX, "dT =", dT_1, "alpha =", alpha_1
print "DTRW subdiff for dX =", dX, "dT =", dT_2, "alpha =", alpha_2

bc = BC_zero_flux_centred()

dtrw = DTRW_diffusive_SIS([S_init, I_init], N, r, beta * dT, mu * dT, gamma * dT, history_length=2, boundary_condition=bc)
dtrw_sub_1 = DTRW_subdiffusive_SIS([S_init, I_init], N_1, alpha_1, beta * dT_1, mu * dT_1, gamma * dT_1, history_length=N_1, boundary_condition=bc)
dtrw_sub_2 = DTRW_subdiffusive_SIS([S_init, I_init], N_2, alpha_2, beta * dT_2, mu * dT_2, gamma * dT_2, history_length=N_2, boundary_condition=bc)

dtrw_file_name = "DTRW_dT_{0:f}_dX_{1:f}.csv".format(dT, dX)
dtrw_file_name_alpha_1 = "DTRW_dT_{0:f}_dX_{1:f}_alpha_{2:f}.csv".format(dT, dX, alpha_1)
dtrw_file_name_alpha_2 = "DTRW_dT_{0:f}_dX_{1:f}_alpha_{2:f}.csv".format(dT, dX, alpha_2)

t_alpha_1 = "T_alpha_{0:f}.csv".format(alpha_1)
t_alpha_2 = "T_alpha_{0:f}.csv".format(alpha_2)
np.savetxt(t_alpha_1, ts_1, delimiter=",")
np.savetxt(t_alpha_2, ts_2, delimiter=",")

##########################################
# Solve the systems and write the results
##########################################

dtrw.solve_all_steps()
print "Exp case solved"
np.savetxt("S_" + dtrw_file_name, dtrw.Xs[0][0,:,:], delimiter=",")
np.savetxt("I_" + dtrw_file_name, dtrw.Xs[1][0,:,:], delimiter=",")

dtrw_sub_1.solve_all_steps()
print "alpha =", alpha_1, "case solved"
np.savetxt("S_" + dtrw_file_name_alpha_1, dtrw_sub_1.Xs[0][0,:,:], delimiter=",")
np.savetxt("I_" + dtrw_file_name_alpha_1, dtrw_sub_1.Xs[1][0,:,:], delimiter=",")

dtrw_sub_2.solve_all_steps()
print "alpha =", alpha_2, "case solved"
np.savetxt("S_" + dtrw_file_name_alpha_2, dtrw_sub_2.Xs[0][0,:,:], delimiter=",")
np.savetxt("I_" + dtrw_file_name_alpha_2, dtrw_sub_2.Xs[1][0,:,:], delimiter=",")
