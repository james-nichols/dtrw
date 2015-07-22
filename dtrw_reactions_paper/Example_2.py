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

class DTRW_diffusive_two_way(DTRW_diffusive):
    
    def __init__(self, X_inits, N, alpha, k_1, k_2, r=1., history_length=0, boltz_beta=0., potential = np.array([]), boundary_condition=BC()):
        
        self.k_1 = k_1
        self.k_2 = k_2

        super(DTRW_diffusive_two_way, self).__init__(X_inits, N, alpha, r=r, history_length=history_length, boltz_beta=boltz_beta, potential=potential, boundary_condition=boundary_condition)
   
    def calc_omega(self):
        """ Probability of death between n and n+1"""
        if self.omegas == None:
            self.omegas = [np.zeros((X.shape[0], X.shape[1])) for X in self.Xs]
            
        # We assume that the calculation has been made for all n up to now, so we simply update the n-th point
        # a 
        self.omegas[0][:,:] = 1. - np.exp(-self.k_1) 
        # b 
        self.omegas[1][:,:] = 1. - np.exp(-self.k_2)

    def calc_nu(self):
        """Likelihood of birth happening at i"""
        if self.nus == None:
            self.nus = [np.zeros((X.shape[0], X.shape[1])) for X in self.Xs]
    
        self.nus[0][:,:] = (1. - np.exp(-self.k_2)) * self.Xs[1][:,:,self.n]
        self.nus[1][:,:] = (1. - np.exp(-self.k_1)) * self.Xs[0][:,:,self.n]

class DTRW_subdiffusive_two_way(DTRW_subdiffusive):
     
    def __init__(self, X_inits, N, alpha, k_1, k_2, r=1., history_length=0, boltz_beta=0., potential = np.array([]), boundary_condition=BC()):
        
        self.k_1 = k_1
        self.k_2 = k_2
 
        super(DTRW_subdiffusive_two_way, self).__init__(X_inits, N, alpha, r=r, history_length=history_length, boltz_beta=boltz_beta, potential=potential, boundary_condition=boundary_condition)
   
    def calc_omega(self):
        """ Probability of death between n and n+1"""
        if self.omegas == None:
            self.omegas = [np.zeros((X.shape[0], X.shape[1])) for X in self.Xs]
            
        # We assume that the calculation has been made for all n up to now, so we simply update the n-th point
        # a 
        self.omegas[0][:,:] = 1. - np.exp(-self.k_1) 
        #self.omegas[0][:,:] = - self.k_1 
        # b 2.
        self.omegas[1][:,:] = 1. - np.exp(-self.k_2)
        #self.omegas[1][:,:] = - self.k_2

    def calc_nu(self):
        """Likelihood of birth happening at i"""
        if self.nus == None:
            self.nus = [np.zeros((X.shape[0], X.shape[1])) for X in self.Xs]
    
        self.nus[0][:,:] = (1. - np.exp(-self.k_2)) * self.Xs[1][:,:,self.n]
        #self.nus[0][:,:] = -self.k_2 * self.Xs[1][:,:,self.n]
        self.nus[1][:,:] = (1. - np.exp(-self.k_1)) * self.Xs[0][:,:,self.n]
        #self.nus[1][:,:] = -self.k_1 * self.Xs[0][:,:,self.n]

alpha_1 = 0.5
D_alpha = 1.
T = 0.1
k_1 = 1.

r = 1.

# Do we want results on the grid points, or in the centroids?
use_grid_aligned = False

if use_grid_aligned:
    dXs = [1./2, 1./5., 1./7., 1./10., 1./15., 1./20., 1./30.]          # Choose this for grid aligned results
else:
    dXs = [2./3., 2./5., 2./7., 2./11., 2./15., 2./21., 2./31., 2./41.] # Choose this for centroid results

soln = np.loadtxt("Soln_at_t01.dat")

for dX in dXs:
     
    dT = pow(dX * dX / (2.0 * D_alpha), 1.0 / alpha_1)

    if use_grid_aligned:
        xs = np.arange(-1.0, 1.0+dX, dX)                        # Choose this for grid aligned results
    else:
        xs = np.arange(-1.0+0.5*dX, 1.0, dX)                    # Choose this for centroid results

    # We make the init conditions slightly bigger for the zero flux boundaries
    a_init = np.zeros(xs.shape)
    if use_grid_aligned:
        a_init[xs.shape[0] / 2 ] = float(xs.shape[0]-1)/2.     # Choose this for grid aligned results
    else:
        a_init[xs.shape[0] / 2 ] = 1./dX                       # Choose this for centroid results
    b_init = np.zeros(xs.shape)

    N = int(round(T / dT) + 1)
    history_length = N + 1

    ts = np.array(np.arange(N) * dT)

    # Calculate r for diffusive case so as to get the *same* dT as the subdiffusive case
    omega = dT / (dX * dX / (2.0 * D_alpha))

    print "Solving DTRW subdiff for dX =", dX, "dT =", dT, "r =", r, "alpha =", alpha_1, "N =", N

    bc = BC_zero_flux()

    dtrw_sub_1 = DTRW_subdiffusive_two_way([a_init, b_init], N, alpha_1, k_1 * dT, k_1 * dT, r=r, history_length=N, boundary_condition=bc)

    dtrw_sub_1.solve_all_steps()
    
    dtrw_file_name = "DTRW_dT_{0:f}_dX_{1:f}.csv".format(dT, dX)
    dtrw_file_name_alpha_1 = "DTRW_dT_{0:f}_dX_{1:f}_alpha_{2:f}.csv".format(dT, dX, alpha_1)
    
    np.savetxt("a_" + dtrw_file_name_alpha_1, dtrw_sub_1.Xs[0][0,:,-1], delimiter=",")
    np.savetxt("b_" + dtrw_file_name_alpha_1, dtrw_sub_1.Xs[1][0,:,-1], delimiter=",")
    np.savetxt("a_2ndLast_" + dtrw_file_name_alpha_1, dtrw_sub_1.Xs[0][0,:,-2], delimiter=",")
    np.savetxt("b_2ndLast_" + dtrw_file_name_alpha_1, dtrw_sub_1.Xs[1][0,:,-2], delimiter=",")

    t_alpha_1 = "T_dT_{0:f}_alpha_{1:f}.csv".format(dT, alpha_1)
    np.savetxt(t_alpha_1, ts, delimiter=",")

