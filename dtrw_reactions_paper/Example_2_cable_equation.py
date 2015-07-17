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
    
    def __init__(self, X_inits, N, alpha, k, r=1., history_length=0, boltz_beta=0., potential = np.array([]), boundary_condition=BC()):
        
        self.k = k

        super(DTRW_diffusive_two_way, self).__init__(X_inits, N, alpha, r=r, history_length=history_length, boltz_beta=boltz_beta, potential=potential, boundary_condition=boundary_condition)
   
    def calc_nu(self):
        """Likelihood of birth happening at i"""
        if self.nus == None:
            self.nus = [np.zeros((X.shape[0], X.shape[1])) for X in self.Xs]
    
            self.nus[0][:,:] = self.k

class DTRW_subdiffusive_two_way(DTRW_subdiffusive):
     
    def __init__(self, X_inits, N, alpha, k, r=1., history_length=0, boltz_beta=0., potential = np.array([]), boundary_condition=BC()):
        
        self.k = k
 
        super(DTRW_subdiffusive_two_way, self).__init__(X_inits, N, alpha, r=r, history_length=history_length, boltz_beta=boltz_beta, potential=potential, boundary_condition=boundary_condition)
   
    def calc_nu(self):
        """Likelihood of birth happening at i"""
        if self.nus == None:
            self.nus = [np.zeros((X.shape[0], X.shape[1])) for X in self.Xs]
    
            self.nus[0][:,:] = self.k

alpha_1 = 0.5
D_alpha = 1.
T = 0.1
k_1 = 1.

r = 1.

dXs = [0.2, 0.1, 0.05, 0.025, 0.0125]

for dX in dXs:
     
    dT = pow(dX * dX / (2.0 * D_alpha), 1.0 / alpha_1)

    xs = np.arange(-1.0, 1.0+dX, dX)
    
    # We make the init conditions slightly bigger for the zero flux boundaries
    a_init = np.zeros(xs.shape)
    a_init[xs.shape[0] / 2 ] = 1.0 / dX #float(n_points)/2.
    b_init = np.zeros(xs.shape)

    N = int(round(T / dT) + 1)
    N_1 = int(round(T / dT) + 1)
    history_length = N + 1

    ts = np.array(np.arange(N) * dT)
    ts_1 = np.array(np.arange(N_1) * dT)

    print "Solving DTRW subdiff for dX =", dX, "dT =", dT, "r =", r, "alpha =", alpha_1, "N =", N_1

    bc = BC_zero_flux()

    dtrw_sub_1 = DTRW_subdiffusive_two_way([a_init], N_1, alpha_1, k_1 * dT, r=r, history_length=N_1, boundary_condition=bc)

    print "Exp case solved"
    dtrw_sub_1.solve_all_steps()
    print "alpha =", alpha_1, "case solved"
    
    dtrw_file_name_alpha_1 = "CableEq_DTRW_dT_{0:f}_dX_{1:f}_alpha_{2:f}.csv".format(dT, dX, alpha_1)
    np.savetxt("z_" + dtrw_file_name_alpha_1, dtrw_sub_1.Xs[0][0,:,-1], delimiter=",")

    t_alpha_1 = "CableEq_T_dT_{0:f}_alpha_{1:f}.csv".format(dT, alpha_1)
    np.savetxt(t_alpha_1, ts_1, delimiter=",")

