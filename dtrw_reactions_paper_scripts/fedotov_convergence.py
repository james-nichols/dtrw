#! /usr/bin/env python

# Libraries are in parent directory
import sys
sys.path.append('../')

import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm

import pdb

from dtrw import *

""" This script tests the convergence of our DTRW solution to a known analytic solution,
    for decreasing spatial (and hence also temporal) grid size """

#L = 10.0
Ls = [5.0, 10.0, 20.0, 50.0]
T = 2.
alpha = 0.9
D_alpha = 1.
g = 1. 
k = 10.0 

dXs = [0.4, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005]#, 0.0025]
dXs = [0.5, 0.25, 0.1, 0.05]

l2_grid = np.zeros([len(dXs), len(Ls)])
linf_grid = np.zeros([len(dXs), len(Ls)])
linf_rel_grid = np.zeros([len(dXs), len(Ls)])

# Analytic solution parameters
mu = 2. - alpha
g_star = g / (math.sqrt(D_alpha * pow(k, 2.-alpha)))
pst_0 = pow(g_star * math.sqrt((mu+2.)/(2.*mu)), 2. / (mu+2.))
x_0 = (2. * mu / (2. - mu)) * pow(g_star, -(2.-mu)/(mu+2.)) * pow((mu+2.)/(2.*mu), mu/(mu+2.)) * math.sqrt(D_alpha/pow(k,alpha))

#for dX in dXs:
#    for L in Ls:
for i in range(len(dXs)):
    for j in range(len(Ls)):
        dX = dXs[i]
        L = Ls[j]

        n_points = int(math.floor(L / dX)) 
        xs = np.linspace(0., L, n_points, endpoint=False)
        
        analytic_soln = pst_0 * pow((1 + xs / x_0), -2./alpha)
        X_init = analytic_soln

        dT = pow(dX * dX / (2. * D_alpha), 1./alpha)
        N = int(math.floor(T / dT))
        history_length = N

        print "Solving ", N, "steps, dT =", dT, "to time T =", T, "with", L, "spatial points, dX =", dX

        # If we even want to use this at all...
        bc_constant = g * dX / (D_alpha * pow(k, 1.-alpha))
        
        # Boundary conditions!
        dir_bc = BC_Dirichelet([analytic_soln[0], analytic_soln[-1]])
        fed_bal_bc = BC_Fedotov_balance()
        fed_bc = BC_Fedotov(alpha, bc_constant, analytic_soln[-1])

        dtrw = DTRW_subdiffusive_fedotov_death(X_init, N, alpha, dT*k, 1.0, history_length, boundary_condition=fed_bal_bc)
        dtrw.solve_all_steps()
        
        dtrw_file_name = "DTRW_Soln_{0:f}_{1:f}_{2:f}_{3:f}_{4:f}".format(alpha, T, k, dX, L)
        np.savetxt(dtrw_file_name, dtrw.Xs[0][:,:,-1], delimiter=',')
        
        analytic_file_name = "Analytic_Soln_{0:f}_{1:f}_{2:f}_{3:f}_{4:f}".format(alpha, T, k, dX, L)
        np.savetxt(analytic_file_name, analytic_soln, delimiter=',')

        l2_dist = dX * np.linalg.norm(dtrw.Xs[0][:,:,-1] - analytic_soln)
        linf_dist = np.linalg.norm((dtrw.Xs[0][:,:,-1] - analytic_soln).T, np.inf)
        linf_rel = np.linalg.norm(((dtrw.Xs[0][:,:,-1] - analytic_soln)/analytic_soln).T, np.inf)
        
        l2 =  np.apply_along_axis(np.linalg.norm, 1, dX * (dtrw.Xs[0][:,:,:-1] - dtrw.Xs[0][:,:,1:])).flatten()
        linf =  np.apply_along_axis(np.linalg.norm, 1, (dtrw.Xs[0][:,:,:-1] - dtrw.Xs[0][:,:,1:]).T, np.inf).flatten()

        l2_grid[i,j] = l2_dist
        linf_grid[i,j] = linf_dist
        linf_rel_grid[i,j] = linf_rel

        print "Balance mthd \t {0:e} \t {1:e} \t {2:e} \t {3:e}".format(l2[-1] * dX, linf[-1], l2_dist, linf_dist)
       
        #print "DTRW GME", dtrw.Xs[0][:,:,-1]
        #print "analytic", analytic_soln
        #print ""
        #print "abs rel", np.abs((dtrw.Xs[0][:,:10,-1] - analytic_soln[:10]) / analytic_soln[:10])
        #print ""
        #pdb.set_trace() 

print l2_grid
print linf_grid
print linf_rel_grid
    
"""
lo = 0.
hi = 1.05
p_0 = analytic_soln[0]

fig = plt.figure(figsize=(8,8))
plt.xlim(lo, hi)
plt.ylim(0,1.2)
plt.xlabel('x')
line1 = plt.plot(xs, analytic_soln/p_0, 'r-') 
line2 = plt.plot(xs, dtrw.Xs[0][:,:,-1].flatten()/p_0, 'gx') 
line3 = plt.plot(xs, dtrw.Xs[0][:,:,-1].flatten()/p_0, 'y^') 
plt.show()
"""


