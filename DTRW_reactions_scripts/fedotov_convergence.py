#! /usr/bin/env python

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

L = 10.0
T = 2.
alpha = 0.9
D_alpha = 1.
g = 10.
k = 1.5

dXs = [0.4, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025]
#dXs = [0.01, 0.005, 0.0025]

# Analytic solution parameters
mu = 2. - alpha
g_star = g / (math.sqrt(D_alpha * pow(k, 2.-alpha)))
pst_0 = pow(g_star * math.sqrt((mu+2.)/(2.*mu)), 2. / (mu+2.))
x_0 = (2. * mu / (2. - mu)) * pow(g_star, -(2.-mu)/(mu+2.)) * pow((mu+2.)/(2.*mu), mu/(mu+2.)) * math.sqrt(D_alpha/pow(k,alpha))

#l2_bal_dist = dX * np.linalg.norm(dtrw_bal.Xs[0][:,:,-1] - analytic_soln)
#linf_bal_dist = np.linalg.norm(dtrw_bal.Xs[0][:,:,-1] - analytic_soln, np.inf)
#l2_neu_dist = dX * np.linalg.norm(dtrw.Xs[0][:,:,-1] - analytic_soln) 
#linf_neu_dist = np.linalg.norm(dtrw.Xs[0][:,:,-1] - analytic_soln, np.inf)
   
#bal_l2 =  np.apply_along_axis(np.linalg.norm, 1, dtrw.Xs[0][:,:,:-1] - dtrw.Xs[0][:,:,1:])
for dX in dXs:

    n_points = int(math.floor(L / dX)) 
    xs = np.linspace(0., L, n_points, endpoint=False)
    
    analytic_soln = pst_0 * pow((1 + xs / x_0), -2./alpha)
    X_init = analytic_soln

    dT = pow(dX * dX / (2. * D_alpha), 1/alpha)
    N = int(math.floor(T / dT))
    history_length = N

    print "Solving ", N, "steps, dT =", dT, "to time T =", T, "dX =", dX

    # If we even want to use this at all...
    bc_constant = g * dX / (D_alpha * pow(k, 1.-alpha))
    
    # Boundary conditions!
    dir_bc = BC_Dirichelet([analytic_soln[0], analytic_soln[-1]])
    fed_bal_bc = BC_Fedotov_balance()
    fed_bc = BC_Fedotov(alpha, bc_constant, analytic_soln[-1])

    dtrw = DTRW_subdiffusive_fedotov_death(X_init, N, alpha, dT*k, history_length, boundary_condition=fed_bc)
    #dtrw = DTRW_subdiffusive_fedotov_death(X_init, N, alpha, 1.-exp(k*dT), history_length, boundary_condition=fed_bc)
    dtrw.solve_all_steps()

    dtrw_bal = DTRW_subdiffusive_fedotov_death(X_init, N, alpha, dT*k, history_length, boundary_condition=fed_bal_bc)
    dtrw_bal.solve_all_steps()
    
    dtrw_bal_file_name = "DTRW_bal_Soln_{0:f}_{1:f}_{2:f}_{3:f}".format(alpha, T, k, dX)
    np.save(dtrw_bal_file_name, dtrw_bal.Xs[0])
    dtrw_neu_file_name = "DTRW_neu_Soln_{0:f}_{1:f}_{2:f}_{3:f}".format(alpha, T, k, dX)
    np.save(dtrw_neu_file_name, dtrw.Xs[0])
    analytic_file_name = "Analytic_Soln_{0:f}_{1:f}_{2:f}_{3:f}".format(alpha, T, k, dX)
    np.save(analytic_file_name, analytic_soln)

    l2_bal_dist = dX * np.linalg.norm(dtrw_bal.Xs[0][:,:,-1] - analytic_soln)
    linf_bal_dist = np.linalg.norm(dtrw_bal.Xs[0][:,:,-1] - analytic_soln, np.inf)
    l2_neu_dist = dX * np.linalg.norm(dtrw.Xs[0][:,:,-1] - analytic_soln) 
    linf_neu_dist = np.linalg.norm(dtrw.Xs[0][:,:,-1] - analytic_soln, np.inf)
   
    bal_l2 =  np.apply_along_axis(np.linalg.norm, 1, dtrw_bal.Xs[0][:,:,:-1] - dtrw_bal.Xs[0][:,:,1:]).flatten()
    bal_linf =  np.apply_along_axis(np.linalg.norm, 1, dtrw_bal.Xs[0][:,:,:-1] - dtrw_bal.Xs[0][:,:,1:], np.inf).flatten()
    neu_l2 =  np.apply_along_axis(np.linalg.norm, 1, dtrw.Xs[0][:,:,:-1] - dtrw.Xs[0][:,:,1:]).flatten()
    neu_linf =  np.apply_along_axis(np.linalg.norm, 1, dtrw.Xs[0][:,:,:-1] - dtrw.Xs[0][:,:,1:], np.inf).flatten()

    print "Balance mthd \t {0:e} \t {1:e} \t {2:e} \t {3:e} \t {4:e}".format(bal_l2[-1], bal_l2[-1] * dX, bal_linf[-1], l2_bal_dist, linf_bal_dist)
    print "Neumann mthd \t {0:e} \t {1:e} \t {2:e} \t {3:e} \t {4:e}".format(neu_l2[-1], neu_l2[-1] * dX, neu_linf[-1], l2_neu_dist, linf_neu_dist)
    
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
    line3 = plt.plot(xs, dtrw_bal.Xs[0][:,:,-1].flatten()/p_0, 'y^') 
    plt.show()
    """


