#!/usr/local/bin/python3

# Libraries are in parent directory
import sys
sys.path.append('../')

import sys
import time
import math
import random
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from dtrw import *
import pdb

import mpmath

def analytic_solve(params, T, xs):
    # Use meijer-G function to calculate subdiffusion for alpha = 1/2
    D_alpha = params[0]
    
    integrand = lambda u: 1. / math.sqrt(8 * pow(math.pi,3) * D_alpha * math.sqrt(T)) * float(mpmath.meijerg([[],[]], [[0, 0.25, 0.5],[]], pow(u,4) / (256. * D_alpha * D_alpha * T)))
     
    return np.vectorize(lambda x: integrand(x))(xs) #1.0 - 2.0 * np.vectorize(lambda x: scipy.integrate.quad(integrand, 0., x)[0])(xs)

####################################################################
# Inverse cumulative Sibuya calculation (several different methods)
####################################################################

def inv_cum_sibuya_slow(x, alpha):
    # Returns an integer - the inverse cumulative Sibuya distribution with parameter alpha
    # SUPER SLOW!!
    sib_n = 1. - alpha
    n = 1
    while x < sib_n and n < num_t:
        n += 1
        sib_n *= (1. - alpha / n)
    return n

def make_survival_probs(alpha, nt):
    survival_probs = np.zeros(nt) # The longest possible jump we can have is the total number of time points!
    survival_probs[0] = 1. - alpha
    for i in range(1, nt):
        survival_probs[i] = survival_probs[i-1] * (1. - alpha/float(i+1)) 
    survival_probs = 1. - survival_probs

    return survival_probs

def inv_cum_sibuya_fast(x, survival):
    return np.searchsorted(survival, 1. - x) + 1 

###############
# MC Approach
###############

def mc_solve(xs, N, nt, alpha):
    density = np.zeros(xs.shape)
    jump_distribution = np.zeros(int(nt))

    survival_probs = make_survival_probs(alpha, nt)

    n = 0
    #while max(abs(density / float(n) - analytic_soln)) > epsilon:
    while n < N:
        # loop through each MC point
        n = n + 1
        t = 0
        
        pos = num_points // 2 
        # 0.5 chance of hitting the other point if we have an evenly spaced grid...
        if num_points % 2 == 0:
            if random.random() < 0.5:
                pos = num_points // 2 - 1
        num_jumps = 0
        
        t += inv_cum_sibuya_fast(random.random(), survival_probs)
        while t < nt:
            jump = random.choice([-1, 1])
           
            # We apply the zero-flux / Neumann boundaries here
            if pos > 0 and pos < num_points:
                pos += jump
            elif pos == 0 and jump == 1:
                pos += jump
            elif pos == num_points and jump == -1:
                pos += jump

            num_jumps += 1 
            
            t += inv_cum_sibuya_fast(random.random(), survival_probs)
        
        if pos >= 0 and pos <= num_points-1:
            # This enacts the absorbing boundary conditions - so
            # should be zero flux
            density[pos] += 1.
        jump_distribution[num_jumps] += 1

    return density, n

################
# DTRW Approach
################

def dtrw_solve(xs, num_t, alpha):

    X_init = np.zeros(xs.shape)
    if num_points % 2 == 0:
        X_init[num_points // 2] = 0.5
        X_init[num_points // 2 - 1] = 0.5
    else:
        X_init[num_points//2] = 1.
   
    bc = BC_zero_flux()
    dtrw = DTRW_subdiffusive(X_init, num_t, alpha, boundary_condition = bc)
    dtrw.solve_all_steps()

    return dtrw.Xs[0][:,:,-1].T

######################
# Compare approaches #
######################

# Number of MC points
N = int(1e6)

T = 0.5 
D_alpha = 0.1
dX = 2.e-2
end_points = [-1., 1.]
num_points = int((end_points[1]-end_points[0])/dX) 
xs = np.linspace(end_points[0] + 0.5 * dX, end_points[1] - 0.5 * dX, num_points)

dX_anal = 1.e-2
num_points_anal = int((end_points[1]-end_points[0])/dX_anal) + 1
xs_anal = np.linspace(end_points[0], end_points[1], num_points_anal)

dtrw_times = []
mc_times = []
dtrw_solns = []
mc_solns = []
dtrw_diffs = []
mc_diffs = []
import pandas as pd

analytic = pd.read_csv('Exact.csv')
at = analytic.transpose()

sns.set_style("whitegrid")

alphas = [0.9, 0.8, 0.7, 0.6, 0.5]
for alpha in alphas:

    dT = pow((dX*dX/ (2.0 * D_alpha)), 1. / alpha)
    num_t = int(T / dT)
    T_grid = dT * num_t
    #print("dT: ", dT, " Difference between final time step and final time: ", num_t * dT - T)
    print(alpha, '\t', num_t * dT, ',\t', num_t, 'x', num_points, 'grid')

    analytic_soln = at.ix[:,(at.ix['Time'] == T) & (at.ix['Alpha'] == alpha)].values[2:].flatten()
    
    mc_start = time.time()
    mc_soln, n = mc_solve(xs, N, num_t, alpha)
    mc_end = time.time()

    dtrw_soln = dtrw_solve(xs, num_t, alpha)
    dtrw_end = time.time()

    mc_times.append(mc_end - mc_start)
    dtrw_times.append(dtrw_end - mc_end)
   
    np.savetxt('DTRW_{0}.csv'.format(alpha), dtrw_soln)
    np.savetxt('MC_{0}_{1}.csv'.format(alpha, N), mc_soln)


    if dX == dX_anal:
        dtrw_diffs.append(np.abs(analytic_soln - np.interp(xs_anal, xs, dtrw_soln.T.flatten() / dX)))
        mc_diffs.append(np.abs(analytic_soln - np.interp(xs_anal, xs, mc_soln.T / (N * dX)) ))

    if dX == 2e-2:
        dtrw_diffs.append(np.abs(analytic_soln[1:-1:2] - dtrw_soln.T.flatten() / dX))
        mc_diffs.append(np.abs(analytic_soln[1:-1:2] - mc_soln.T / (N * dX)))
    
    print("        DTRW      \t MC")
    print("L_inf = {0:6.4e} \t {1:6.4e}".format(dtrw_diffs[-1].max(), mc_diffs[-1].max()))
    print("l_2   = {0:6.4e} \t {1:6.4e}".format(np.linalg.norm(dtrw_diffs[-1]), np.linalg.norm(mc_diffs[-1])))
    print("Time  = {0:6.4e} \t {1:6.4e}".format(dtrw_times[-1], mc_times[-1]))

    c = sns.color_palette("colorblind", 10)
    c = sns.color_palette("deep", 10)

    f, ax = plt.subplots()#figsize=(4, 3))
    plt.xlabel('x')
    plt.ylabel('Solution')
    plt.plot(xs_anal, analytic_soln.T, color=c[0], lw=5, label=r'Analytic')
    plt.plot(xs, mc_soln / (N * dX), '-s', lw=1.5, ms=4, color=c[1], label='Monte Carlo')
    plt.plot(xs, dtrw_soln / dX, '-o', lw=1.5, ms=4, color=c[2], label='DTRW')
    plt.legend()
    plt.savefig('soln_{0}_{1}.pdf'.format(alpha, N))

    f, ax = plt.subplots()#figsize=(4, 3))
    ax.set(yscale="log")
    plt.xlabel('x')
    plt.ylabel('Difference, log scale')
    plt.plot(xs, mc_diffs[-1], 's', markerfacecolor='none', mew=2, mec=c[1], label=r'|Monte Carlo - Analytic|')
    plt.plot(xs, dtrw_diffs[-1], 'o', markerfacecolor='none', mew=2, mec=c[2],  label='|DTRW - Analytic|')
    plt.legend()
    plt.savefig('diff_{0}_{1}.pdf'.format(alpha, N))


#sns.set(style="whitegrid", palette="muted")
f, ax = plt.subplots()#figsize=(4, 3))
ax.set(yscale="log")
plt.plot(alphas, mc_times, 's', markerfacecolor='none', mew=2, markersize=10, mec=c[1], label='Monte Carlo')
plt.plot(alphas, dtrw_times, 'o', markerfacecolor='none', mew=2, markersize=10, mec=c[2], label='DTRW')
plt.xlabel('Alpha')
plt.ylabel('Time, log scale')
plt.legend()
plt.savefig('dtrw_mc_time_{0}.pdf'.format(N))
