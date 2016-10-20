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

def mc_solve(N, nt, alpha):
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

def dtrw_solve():

    X_init = np.zeros(xs.shape)
    X_init[num_points//2] = 1.
   
    bc = BC_zero_flux()
    dtrw = DTRW_subdiffusive(X_init, num_t, alpha, boundary_condition = bc)
    dtrw.solve_all_steps()

    return dtrw.Xs[0][:,:,-1].T

#####################
# Compare approaches
#####################

# Number of MC points
N = 100000

T = 1.0 
D_alpha = 0.1
dX = 1.e-2
end_points = [-1., 1.]
num_points = int((end_points[1]-end_points[0])/dX) + 1
xs = np.linspace(end_points[0], end_points[1], num_points)

dtrw_times = []
mc_times = []
dtrw_solns = []
mc_solns = []
dtrw_diffs = []
mc_diffs = []
import pandas as pd

analytic = pd.read_csv('Exact.csv')
at = analytic.transpose()

alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
for alpha in alphas:

    dT = pow((dX*dX/ (2.0 * D_alpha)), 1. / alpha)
    num_t = int(T / dT)
    T_grid = dT * num_t
    #print("dT: ", dT, " Difference between final time step and final time: ", num_t * dT - T)
    print(alpha, '\t', num_t * dT, ',\t', num_t, 'x', num_points, 'grid')

    analytic_soln = at.ix[:,(at.ix['Time'] == 1.0) & (at.ix['Alpha'] == alpha)].values[2:].flatten()
    
    mc_start = time.time()
    mc_soln, n = mc_solve(N, num_t, alpha)
    mc_end = time.time()

    dtrw_soln = dtrw_solve()
    dtrw_end = time.time()

    mc_times.append(mc_end - mc_start)
    dtrw_times.append(dtrw_end - mc_end)

    dtrw_diffs.append(np.abs(analytic_soln - dtrw_soln))
    mc_diffs.append(np.abs(analytic_soln - mc_soln))
    
    plt.plot(xs, mc_soln / (N * dX))
    plt.plot(xs, dtrw_soln / dX)
    plt.plot(xs, analytic_soln)
    plt.show()

#sns.set(style="whitegrid", palette="muted")
f, ax = plt.subplots(figsize=(7, 7))
ax.set(yscale="log")
plt.scatter(alphas, np.log(mc_times))
plt.scatter(alphas, np.log(dtrw_times))
plt.show()
