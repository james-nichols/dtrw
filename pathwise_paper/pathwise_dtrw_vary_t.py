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
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from dtrw import *
import pdb

from pathwise_dtrw import * 

######################
# Compare approaches #
######################
if __name__ == "__main__":
    
    # Number of MC points
    N = int(1e6)

    alpha = 0.7
    D_alpha = 0.1
    dX = 2.e-2
    end_points = [-1., 1.]
    num_points = int((end_points[1]-end_points[0])/dX) 
    xs = np.linspace(end_points[0] + 0.5 * dX, end_points[1] - 0.5 * dX, num_points)

    dtrw_times = []
    mc_times = []
    dtrw_solns = []
    mc_solns = []
    dtrw_diffs = []
    mc_diffs = []

    sns.set_style("whitegrid")

    Ts = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for T in Ts:
        
        dT = pow((dX*dX/ (2.0 * D_alpha)), 1. / alpha)
        num_t = int(T / dT)
        T_grid = dT * num_t
        
        print(T, '\t', num_t * dT, ',\t', num_t, 'x', num_points, 'grid')
        
        mc_start = time.clock()
        mc_soln, n = mc_solve(xs, N, num_t, alpha)
        mc_end = time.clock()

        dtrw_soln = dtrw_solve(xs, num_t, alpha)
        dtrw_end = time.clock()

        mc_times.append(mc_end - mc_start)
        dtrw_times.append(dtrw_end - mc_end)
       
    c = sns.color_palette("deep", 10)

    f, ax = plt.subplots()#figsize=(4, 3))
    ax.set(yscale="log")
    plt.plot(Ts, mc_times, 's', markerfacecolor='none', mew=2, markersize=10, mec=c[1], label='Monte Carlo')
    plt.plot(Ts, dtrw_times, 'o', markerfacecolor='none', mew=2, markersize=10, mec=c[2], label='DTRW')
    plt.xlabel('T')
    plt.ylabel('Computation time, log scale')
    plt.legend()
    plt.savefig('dtrw_mc_T_comp_time_{0}.pdf'.format(N))
