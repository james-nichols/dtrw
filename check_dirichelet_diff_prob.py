#! /usr/bin/env python

import numpy as np
import scipy.interpolate, scipy.special
import time, csv, math
from dtrw import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm

import pdb

T = 1.0
L = 2.0
LHS = 1.0
D_alpha = 0.1 
alpha = 0.7

n_points = [10, 20, 40, 80, 160]

fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

for l in n_points:

    dX = L / float(l)
    dT = dX * dX / (2.0 * D_alpha)
    N = int(math.ceil(T / dT))

    # Calculate r for diffusive case so as to get the *same* dT as the subdiffusive case
    r = dT / (dX * dX / (2.0 * D_alpha))
    X_init = np.zeros(np.floor(L / dX))
    xs = np.arange(0.0, L, dX)
        
    print l, N, r

    dtrw = DTRW_diffusive(X_init, N, r, history_length=N, boundary_condition=BC_Dirichelet([LHS, 0.0]))
    dtrw.solve_all_steps()

    diff_analytic_soln = LHS * (1.0 - scipy.special.erf(xs / math.sqrt(4. * D_alpha * T)))
        
    ax1.plot(xs, dtrw.Xs[0][:,:,-1].T, 'g.-')
    ax2.semilogy(xs, dtrw.Xs[0][:,:,-1].T, 'g.-')
    
ax1.plot(xs, diff_analytic_soln, 'gx')
ax2.semilogy(xs, diff_analytic_soln, 'gx')
plt.show()
