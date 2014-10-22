#! /usr/bin/env python

import numpy as np
import time
from dtrw import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm

import pdb

n_points = 100
L = 5.0
dX = L / n_points

X_init = np.zeros(n_points)
X_init[n_points / 2] = 1.0

T = 10.0

alpha_1 = 0.9
alpha_2 = 0.75
alpha_3 = 0.6
D_alpha = 0.1

dT_1 = pow((dX * dX / (2.0 * D_alpha)), 1./alpha_1)
dT_2 = pow((dX * dX / (2.0 * D_alpha)), 1./alpha_2)
dT_3 = pow((dX * dX / (2.0 * D_alpha)), 1./alpha_3)
N_1 = int(math.floor(T / dT_1))
N_2 = int(math.floor(T / dT_2))
N_3 = int(math.floor(T / dT_3))

# Calculate r for diffusive case so as to get the *same* dT as the subdiffusive case
#r = dT / (dX * dX / (2.0 * D_alpha))

dtrw_sub_1 = DTRW_subdiffusive(X_init, N_1, alpha_1, N_1)
dtrw_sub_2 = DTRW_subdiffusive(X_init, N_2, alpha_2, N_2)
dtrw_sub_3 = DTRW_subdiffusive(X_init, N_3, alpha_3, N_3)

dtrw_sub_1.solve_all_steps()
dtrw_sub_2.solve_all_steps()
dtrw_sub_3.solve_all_steps()

ts_1 = np.arange(0., T, dT_1)[:-1]
ts_2 = np.arange(0., T, dT_2)[:-1]
ts_3 = np.arange(0., T, dT_3)[:-1]

print "Solutions computed, now creating animation..."

xs = np.linspace(0., L, n_points, endpoint=False)
msd_1 = np.zeros(N_1)
msd_2 = np.zeros(N_2)
msd_3 = np.zeros(N_3)
for i in range(N_1):
    msd_1[i] = (dtrw_sub_1.Xs[0][:,:,i] * (xs - xs[n_points/2]) * (xs - xs[n_points/2])).sum()
for i in range(N_2):
    msd_2[i] = (dtrw_sub_2.Xs[0][:,:,i] * (xs - xs[n_points/2]) * (xs - xs[n_points/2])).sum()
for i in range(N_3):
    msd_3[i] = (dtrw_sub_3.Xs[0][:,:,i] * (xs - xs[n_points/2]) * (xs - xs[n_points/2])).sum()

plt.loglog(ts_1, msd_1, ts_2, msd_2, ts_3, msd_3)
plt.show()
