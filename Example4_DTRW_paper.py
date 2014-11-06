#! /usr/bin/env python

import numpy as np
import time
from dtrw import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
import pdb

n_points = 40
L = 1.0
dX = L / n_points  
X_init = np.zeros(n_points)
X_init[0] = 1.0 / dX

T = 0.01
alpha = 0.55
D_alpha = 1.

dT = pow((dX * dX / (2.0 * D_alpha)), 1./alpha)
N = int(math.floor(T / dT))
history_length = N

xs = np.linspace(0., L, n_points, endpoint=False)

boltz_beta = 0.5
potential = np.cos(2. * math.pi * xs)

omega = 0.0 #0.05
nu = 0.0 #0.0005

boundary_condition = BC_periodic()

#dtrw = DTRW_diffusive(X_init, N, dT, tau, omega, nu, history_length, boundary_condition)
dtrw_sub = DTRW_subdiffusive(X_init, N, alpha, history_length, boltz_beta, potential, boundary_condition)
#dtrw.solve_all_steps()

print "Solving DTRW for", N, "time steps to time", T, ". dT =", dT

dtrw_sub.solve_all_steps()

np.save("Example4_results", dtrw_sub.X)

fig = plt.figure(figsize=(8,8))
plt.xlim(0,1)
plt.ylim(dtrw_sub.X.min(), dtrw_sub.X.max())
plt.xlabel('x')
line1, = plt.plot([],[],'r-')
#line2, = plt.plot([],[],'g-')

def update(i, line1):
    #line1.set_data(xs,dtrw.X[:,:,i])
    line1.set_data(xs,dtrw_sub.X[:,:,i])
    return line1, 

# call the animator. blit=True means only re-draw the parts that have changed.
#anim = animation.FuncAnimation(fig, update, 
#        frames=N, fargs=(line1,), interval=10)

#anim.save('Example4.mp4', fps=24)
#plt.show()

