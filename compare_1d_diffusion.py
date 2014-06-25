#! /usr/bin/env python

import numpy as np
import time
from dtrw import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
import pdb
X_init = np.zeros(100)
X_init[50] = 1.0

N = 500
history_length = N
dT = 0.5
alpha = 0.5
tau = 0.5
omega = 0.0 #0.05
nu = 0.0 #0.0005

dtrw = DTRW_diffusive(X_init, N, dT, tau, omega, nu, history_length)
dtrw_sub = DTRW_subdiffusive(X_init, N, alpha, omega, nu, history_length)
dtrw.solve_all_steps()
dtrw_sub.solve_all_steps()

xs = np.linspace(0, 1, X_init.shape[0])

fig = plt.figure(figsize=(8,8))
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('x')
line1, = plt.plot([],[],'r-')
line2, = plt.plot([],[],'g-')

def update(i, line1, line2):
    line1.set_data(xs,dtrw.X[:,:,i])
    line2.set_data(xs,dtrw_sub.X[:,:,i])
    return line1, line2

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, update, 
        frames=N, fargs=(line1, line2), interval=10)

anim.save('subdiffusion_line.mp4', fps=24)
plt.show()

