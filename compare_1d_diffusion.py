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
X_init[n_points / 2] = 1.0

T = 1.0

alpha = 0.75
D_alpha = 0.1

dT = pow((dX * dX / (2.0 * D_alpha)), 1./alpha)
N = int(math.floor(T / dT))
history_length = N

# Calculate r for diffusive case so as to get the *same* dT as the subdiffusive case
r = dT / (dX * dX / (2.0 * D_alpha))

print "Diffusive sim with dT =", dT, "N =", N, "alpha =", alpha, "diffusion matching r =", r

dtrw = DTRW_diffusive(X_init, N, r, history_length)
dtrw_sub = DTRW_subdiffusive(X_init, N, alpha, history_length)

print "Left jump probs: ", dtrw.lam[:,:,0]
print "Right jump probs: ", dtrw.lam[:,:,1]

dtrw.solve_all_steps()
dtrw_sub.solve_all_steps()

print "Solutions computed, now creating animation..."

xs = np.linspace(0., L, n_points, endpoint=False)

fig = plt.figure(figsize=(8,8))
plt.xlim(0,1)
plt.ylim(0,0.2)
plt.xlabel('x')
line1, = plt.plot([],[],'r-')
line2, = plt.plot([],[],'g-')
line3, = plt.plot([],[],'b-')

def update(i, line1, line2, line3):
    line1.set_data(xs,dtrw.Xs[0][:,:,i])
    line2.set_data(xs,dtrw_sub.Xs[0][:,:,i])
    if i == 0:
        analytic_soln = X_init
    else:
        analytic_soln = (1./math.sqrt(4. * math.pi * float(i) * dT * D_alpha)) * np.exp( - (xs - 0.5) * (xs - 0.5) / (4. * D_alpha * float(i) * dT)) * dX
    line3.set_data(xs, analytic_soln)
    return line1, line2, line3

# call the animator. blit=True means only re-draw the parts that have changed.
# NOTE the comparison between diffusive and subdiffusive behaviour in this animator 
# is NOT correct (for the time being) as time steps are fundamentally different between the two.
# Perhaps in future it might be good to include a time step interpolator for comparing the behaviours.
anim = animation.FuncAnimation(fig, update, 
        frames=N, fargs=(line1, line2, line3), interval=10)

import inspect, os, subprocess
exec_name =  os.path.splitext(os.path.basename(inspect.getfile(inspect.currentframe())))[0]
git_tag = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).replace('\n', '')

file_name = '{0}_{1}.mp4'.format(exec_name, git_tag)
print "Saving animation to", file_name

anim.save(file_name, fps=24)
plt.show()

