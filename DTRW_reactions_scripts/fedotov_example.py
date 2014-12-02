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

""" Here we start the script that does the simulation, complete with plotting routine """

L = 10.0
dX = 0.01
n_points = int(math.floor(L / dX)) 

X_init = np.zeros(n_points) # Adding -dX point for BC
X_init[1] = 1.0 / dX

T = 1.

alpha = 0.9
# Note that it works to take dT = tau
dT = 0.001 #pow((dX * dX / (2.0 * D_alpha)), 1./alpha)
N = int(math.floor(T / dT))
history_length = N

D_alpha = dX * dX / (2.0 * pow(dT, alpha))

# Calculate r for diffusive case so as to get the *same* dT as the subdiffusive case
r = dT / (dX * dX / (2.0 * D_alpha))

print "Diffusive sim with D_alpha =", D_alpha, "dT =", dT, "N =", N, "alpha =", alpha, "diffusion matching r =", r
g = 10.
k = 1.5
bc_constant = g * dX / (D_alpha * pow(k, 1.-alpha))

# Analytic solution
xs = np.linspace(0., L, n_points, endpoint=False)
mu = 2. - alpha
g_star = g / (math.sqrt(D_alpha * pow(k, 2.-alpha)))
pst_0 = pow(g_star * math.sqrt((mu+2.)/(2.*mu)), 2. / (mu+2.))
x_0 = (2. * mu / (2. - mu)) * pow(g_star, -(2.-mu)/(mu+2.)) * pow((mu+2.)/(2.*mu), mu/(mu+2.)) * math.sqrt(D_alpha/pow(k,alpha))
analytic_soln = pst_0 * pow((1 + xs / x_0), -2./alpha)

# Boundary conditions!
fed_bc = BC_Fedotov(alpha, bc_constant, analytic_soln[-1])
dir_bc = BC_Dirichelet([analytic_soln[0], analytic_soln[-1]])
fed_bal_bc = BC_Fedotov_balance()

X_init = analytic_soln

dtrw = DTRW_subdiffusive_fedotov_death(X_init, N, alpha, dT*k, history_length, boundary_condition=fed_bc)
#dtrw = DTRW_subdiffusive_fedotov_death(X_init, N, alpha, 1.-exp(k*dT), history_length, boundary_condition=fed_bc)
dtrw.solve_all_steps()

dtrw_dir = DTRW_subdiffusive_fedotov_death(X_init, N, alpha, dT*k, history_length, boundary_condition=dir_bc)
dtrw_dir.solve_all_steps()

dtrw_bal = DTRW_subdiffusive_fedotov_death(X_init, N, alpha, dT*k, history_length, boundary_condition=fed_bal_bc)
dtrw_bal.solve_all_steps()

print "Solutions computed, now creating animation..."

lo = n_points - 10 
hi = n_points-1 

fig = plt.figure(figsize=(8,8))
plt.xlim(0,L)
plt.xlim(xs[lo], xs[hi])
plt.ylim(0,1.0)
plt.xlabel('x')
line1, = plt.plot([],[],'r-')
line2, = plt.plot([],[],'gx')
line3, = plt.plot([],[],'bo')
line4, = plt.plot([],[],'y^')
plt.legend([line1, line2, line3, line4], ["Analytic Steady State Sol'n", "DTRW Sol'n 1", "Balancing ", "DTRW Sol'n 3" ])

def update(i, line1, line2, line3, line4):
    p_0 = analytic_soln[0]
    p_0_dtrw = dtrw.Xs[0][:,0,i]
    line1.set_data(xs[lo:hi], analytic_soln[lo:hi]/p_0)
    line2.set_data(xs[lo:hi], dtrw.Xs[0][:,lo:hi,i]/p_0)
    line3.set_data(xs[lo:hi], dtrw_bal.Xs[0][:,lo:hi,i]/p_0)
    line4.set_data(xs[lo:hi], dtrw_dir.Xs[0][:,lo:hi,i]/p_0)
    plt.ylim(0., max(dtrw.Xs[0][:,:,i].max(), analytic_soln.max()))
    plt.ylim(0., max(dtrw.Xs[0][:,lo:hi,i].max()/p_0, analytic_soln[lo:hi].max()/p_0))

    print "Fed", (dtrw.Xs[0][:,:,i] - analytic_soln).sum()
    print "Bal", (dtrw_bal.Xs[0][:,:,i] - analytic_soln).sum()
    print "Dir", (dtrw_dir.Xs[0][:,:,i] - analytic_soln).sum()
    return line1, line2, line3, line4

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, update, 
        frames=N, fargs=(line1, line2, line3, line4), interval=10)

import inspect, os, subprocess
exec_name =  os.path.splitext(os.path.basename(inspect.getfile(inspect.currentframe())))[0]
git_tag = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).replace('\n', '')

file_name = '{0}_{1}.mp4'.format(exec_name, git_tag)
print "Saving animation to", file_name

#anim.save(file_name, fps=24)
plt.show()

