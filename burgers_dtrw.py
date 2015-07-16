#! /usr/bin/env python

import numpy as np
import time
from dtrw import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm

class DTRW_burgers(DTRW_diffusive):
    
    def __init__(self, X_inits, N, omega, k, r = 1., history_length = 0, boltz_beta = 0., potential = np.array([]), boundary_condition=BC()):
        
        self.k = k

        super(DTRW_burgers, self).__init__(X_inits, N, omega, r, history_length, boltz_beta, potential, boundary_condition)
  
    def calc_potential(self):
        #return self.Xs[0][::-1,::-1,self.n] #* self.Xs[0][:,:,self.n]
        return np.cumsum(self.Xs[0][:,:,self.n])

    def calc_omega(self):
        """ Probability of death between n and n+1"""
        if self.omegas == None:
            self.omegas = [ (1. - np.exp(-self.k)) * np.ones((X.shape[0], X.shape[1])) for X in self.Xs]


n_points = 101
L = 10.0
dX = L / (n_points-1)

xs = np.linspace(0, 1, n_points, endpoint=True)

sd = 0.05
X_init = np.exp(-(xs - 0.5) * (xs - 0.5) / (sd * sd)) / (sd * np.sqrt(2. * math.pi))
X_init[10] = 10.
X_init[22] = 10.
T = 100.0 

alpha = 1.0 
D_alpha = 0.01
r = 0.1

dT = pow((r * dX * dX / (2.0 * D_alpha)), 1./alpha)
N = int(math.floor(T / dT))
history_length = N

# Calculate r for diffusive case so as to get the *same* dT as the subdiffusive case
#r = dT / (dX * dX / (2.0 * D_alpha))

k = dT * 0.
beta = 2.0 * dX
dtrw = DTRW_burgers([X_init], N, r, k, history_length=history_length, boltz_beta=beta, boundary_condition = BC_zero_flux())
dtrw_no_boltz = DTRW_burgers([X_init], N, r, k, history_length=history_length, boltz_beta=0.0, boundary_condition = BC_zero_flux())

print "Solving for", N, "steps, dT =", dT, ", diffusion matching gives r =", r

start = time.clock()
dtrw.solve_all_steps()
dtrw_no_boltz.solve_all_steps()
end = time.clock()

print "Time for solution: ", end - start

X = dtrw.Xs[0]
Xb = dtrw_no_boltz.Xs[0]

fig = plt.figure(figsize=(16,16))
plt.xlabel('x')
line1, = plt.plot(xs,X[:,:,0].T,'r-')
line2, = plt.plot(xs,Xb[:,:,0].T,'g-')

def update(i, line1, line2):

    line1.set_data(xs,X[:,:,i])
    line2.set_data(xs,Xb[:,:,i])

    return line1, line2

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, update, 
        frames=N, 
        fargs=(line1, line2), interval=10)

import inspect, os, subprocess
exec_name =  os.path.splitext(os.path.basename(inspect.getfile(inspect.currentframe())))[0]
git_tag = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).replace('\n', '')

file_name = '{0}_{1}.mp4'.format(exec_name, git_tag)
print "Saving animation to", file_name

#anim.save(file_name, fps=24)#, extra_args=['-vcodec', 'libx264'])
plt.show()
