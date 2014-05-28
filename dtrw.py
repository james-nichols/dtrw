#! /usr/bin/env python

import math
import numpy as np
import scipy as sp
import scipy.signal
from itertools import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation

import pdb

class DTRW_with_reactions(object):
    """ Base definition of a DTRW with arbitrary wait-times
        for reactions and jumps """

    def __init__(self, X, N, dT, tau, omega):
        """X is the initial concentration field, N is the number of time steps to simulate"""
        self.X = np.dstack([X])
        self.Q = np.dstack([X])
        self.N = N
        self.dT = dT
        self.tau = tau
        
        # How many time steps have preceded the current one?
        self.n = 0 # self.X.shape[-1]

        self.calc_lambda()
        self.calc_psi()
        self.calc_Phi()
        self.calc_omega(omega * self.dT)
        self.calc_theta()

        # Can't actually compute the memory kernel for now...
        # self.calc_mem_kernel()
    
    def calc_mem_kernel(self):
        """Once off call to calculate the memory kernel and store it"""
        self.K = np.zeros(self.N)
        
        # TODO FILL THIS IN!!!

    def calc_lambda(self):
        """ Basic equal finite mean diffusion """
        self.lam = np.array( [[0.00, 0.25, 0.00],
                              [0.25, 0.00, 0.25],
                              [0.00, 0.25, 0.00]])

    def calc_psi(self):
        """Waiting time distribution for spatial jumps"""
        t = np.array(range(self.N)) * self.dT
        #self.psi = (1. / self.tau) * np.exp(-t / self.tau) 
        self.psi = np.exp(-t / self.tau)
        self.psi[:-1] -= np.exp(-t[1:] / self.tau)
         
    def calc_Phi(self):
        """PDF of not jumping up to time n"""
        self.Phi = np.ones(N)
        for i in range(N):
            # unsure about ending index - might need extra point...
            self.Phi[i] -= sum(self.psi[:i])

    def calc_omega(self, om):
        """ Likelihood of surviving death between n and n+1"""

        # For now just constant for the length of time
        self.omega = om * np.ones(self.N)
    
    def calc_theta(self):
        """Likelihood of death happening between 0 and n"""
        self.theta = np.zeros(self.N)
        
        # Note that this only works for constant theta at the moment
        for i in range(N):
            self.theta[i] = 1.0 - sum(self.omega[:i])
    
    def time_step(self):
        """Take a time step forward"""
        
        # How many time steps have we had?
        self.n = self.X.shape[2] 
        
        next_Q = np.zeros(self.Q.shape[:2]) 
        for i in range(self.Q.shape[0]):
            for j in range(self.Q.shape[1]):
                next_Q[i,j] = sum(self.Q[i, j, :] * self.psi[:self.n][::-1] * self.theta[:self.n][::-1])
        
        # Now apply lambda jump probabilities
        next_Q = sp.signal.convolve2d(next_Q, self.lam, 'same')
        
        # Add Q to the list of Qs over time
        self.Q = np.dstack([self.Q, next_Q])

        # Now find X from Q 
        next_X = np.zeros(self.Q.shape[:2])
        for i in range(self.Q.shape[0]):
            for j in range(self.Q.shape[1]):
                next_X[i,j] = sum(self.Q[i, j, :] * self.Phi[:self.n+1][::-1] * self.theta[:self.n+1][::-1])
        
        self.X = np.dstack([self.X, next_X])
        
        """
        # outward flux 
        flux = np.zeros(self.X.shape[:2])

        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                flux(i,j) = sum(self.X[i, j, :] * self.K[:self.n][::-1] * self.theta[:self.n][::-1])
        
        # update the field
        next_X = sp.signal.convolve2d(flux, self.lam, 'same') - i - self.omega(n) * self.X[:,:,-1]
       
        # stack next_X on to the list of fields X - giving us another layer in the 3d array of spatial results
        X = np.dstack((X, next_X))
        """
    def solve_all_steps(self):

        for i in range(self.N-1):
            self.time_step()


X_init = np.zeros([100, 100])
X_init[49,49] = 1.0

N = 100
dT = 0.1
tau = 0.1
omega = 0.01
dtrw = DTRW_with_reactions(X_init, N, dT, tau, omega)

dtrw.solve_all_steps()

fig = plt.figure()
ax = Axes3D(fig)
xs = np.linspace(0, 1, X_init.shape[0])
ys = np.linspace(0, 1, X_init.shape[1])
Xs, Ys = np.meshgrid(xs, ys)

wframe = ax.plot_wireframe(Xs, Ys, X_init, rstride=2, cstride=2)
ax.set_zlim(0.0,1)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Particle density')
#title = ax.text(.5, 1.05, '', transform = ax.transAxes, va='center')
#title = ax.text(0.5, 0.5, 0., 'Hi', zdir=None)

# initialization function: plot the background of each frame
def init():
    surface1.set_data([], [], [])
    return surface1

def update(i, ax, fig):
    ax.cla()
    wframe = ax.plot_wireframe(Xs, Ys, dtrw.X[:,:,i], rstride=2, cstride=2)
    ax.set_zlim(0.0,1.1 * dtrw.X[:,:,i].max())
    #title.set_label('Total = ' + str(dtrw.X[:,:,i].sum()))
    print dtrw.X[:,:,i].sum()
    print dtrw.X[47:53,47:53,i]
    return wframe,

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, update, 
        frames=N, 
        fargs=(ax, fig), interval=100)
plt.show()

