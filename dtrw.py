#! /usr/bin/env python

import math
import numpy as np
import scipy as sp
import scipy.signal
import scipy.special
from itertools import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm

import pdb

class DTRW(object):
    """ Base definition of a DTRW with arbitrary wait-times
        for reactions and jumps """

    def __init__(self, X, N, dT, tau, omega, nu):
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
        self.calc_nu(nu * self.dT)

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
        
        #self.psi = np.exp(-t / self.tau)
        #self.psi[:-1] -= np.exp(-t[1:] / self.tau)
        self.psi = np.zeros(self.N) 
        self.psi[1:] = np.exp(-t[:-1] / self.tau)
        self.psi[1:-1] -= np.exp(-t[1:-1] / self.tau)

    def calc_Phi(self):
        """PDF of not jumping up to time n"""
        self.Phi = np.ones(N)
        for i in range(N):
            # unsure about ending index - might need extra point...
            self.Phi[i] -= sum(self.psi[1:i+1])

    def calc_omega(self, om):
        """ Likelihood of surviving between n and n+1"""

        # For now just constant for the length of time
        #self.omega = om * np.ones(self.N)
        self.omega = np.exp(-om * self.dT) * np.ones(self.N)

    def calc_theta(self):
        """Likelihood of surviving between 0 and n"""
        self.theta = np.zeros(self.N)
        
        # Note that this only works for constant theta at the moment
        for i in range(N):
            self.theta[i] = self.omega[:i].prod()
    
    def calc_nu(self, birth):
        """Likelihood of birth happening at i"""
        
        # For now just constant for the length of time
        self.nu = birth * np.ones(self.N)
    
    def time_step(self):
        """Take a time step forward"""
        
        # How many time steps have we had?
        self.n = self.X.shape[2] 
        
        next_Q = np.zeros(self.Q.shape[:2]) 
        for i in range(self.Q.shape[0]):
            for j in range(self.Q.shape[1]):
                # Note from this equation that psi[0] is applied to Q[n], that is psi[0] is equivalent to psi(0) from the paper... ((1))
                next_Q[i,j] = sum(self.Q[i, j, :] * self.psi[1:self.n+1][::-1] * self.theta[1:self.n+1][::-1]) + self.nu[self.n] * self.X[i,j,-1]
                
        # Now apply lambda jump probabilities
        next_Q = sp.signal.convolve2d(next_Q, self.lam, 'same')
        
        # Add Q to the list of Qs over time
        self.Q = np.dstack([self.Q, next_Q])

        # Now find X from Q 
        next_X = np.zeros(self.Q.shape[:2])
        for i in range(self.Q.shape[0]):
            for j in range(self.Q.shape[1]):
                # Similarly to point ((1)), here Phi[0] is applied to Q[n+1], therefore Phi[0] is equivalent to Phi(0) from the paper ((2))
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

class DTRW_subdiffusive(DTRW):

    def __init__(self, X, N, dT, tau, omega, nu):
        super(DTRW_subdiffusive, self).__init__(X, N, dT, tau, omega, nu)

    def calc_psi(self):
        """Waiting time distribution for spatial jumps"""
        t = np.array(range(self.N)) * self.dT
        
        tn = np.array(range(self.N))
        self.psi = pow(-1,tn+1) * scipy.special.gamma(self.tau + 1) / (scipy.special.gamma(tn + 1) * scipy.special.gamma(self.tau - tn + 1))
        self.psi[0] = 0.
        #self.psi[:-1] -= self.psi[1:]
         
    def calc_Phi(self):
        """PDF of not jumping up to time n"""
        
        tn = np.array(range(0,self.N))
        self.Phi = pow(-1,tn) * scipy.special.gamma(self.tau ) / (scipy.special.gamma(tn + 1) * scipy.special.gamma(self.tau - tn))
        self.Phi[0] = 1.
        #self.Phi = np.ones(N)
        #for i in range(N):
            # unsure about ending index - might need extra point...
        #    self.Phi[i] -= sum(self.psi[:i])

    

X_init = np.zeros([100, 100])
X_init[50,50] = 1.0
X_init[50,10] = 1.0
X_init[80,85] = 1.0

N = 10
dT = 0.5
tau = 0.5
alpha = 0.9
tau2 = 0.2
omega = 0.0
nu = 0.0
dtrw = DTRW(X_init, N, dT, tau, omega, nu)
#dtrw_sub = DTRW(X_init, N, dT, tau2, omega, nu)
dtrw_sub = DTRW_subdiffusive(X_init, N, dT, alpha, omega, nu)
pdb.set_trace()
print dtrw.psi, dtrw.psi.sum()
print dtrw_sub.psi, dtrw_sub.psi.sum()
print dtrw.Phi
print dtrw_sub.Phi

dtrw.solve_all_steps()
dtrw_sub.solve_all_steps()

xs = np.linspace(0, 1, X_init.shape[0])
ys = np.linspace(0, 1, X_init.shape[1])
Xs, Ys = np.meshgrid(xs, ys)

fig = plt.figure()
ax = Axes3D(fig)
wframe = ax.plot_surface(Xs, Ys, X_init, rstride=5, cstride=5)
ax.set_zlim(-0.1,1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Particle density')

#title = ax.text(.5, 1.05, '', transform = ax.transAxes, va='center')
#title = ax.text(0.5, 0.5, 0., 'Hi', zdir=None)

def update(i, ax, fig):
    
    ax.cla()
    wframe = ax.plot_surface(Xs, Ys, dtrw.X[:,:,i], rstride=5, cstride=5, color='Blue', alpha=0.2)
    wframe2 = ax.plot_surface(Xs, Ys, dtrw_sub.X[:,:,i], rstride=5, cstride=5, color='Red', alpha=0.2)

    plot_max = max(dtrw.X[:,:,i].max(),dtrw_sub.X[:,:,i].max())
    cset = ax.contour(Xs, Ys, dtrw.X[:,:,i], zdir='z', offset=-0.1 * plot_max, cmap=cm.coolwarm)
    cset = ax.contour(Xs, Ys, dtrw.X[:,:,i], zdir='x', offset=0., cmap=cm.coolwarm)
    cset = ax.contour(Xs, Ys, dtrw.X[:,:,i], zdir='y', offset=1., cmap=cm.coolwarm)
    ax.set_zlim(-0.1 * plot_max,1.1 * plot_max)
    #ax.set_zlim(0.0,1.1 * dtrw_sub.X[:,:,i].max())
    
    return wframe, wframe2

def update2(i, ax, fig):
    ax.cla()
    wframe = ax.plot_wireframe(Xs, Ys, dtrw.X[:,:,i] - dtrw_sub.X[:,:,i], rstride=5, cstride=5, color='Red')
    ax.set_zlim(-0.1, 1.1 * (dtrw.X[:,:,i]-dtrw_sub.X[:,:,i]).max())
    return wframe,

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, update, 
        frames=N, 
        fargs=(ax, fig), interval=100)

#anim.save('basic_animation.mp4', fps=30)
plt.show()

