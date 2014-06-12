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

    def __init__(self, X, N, dT, tau, omega, nu, hist_length=2):
        """X is the initial concentration field, N is the number of time steps to simulate"""
        self.X = np.dstack([X])
        self.Q = np.dstack([X])
        self.N = N
        self.dT = dT
        self.tau = tau

        self.history_length = hist_length

        # How many time steps have preceded the current one?
        self.n = 0 # self.X.shape[-1]

        self.calc_lambda()
        self.calc_psi()
        self.calc_Phi()
        self.calc_omega(omega * self.dT)
        self.calc_theta()
        self.calc_nu(nu * self.dT)
        
        self.calc_mem_kernel()
    
    def calc_mem_kernel(self):
        """Once off call to calculate the memory kernel and store it"""
        self.K = np.zeros(self.history_length+1)
         
        # In the exponential case it's quite simply...
        self.K[1] = (1.0 - np.exp(-self.dT / self.tau))

    def calc_lambda(self):
        """ Basic equal finite mean diffusion """
        self.lam = np.array( [[0.00, 0.24, 0.00],
                              [0.24, 0.00, 0.26],
                              [0.00, 0.26, 0.00]])

    def calc_psi(self):
        """Waiting time distribution for spatial jumps"""
        t = np.array(range(self.history_length+1)) * self.dT
        
        self.psi = np.zeros(self.history_length+1) 
        self.psi[1:] = np.exp(-t[:-1] / self.tau)
        self.psi[1:-1] -= np.exp(-t[1:-1] / self.tau)

    def calc_Phi(self):
        """PDF of not jumping up to time n"""
        self.Phi = np.ones(self.history_length+1)
        for i in range(self.history_length+1):
            # unsure about ending index - might need extra point...
            self.Phi[i] -= sum(self.psi[1:i+1])

    def calc_omega(self, om):
        """ Likelihood of surviving between n and n+1"""

        # For now just constant for the length of time
        self.omega = (1.0 - np.exp(-om * self.dT)) * np.ones(self.N)

    def calc_theta(self):
        """Likelihood of surviving between 0 and n"""
        self.theta = np.zeros(self.N)
        
        # Note that this only works for constant theta at the moment
        for i in range(self.N):
            self.theta[i] = (1.0 - self.omega[:i]).prod()
    
    def calc_nu(self, birth):
        """Likelihood of birth happening at i"""
        
        # For now just constant for the length of time
        self.nu = birth * np.ones(self.N)
    
    def time_step(self):
        """Take a time step forward"""
        
        # How many time steps have we had?
        self.n = self.X.shape[2] 
       
        lookback = min( self.n, self.history_length )

        next_Q = np.zeros(self.Q.shape[:2]) 
        for i in range(self.Q.shape[0]):
            for j in range(self.Q.shape[1]):
                # Note from this equation that psi[0] is applied to Q[n], that is psi[0] is equivalent to psi(0) from the paper... ((1))
                next_Q[i,j] = sum(self.Q[i, j, -lookback:] * self.psi[1:lookback+1][::-1] * self.theta[-lookback:]) + self.nu[self.n] #* self.X[i,j,-1]
            
        # Now apply lambda jump probabilities
        next_Q = sp.signal.convolve2d(next_Q, self.lam, 'same')
        
        # Applying zero-flux boundary condition
        #next_Q[:, 0] = next_Q[:, 1]
        #next_Q[0, :] = next_Q[1, :]
        #next_Q[:, -1] = next_Q[:, -2]
        #next_Q[-1, :] = next_Q[-2, :]
            
        # Add Q to the list of Qs over time
        self.Q = np.dstack([self.Q, next_Q])

        lookback = min( self.n+1, self.history_length )
        
        # Now find X from Q 
        next_X = np.zeros(self.Q.shape[:2])
        for i in range(self.Q.shape[0]):
            for j in range(self.Q.shape[1]):
                # Similarly to point ((1)), here Phi[0] is applied to Q[n+1], therefore Phi[0] is equivalent to Phi(0) from the paper ((2))               
                next_X[i,j] = sum(self.Q[i, j, -lookback:] * self.Phi[:lookback][::-1] * self.theta[-lookback:])
        
        self.X = np.dstack([self.X, next_X])
        
    def time_step_with_K(self):
        """ Step forwards directly with X using the memory kernel K, this
            method is only available in cases where we can calculate K analytically"""
 
        # How many time steps have we had?
        self.n = self.X.shape[2] 
        lookback = min(self.n, self.history_length)

        # outward flux 
        flux = np.zeros(self.X.shape[:2])
    
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                # Note that we are assuming the K(0)=0 here, as we only apply K(1) onwards to X...
                flux[i,j] = sum(self.X[i, j, -lookback:] * self.K[1:lookback+1][::-1] * self.theta[-lookback:])
        
        # update the field
        next_X = self.X[:,:,-1] + sp.signal.convolve2d(flux, self.lam, 'same') - flux - self.omega[self.n] * self.X[:,:,-1]
      
        # BUT WHAT ABOUT THE BOUNDARY CONDITIONS!!!!!!!

        # stack next_X on to the list of fields X - giving us another layer in the 3d array of spatial results
        self.X = np.dstack((self.X, next_X))

    def solve_all_steps(self):

        for i in range(self.N-1):
            self.time_step()
    
    def solve_all_steps_with_K(self):
        """Solve the time steps using the memory kernel, available only if we know how to calculate K"""
        for i in range(self.N-1):
            self.time_step_with_K()


class DTRW_subdiffusive(DTRW):

    def __init__(self, X, N, dT, tau, omega, nu, history_length):
        super(DTRW_subdiffusive, self).__init__(X, N, dT, tau, omega, nu, history_length)

    def calc_psi(self):
        """Waiting time distribution for spatial jumps"""
        tn = np.array(range(self.history_length+1))
        self.psi = pow(-1,tn+1) * scipy.special.gamma(self.tau + 1) / (scipy.special.gamma(tn + 1) * scipy.special.gamma(self.tau - tn + 1))
        self.psi[0] = 0.
        # Highly dubious: Normalising psi so that we conservation of particles
        self.psi[-1] = 1.0 - self.psi[:-1].sum()
         
    def calc_Phi(self):
        """PDF of not jumping up to time n"""
        tn = np.array(range(self.history_length+1))
        self.Phi = pow(-1,tn) * scipy.special.gamma(self.tau ) / (scipy.special.gamma(tn + 1) * scipy.special.gamma(self.tau - tn))
        self.Phi[0] = 1.
        #self.Phi = np.ones(N)
        #for i in range(N):
            # unsure about ending index - might need extra point...
        #    self.Phi[i] -= sum(self.psi[:i])

    def calc_mem_kernel(self):
        """Once off call to calculate the memory kernel and store it"""
        self.K = np.zeros(self.history_length+1)
        
        self.K[0] = 0.0
        self.K[1] = scipy.special.gamma(self.tau) / (scipy.special.gamma(self.tau - 1.0) * 2.0) + 1.0

        for i in range(2,self.history_length+1):
            self.K[i] = (float(i) + self.tau - 2.0) * self.K[i-1] / float(i)
    

X_init = np.zeros([100, 100])
X_init[50,50] = 1.0
X_init[50,10] = 1.0
X_init[80,85] = 1.0

N = 1000
dT = 0.5
tau = 0.5
alpha = 0.5
tau2 = 0.2
omega = 0.0 #0.05
nu = 0.0 #0.0005

history_length = 40

dtrw = DTRW(X_init, N, dT, tau, omega, nu, history_length)
dtrw_X = DTRW(X_init, N, dT, tau, omega, nu, history_length)
dtrw_sub = DTRW_subdiffusive(X_init, N, dT, alpha, omega, nu, history_length)
dtrw_sub_X = DTRW_subdiffusive(X_init, N, dT, alpha, omega, nu, history_length)

print dtrw.psi, dtrw.psi.sum()
print dtrw_sub.psi, dtrw_sub.psi.sum()
print dtrw.Phi
print dtrw_sub.Phi

#dtrw.solve_all_steps()
#dtrw_X.solve_all_steps_with_K()
dtrw_sub.solve_all_steps()
dtrw_sub_X.solve_all_steps_with_K()

xs = np.linspace(0, 1, X_init.shape[0])
ys = np.linspace(0, 1, X_init.shape[1])
Xs, Ys = np.meshgrid(xs, ys)

fig = plt.figure(figsize=(24,8))
#ax = Axes3D(fig)
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
wframe = ax1.plot_surface(Xs, Ys, X_init, rstride=5, cstride=5)
ax1.set_zlim(-0.1,1)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('Particle density')

ax2 = fig.add_subplot(1, 3, 2, projection='3d')
wframe2 = ax2.plot_surface(Xs, Ys, X_init, rstride=5, cstride=5)
ax2.set_zlim(-0.1,1)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('Particle density')

ax3 = fig.add_subplot(1, 3, 3, projection='3d')
wframe3 = ax3.plot_surface(Xs, Ys, X_init - X_init, rstride=5, cstride=5)
ax3.set_zlim(-0.1,1)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('Particle density')
#title = ax.text(.5, 1.05, '', transform = ax.transAxes, va='center')
#title = ax.text(0.5, 0.5, 0., 'Hi', zdir=None)

def update(i, ax1, ax2, ax3, fig):
    
    #ax = fig.get_axes(1, 3, 1)
    ax1.cla()
    wframe = ax1.plot_surface(Xs, Ys, dtrw_sub.X[:,:,i], rstride=5, cstride=5, color='Blue', alpha=0.2)
    ax1.set_zlim(-0.1 * dtrw_sub.X[:,:,i].max(), 1.1 * dtrw_sub.X[:,:,i].max())

    #ax = fig.get_axes(1, 3, 2) 
    ax2.cla()
    wframe2 = ax2.plot_surface(Xs, Ys, dtrw_sub_X.X[:,:,i], rstride=5, cstride=5, color='Red', alpha=0.2)
    ax2.set_zlim(-0.1 * dtrw_sub_X.X[:,:,i].max(), 1.1 * dtrw_sub_X.X[:,:,i].max())
    #wframe2 = ax2.plot_surface(Xs, Ys, dtrw_sub.X[:,:,i], rstride=5, cstride=5, color='Red', alpha=0.2)
    #ax2.set_zlim(-0.1 * dtrw_sub.X[:,:,i].max(), 1.1 * dtrw_sub.X[:,:,i].max())

    #ax = fig.get_axes(1, 3, 3) 
    ax3.cla()
    wframe3 = ax3.plot_surface(Xs, Ys, dtrw_sub.X[:,:,i] - dtrw_sub_X.X[:,:,i], rstride=5, cstride=5, color='Purple', alpha=0.2)
    plot_max = (dtrw_sub.X[:,:,i] - dtrw_sub_X.X[:,:,i]).max()
    plot_min = (dtrw_sub.X[:,:,i] - dtrw_sub_X.X[:,:,i]).min()
    #wframe3 = ax3.plot_surface(Xs, Ys, dtrw.X[:,:,i] - dtrw_sub.X[:,:,i], rstride=5, cstride=5, color='Purple', alpha=0.2)
    #plot_max = (dtrw.X[:,:,i] - dtrw_sub.X[:,:,i]).max()
    #plot_min = (dtrw.X[:,:,i] - dtrw_sub.X[:,:,i]).min()
    ax3.set_zlim(1.1 * plot_min, 1.1 * plot_max)

    #cset = ax.contour(Xs, Ys, dtrw.X[:,:,i], zdir='z', offset=plot_min, cmap=cm.coolwarm)
    #cset = ax.contour(Xs, Ys, dtrw.X[:,:,i], zdir='x', offset=0., cmap=cm.coolwarm)
    #cset = ax.contour(Xs, Ys, dtrw.X[:,:,i], zdir='y', offset=1., cmap=cm.coolwarm)
    #ax.set_zlim(-0.1 * plot_max,1.1 * plot_max)
   
    #print dtrw.X[:,:,i].sum(), dtrw_sub.X[:,:,i].sum()

    return wframe, wframe2, wframe3

def update2(i, ax, fig):
    ax.cla()
    wframe = ax.plot_wireframe(Xs, Ys, dtrw.X[:,:,i] - dtrw_sub.X[:,:,i], rstride=5, cstride=5, color='Red')
    ax.set_zlim(-0.1, 1.1 * (dtrw.X[:,:,i]-dtrw_sub.X[:,:,i]).max())
    return wframe,

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, update, 
        frames=N, 
        fargs=(ax1, ax2, ax3, fig), interval=100)

anim.save('basic_animation.mp4', fps=30)
plt.show()

