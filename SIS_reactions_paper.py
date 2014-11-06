#! /usr/bin/env python

import numpy as np
import time
from dtrw import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm

class DTRW_diffusive_SIS(DTRW_diffusive):
    
    def __init__(self, X_inits, N, alpha, beta, mu, gamma, history_length=0, boltz_beta=0., potential = np.array([]), boundary_condition=BC()):
        
        self.beta = beta
        self.mu = mu
        self.gamma = gamma
            
        super(DTRW_diffusive_SIS, self).__init__(X_inits, N, alpha, history_length, boltz_beta, potential, boundary_condition)
   
        self.has_spatial_reactions = True

    def calc_omega(self):
        """ Probability of death between n and n+1"""
        if self.omegas == None:
            self.omegas = [np.zeros((X.shape[0], X.shape[1], self.N)) for X in self.Xs]
            
        # We assume that the calculation has been made for all n up to now, so we simply update the n-th point
        # S 
        self.omegas[0][:,:,self.n] = 1. - np.exp(- self.beta * self.Xs[1][:,:,self.n] - mu) 
        self.omegas[0][:,:,self.n] = self.beta * self.Xs[1][:,:,self.n] + self.mu
        # I 
        self.omegas[1][:,:,self.n] = 1. - np.exp(-(self.mu + self.gamma))
        self.omegas[1][:,:,self.n] = self.mu + self.gamma

    def calc_theta(self):
        """ Probability of surviving between 0 and n"""
        if self.thetas == None:
            self.thetas = [np.ones((X.shape[0], X.shape[1], self.N)) for X in self.Xs]

        for i in range(len(self.Xs)):
            # THIS IS WRONG! 
            #self.thetas[i][:,:,self.n] = (1. - self.omegas[i][:,:,:self.n+1]).prod(2)
            # This is right!!!! remember you want theta between m and n, not 0 and m!!!!
            self.thetas[i][:,:,:self.n+1] = self.thetas[i][:,:,:self.n+1] * np.dstack([1. - self.omegas[i][:,:,self.n]])

    def calc_nu(self):
        """Likelihood of birth happening at i"""
        if self.nus == None:
            self.nus = [np.zeros((X.shape[0], X.shape[1], self.N)) for X in self.Xs]
    
        self.nus[0][:,:,self.n] = 1. - np.exp(-(self.mu + self.gamma))
        self.nus[0][:,:,self.n] = self.gamma * self.Xs[1][:,:,self.n] + self.mu
        self.nus[1][:,:,self.n] = 1. - np.exp(-self.beta * self.Xs[1][:,:,self.n]) 
        self.nus[1][:,:,self.n] = self.beta * self.Xs[1][:,:,self.n] * self.Xs[0][:,:,self.n]



class DTRW_subdiffusive_SIS(DTRW_subdiffusive):
    
    def __init__(self, X_inits, N, alpha, beta, mu, gamma, history_length=0, boltz_beta=0., potential = np.array([]), boundary_condition=BC()):
        
        self.beta = beta
        self.mu = mu
        self.gamma = gamma
            
        super(DTRW_subdiffusive_SIS, self).__init__(X_inits, N, alpha, history_length, boltz_beta, potential, boundary_condition)
   
        self.has_spatial_reactions = True

    def calc_omega(self):
        """ Probability of death between n and n+1"""
        if self.omegas == None:
            self.omegas = [np.zeros((X.shape[0], X.shape[1], self.N)) for X in self.Xs]
            
        # We assume that the calculation has been made for all n up to now, so we simply update the n-th point
        # S 
        #self.omegas[0][:,:,self.n] = 1. - np.exp(- self.beta * self.Xs[1][:,:,self.n] - mu) 
        self.omegas[0][:,:,self.n] = self.beta * self.Xs[1][:,:,self.n] + self.mu
        # I 
        #self.omegas[1][:,:,self.n] = 1. - np.exp(-(self.mu + self.gamma))
        self.omegas[1][:,:,self.n] = self.mu + self.gamma

    def calc_theta(self):
        """ Probability of surviving between 0 and n"""
        if self.thetas == None:
            self.thetas = [np.ones((X.shape[0], X.shape[1], self.N)) for X in self.Xs]

        for i in range(len(self.Xs)):
            # THIS IS WRONG! 
            #self.thetas[i][:,:,self.n] = (1. - self.omegas[i][:,:,:self.n+1]).prod(2)
            # This is right!!!! remember you want theta between m and n, not 0 and m!!!!
            self.thetas[i][:,:,:self.n+1] = self.thetas[i][:,:,:self.n+1] * np.dstack([1. - self.omegas[i][:,:,self.n]])

    def calc_nu(self):
        """Likelihood of birth happening at i"""
        if self.nus == None:
            self.nus = [np.zeros((X.shape[0], X.shape[1], self.N)) for X in self.Xs]
    
        #self.nus[0][:,:,self.n] = 1. - np.exp(-self.mu)
        self.nus[0][:,:,self.n] = self.gamma * self.Xs[1][:,:,self.n] + self.mu
        #self.nus[1][:,:,self.n] = 1. - np.exp(-self.beta * self.Xs[1][:,:,self.n]) 
        self.nus[1][:,:,self.n] = self.beta * self.Xs[1][:,:,self.n] * self.Xs[0][:,:,self.n]

n_points = 101
L = 1.0
dX = L / n_points
xs = np.linspace(0., 1., n_points, endpoint=True)

S_init = np.ones(n_points)
I_init = np.exp(- 5. * (xs - 0.5) * (xs - 0.5))

alpha = 0.8
D_alpha = 0.005

T = 10.
dT = pow((dX * dX / (2.0 * D_alpha)), 1./alpha)
N = int(math.floor(T / dT))
history_length = N
ts = np.array(np.arange(N) * dT)

beta = 0.5 * dT
mu = 0.2 * dT
gamma = 0.01 * dT

# Calculate r for diffusive case so as to get the *same* dT as the subdiffusive case
r = dT / (dX * dX / (2.0 * D_alpha))

print "DTRW for dX =", dX, "dT =", dT, "r =", r

bc = BC_zero_flux()

dtrw_sub = DTRW_subdiffusive_SIS([S_init, I_init], N, alpha, beta, mu, gamma, history_length, boundary_condition=bc)
dtrw = DTRW_diffusive_SIS([S_init, I_init], N, r, beta, mu, gamma, history_length, boundary_condition=bc)

dtrw.solve_all_steps()
dtrw_sub.solve_all_steps()

Xs, Ts = np.meshgrid(xs, ts)

fig = plt.figure(figsize=(8,8))

ax1 = fig.add_subplot(2, 2, 1)
pc1 = ax1.pcolor(ts, xs, dtrw.Xs[0][0,:,:])
#CS1 = ax1.plot_surface(Ts, Xs, dtrw.Xs[0][0,:,:])
#CS1 = ax1.contour(Ts, Xs, dtrw.Xs[0][0,:,:])

ax2 = fig.add_subplot(2, 2, 2)
pc2 = ax2.pcolor(ts, xs, dtrw.Xs[1][0,:,:])
#CS2 = ax2.contour(Ts, Xs, dtrw.Xs[1][0,:,:])

ax3 = fig.add_subplot(2, 2, 3)
pc3 = ax3.pcolor(ts, xs, dtrw_sub.Xs[0][0,:,:])
#CS3 = ax3.contour(Ts, Xs, dtrw_sub.Xs[0][0,:,:])

ax4 = fig.add_subplot(2, 2, 4)
pc4 = ax4.pcolor(ts, xs, dtrw_sub.Xs[1][0,:,:])
#CS4 = ax4.contour(Ts, Xs, dtrw_sub.Xs[1][0,:,:])

plt.show()

import inspect, os, subprocess
exec_name =  os.path.splitext(os.path.basename(inspect.getfile(inspect.currentframe())))[0]
git_tag = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).replace('\n', '')

