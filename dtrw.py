
import math
import numpy as np
import scipy as sp
from itertools import *

import pdb

class DTRW_with_reactions(object):
    """ Base definition of a DTRW with arbitrary wait-times
        for reactions and jumps """

    def __init__(self, X, N, dT)
        """X is the initial concentration field, N is the number of time steps to simulate"""
        self.N = N
        self.dT = dT
        self.X = np.dstack(self.X)
        
        # How many time steps have preceded the current one?
        self.n = 0 # self.X.shape[-1]

        self.calc_lambda()
        self.calc_psi()
        self.calc_Phi()
        self.calc_omega(0.01 * self.dT)
        self.calc_theta()
        self.calc_mem_kernel()
    
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
        t = np.array(range(N)) * self.dT
        self.psi = (1 / self.tau) * np.exp(-t / self.tau) 

    def calc_Phi(self):
        """PDF of not jumping up to time n"""
        self.Phi = np.ones(N)
        for i in range(N):
            # unsure about ending index - might need extra point...
            self.Phi(i) -= sum(self.psi[:i])

    def calc_omega(self, om):
        """ Likelihood of surviving death between n and n+1"""

        # For now just constant for the length of time
        self.omega = om * np.ones(self.N)
    
    def calc_theta(self):
        """Likelihood of death happening between 0 and n"""
        self.theta = np.zeros(self.N)
        
        # Note that this only works for constant theta at the moment
        for i in range(N):
            self.theta = 1.0 - sum(self.omega[:i])
    
    def time_step(self):
        """Take a time step forward"""
        
        # How many time steps have we had?
        self.n = self.X.shape[2] + 1
       
        self.update_theta()
        
        # outward flux 
        flux = np.zeros(self.X.shape[:2])

        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                flux(i,j) = sum(self.X[i, j, :] * self.K[:self.n][::-1] * self.theta[:self.n][::-1])
        
        # update the field
        next_X = sp.signal.convolve2d(flux, self.lam, 'same') - i - self.omega(n) * self.X[:,:,-1]
       
        # stack next_X on to the list of fields X - giving us another layer in the 3d array of spatial results
        X = np.dstack((X, next_X))


