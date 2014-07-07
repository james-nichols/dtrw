#! /usr/bin/env python

import math
import numpy as np
import scipy as sp
import scipy.signal
import scipy.special
from itertools import *

import pdb

class DTRW(object):
    """ Base definition of a DTRW with arbitrary wait-times
        for reactions and jumps """

    def __init__(self, X, N, omega, nu, history_length = 2, beta = 0., potential = np.array([]), is_periodic=False):
        """X is the initial concentration field, N is the number of time steps to simulate"""
        self.X = np.dstack([X])
        self.Q = np.dstack([X])
        self.N = N
        self.history_length = history_length
        self.is_periodic = is_periodic

        # How many time steps have preceded the current one?
        self.n = 0 # self.X.shape[-1]
    
        self.beta = beta
        self.calc_lambda(potential)
        self.calc_psi()
        self.calc_Phi()
        self.calc_omega(omega)
        self.calc_theta()
        self.calc_nu(nu)
        
        self.calc_mem_kernel()
    
    def calc_mem_kernel(self):
        """Once off call to calculate the memory kernel and store it"""
        self.K = np.zeros(self.history_length+1)

    def calc_lambda(self, potential):
        """ Basic equal prob diffusion. NOTE: boundary conditions are not taken care of here! """
        # New regime - create a sequence of probabilities for left, right, up, down (in that order)
        
        if potential.size:
            if potential.size != self.X[:,:,0].size: # and potential.shape != (self.X[:,:,0].shape + (2,2)):
                # NOT CURRENTLY TRUE BUT MIGHT BE AN OPTION: We allow the potential to have a to allow for escape fromt the domain
                raise NameError("potential not the same shape as initial conditions") 

            if len(potential.shape) == 1: # As np.dstack introduces two more dimensions, in the 1-D case we need to as well
                potential = potential[np.newaxis]
            # Calculate the transition probabilities Boltzmann style based on potentials.
            boltz_func = np.exp(-self.beta * potential)
            boltz_denom = np.zeros(boltz_func.shape)
            boltz_denom[:,:-1] += boltz_func[:,1:]
            boltz_denom[:,1:] += boltz_func[:,:-1]
            if self.is_periodic:
                boltz_denom[:,-1] += boltz_func[:,0]
                boltz_denom[:,0] += boltz_func[:,-1]
            if self.X.shape[0] > 1:
                boltz_denom[:-1,:] += boltz_func[1:,:]
                boltz_denom[1:,:] += boltz_func[:-1,:]
                if self.is_periodic:
                    boltz_denom[-1,:] += boltz_func[0,:]
                    boltz_denom[0,:] += boltz_func[-1,:]

            # Now calculate the left, right (and if 2D, up and down) probabilities
            self.lam = np.zeros(self.X[:,:,0].shape)
            # left
            self.lam = np.dstack([self.lam])
            self.lam[:,1:,0] += boltz_func[:,:-1] / boltz_denom[:,1:]
            # right
            self.lam = np.dstack([self.lam, np.zeros(self.X[:,:,0].shape)])
            self.lam[:,:-1,1] += boltz_func[:,1:] / boltz_denom[:,:-1]
            if self.is_periodic:
                self.lam[:,0,0] += boltz_func[:,-1] / boltz_denom[:,0]
                self.lam[:,-1,1] += boltz_func[:,0] / boltz_denom[:,-1]

            if self.X.shape[0] > 1:
                # up
                self.lam = np.dstack([self.lam, np.zeros(self.X[:,:,0].shape)])
                self.lam[1:,:,2] += boltz_func[:-1,:] / boltz_denom[1:,:]
                # down
                self.lam = np.dstack([self.lam, np.zeros(self.X[:,:,0].shape)])
                self.lam[:-1,:,3] += boltz_func[1:,:] / boltz_denom[:-1,:]
                if self.is_periodic:
                    self.lam[0,:,2] += boltz_func[-1,:] / boltz_denom[0,:]
                    self.lam[-1,:,3] += boltz_func[0,:] / boltz_denom[-1,:]
        
            if False:
                # Fill in the boundary conditions to allow for jumps out of the region
                self.lam[:,1,0] = 1.0 - (self.lam[:,1,1] + self.lam[:,1,2] + self.lam[:,1,3])
                self.lam[:,-1,1] = 1.0 - (self.lam[:,-1,0] + self.lam[:,-1,2] + self.lam[:,-1,3]) 
                self.lam[1,:,2] = 1.0 - (self.lam[1,:,0] + self.lam[1,:,1] + self.lam[1,:,3]) 
                self.lam[-1,:,3] = 1.0 - (self.lam[-1,:,0] + self.lam[-1,:,1] + self.lam[-1,:,2]) 

        else:
            self.lam = np.ones(self.X[:,:,0].shape)
            self.lam = np.dstack([self.lam, self.lam]) 

            # now - if there's the second dimension, then we add the up & right properties
            if self.X.shape[0] > 1:
                # up probability
                self.lam = np.dstack([self.lam, np.ones(self.X[:,:,0].shape)])
                # down probability
                self.lam = np.dstack([self.lam, np.ones(self.X[:,:,0].shape)])
                self.lam = 0.25 * self.lam
            else:
                # if there isn't the second dimension, prob's are 0.5
                self.lam = 0.5 * self.lam


    def calc_psi(self):
        """Waiting time distribution for spatial jumps"""
        self.psi = np.zeros(self.history_length+1) 

    def calc_Phi(self):
        """PDF of not jumping up to time n"""
        self.Phi = np.ones(self.history_length+1)

    def calc_omega(self, om):
        """ Likelihood of surviving between n and n+1"""
        self.omega = om * np.ones(self.N)

    def calc_theta(self):
        """Likelihood of surviving between 0 and n"""
        self.theta = np.zeros(self.N)
        
        # Note that this only works for constant theta at the moment
        for i in range(self.N):
            self.theta[i] = (1.0 - self.omega[:i]).prod()
    
    def calc_nu(self, birth):
        """Likelihood of birth happening at i"""
        self.nu = birth * np.ones(self.N)
    
    def time_step(self):
        """Take a time step forward"""
        
        # How many time steps have we had?
        self.n = self.X.shape[2] 
       
        lookback = min( self.n, self.history_length )

        # Matrix methods to calc Q as in eq. (9) in the J Comp Phys paper
        flux = (self.Q[:, :, -lookback:] * self.psi[1:lookback+1][::-1] * self.theta[-lookback:]).sum(2) + self.nu[self.n]
        
        # Now apply lambda jump probabilities
        next_Q = np.zeros(self.X[:,:,0].shape)
        
        # First multiply by all the left jump probabilities
        next_Q[:,:-1] += (self.lam[:,:,0] * flux)[:,1:]
        # then all the right jump 
        next_Q[:,1:] += (self.lam[:,:,1] * flux)[:,:-1]
        if self.is_periodic:
            next_Q[:,-1] += self.lam[:,0,0] * flux[:,0]
            next_Q[:,0] += self.lam[:,-1,1] * flux[:,-1]
        if self.X.shape[0] > 1:
            # The up jump
            next_Q[:-1,:] += (self.lam[:,:,2] * flux)[1:,:]
            # The down jump
            next_Q[1:,:] += (self.lam[:,:,3] * flux)[:-1,:]
            if self.is_periodic:
                next_Q[-1,:] += self.lam[0,:,2] * flux[0,:]
                next_Q[0,:] += self.lam[-1,:,3] * flux[-1,:]

        # Applying zero-flux boundary condition
        #next_Q[:, 0] = next_Q[:, 1]
        #next_Q[0, :] = next_Q[1, :]
        #next_Q[:, -1] = next_Q[:, -2]
        #next_Q[-1, :] = next_Q[-2, :]
            
        # Add Q to the list of Qs over time
        self.Q = np.dstack([self.Q, next_Q])

        lookback = min( self.n+1, self.history_length )
        
        # Matrix methods to calc X as in eq. (11) in the J Comp Phys paper
        next_X = (self.Q[:, :, -lookback:] * self.Phi[:lookback][::-1] * self.theta[-lookback:]).sum(2)
        self.X = np.dstack([self.X, next_X])
        
    def time_step_with_K(self):
        """ Step forwards directly with X using the memory kernel K, this
            method is only available in cases where we can calculate K analytically"""
 
        # How many time steps have we had?
        self.n = self.X.shape[2] 
        lookback = min(self.n, self.history_length)

        # Matrix multiply to calculate flux (using memory kernel), then sum in last dimension (2), to get outward flux
        flux = (self.X[:, :, -lookback:] * self.K[1:lookback+1][::-1] * self.theta[-lookback:]).sum(2)
        
        next_X = self.X[:,:,-1] - flux - self.omega[self.n] * self.X[:,:,-1]

        # Now we add the spatial jumps
        # First multiply by all the left jump probabilities
        next_X[:,:-1] += (self.lam[:,:,0] * flux)[:,1:]
        # then all the right jump 
        next_X[:,1:] += (self.lam[:,:,1] * flux)[:,:-1]
        if self.is_periodic:
            next_X[:,-1] += self.lam[:,0,0] * flux[:,0]
            next_X[:,0] += self.lam[:,-1,1] * flux[:,-1]
        if self.X.shape[0] > 1:
            # The up jump
            next_X[:-1,:] += (self.lam[:,:,2] * flux)[1:,:]
            # The down jump
            next_X[1:,:] += (self.lam[:,:,3] * flux)[:-1,:]
            if self.is_periodic:
                next_X[-1,:] += self.lam[0,:,2] * flux[0,:]
                next_X[0,:] += self.lam[-1,:,3] * flux[-1,:]
               
        # BUT WHAT ABOUT THE BOUNDARY CONDITIONS!!!!!!!

        # stack next_X on to the list of fields X - giving us another layer in the 3d array of spatial results
        self.X = np.dstack((self.X, next_X))

    def solve_all_steps(self):

        for i in range(self.N-1):
            self.time_step()
    
    def solve_all_steps_with_K(self):
        """Solve the time steps using the memory kernel, available only if we know how to calculate K"""
        for i in range(self.N-1):
            if i % 100 == 0:
                print "Solved to step", i
            self.time_step_with_K()

class DTRW_diffusive(DTRW):

    def __init__(self, X, N, dT, tau, omega, nu, history_length=2, beta = 0., potential = np.array([]), is_periodic=False):
        
        self.dT = dT
        self.tau = tau
        
        super(DTRW_diffusive, self).__init__(X, N, omega, nu, history_length, beta, potential, is_periodic)

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
    
    def calc_mem_kernel(self):
        """Once off call to calculate the memory kernel and store it"""
        self.K = np.zeros(self.history_length+1)
         
        # In the exponential case it's quite simply...
        self.K[1] = (1.0 - np.exp(-self.dT / self.tau))

class DTRW_subdiffusive(DTRW):

    def __init__(self, X, N, alpha, omega, nu, history_length, beta = 0., potential = np.array([]), is_periodic=False):
        
        self.alpha = alpha
        
        super(DTRW_subdiffusive, self).__init__(X, N, omega, nu, history_length, beta, potential, is_periodic)

    def calc_psi(self):
        """Waiting time distribution for spatial jumps"""
        #tn = np.array(range(self.history_length+1))
        #self.psi = pow(-1,tn+1) * scipy.special.gamma(self.alpha + 1) / (scipy.special.gamma(tn + 1) * scipy.special.gamma(self.alpha - tn + 1))
        #psi[0] = 0.
        
        self.psi = np.zeros(self.history_length+1)
        self.psi[0] = 0.
        
        self.psi[1] = self.alpha
        for i in range(2,self.history_length+1):
            self.psi[i] = -self.psi[i-1] * (self.alpha - float(i) + 1.) / float(i)
        
        # Highly dubious: Normalising psi so that we conservation of particles
        # self.psi[-1] = 1.0 - self.psi[:-1].sum()
        
    
    def calc_Phi(self):
        """PDF of not jumping up to time n"""
        #tn = np.array(range(self.history_length+1))
        #self.Phi = pow(-1,tn) * scipy.special.gamma(self.alpha ) / (scipy.special.gamma(tn + 1) * scipy.special.gamma(self.alpha - tn))
        #self.Phi[0] = 1.

        self.Phi = np.ones(self.history_length+1)
        self.Phi[1] = 1.0 - self.alpha
        for i in range(2, self.history_length+1):
            self.Phi[i] = - self.Phi[i-1] * (self.alpha - float(i)) / float(i)
        
        #for i in range(self.history_length+1):
            # unsure about ending index - might need extra point...
        #    self.Phi[i] = 1.0-sum(self.psi[:i+1])

    def calc_mem_kernel(self):
        """Once off call to calculate the memory kernel and store it"""
        self.K = np.zeros(self.history_length+1)
        
        self.K[0] = 0.0
        self.K[1] = self.alpha 
        self.K[2] = self.alpha * 0.5 * (self.alpha - 1.0)
        for i in range(3,self.history_length+1):
            self.K[i] = (float(i) + self.alpha - 2.0) * self.K[i-1] / float(i)
