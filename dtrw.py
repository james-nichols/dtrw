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

    def __init__(self, X_inits, N, death, birth, history_length = 2, beta = 0., potential = np.array([]), is_periodic=False):
        """X is the initial concentration field, N is the number of time steps to simulate"""
        # Xs is either a single initial condition, or a list of initial conditions,
        # for multi-species calculations, so we check first and act accordingly
        self.Xs = []
        self.Qs = []
        if type(X_inits) == list:
            for X in X_inits:
                X_init = np.dstack([X])
                # We allocate the whole history here to store 
                self.Xs.append(np.zeros((X_init.shape[0], X_init.shape[1], N)))
                self.Xs[-1][:,:,0] = X_init[:,:,0]
                
                Q_init = np.dstack([X])
                self.Qs.append(np.zeros((Q_init.shape[0], Q_init.shape[1], N)))
                self.Qs[-1][:,:,0] = Q_init[:,:,0]
        else:
            X_init = np.dstack([X_inits])
            self.Xs.append(np.zeros((X_init.shape[0], X_init.shape[1], N)))
            self.Xs[0][:,:,0] = X_init[:,:,0]
            
            Q_init = np.dstack([X_inits])
            self.Qs.append(np.zeros((Q_init.shape[0], Q_init.shape[1], N)))
            self.Qs[0][:,:,0] = Q_init[:,:,0]

        self.shape = self.Xs[0][:,:,0].shape
        self.size = self.Xs[0][:,:,0].size

        self.N = N
        self.history_length = history_length
        
        self.is_periodic = is_periodic
        self.has_spatial_reactions = False

        # This is the time-step counter
        self.n = 0
    
        self.beta = beta
        self.calc_lambda(potential)
        self.calc_psi()
        self.calc_Phi()
      
        self.death_rate = death
        self.birth_rate = birth
        
        self.omegas = None
        self.thetas = None
        self.nus = None
        self.calc_omega()
        self.calc_theta()
        self.calc_nu()
        
        self.calc_mem_kernel()
    
    def calc_mem_kernel(self):
        """Once off call to calculate the memory kernel and store it"""
        self.K = np.zeros(self.history_length+1)

    def calc_lambda(self, potential):
        """ Basic equal prob diffusion. NOTE: boundary conditions are not taken care of here! """
        # New regime - create a sequence of probabilities for left, right, up, down (in that order)
        
        if potential.size:
            if potential.size != self.size: # and potential.shape != (self.X[:,:,0].shape + (2,2)):
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
            if self.shape[0] > 1:
                boltz_denom[:-1,:] += boltz_func[1:,:]
                boltz_denom[1:,:] += boltz_func[:-1,:]
                if self.is_periodic:
                    boltz_denom[-1,:] += boltz_func[0,:]
                    boltz_denom[0,:] += boltz_func[-1,:]

            # Now calculate the left, right (and if 2D, up and down) probabilities
            self.lam = np.zeros(self.shape)
            # left
            self.lam = np.dstack([self.lam])
            self.lam[:,1:,0] += boltz_func[:,:-1] / boltz_denom[:,1:]
            # right
            self.lam = np.dstack([self.lam, np.zeros(self.shape)])
            self.lam[:,:-1,1] += boltz_func[:,1:] / boltz_denom[:,:-1]
            if self.is_periodic:
                self.lam[:,0,0] += boltz_func[:,-1] / boltz_denom[:,0]
                self.lam[:,-1,1] += boltz_func[:,0] / boltz_denom[:,-1]

            if self.shape[0] > 1:
                # up
                self.lam = np.dstack([self.lam, np.zeros(self.shape)])
                self.lam[1:,:,2] += boltz_func[:-1,:] / boltz_denom[1:,:]
                # down
                self.lam = np.dstack([self.lam, np.zeros(self.shape)])
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
            self.lam = np.ones(self.shape)
            self.lam = np.dstack([self.lam, self.lam]) 

            # now - if there's the second dimension, then we add the up & right properties
            if self.shape[0] > 1:
                # up probability
                self.lam = np.dstack([self.lam, np.ones(self.shape)])
                # down probability
                self.lam = np.dstack([self.lam, np.ones(self.shape)])
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

    def calc_omega(self):
        """ Likelihood of surviving between n and n+1"""
        if self.omegas == None:
            self.omegas = [self.death_rate * np.ones(self.N) for i in range(len(self.Xs))]

    def calc_theta(self):
        """Likelihood of surviving between 0 and n"""
        if self.thetas == None:
            self.thetas = [np.zeros(self.N) for i in range(len(self.Xs))]

            for i in range(len(self.Xs)):
                # Note that this only works for constant theta at the moment
                for j in range(self.N):
                    self.thetas[i][j] = (1.0 - self.omegas[i][:j]).prod()
    
    def calc_nu(self):
        """Likelihood of birth happening at i"""
        if self.nus == None:
            self.nus = [self.birth_rate * np.ones(self.N) for i in range(len(self.Xs))]
    
    def time_step_with_Q(self):
        """Take a time step forward using arrival densities. NOTE in the diffusive case 
        it is necessary to use a full history to get this one right"""
        
        # First we increment the time counter!
        self.n += 1
        # How many time steps have we had?
        lookback = min(self.n, self.history_length)
 
        for i in range(len(self.Xs)):
            Q = self.Qs[i]
            theta = self.thetas[i]
            omega = self.omegas[i]
            nu = self.nus[i]

            # Matrix methods to calc Q as in eq. (9) in the J Comp Phys paper
            flux = (Q[:, :, self.n-lookback:self.n] * self.psi[1:lookback+1][::-1] * theta[-lookback:]).sum(2) + nu[self.n]
            
            # Now apply lambda jump probabilities
            next_Q = np.zeros(self.shape)
            
            # First multiply by all the left jump probabilities
            next_Q[:,:-1] += (self.lam[:,:,0] * flux)[:,1:]
            # then all the right jump 
            next_Q[:,1:] += (self.lam[:,:,1] * flux)[:,:-1]
            if self.is_periodic:
                next_Q[:,-1] += self.lam[:,0,0] * flux[:,0]
                next_Q[:,0] += self.lam[:,-1,1] * flux[:,-1]
            if self.shape[0] > 1:
                # The up jump
                next_Q[:-1,:] += (self.lam[:,:,2] * flux)[1:,:]
                # The down jump
                next_Q[1:,:] += (self.lam[:,:,3] * flux)[:-1,:]
                if self.is_periodic:
                    next_Q[-1,:] += self.lam[0,:,2] * flux[0,:]
                    next_Q[0,:] += self.lam[-1,:,3] * flux[-1,:]

            # Add Q to the list of Qs over time
            self.Qs[i][:,:,self.n] = next_Q

            lookback = min(self.n+1, self.history_length)
            
            # Matrix methods to calc X as in eq. (11) in the J Comp Phys paper
            next_X = (Q[:, :, self.n+1-lookback:self.n+1] * self.Phi[:lookback][::-1] * theta[-lookback:]).sum(2)
            self.Xs[i][:,:,self.n] = next_X
       
        self.calc_omega()
        self.calc_theta()
        self.calc_nu() 

    def time_step(self):
        """ Step forwards directly with X using the memory kernel K, this
            method is only available in cases where we can calculate K analytically!"""
 
        # First we increment the time counter!
        self.n += 1
        # This allows for the limited history, an approximation to speed things up.
        lookback = min(self.n, self.history_length)
        
        for i in range(len(self.Xs)):
            X = self.Xs[i]
            theta = self.thetas[i]
            omega = self.omegas[i]
            nu = self.nus[i]
        
            if self.has_spatial_reactions:
                # Matrix multiply to calculate flux (using memory kernel), then sum in last dimension (2), to get outward flux
                flux = (X[:, :, self.n-lookback:self.n] * theta[:,:,self.n-lookback:self.n] * self.K[1:lookback+1][::-1]).sum(2)
                next_X = X[:,:,self.n-1] - flux - omega[:,:,self.n-1] * X[:,:,self.n-1]
            else:
                flux = (X[:, :, self.n-lookback:self.n] * theta[self.n-lookback:self.n] * self.K[1:lookback+1][::-1]).sum(2)
                next_X = X[:,:,self.n-1] - flux - omega[self.n-1] * X[:,:,self.n-1]

            # Now we add the spatial jumps
            # First multiply by all the left jump probabilities
            next_X[:,:-1] += (self.lam[:,:,0] * flux)[:,1:]
            # then all the right jump 
            next_X[:,1:] += (self.lam[:,:,1] * flux)[:,:-1]
            if self.is_periodic:
                next_X[:,-1] += self.lam[:,0,0] * flux[:,0]
                next_X[:,0] += self.lam[:,-1,1] * flux[:,-1]
            if X.shape[0] > 1:
                # The up jump
                next_X[:-1,:] += (self.lam[:,:,2] * flux)[1:,:]
                # The down jump
                next_X[1:,:] += (self.lam[:,:,3] * flux)[:-1,:]
                if self.is_periodic:
                    next_X[-1,:] += self.lam[0,:,2] * flux[0,:]
                    next_X[0,:] += self.lam[-1,:,3] * flux[-1,:]
                   
            # BUT WHAT ABOUT THE BOUNDARY CONDITIONS!!!!!!!

            # stack next_X on to the list of fields X - giving us another layer in the 3d array of spatial results
            self.Xs[i][:,:,self.n] = next_X
   
            self.calc_omega()
            self.calc_theta()
            self.calc_nu() 

    def solve_all_steps_with_Q(self):

        for i in range(self.N-1):
            self.time_step_with_Q()
    
    def solve_all_steps(self):
        """Solve the time steps using the memory kernel, available only if we know how to calculate K"""
        for i in range(self.N-1):
            if i % 100 == 0:
                print "Solved to step", i
            self.time_step()

class DTRW_diffusive(DTRW):

    def __init__(self, X_inits, N, r, omega, nu, history_length=2, beta = 0., potential = np.array([]), is_periodic=False):
        # Probability of jumping in one step 
        self.r = r

        super(DTRW_diffusive, self).__init__(X_inits, N, omega, nu, history_length, beta, potential, is_periodic)

    def calc_psi(self):
        """Waiting time distribution for spatial jumps"""
        self.psi = np.array([self.r * pow(1. - self.r, i-1) for i in range(self.history_length+1)])
        self.psi[0] = 0.

    def calc_Phi(self):
        """PDF of not jumping up to time n"""
        self.Phi = np.array([pow(1. - self.r, i) for i in range(self.history_length+1)])
    
    def calc_mem_kernel(self):
        """Once off call to calculate the memory kernel and store it"""
        self.K = np.zeros(self.history_length+1)
         
        # In the exponential case it's quite simply...
        self.K[1] = self.r

class DTRW_subdiffusive(DTRW):

    def __init__(self, X_inits, N, alpha, omega, nu, history_length, beta = 0., potential = np.array([]), is_periodic=False):
        
        self.alpha = alpha
        
        super(DTRW_subdiffusive, self).__init__(X_inits, N, omega, nu, history_length, beta, potential, is_periodic)

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

