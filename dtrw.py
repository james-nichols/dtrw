#! /usr/bin/env python

import math
import numpy as np
import scipy as sp
import scipy.signal
import scipy.special
import scipy.optimize
from itertools import *

import pdb

# TODO LIST
# o Inherit theta builder from base class
# o Potentially remove the has_spatial_reactions property and unify the approach
# o Make the SIR / compartment model work for testing

class BC(object):
    def __init__(self):
        pass
    def apply_BCs(self, next_X, flux, dtrw):
        # Apply the boundary conditions, called from the time step routine in DTRW
        pass

class BC_Dirichelet(BC):
    def __init__(self, data):
        self.data = data

    def apply_BCs(self, next_X, flux, dtrw):
        if self.data[0].shape != next_X[:,0].shape and self.data[0].shape:
            pdb.set_trace()
            raise Exception('Boundary conditions are incorrect shape, BC data is ' + str(self.data[0].shape) + ' Sol\'n is ' + str(next_X[:,0].shape))

        next_X[:,0] = self.data[0]
        if len(self.data) > 1: 
            #next_X[:,-1] = self.data[1]
            if next_X.shape[0] > 1:
                next_X[0,:] = self.data[2]
                next_X[-1,:] = self.data[3]

class BC_Fedotov(BC):

    def __init__(self, alpha, constant, right):
        self.alpha = alpha
        self.constant = constant
        self.right = right

    def apply_BCs(self, next_X, flux, dtrw):
        # Only left side has BC
        for i in range(next_X.shape[0]):
            #next_X[i,0] = scipy.optimize.newton(self.fedotov_func, next_X[i, 1], fprime=self.fedotov_func_prime, args=(next_X[i,1], self.alpha, self.constant))
            next_X[i,0] = pow(self.constant + pow(next_X[i, 1], 2.-self.alpha), 1. / (2.-self.alpha)) 
            #next_X[i,-1] = self.right
        
    def fedotov_func(self, x_0, x_1, alpha, const):
        # Annoying function to find Fedotov's boundary condition
        return pow(x_0, 1.-alpha) * (x_1 - x_0) + const 
        #return pow(x_1, 1.-alpha) * (x_1 - x_0) + const 

    def fedotov_func_prime(self, x_0, x_1, alpha, const):
        # Annoying function to find Fedotov's boundary condition
        return (1.-alpha) * pow(x_0, -alpha) * (x_1 - x_0) - pow(x_0, 1.-alpha) 
        #return - pow(x_1, 1.-alpha)

class BC_Fedotov_balance(BC):

    def __init__(self):
        pass

    def apply_BCs(self, next_X, flux, dtrw):
        # Only left side has BC
        if dtrw.has_spatial_reactions:
            next_X[:,0] += (dtrw.omegas[0][:,:,dtrw.n-1] * dtrw.Xs[0][:,:,dtrw.n-1]).sum()
            next_X[:,0] += dtrw.lam[:,0,0] * flux[:,0] 
        else:
            next_X[:,0] += (dtrw.omegas[0][dtrw.n-1] * dtrw.Xs[0][:,:,dtrw.n-1]).sum()
            next_X[:,0] += dtrw.lam[:,0,0] * flux[:,0] 

class BC_periodic(BC):
    
    def __init__(self):
        pass 

    def apply_BCs(self, next_X, flux, dtrw):
        # Apply the boundary conditions
        next_X[:,-1] += dtrw.lam[:,0,0] * flux[:,0]
        next_X[:,0] += dtrw.lam[:,-1,1] * flux[:,-1]

        if next_X.shape[0] > 1:
            next_X[-1,:] += dtrw.lam[0,:,2] * flux[0,:]
            next_X[0,:] += dtrw.lam[-1,:,3] * flux[-1,:]

class DTRW(object):
    """ Base definition of a DTRW with arbitrary wait-times
        for reactions and jumps """

    def __init__(self, X_inits, N, history_length = 2, beta = 0., potential = np.array([]), boundary_condition=BC()):
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
        if history_length == 0:
            self.history_length = N
        else:
            self.history_length = history_length
        
        self.boundary_condition = boundary_condition
        self.has_spatial_reactions = False

        # This is the time-step counter
        self.n = 0
    
        self.beta = beta
        self.calc_lambda(potential)
        self.calc_psi()
        self.calc_Phi()
      
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
            if self.boundary_condition is BC_periodic:
                boltz_denom[:,-1] += boltz_func[:,0]
                boltz_denom[:,0] += boltz_func[:,-1]
            if self.shape[0] > 1:
                boltz_denom[:-1,:] += boltz_func[1:,:]
                boltz_denom[1:,:] += boltz_func[:-1,:]
                if self.boundary_condition is BC_periodic:
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
            if self.boundary_condition is BC_periodic:
                self.lam[:,0,0] += boltz_func[:,-1] / boltz_denom[:,0]
                self.lam[:,-1,1] += boltz_func[:,0] / boltz_denom[:,-1]

            if self.shape[0] > 1:
                # up
                self.lam = np.dstack([self.lam, np.zeros(self.shape)])
                self.lam[1:,:,2] += boltz_func[:-1,:] / boltz_denom[1:,:]
                # down
                self.lam = np.dstack([self.lam, np.zeros(self.shape)])
                self.lam[:-1,:,3] += boltz_func[1:,:] / boltz_denom[:-1,:]
                if self.boundary_condition is BC_periodic:
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
            self.omegas = [np.zeros(self.N) for i in range(len(self.Xs))]

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
            self.nus = [np.zeros(self.N) for i in range(len(self.Xs))]
    
    def time_step_with_Q(self):
        """Take a time step forward using arrival densities. NOTE in the diffusive case 
        it is necessary to use a full history to get this one right"""
        
        # First we increment the time counter!
        self.n += 1
 
        for i in range(len(self.Xs)):
            Q = self.Qs[i]
            theta = self.thetas[i]
            omega = self.omegas[i]
            nu = self.nus[i]

            # How many time steps have we had?
            lookback = min(self.n, self.history_length)
            
            # Matrix methods to calc Q as in eq. (9) in the J Comp Phys paper
            if self.has_spatial_reactions:
                # Matrix multiply to calculate flux (using memory kernel), then sum in last dimension (2), to get outward flux
                flux = (Q[:, :, self.n-lookback:self.n] * theta[:,:,self.n-lookback:self.n] * self.psi[1:lookback+1][::-1]).sum(2) 
            else:
                flux = (Q[:, :, self.n-lookback:self.n] * theta[self.n-lookback:self.n] * self.psi[1:lookback+1][::-1]).sum(2) 

            # Now apply lambda jump probabilities
            next_Q = np.zeros(self.shape)
            
            # First multiply by all the left jump probabilities
            next_Q[:,:-1] += (self.lam[:,:,0] * flux)[:,1:]
            # then all the right jump 
            next_Q[:,1:] += (self.lam[:,:,1] * flux)[:,:-1]
            if self.boundary_condition is BC_periodic:
                next_Q[:,-1] += self.lam[:,0,0] * flux[:,0]
                next_Q[:,0] += self.lam[:,-1,1] * flux[:,-1]
            if self.shape[0] > 1:
                # The up jump
                next_Q[:-1,:] += (self.lam[:,:,2] * flux)[1:,:]
                # The down jump
                next_Q[1:,:] += (self.lam[:,:,3] * flux)[:-1,:]
                if self.boundary_condition is BC_periodic:
                    next_Q[-1,:] += self.lam[0,:,2] * flux[0,:]
                    next_Q[0,:] += self.lam[-1,:,3] * flux[-1,:]

            # Add Q to the list of Qs over time
            self.Qs[i][:,:,self.n] = next_Q

            lookback = min(self.n+1, self.history_length)
            
            # Matrix methods to calc X as in eq. (11) in the J Comp Phys paper
            if self.has_spatial_reactions:
                # Matrix multiply to calculate flux (using memory kernel), then sum in last dimension (2), to get outward flux
                next_X = (Q[:, :, self.n+1-lookback:self.n+1] * theta[:,:,self.n+1-lookback:self.n+1] * self.Phi[:lookback][::-1]).sum(2) + nu[:,:,self.n-1]
            else:
                next_X = (Q[:, :, self.n+1-lookback:self.n+1] * theta[self.n+1-lookback:self.n+1] * self.Phi[:lookback][::-1]).sum(2) + nu[self.n-1]

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
                next_X = X[:,:,self.n-1] - flux - omega[:,:,self.n-1] * X[:,:,self.n-1] + nu[:,:,self.n-1]
            else:
                flux = (X[:, :, self.n-lookback:self.n] * theta[self.n-lookback:self.n] * self.K[1:lookback+1][::-1]).sum(2)
                next_X = X[:,:,self.n-1] - flux - omega[self.n-1] * X[:,:,self.n-1] + nu[self.n-1]

            # Now we add the spatial jumps
            # First multiply by all the left jump probabilities
            next_X[:,:-1] += (self.lam[:,:,0] * flux)[:,1:]
            # then all the right jump 
            next_X[:,1:] += (self.lam[:,:,1] * flux)[:,:-1]
             
            if X.shape[0] > 1:
                # The up jump
                next_X[:-1,:] += (self.lam[:,:,2] * flux)[1:,:]
                # The down jump
                next_X[1:,:] += (self.lam[:,:,3] * flux)[:-1,:]
            
            # Apply the boundary conditions
            self.boundary_condition.apply_BCs(next_X, flux, self)

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
            #if i % 100 == 0:
            #    print "Solved to step", i
            self.time_step()

class DTRW_diffusive(DTRW):

    def __init__(self, X_inits, N, r, history_length=2, beta = 0., potential = np.array([]), boundary_condition=BC()):
        # Probability of jumping in one step 
        self.r = r

        super(DTRW_diffusive, self).__init__(X_inits, N, history_length, beta, potential, boundary_condition)

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

    def __init__(self, X_inits, N, alpha, history_length = 0, beta = 0., potential = np.array([]), boundary_condition=BC()):
        
        self.alpha = alpha
        super(DTRW_subdiffusive, self).__init__(X_inits, N, history_length, beta, potential, boundary_condition)

    def calc_psi(self):
        """Waiting time distribution for spatial jumps"""
        self.psi = np.zeros(self.history_length+1)
        self.psi[0] = 0.
        
        self.psi[1] = self.alpha
        for i in range(2,self.history_length+1):
            self.psi[i] = self.psi[i-1] * (float(i) - 1. - self.alpha) / float(i)
        
        # Highly dubious: Normalising psi so that we conservation of particles
        # self.psi[-1] = 1.0 - self.psi[:-1].sum()
        
    
    def calc_Phi(self):
        """PDF of not jumping up to time n"""
        self.Phi = np.ones(self.history_length+1)
        self.Phi[1] = 1.0 - self.alpha
        for i in range(2, self.history_length+1):
            self.Phi[i] = self.Phi[i-1] * (float(i) - self.alpha) / float(i)
        
    def calc_mem_kernel(self):
        """Once off call to calculate the memory kernel and store it"""
        self.K = np.zeros(self.history_length+1)
        
        self.K[0] = 0.0
        self.K[1] = self.alpha 
        self.K[2] = self.alpha * 0.5 * (self.alpha - 1.0)
        for i in range(3,self.history_length+1):
            self.K[i] = (float(i) + self.alpha - 2.0) * self.K[i-1] / float(i)

class DTRW_diffusive_with_transition(DTRW_diffusive):
    
    def __init__(self, X_inits, N, r, k_1, k_2, clearance_rate, infection_rate, history_length=2, beta = 0., potential = np.array([]), boundary_condition=BC()):
        
        self.k_1 = k_1 
        self.k_2 = k_2
        # For now NO DEATH PROCESS - FOR TESTING CONSERVATION OF PARTICLES
        self.clearance_rate = 0. #clearance_rate 
        self.infection_rate = infection_rate

        super(DTRW_diffusive_with_transition, self).__init__(X_inits, N, r, history_length, beta, potential, boundary_condition)
   
        self.has_spatial_reactions = True

    def calc_omega(self):
        """ Probability of death between n and n+1"""
        if self.omegas == None:
            self.omegas = [np.zeros((X.shape[0], X.shape[1], self.N)) for X in self.Xs]
            
        # We assume that the calculation has been made for all n up to now, so we simply update the n-th point
        # Virions, layer 1
        self.omegas[0][:,:,self.n] = 1. - np.exp(-self.k_1) 
        
        # Virions, layer 2 
        self.omegas[1][:,:,self.n] = 1. - np.exp(-self.k_2 - self.clearance_rate)

        # Target CD4+ cells, layer 2
        self.omegas[2][:,:,self.n] = 1. - np.exp(-self.infection_rate * self.Xs[1][:,:,self.n])
        
        # Infected CD4+ cells, layer 2
        # For now NO DEATH PROCESS - FOR TESTING CONSERVATION OF PARTICLES


    def calc_theta(self):
        """ Probability of surviving between 0 and n"""
        if self.thetas == None:
            self.thetas = [np.ones((X.shape[0], X.shape[1], self.N)) for X in self.Xs]

        for i in range(len(self.Xs)):
            #self.thetas[i][:,:,self.n] = (1. - self.omegas[i][:,:,:self.n+1]).prod(2)
            # Above is wrong! Below is right!!!! remember you want theta between m and n, not 0 and m!!!!
            self.thetas[i][:,:,:self.n+1] = self.thetas[i][:,:,:self.n+1] * np.dstack([1. - self.omegas[i][:,:,self.n]])

    def calc_nu(self):
        """Likelihood of birth happening at i"""
        if self.nus == None:
            self.nus = [np.zeros((X.shape[0], X.shape[1], self.N)) for X in self.Xs]
    
        # Here we get birth rates that reflect death rates, that is, everything balances out in the end.
        self.nus[0][:,:,self.n] = (1. - np.exp(-self.k_2)) * self.Xs[1][:,:,self.n]
        self.nus[1][:,:,self.n] = (1. - np.exp(-self.k_1)) * self.Xs[0][:,:,self.n]
        # No birth proces for target cells
        #self.nus[2][:,:,self.n]
        self.nus[3][:,:,self.n] = (1. - np.exp(-self.infection_rate * self.Xs[1][:,:,self.n])) * self.Xs[2][:,:,self.n]


class DTRW_subdiffusive_with_transition(DTRW_subdiffusive):
    
    def __init__(self, X_inits, N, alpha, k_1, k_2, clearance_rate, infection_rate, history_length = 0, beta = 0., potential = np.array([]), boundary_condition=BC()):
        
        self.k_1 = k_1 
        self.k_2 = k_2
        # For now NO DEATH PROCESS - FOR TESTING CONSERVATION OF PARTICLES
        self.clearance_rate = 0. #clearance_rate 
        self.infection_rate = infection_rate

        super(DTRW_subdiffusive_with_transition, self).__init__(X_inits, N, alpha, history_length, beta, potential, boundary_condition)
   
        self.has_spatial_reactions = True

    def calc_omega(self):
        """ Probability of death between n and n+1"""
        if self.omegas == None:
            self.omegas = [np.zeros((X.shape[0], X.shape[1], self.N)) for X in self.Xs]
            
        # We assume that the calculation has been made for all n up to now, so we simply update the n-th point
        # Virions, layer 1
        self.omegas[0][:,:,self.n] = 1. - np.exp(-self.k_1) 
        
        # Virions, layer 2 
        self.omegas[1][:,:,self.n] = 1. - np.exp(-self.k_2 - self.clearance_rate)

        # Target CD4+ cells, layer 2
        self.omegas[2][:,:,self.n] = 1. - np.exp(-self.infection_rate * self.Xs[1][:,:,self.n])
        
        # Infected CD4+ cells, layer 2
        # For now NO DEATH PROCESS - FOR TESTING CONSERVATION OF PARTICLES


    def calc_theta(self):
        """ Probability of surviving between 0 and n"""
        if self.thetas == None:
            self.thetas = [np.ones((X.shape[0], X.shape[1], self.N)) for X in self.Xs]

        for i in range(len(self.Xs)):
            #self.thetas[i][:,:,self.n] = (1. - self.omegas[i][:,:,:self.n+1]).prod(2)
            # Above is wrong! Below is right!!!! remember you want theta between m and n, not 0 and m!!!!
            self.thetas[i][:,:,:self.n+1] = self.thetas[i][:,:,:self.n+1] * np.dstack([1. - self.omegas[i][:,:,self.n]])

    def calc_nu(self):
        """Likelihood of birth happening at i"""
        if self.nus == None:
            self.nus = [np.zeros((X.shape[0], X.shape[1], self.N)) for X in self.Xs]
    
        # Here we get birth rates that reflect death rates, that is, everything balances out in the end.
        self.nus[0][:,:,self.n] = (1. - np.exp(-self.k_2)) * self.Xs[1][:,:,self.n]
        self.nus[1][:,:,self.n] = (1. - np.exp(-self.k_1)) * self.Xs[0][:,:,self.n]
        # No birth proces for target cells
        #self.nus[2][:,:,self.n]
        self.nus[3][:,:,self.n] = (1. - np.exp(-self.infection_rate * self.Xs[1][:,:,self.n])) * self.Xs[2][:,:,self.n]

class DTRW_subdiffusive_fedotov_death(DTRW_subdiffusive):
    """ A subdiffusive system as outlined in Fedotov & Falconer, 2014. We check the results
        against a known stationary solution """

    def __init__(self, X_inits, N, alpha, k, history_length = 0, beta = 0., potential = np.array([]), boundary_condition=BC()):
        
        self.k = k
        super(DTRW_subdiffusive_fedotov_death, self).__init__(X_inits, N, alpha, history_length, beta, potential, boundary_condition)
        self.has_spatial_reactions = True

    def calc_omega(self):
        """ Probability of death between n and n+1"""
        if self.omegas == None:
            self.omegas = [np.zeros((X.shape[0], X.shape[1], self.N)) for X in self.Xs]
        
        self.omegas[0][:,:,self.n] = self.k * self.Xs[0][:,:,self.n] #1. - np.exp(-self.k * self.Xs[0][:,:,self.n] * self.Xs[0][:,:,self.n])

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
        """Likelihood of birth happening at i, just zero as there's no births """
        if self.nus == None:
            self.nus = [np.zeros((X.shape[0], X.shape[1], self.N)) for X in self.Xs]

class DTRW_ODE(object):
    """ A DTRW class for non spatial, compartment only models, so reactions are easily given either an anomalous  """

    def __init__(self, X_inits, N):
        
        self.N = N
        self.n = 0
        self.Xs = np.vstack(X_inits)
        # We define no kernels and no fluxes.

    def out_flux(self, i):
        """ defines the outward flux of a compartment """
        return np.zeros(len(self.Xs[-1]))

    def in_flux(self, i):
        """ defines the inward flux of a compartment """
        return np.zeros(len(self.Xs[-1]))
    
    def time_step(self):
        """ Step forwards directly with X using the memory kernels K, this
            method is only available in cases where we can calculate K analytically!"""
 
        # First we increment the time counter!
        self.n += 1

        next_X = self.Xs[:,-1] + self.in_flux() - self.out_flux()
         
        self.Xs = np.column_stack([self.Xs, next_X])
   

    def calc_sibuya_kernel(self, N, alpha):
        """Once off call to calculate the memory kernel and store it"""
        result = np.zeros(N+1)
        
        result[0] = 0.0
        result[1] = alpha 
        result[2] = alpha * 0.5 * (self.alpha - 1.0)
        for i in range(3,N+1):
            result = (float(i) + alpha - 2.0) * result[i-1] / float(i)
 
    def calc_diff_kernel(self, N, r):
        """Once off call to calculate the memory kernel and store it"""
        result = np.zeros(N+1)
         
        # In the exponential case it's quite simply...
        result[1] = r

class DTRW_two_compartment_test(DTRW_ODE):

    def __init__(self, X_inits, N, dt, alpha, delta):

        self.transition_K = self.calc_sibuya_kernel(N)
        self.death_K = self.calc_diff_kernel(2, delta)

    def out_flux(self):
        
        out_f = np.zeros(len(self.Xs[:,-1]))
        
        out_f[0] = self.Xs[0, :] * self.transition_K[:self.n][::-1]
        out_f[1] = self.Xs[1, :-2] * self.death_K[::-1]
