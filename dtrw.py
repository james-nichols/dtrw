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
# o Make the SIR / compartment model work for testing
# o Potentially put r back in to lambda and do it that way, remove from apply_BC argument

def calc_sibuya_kernel(N, alpha):
    """ The Sibuya Kernel as defined in J Phys Comp """
    K = np.zeros(N+1)
    
    K[0] = 0.0
    K[1] = alpha 
    K[2] = alpha * 0.5 * (alpha - 1.0)
    for i in range(3,N+1):
        K[i] = (float(i) + alpha - 2.0) * K[i-1] / float(i)

    return K

def calc_exp_kernel(N, r):
    """ Exponential waiting times leads to delta function kernel """
    K = np.zeros(N+1)
     
    # In the exponential case it's quite simply...
    K[1] = r

    return K

class BC(object):
    def __init__(self):
        pass
    def apply_BCs(self, next_X, flux, r, dtrw):
        # Apply the boundary conditions, called from the time step routine in DTRW
        pass

class BC_Dirichelet(BC):
    def __init__(self, data):
        self.data = data

    def apply_BCs(self, next_X, flux, r, dtrw):
        if self.data[0].shape != next_X[:,0].shape and self.data[0].shape:
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

    def apply_BCs(self, next_X, flux, r, dtrw):
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

    def apply_BCs(self, next_X, flux, r, dtrw):
        # Only left side has BC
        next_X[:,0] += (dtrw.omegas[0][:,:,dtrw.n-1] * dtrw.Xs[0][:,:,dtrw.n-1]).sum()
        next_X[:,0] += r * dtrw.lam[:,0,0] * flux[:,0] 

class BC_periodic(BC):
    
    def __init__(self):
        pass 

    def apply_BCs(self, next_X, flux, r, dtrw):
        # Apply the boundary conditions
        next_X[:,-1] += dtrw.lam[:,0,0] * r * flux[:,0]
        next_X[:,0] += dtrw.lam[:,-1,1] * r * flux[:,-1]

        if next_X.shape[0] > 1:
            next_X[-1,:] += dtrw.lam[0,:,2] * r * flux[0,:]
            next_X[0,:] += dtrw.lam[-1,:,3] * r * flux[-1,:]

class BC_zero_flux(BC):
    
    def __init__(self):
        pass 

    def apply_BCs(self, next_X, flux, r, dtrw):
        # Apply the boundary conditions
        next_X[:,-1] = next_X[:,-3]
        next_X[:,0] = next_X[:,2]
        
        # New regime: simply reflect flux back where it came from!
        #next_X[:,0] += r * dtrw.lam[:,0,0] * flux[:,0]
        #next_X[:,-1] += r * dtrw.lam[:,-1,1] * flux[:,-1]

        if next_X.shape[0] > 1:
            next_X[-1,:] = next_X[-3,:]
            next_X[0,:] = next_X[2,:]
            #next_X[0,:] += r * dtrw.lam[0,:,2] * flux[0,:]
            #next_X[-1,:] += r * dtrw.lam[-1,:,3] * flux[-1,:]

class BC_zero_flux_centred(BC):
    
    def __init__(self):
        pass 

    def apply_BCs(self, next_X, flux, r, dtrw):
        # Apply the boundary conditions
        
        # New regime: simply reflect flux back where it came from!
        next_X[:,0] += r * dtrw.lam[:,0,0] * flux[:,0]
        next_X[:,-1] += r * dtrw.lam[:,-1,1] * flux[:,-1]

        if next_X.shape[0] > 1:
            next_X[0,:] += r * dtrw.lam[0,:,2] * flux[0,:]
            next_X[-1,:] += r * dtrw.lam[-1,:,3] * flux[-1,:]

class BC_zero_flux_centred(BC):
    
    def __init__(self):
        pass 

    def apply_BCs(self, next_X, flux, r, dtrw):
        # Apply the boundary conditions
        next_X[:,-1] = next_X[:,-3]
        next_X[:,0] = next_X[:,2]
        
        if next_X.shape[0] > 1:
            next_X[-1,:] = next_X[-3,:]
            next_X[0,:] = next_X[2,:]

class DTRW(object):
    """ Base definition of a DTRW with arbitrary wait-times
        for reactions and jumps """

    def __init__(self, X_inits, N, r = 1., history_length = 2, boltz_beta = 0., potential = np.array([]), boundary_condition=BC()):
        """X is the initial concentration field, N is the number of time steps to simulate"""

        if isinstance(boundary_condition, BC_zero_flux) or isinstance(boundary_condition, BC_zero_flux_centred): 
            # Make ghost points...
            for i in range(len(X_inits)):
                if len(X_inits[i].shape) > 1:
                    X_inits[i] = np.pad(X_inits[i], [[1,1],[1,1]], 'edge')
                else:
                    X_inits[i] = np.pad(X_inits[i], [1,1], 'edge')

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
       
        if isinstance(r, (int, float)): 
            self.r = [r for i in range(len(self.Xs))]
        elif len(r)==len(self.Xs):
            self.r = r
        
        self.boundary_condition = boundary_condition

        # This is the time-step counter
        self.n = 0
    
        self.boltz_beta = boltz_beta
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
            boltz_func = np.exp(-self.boltz_beta * potential)
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
            self.omegas = [np.zeros(X.shape) for X in self.Xs]

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
            self.nus = [np.zeros(X.shape) for X in self.Xs]

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

            # Matrix multiply to calculate flux (using memory kernel), then sum in last dimension (2), to get outward flux
            flux = (Q[:, :, self.n-lookback:self.n] * theta[:,:,self.n-lookback:self.n] * self.psi[1:lookback+1][::-1]).sum(2) 

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
  
            # Matrix multiply to calculate flux (using memory kernel), then sum in last dimension (2), to get outward flux
            next_X = (Q[:, :, self.n+1-lookback:self.n+1] * theta[:,:,self.n+1-lookback:self.n+1] * self.Phi[:lookback][::-1]).sum(2) + nu[:,:,self.n-1]

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
            r = self.r[i]

            # Matrix multiply to calculate flux (using memory kernel), then sum in last dimension (2), to get outward flux
            flux = (X[:, :, self.n-lookback:self.n] * theta[:,:,self.n-lookback:self.n] * self.K[1:lookback+1][::-1]).sum(2)
            next_X = X[:,:,self.n-1] - r * flux - omega[:,:,self.n-1] * X[:,:,self.n-1] + nu[:,:,self.n-1]
           
            if (flux < 0.).sum() == True:
                print "Step", i, "has -ve flux"
                print flux

            # Now we add the spatial jumps
            # First multiply by all the left jump probabilities
            next_X[:,:-1] += r * (self.lam[:,:,0] * flux)[:,1:]
            # then all the right jump 
            next_X[:,1:] += r * (self.lam[:,:,1] * flux)[:,:-1]
            
            if X.shape[0] > 1:
                # The up jump
                next_X[:-1,:] += r * (self.lam[:,:,2] * flux)[1:,:]
                # The down jump
                next_X[1:,:] += r * (self.lam[:,:,3] * flux)[:-1,:]
            
            # Apply the boundary conditions
            self.boundary_condition.apply_BCs(next_X, flux, r, self)
            
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
        
        if isinstance(self.boundary_condition, BC_zero_flux) or isinstance(self.boundary_condition, BC_zero_flux_centred): 
            # Get rid of ghost points...
            for i in range(len(self.Xs)):
                if self.Xs[i].shape[0] > 1:
                    self.Xs[i] = self.Xs[i][1:-1,1:-1,:]
                else:
                    self.Xs[i] = self.Xs[i][:,1:-1,:]

class DTRW_diffusive(DTRW):

    def __init__(self, X_inits, N, omega, r = 1.0, history_length=2, boltz_beta = 0., potential = np.array([]), boundary_condition=BC()):
        # Probability of jumping in one step 
        self.omega = omega

        super(DTRW_diffusive, self).__init__(X_inits, N, r, history_length, boltz_beta, potential, boundary_condition)

    def calc_psi(self):
        """Waiting time distribution for spatial jumps"""
        self.psi = np.array([self.omega * pow(1. - self.omega, i-1) for i in range(self.history_length+1)])
        self.psi[0] = 0.

    def calc_Phi(self):
        """PDF of not jumping up to time n"""
        self.Phi = np.array([pow(1. - self.omega, i) for i in range(self.history_length+1)])
    
    def calc_mem_kernel(self):
        """Once off call to calculate the memory kernel and store it"""
        self.K = np.zeros(self.history_length+1)
         
        # In the exponential case it's quite simply...
        self.K[1] = self.omega

class DTRW_subdiffusive(DTRW):

    def __init__(self, X_inits, N, alpha, r = 1.0, history_length = 0, boltz_beta = 0., potential = np.array([]), boundary_condition=BC()):
        self.alpha = alpha
        super(DTRW_subdiffusive, self).__init__(X_inits, N, r, history_length, boltz_beta, potential, boundary_condition)

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

class DTRW_subdiffusive_with_death(DTRW_subdiffusive):
    
    def __init__(self, X_inits, N, alpha, k, r = 1., history_length = 0, boltz_beta = 0., potential = np.array([]), boundary_condition=BC()):
        
        self.k = k

        super(DTRW_subdiffusive_with_death, self).__init__(X_inits, N, alpha, r, history_length, boltz_beta, potential, boundary_condition)
   
    def calc_omega(self):
        """ Probability of death between n and n+1"""
        if self.omegas == None:
            self.omegas = [np.zeros((X.shape[0], X.shape[1], self.N)) for X in self.Xs]

        self.omegas[0][:,:,self.n] = 1. - np.exp(-self.k) 
        
class DTRW_diffusive_with_transition(DTRW_diffusive):
    
    def __init__(self, X_inits, N, omega, k_1, k_2, clearance_rate, infection_rate, r = 1., history_length=2, boltz_beta = 0., potential = np.array([]), boundary_condition=BC()):
        
        self.k_1 = k_1 
        self.k_2 = k_2
        self.clearance_rate = 0. #clearance_rate 
        self.infection_rate = infection_rate

        super(DTRW_diffusive_with_transition, self).__init__(X_inits, N, omega, r, history_length, boltz_beta, potential, boundary_condition)
   
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
    
    def __init__(self, X_inits, N, alpha, k_1, k_2, clearance_rate, infection_rate, r = 1., history_length = 0, boltz_beta = 0., potential = np.array([]), boundary_condition=BC()):
        
        self.k_1 = k_1 
        self.k_2 = k_2
        self.clearance_rate = 0. #clearance_rate 
        self.infection_rate = infection_rate

        super(DTRW_subdiffusive_with_transition, self).__init__(X_inits, N, alpha, r, history_length, boltz_beta, potential, boundary_condition)
   
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

    def __init__(self, X_inits, N, alpha, k, r=1., history_length = 0, boltz_beta = 0., potential = np.array([]), boundary_condition=BC()):
        
        self.k = k
        super(DTRW_subdiffusive_fedotov_death, self).__init__(X_inits, N, alpha, r, history_length, boltz_beta, potential, boundary_condition)

    def calc_omega(self):
        """ Probability of death between n and n+1"""
        if self.omegas == None:
            self.omegas = [np.zeros((X.shape[0], X.shape[1], self.N)) for X in self.Xs]
        
        self.omegas[0][:,:,self.n] = self.k * self.Xs[0][:,:,self.n] #1. - np.exp(-self.k * self.Xs[0][:,:,self.n] * self.Xs[0][:,:,self.n])

    def calc_nu(self):
        """Likelihood of birth happening at i, just zero as there's no births """
        if self.nus == None:
            self.nus = [np.zeros((X.shape[0], X.shape[1], self.N)) for X in self.Xs]

#####################
# COMPARTMENT MODELS
#####################

class DTRW_compartment(object):
    """ A DTRW class for non spatial, compartment only models, so reactions are easily given either an anomalous  """

    def __init__(self, X_inits, T, dT):
        
        self.N = int(math.ceil(float(T) / dT))
        self.T = T
        self.dT = dT
        self.n = 0 # Time point counter

        self.n_species = len(X_inits)
       
        self.Ks = [None] * self.n_species
        self.anom_rates = np.zeros(self.n_species)

        self.Xs = np.zeros([self.n_species, self.N])
        self.Xs[:,0] = X_inits
 
        # Need the survival probs for each compartment for the anomalous removals.
        self.survival_probs = np.ones([self.n_species, self.N])

    def cration_rate(self, n):
        """ Define a simple creation rate """
        return np.zeros(self.n_species)

    def creation_flux(self, n):
        """ Defines all creation processes for each compartment """
        return np.zeros(self.n_species)

    def removal_rate(self, n):
        """ Defines the removal rate AT time point n """
        return np.zeros(self.n_species)
    
    def removal_prob_markovian(self, n):
        """ Simple markovian probability calc - like an exponential interpolation """
        return (1. - np.exp(-self.dT * self.removal_rate(n)))

    def removal_flux_markovian(self, n):
        """ Defines all Markovian removal processes """
        return self.removal_prob_markovian(n) * self.Xs[:,n]
    
    def removal_flux_anomalous(self, n):
        """ Defines all outward anomalous removal processes """
        flux = np.zeros(self.n_species)
        for i in range(self.n_species):
            if self.Ks[i] != None:
                flux[i] = self.dT * self.anom_rates[i] * (self.Xs[i,:n] * self.survival_probs[i,:n] * self.Ks[i][1:n+1][::-1]).sum()
        return flux 

    def calc_anom_flux(self, K, X, survival_prob, n):
        """ Handy function to calculate outward flux due to a kernel 
            Theta is the survival prob. of all other compartments """
        return (X[:n] * survival_prob[:n] * K[1:n+1][::-1]).sum()

    def time_step(self):
        """ Step forwards directly with X using the memory kernels K, this
            method is only available in cases where we can calculate K analytically!"""
 
        # Increment the time counter
        self.n += 1
        
        # Update survival probabilities
        self.survival_probs[:,:self.n] = self.survival_probs[:,:self.n] * np.vstack(1. - self.removal_prob_markovian(self.n-1))
        # Update popultions
        self.Xs[:,self.n] = self.Xs[:,self.n-1] + self.creation_flux(self.n-1) - self.removal_flux_markovian(self.n-1) - self.removal_flux_anomalous(self.n)

    def solve_all_steps(self):
        """Solve the time steps using the memory kernel, available only if we know how to calculate K"""
        
        for i in range(self.N-1):
            self.time_step()
        
class DTRW_SIR(DTRW_compartment):

    def __init__(self, X_inits, T, dT, lam, omega, gamma, mu, alpha):

        if len(X_inits) != 3:
            # Error!
            print "Need three initial points"
            raise SystemExit 

        super(DTRW_SIR, self).__init__(X_inits, T, dT)
        
        self.lam = lam
        self.omega = omega
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

        self.Ks[1] = calc_sibuya_kernel(self.N+1, self.alpha)
        self.anom_rates[1] = self.mu

    def creation_flux(self, n):
        return np.array([(1. - np.exp(-self.dT * self.lam)), \
                         (1. - np.exp(-self.dT * self.omega * self.Xs[1,n])) * self.Xs[0,n], \
                         self.removal_flux_anomalous(n)[1] ])


    def removal_rate(self, n):
        return np.array([self.omega * self.Xs[1, n] + self.gamma, \
                         self.gamma, \
                         self.gamma])

     
class DTRW_PBPK(DTRW_compartment):

    def __init__(self, X_inits, T, dT, V, Q, R, mu, Vmax, Km, alpha):
        
        if len(X_inits) != 6:
            # Error!
            print "Need six initial points"
            raise SystemExit

        super(DTRW_PBPK, self).__init__(X_inits, T, dT)
          
        self.Vs = V
        self.Qs = Q
        self.Rs = R
        self.mu = mu
        self.Vmax = Vmax
        self.Km = Km
        self.alpha = alpha

        self.Ks[1] = calc_sibuya_kernel(self.N+1, self.alpha)
        self.anom_rates[1] = self.mu

    def creation_flux(self, n):
        return np.array([(1. - np.exp(-self.dT * self.lam)), \
                         (1. - np.exp(-self.dT * self.omega * self.Xs[1,n])) * self.Xs[0,n], \
                         self.removal_flux_anomalous(n)[1] ])


    def removal_rate(self, n):
        return np.array([self.Qs[0] / (self.Vs[0] * self.Rs[0]),
                         self.Qs[1] / (self.Vs[1] * self.Rs[1]),
                         self.Qs[2] / (self.Vs[2] * self.Rs[2]),
                         self.Qs[3] / (self.Vs[3] * self.Rs[3]) + self.mu,
                         self.Qs[4] / (self.Vs[4] * self.Rs[4]) + self.Vmax / (self.Km + self.Xs[4,n]),
                         self.Qs.sum() / self.Vs[5])
                        
