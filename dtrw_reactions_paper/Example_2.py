#! /usr/bin/env python

# Libraries are in parent directory
import sys
sys.path.append('../')

import numpy as np
import time
from dtrw import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm

class DTRW_diffusive_two_way(DTRW_diffusive):
    
    def __init__(self, X_inits, N, alpha, k_1, k_2, r=1., history_length=0, boltz_beta=0., potential = np.array([]), boundary_condition=BC()):
        
        self.k_1 = k_1
        self.k_2 = k_2

        super(DTRW_diffusive_two_way, self).__init__(X_inits, N, alpha, r=r, history_length=history_length, boltz_beta=boltz_beta, potential=potential, boundary_condition=boundary_condition)
   
    def calc_omega(self):
        """ Probability of death between n and n+1"""
        if self.omegas == None:
            self.omegas = [np.zeros((X.shape[0], X.shape[1])) for X in self.Xs]
            
        # We assume that the calculation has been made for all n up to now, so we simply update the n-th point
        # a 
        self.omegas[0][:,:] = 1. - np.exp(-self.k_1) 
        # b 
        self.omegas[1][:,:] = 1. - np.exp(-self.k_2)

    def calc_nu(self):
        """Likelihood of birth happening at i"""
        if self.nus == None:
            self.nus = [np.zeros((X.shape[0], X.shape[1])) for X in self.Xs]
    
        self.nus[0][:,:] = (1. - np.exp(-self.k_2)) * self.Xs[1][:,:,self.n]
        self.nus[1][:,:] = (1. - np.exp(-self.k_1)) * self.Xs[0][:,:,self.n]

class DTRW_subdiffusive_two_way(DTRW_subdiffusive):
     
    def __init__(self, X_inits, N, alpha, k_1, k_2, r=1., history_length=0, boltz_beta=0., potential = np.array([]), boundary_condition=BC()):
        
        self.k_1 = k_1
        self.k_2 = k_2
 
        super(DTRW_subdiffusive_two_way, self).__init__(X_inits, N, alpha, r=r, history_length=history_length, boltz_beta=boltz_beta, potential=potential, boundary_condition=boundary_condition)
   
    def calc_omega(self):
        """ Probability of death between n and n+1"""
        if self.omegas == None:
            self.omegas = [np.zeros((X.shape[0], X.shape[1])) for X in self.Xs]
            
        # We assume that the calculation has been made for all n up to now, so we simply update the n-th point
        # a 
        self.omegas[0][:,:] = 1. - np.exp(-self.k_1) 
        #self.omegas[0][:,:] = - self.k_1 
        # b 2.
        self.omegas[1][:,:] = 1. - np.exp(-self.k_2)
        #self.omegas[1][:,:] = - self.k_2

    def calc_nu(self):
        """Likelihood of birth happening at i"""
        if self.nus == None:
            self.nus = [np.zeros((X.shape[0], X.shape[1])) for X in self.Xs]
    
        self.nus[0][:,:] = (1. - np.exp(-self.k_2)) * self.Xs[1][:,:,self.n]
        #self.nus[0][:,:] = -self.k_2 * self.Xs[1][:,:,self.n]
        self.nus[1][:,:] = (1. - np.exp(-self.k_1)) * self.Xs[0][:,:,self.n]
        #self.nus[1][:,:] = -self.k_1 * self.Xs[0][:,:,self.n]


alpha_1 = 0.5
D_alpha = 1.
T = 0.1
k_1 = 1.

r = 1.

#dXs = [0.2, 0.1, 1.0/15.0, 0.05, 0.025, 0.0125]
dXs = [1./2, 1./5., 1./7., 1./10., 1./15., 1./20., 1./30.]

for dX in dXs:
     
    dT = pow(dX * dX / (2.0 * D_alpha), 1.0 / alpha_1)

    xs = np.arange(-1.0, 1.0+dX, dX)
    
    # We make the init conditions slightly bigger for the zero flux boundaries
    a_init = np.zeros(xs.shape)
    a_init[xs.shape[0] / 2 ] = float(xs.shape[0])/2.
    b_init = np.zeros(xs.shape)

    N = int(round(T / dT) + 1)
    N_1 = int(round(T / dT) + 1)
    history_length = N + 1

    ts = np.array(np.arange(N) * dT)
    ts_1 = np.array(np.arange(N_1) * dT)

    # Calculate r for diffusive case so as to get the *same* dT as the subdiffusive case
    omega = dT / (dX * dX / (2.0 * D_alpha))

    #print "DTRW for dX =", dX, "dT =", dT, "omega =", omega, "N =", N
    print "Solving DTRW subdiff for dX =", dX, "dT =", dT, "r =", r, "alpha =", alpha_1, "N =", N_1

    bc = BC_zero_flux()

    dtrw_sub_1 = DTRW_subdiffusive_two_way([a_init, b_init], N_1, alpha_1, k_1 * dT, k_1 * dT, r=r, history_length=N_1, boundary_condition=bc)
    #dtrw = DTRW_diffusive_two_way([a_init, b_init], N, omega, k_1 * dT, k_1 * dT, history_length=2, boundary_condition=bc)

    #dtrw.solve_all_steps()
    #print "Exp case solved"
    dtrw_sub_1.solve_all_steps()
    
    dtrw_file_name = "DTRW_dT_{0:f}_dX_{1:f}.csv".format(dT, dX)

    #np.savetxt("a_" + dtrw_file_name, dtrw.Xs[0][0,:,-1], delimiter=",")
    #np.savetxt("b_" + dtrw_file_name, dtrw.Xs[1][0,:,-1], delimiter=",")

    dtrw_file_name_alpha_1 = "DTRW_dT_{0:f}_dX_{1:f}_alpha_{2:f}.csv".format(dT, dX, alpha_1)
    np.savetxt("a_" + dtrw_file_name_alpha_1, dtrw_sub_1.Xs[0][0,:,-1], delimiter=",")
    np.savetxt("b_" + dtrw_file_name_alpha_1, dtrw_sub_1.Xs[1][0,:,-1], delimiter=",")
    np.savetxt("a_2ndLast_" + dtrw_file_name_alpha_1, dtrw_sub_1.Xs[0][0,:,-2], delimiter=",")
    np.savetxt("b_2ndLast_" + dtrw_file_name_alpha_1, dtrw_sub_1.Xs[1][0,:,-2], delimiter=",")

    t_alpha_1 = "T_dT_{0:f}_alpha_{1:f}.csv".format(dT, alpha_1)
    np.savetxt(t_alpha_1, ts_1, delimiter=",")

#fig = plt.figure(figsize=(8,8))

#ax1 = fig.add_subplot(1,2,1)
#plt.xlim(-1., 1.)
#plt.ylim(0., 5.)
#line1, = plt.plot([],[],'r-')
#line2, = plt.plot([],[],'g-')
#line3, = plt.plot([],[],'k-')
#plt.legend([line1, line2, line3], ["a", "b", "sum"])

#ax2 = fig.add_subplot(1,2,2)
#line4, = ax2.plot([],[],'r-')
#line5, = ax2.plot([],[],'g-')
#line6, = ax2.plot([],[],'k-')
#plt.legend([line4, line5, line6], ["a", "b", "sum"])

def update(i, line1, line2, line3):
    line1.set_data(xs,dtrw_sub_1.Xs[0][:,:,i])
    line2.set_data(xs,dtrw_sub_1.Xs[1][:,:,i])
    line3.set_data(xs,dtrw_sub_1.Xs[0][:,:,i] + dtrw_sub_1.Xs[1][:,:,i])

    return line1, line2, line3

# call the animator. blit=True means only re-draw the parts that have changed.
#anim = animation.FuncAnimation(fig, update, 
#        frames=N_1, fargs=(line1, line2, line3), interval=10)

import inspect, os, subprocess
exec_name =  os.path.splitext(os.path.basename(inspect.getfile(inspect.currentframe())))[0]
git_tag = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).replace('\n', '')

#file_name = '{0}_{1}.mp4'.format(exec_name, git_tag)
#print "Saving animation to", file_name

#anim.save(file_name, fps=24)
#plt.show()

