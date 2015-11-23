#! /usr/bin/env python

# Libraries are in parent directory
import sys
sys.path.append('../')
from os import listdir

import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.optimize
from matplotlib.backends.backend_pdf import PdfPages
import scipy.io

import pdb

from dtrw import *
 
pp = PdfPages('PatientFitPlots_PINTDataLog.pdf')
plot_linear = False

class DTRW_HIV(DTRW_compartment):

    def __init__(self, X_inits, T, dT, \
                 k, T_cells, eff, delta_I, delta_V, burst, alpha):

        if len(X_inits) != 2:
            # Error!
            print "Need two initial points"
            raise SystemExit 

        super(DTRW_HIV, self).__init__(X_inits, T, dT)
        
        self.T_cells = T_cells
        self.k = k
        self.eff = eff

        self.delta_I = delta_I
        self.delta_V = delta_V
        self.burst = burst
        self.alpha = alpha

        self.alphas[0] = alpha
        self.Ks[0] = calc_sibuya_kernel(self.N+1, self.alpha)
    
    def creation_flux(self, n):
        return np.array([(1.0 - self.eff) * self.k * self.T_cells * self.Xs[1,n] * self.dT, 
                         self.removal_flux_anomalous(n)[0] * self.burst])
                         #self.Xs[0,n] * self.burst * self.dT])

    def anomalous_rates(self, n):
        return np.array([pow(self.delta_I, self.alpha), 0.])

    def removal_rates(self, n):
        return np.array([[0.], \
                         [self.delta_V]])

def produce_soln(params, k, T_cells, I_init, V_init, eff, virus, num_ts):
    T = virus[-1,0] #time_series[-1,0]
    dT = T / float(num_ts)
    
    t = np.arange(0., T, dT)

    delta_I, delta_V, burst, alpha = params
    
    dtrw_anom = DTRW_HIV([I_init, V_init], T, dT, k, T_cells, eff, delta_I, delta_V, burst, alpha)
    dtrw_anom.solve_all_steps()
   
    print "Soln with params: ", delta_I, delta_V, burst, alpha
    print virus[:,0]
    print virus[:,1]
    print np.interp(virus[:,0], t, dtrw_anom.Xs[1,:])

    return np.log(virus[:,1]) - np.log(np.interp(virus[:,0], t, dtrw_anom.Xs[1,:]))

delta_I = 1.0 
delta_V = 1.0 
V0 = 1.
I0 = 1.
alpha = 0.9
burst = 10.0
T = 100.
dT = 0.01 
ts = np.arange(0., T, dT)
t_sample = np.arange(0., T, 1.0)
T_cells = 0.0
k=0.0
eff=1.0


# This data was generated using a matlab ML function solver, for checking against a known
# "analytic" solution
matlab_ML = np.loadtxt('matlab_ML.csv', delimiter=',')
matlab_V = np.loadtxt('matlab_V.csv', delimiter=',')
plt.semilogy(t_sample, matlab_ML[:-1])
plt.semilogy(t_sample, matlab_V[:-1])

dTs = [1.0, 0.1, 0.01]

for dT in dTs:
    dtrw_hiv = DTRW_HIV([I0, V0], T, dT, k, T_cells, eff, delta_I, delta_V, burst, alpha)
    dtrw_hiv.solve_all_steps()

    ts = np.arange(0., T, dT)
    plt.semilogy(ts, dtrw_hiv.Xs[0,:], '--')
    plt.semilogy(ts, dtrw_hiv.Xs[1,:], 'x')

plt.show()


data = scipy.io.loadmat('Data/PINT_data.mat')

numpats = data['nopats']
patall = np.unique
raw = []
clean_data = []
for i in range(numpats):
    pat_data = data['dataprim'][i, 3][: ,[0,1,2,4,5]]
    clean_data.append(pat_data[~np.isnan(pat_data[:,1])])

for i in range(1): # range(numpats):
    # We don't have CD4 cell counts...
    #cells = data[0][data_label[0].index(pat)]

    hiv_ts = clean_data[i][:,[0,1]]
    
    I_init = 1e4
    V_init = hiv_ts[0,1]

    k = 2.4e-5
    T_cells = 1000.
    eff = 1.0
    delta_I = 0.2
    delta_V = 5.4
    burst = 100. # This "Varies" according to Perelson '93 
    alpha = 0.9
    
    init_params = [delta_I, delta_V, burst, alpha]
    fit = scipy.optimize.leastsq(produce_soln, init_params, args=(k, T_cells, I_init, V_init, eff, hiv_ts, 1e4))
    fit = fit[0]

    T = hiv_ts[-1,0]
    dT = T / 1.e4
    ts = np.arange(0., T, dT)

    dtrw_hiv = DTRW_HIV([I_init, V_init], T, dT, k, T_cells, eff, fit[0], fit[1], fit[2], fit[3])
    dtrw_hiv.solve_all_steps()
    
    if plot_linear:
        fig = plt.figure(figsize=(16,8))

        ax2 = fig.add_subplot(1, 2, 1)
        ax2.set_xlim([0,T*1.05])
        ax2.set_ylim([0,V_init*1.1])
        ax2.set_xlabel('Days')
        ax2.set_ylabel('Virus')
        V_line, = ax2.plot(ts, dtrw_hiv.Xs[1,:], lw=2)
        V_data, = ax2.plot(hiv_ts[:,0], hiv_ts[:,1], 'o', markersize=10)
    
        ax4 = fig.add_subplot(1, 2, 2)
    else:
        fig = plt.figure(figsize=(8,8))
        ax4 = fig.add_subplot(1,1,1)

    ax4.set_xlim([0,T*1.05])
    ax4.set_ylim([0,V_init*1.1])
    ax4.set_xlabel('Days')
    ax4.set_ylabel('Virus')
    ax4.grid(True) 
    ax4.set_yscale("log", nonposy='clip')
    
    V_line_log, = ax4.semilogy(ts, dtrw_hiv.Xs[1,:], lw=2)
    V_data_log, = ax4.semilogy(hiv_ts[:,0], hiv_ts[:,1], 'o', markersize=10)

    fig.suptitle('Patient {0}, d_I={1} d_V={2} burst={3} alpha={4}'.format(i, fit[0], fit[1], fit[2], fit[3]))
    pp.savefig()
    #plt.show()

pp.close()
