#!/usr/local/bin/python3

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

import pdb

from dtrw import *
 
pp = PdfPages('PatientFitPlots.pdf')

class DTRW_HIV(DTRW_compartment):

    def __init__(self, X_inits, T, dT, \
                 k, T_cells, eff, delta_I, delta_V, burst, alpha):

        if len(X_inits) != 2:
            # Error!
            print("Need two initial points")
            raise SystemExit 

        super(DTRW_HIV, self).__init__(X_inits, T, dT)
        
        self.T_cells = T_cells
        self.k = k
        self.eff = eff

        self.delta_I = delta_I
        self.delta_V = delta_V
        self.burst = burst 
        self.alpha = alpha

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

def produce_soln(params, k, T_cells, I_init, V_init, eff, cells, virus, num_ts):
    T = max(cells[-1,0], virus[-1,0]) #time_series[-1,0]
    dT = T / float(num_ts)
    
    t = np.arange(0., T, dT)

    delta_I, delta_V, burst, alpha = params
    
    dtrw_anom = DTRW_HIV([I_init, V_init], T, dT, k, T_cells, eff, delta_I, delta_V, burst, alpha)
    dtrw_anom.solve_all_steps()
    
    print("Soln with params: ", delta_I, delta_V, burst, alpha)
    print(cells[:,0])
    print(cells[:,1])
    print(np.interp(cells[:,0], t, dtrw_anom.Xs[0,:]))
    print(rna[:,0])
    print(rna[:,1])
    print(np.interp(rna[:,0], t, dtrw_anom.Xs[1,:]))

    return np.append(np.log(cells[:,1]) - np.log(np.interp(cells[:,0], t, dtrw_anom.Xs[0,:])), \
                     np.log(rna[:,1]) - np.log(np.interp(rna[:,0], t, dtrw_anom.Xs[1,:])))


data_dirs = ['Perelson1997Nature/Cells', 'Perelson1997Nature/DNA', 'Perelson1997Nature/RNA']
data = []
data_label = []
for data_dir in data_dirs:
    data_files = [ f for f in listdir(data_dir) ]
    data_files.sort()
    
    data_sub = []
    data_label_sub = []
    for data_file in data_files:
        print(data_file)
        pat_num = int(list(filter(str.isdigit, data_file)))
        data_sub.append(np.loadtxt(data_dir + '/' + data_file))
        data_label_sub.append(pat_num)

    data.append(data_sub)
    data_label.append(data_label_sub)

common_patients = np.intersect1d(np.intersect1d(data_label[0], data_label[1]), data_label[2])

for pat in common_patients:
    cells = data[0][data_label[0].index(pat)]
    dna = data[1][data_label[1].index(pat)]
    rna = data[2][data_label[2].index(pat)]

    I_init = cells[0,1]
    V_init = rna[0,1]

    k = 2.4e-5
    T_cells = 1000.
    eff = 1.0
    delta_I = 0.2
    delta_V = 5.4
    burst = 100. # This "Varies" according to Perelson '93 
    alpha = 0.9
    
    init_params = [delta_I, delta_V, burst, alpha]
    fit = scipy.optimize.leastsq(produce_soln, init_params, args=(k, T_cells, I_init, V_init, eff, cells, rna, 1e4))
    fit = fit[0]

    T = rna[-1,0]
    dT = T / 1.e4
    ts = np.arange(0., T, dT)

    dtrw_hiv = DTRW_HIV([I_init, V_init], T, dT, k, T_cells, eff, fit[0], fit[1], fit[2], fit[3])
    dtrw_hiv.solve_all_steps()

    fig = plt.figure(figsize=(16,16))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_xlim([0,T*1.05])
    ax1.set_ylim([0,I_init*1.1])
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Infected T-cells')
    I_line, = ax1.plot(ts, dtrw_hiv.Xs[0,:], lw=2)
    I_data, = ax1.plot(cells[:,0], cells[:,1], 'o', markersize=10)
    
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_xlim([0,T*1.05])
    ax2.set_ylim([0,V_init*1.1])
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Virus')
    V_line, = ax2.plot(ts, dtrw_hiv.Xs[1,:], lw=2)
    V_data, = ax2.plot(rna[:,0], rna[:,1], 'o', markersize=10)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_xlim([0,T*1.05])
    ax3.set_ylim([0,I_init*1.1])
    ax3.set_xlabel('Days')
    ax3.set_ylabel('Infected T-cells')
    I_line_log, = ax3.semilogy(ts, dtrw_hiv.Xs[0,:], lw=2)
    I_data_log, = ax3.semilogy(cells[:,0], cells[:,1], 'o', markersize=10)
    
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_xlim([0,T*1.05])
    ax4.set_ylim([0,V_init*1.1])
    ax4.set_xlabel('Days')
    ax4.set_ylabel('Virus')
    V_line_log, = ax4.semilogy(ts, dtrw_hiv.Xs[1,:], lw=2)
    V_data_log, = ax4.semilogy(rna[:,0], rna[:,1], 'o', markersize=10)

    fig.suptitle('Patient {0}, d_I={1} d_V={2} burst={3} alpha={4}'.format(pat, fit[0], fit[1], fit[2], fit[3]))
    pp.savefig()
    #plt.show()

pp.close()
