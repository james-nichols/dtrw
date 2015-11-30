#! /usr/bin/env python

# Libraries are in parent directory
import sys
sys.path.append('../')

import numpy as np
import scipy
import time, csv, math
from dtrw import *
# Local fit functions for a variety of scripts
from fit_functions import *

import mpmath
import scipy.integrate

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages

import pdb
pp = PdfPages(sys.argv[1])

def append_string_as_int(array, item):
    try:
        array = np.append(array, np.int32(item))
    except ValueError:
        array = np.append(array, np.nan)
    return array

def append_string_as_float(array, item):
    try:
        array = np.append(array, np.float64(item))
    except ValueError:
        array = np.append(array, np.nan)
    return array

labels = []

image_index = []
p24 = np.array([], dtype=np.int32)
virions = np.array([], dtype=np.int32)
penetrators = np.array([], dtype=np.int32)
depth = np.array([], dtype=np.float64)

with open('SMEG_Data/PenetrationMLoadnewestOMITAngelafixed.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    labels = reader.next()
    for row in reader:
        image_index.append(row[0])
        p24 = append_string_as_int(p24, row[4]) 
        virions = append_string_as_int(virions, row[5]) 
        penetrators = append_string_as_int(penetrators, row[6]) 
        depth = append_string_as_float(depth, row[7]) 

# Ok, data loaded, now lets get to business. Prune zeros and NaNs from depth 
# (may want to infact double check a 0.0 depth is valid if it's seen as a penetrator)
nz_depth = depth[np.nonzero(depth)]
nz_depth = nz_depth[~np.isnan(nz_depth)]
num_depth_bins = 20

# Depth Histogram
depth_hist, depth_bins = np.histogram(nz_depth, num_depth_bins, density=True)
bin_cent = (depth_bins[1:]+depth_bins[:-1])/2.0

# Depth based survival function - sometimes a better function to fit to, and we certainly don't lose resolution
surv_func = scipy.stats.itemfreq(nz_depth)
surv_func_x = surv_func[:,0]
surv_func_y = 1.0 - np.cumsum(surv_func[:,1]) / surv_func[:,1].sum()   #np.array([np.cumsum(surv_func[i:,1]) for i in range(surv_func[:,1].size)])
#surv_func_x = np.insert(surv_func_x[:-1], 0, 0.0)
#surv_func_y = np.insert(surv_func_y[:-1], 0, 1.0)

surv_func = scipy.stats.itemfreq(nz_depth-1.0) 
surv_func_x2 = surv_func[:,0]
surv_func_y2 = 1.0 - np.cumsum(surv_func[:,1]) / surv_func[:,1].sum()   #np.array([np.cumsum(surv_func[i:,1]) for i in range(surv_func[:,1].size)])

plt.plot(surv_func_x, surv_func_y)
plt.plot(surv_func_x2, surv_func_y2)

T = 10.0
L = nz_depth.max()
dX = L / 100.0

D_alpha = 17.0
alpha = 0.7
# Last minimisation got close to:
#diff_fit = [ 5.28210775, 0.95847065]
#subdiff_fit = [ 15.07811124, 0.55, 0.99997347]
xs = np.arange(0.0, L+dX, dX)

#
# FIT Diffusion model - analytic
#
diff_init_params = [D_alpha]
diff_fit = scipy.optimize.fmin_slsqp(lsq_diff, diff_init_params, args=(T, surv_func_x, surv_func_y), \
                                bounds=[(0.0, np.Inf)], epsilon = 1.0e-8, acc=1.0e-9)
print 'Diffusion fit parameters:', diff_fit
diff_analytic_soln_survival = produce_diff_soln_survival(diff_fit, T, xs) 
diff_analytic_soln = produce_diff_soln(diff_fit, T, xs) 

#
# FIT Subdiffusion model - numerical (DTRW algorithm)
#
history_truncation = 1000
subdiff_init_params = [D_alpha, alpha]
subdiff_fit = scipy.optimize.fmin_slsqp(lsq_subdiff, subdiff_init_params, args=(T, 4.0 * L, dX, surv_func_x, surv_func_y, history_truncation), \
                                bounds=[(0.0, 50.0),(0.48, 1.0)], epsilon = 1.0e-8, acc=1.0e-6)
subdiff_fit = [16.99999094, 0.69999606]
print 'Subdiffusion fit parameters:', subdiff_fit

dtrw_sub_soln = produce_subdiff_soln(subdiff_fit, T, 4.0*L, dX)
dtrw_sub_soln_survival = produce_subdiff_soln_survival(subdiff_fit, T, 4.0*L, dX)

#
# FIT Subdiffusion model - analytic
#
subdiff_anal_init_params = [D_alpha]
subdiff_anal_fit = scipy.optimize.fmin_slsqp(lsq_subdiff_analytic, subdiff_anal_init_params, args=(T, surv_func_x, surv_func_y), \
                                bounds=[(0.0, np.Inf)], epsilon = 1.0e-6, acc=1.0e-6)
print 'Subdiffusion analytic fit parameters:', subdiff_anal_fit
anal_sub_soln = produce_subdiff_analytic_soln(subdiff_anal_fit, T, xs)
anal_sub_soln_survival = produce_subdiff_analytic_survival(subdiff_anal_fit, T, xs)

#
# FIT Exponential... for fun
#
slope, offset = np.linalg.lstsq(np.vstack([surv_func_x, np.ones(len(surv_func_x))]).T, np.log(surv_func_y).T)[0]
exp_fit = np.exp(offset + xs * slope)

#
# PLOT IT ALL
#
fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(1, 2, 1)
bar1, = ax1.plot(surv_func_x, surv_func_y, 'b.-')
line1, = ax1.plot(xs, dtrw_sub_soln_survival.T[:xs.size], 'r.-')
line2, = ax1.plot(xs, anal_sub_soln_survival, 'y.-')
line3, = ax1.plot(xs, diff_analytic_soln_survival, 'g.-')
line4, = ax1.plot(xs, exp_fit, 'b')
ax1.set_title('Survival function vs fits')

ax2 = fig.add_subplot(1, 2, 2)
ax2.semilogy(surv_func_x, surv_func_y, 'b.-')
ax2.semilogy(xs, dtrw_sub_soln_survival.T[:xs.size], 'r.-')
ax2.semilogy(xs, anal_sub_soln_survival, 'y.-')
ax2.semilogy(xs, diff_analytic_soln_survival, 'g.-')
ax2.semilogy(xs, exp_fit, 'b')
ax2.set_title('Logarithm of survival function vs fits')

plt.legend([bar1, line1, line2, line3, line4], ["Viral survival func", "Subdiffusion fit, alpha={0}, D_alpha={1}".format(subdiff_fit[1],subdiff_fit[0]), "Analytic subdiff fit, alpha=1/2, D_alpha={0}".format(subdiff_anal_fit[0]), "Diffusion fit, D_alpha={0}".format(diff_fit[0]), "Exponential fit"], loc=3)

pp.savefig()
pp.close()

#plt.show()

