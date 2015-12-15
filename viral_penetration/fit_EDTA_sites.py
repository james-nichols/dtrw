#! /usr/bin/env python

# Libraries are in parent directory
import sys
sys.path.append('../')

import numpy as np
import scipy
import time, csv, math, collections
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

output_pdf = sys.argv[1]

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
cervix = []
EDTA = []
p24 = np.array([], dtype=np.int32)
virions = np.array([], dtype=np.int32)
penetrators = np.array([], dtype=np.int32)
depth = np.array([], dtype=np.float64)

no_mucous_data = 'SMEG_Data/NeuraminidaseNOBAFLinear.csv'
with_mucous_data = 'SMEG_Data/PenetrationMLoadnewestOMITAngelafixed.csv'
EDTA_data = 'SMEG_Data/EctoCervixEDTABaLAngelafixed.csv'

with open(EDTA_data, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    labels = reader.next()
    for row in reader:
        image_index.append(row[0])
        cervix.append(row[1])
        EDTA.append(row[2])
        p24 = append_string_as_int(p24, row[3]) 
        virions = append_string_as_int(virions, row[4]) 
        penetrators = append_string_as_int(penetrators, row[5]) 
        depth = append_string_as_float(depth, row[6]) 

count = collections.Counter(image_index)

image_yes = collections.Counter([image_index[i] for i in range(len(image_index)) if EDTA[i] == 'Y'])
image_no = collections.Counter([image_index[i] for i in range(len(image_index)) if EDTA[i] == 'N'])

# The number of sites we analyse...
number_of_sites = 10
# Pick out the most common sites
sites = count.most_common(number_of_sites)
sites_yes = image_yes.most_common(number_of_sites)
sites_no = image_no.most_common(number_of_sites)
pdb.set_trace()
pp = PdfPages(output_pdf + '{0}.pdf'.format(sys.argv[2]))
#for site in [sites_yes, sites_no]:
#    for site in sites:
if sys.argv[3] == 'Y':
    site = sites_yes[int(sys.argv[2])]
else:
    site = sites_no[int(sys.argv[2])]
# All locations of this particular image
img_loc = [i for i,x in enumerate(image_index) if x == site[0]]

depth_loc = depth[img_loc]

# Ok, data loaded, now lets get to business. Prune zeros and NaNs from depth 
# (may want to infact double check a 0.0 depth is valid if it's seen as a penetrator)
nz_depth = depth_loc[np.nonzero(depth_loc)]
nz_depth = nz_depth[~np.isnan(nz_depth)]
num_depth_bins = 20

# Depth Histogram
depth_hist, depth_bins = np.histogram(nz_depth, num_depth_bins, density=True)
bin_cent = (depth_bins[1:]+depth_bins[:-1])/2.0

# Depth based survival function - sometimes a better function to fit to, and we certainly don't lose resolution
surv_func = scipy.stats.itemfreq(nz_depth-1.0)
surv_func_x = surv_func[:,0]
surv_func_y = 1.0 - np.insert(np.cumsum(surv_func[:,1]), 0, 0.0)[:-1] / surv_func[:,1].sum() 
if surv_func_x[0] != 0.0:
    surv_func_x = np.insert(surv_func_x, 0, 0.0)
    surv_func_y = np.insert(surv_func_y, 0, 1.0)

T = 4.0
L = surv_func_x.max() #nz_depth.max()
dX = L / 100.0

D_alpha = 20.0
alpha = 0.75
# Last minimisation got close to:
#diff_fit = [ 5.28210775, 0.95847065]
#subdiff_fit = [ 15.07811124, 0.55, 0.99997347]
xs = np.arange(0.0, L+dX, dX)

#
# FIT Diffusion model - analytic
#
diff_init_params = [D_alpha]
diff_fit = scipy.optimize.fmin_slsqp(lsq_diff, diff_init_params, args=(T, surv_func_x, surv_func_y), \
        bounds=[(0.0, np.Inf)], epsilon = 1.0e-8, acc=1.0e-6, full_output=True)
diff_sq_err = diff_fit[1]
diff_fit = diff_fit[0]
print 'Diffusion fit parameters:', diff_fit
diff_analytic_soln_survival = produce_diff_soln_survival(diff_fit, T, xs)
diff_analytic_soln = produce_diff_soln(diff_fit, T, xs)

#
# FIT Subdiffusion model - numerical (DTRW algorithm)
#
#history_truncation = 0
# New regime: start at diff parameter fit
#subdiff_init_params = [diff_fit[0], alpha]
#subdiff_fit = scipy.optimize.fmin_slsqp(lsq_subdiff, subdiff_init_params, args=(T, 4.0 * L, dX, surv_func_x, surv_func_y, history_truncation), \
#                                bounds=[(0.0, 50.0),(0.51, 1.0)], epsilon = 1.0e-3, acc=1.0e-6, full_output=True)
#subdiff_sq_err = subdiff_fit[1]
#subdiff_fit = subdiff_fit[0]
#print 'Subdiffusion fit parameters:', subdiff_fit
#dtrw_sub_soln = produce_subdiff_soln(subdiff_fit, T, 4.0*L, dX)
#dtrw_sub_soln_survival = produce_subdiff_soln_survival(subdiff_fit, T, 4.0*L, dX)

#
# FIT Subdiffusion model - analytic
#
subdiff_anal_init_params = [D_alpha]
subdiff_anal_fit = scipy.optimize.fmin_slsqp(lsq_subdiff_analytic, subdiff_anal_init_params, args=(T, surv_func_x, surv_func_y), \
                                bounds=[(0.0, np.Inf)], epsilon = 1.0e-3, acc=1.0e-6, full_output=True)
subdiff_anal_sq_err = subdiff_anal_fit[1]
subdiff_anal_fit = subdiff_anal_fit[0]
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
#line1, = ax1.plot(xs, dtrw_sub_soln_survival.T[:xs.size], 'r.-')
line2, = ax1.plot(xs, anal_sub_soln_survival, 'y.-')
line3, = ax1.plot(xs, diff_analytic_soln_survival, 'g.-')
line4, = ax1.plot(xs, exp_fit, 'b')
ax1.set_title('Survival function vs fits, ' + site[0] + ', {0} virions'.format(site[1]))

ax2 = fig.add_subplot(1, 2, 2)
ax2.semilogy(surv_func_x, surv_func_y, 'b.-')
#ax2.semilogy(xs, dtrw_sub_soln_survival.T[:xs.size], 'r.-')
ax2.semilogy(xs, anal_sub_soln_survival, 'y.-')
ax2.semilogy(xs, diff_analytic_soln_survival, 'g.-')
ax2.semilogy(xs, exp_fit, 'b')
ax2.set_title('Logarithm of survival function vs fits, ' + site[0] + ', {0} virions'.format(site[1]))

#plt.legend([bar1, line1, line2, line3, line4], ["Viral survival func", "Subdiffusion fit, alpha={0:.2f}, D_alpha={1:.2f}, sq_err={2:.4f}".format(subdiff_fit[1],subdiff_fit[0],subdiff_sq_err), \
plt.legend([bar1, line2, line3, line4], ["Viral survival func", \
                                                                     "Analytic subdiff fit, alpha=1/2, D_alpha={0:.2f}, sq_err={1:.4f}".format(subdiff_anal_fit[0], subdiff_anal_sq_err), \
                                                                     "Diffusion fit, D_alpha={0:.2f}, sq_err={1:.2f}".format(diff_fit[0], diff_sq_err), "Exponential fit"], loc=3)
pp.savefig()

pp.close()

#plt.show()

