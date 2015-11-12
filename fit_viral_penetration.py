#! /usr/bin/env python

import numpy as np
import scipy.interpolate, scipy.special
import time, csv, math
from dtrw import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm

import pdb

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

def produce_soln(params, T, L, dX, x_fit, y_fit):
    
    D_alpha, alpha, LHS = params

    dT_pre = pow((dX * dX / (2.0 * D_alpha)), 1./alpha)
    N = int(math.ceil(T / dT_pre))
    
    # Now find r to get an *exact* D_alpha
    dT = T / N

    # Calculate r for diffusive case so as to get the *same* dT as the subdiffusive case
    r = dT / (dX * dX / (2.0 * D_alpha))
    X_init = np.zeros(np.floor(L / dX))
    
    print "Params: ", D_alpha, alpha, LHS, " N =", N,
    
    dtrw_sub = DTRW_subdiffusive(X_init, N, alpha, r = r, history_length=N, boundary_condition=BC_Dirichelet([LHS, 0.0]))
    dtrw_sub.solve_all_steps()
    
    interp = scipy.interpolate.interp1d(np.arange(0., L, dX), dtrw_sub.Xs[0][:,:,-1])
    print "Square err:", ((interp(x_fit).flatten() - y_fit) * (interp(x_fit).flatten() - y_fit)).sum()
    
    fit = (interp(x_fit).flatten())
    goal = (y_fit)

    #return ((interp(x_fit).flatten() - y_fit) * (interp(x_fit).flatten() - y_fit)).sum()
    return ((fit - goal) * (fit - goal)).sum()

def produce_diff_soln(params, T, L, dX, x_fit, y_fit):
    
    D_alpha, LHS = params

    dT_pre = dX * dX / (2.0 * D_alpha)
    N = int(math.ceil(T / dT_pre))

    dT = T / N

    # Calculate r for diffusive case so as to get the *same* dT as the subdiffusive case
    r = dT / (dX * dX / (2.0 * D_alpha))

    X_init = np.zeros(np.floor(L / dX))
    
    print "Params: ", D_alpha, LHS, " N =", N,
    
    dtrw = DTRW_diffusive(X_init, N, r, history_length=N, boundary_condition=BC_Dirichelet([LHS, 0.0]))
    dtrw.solve_all_steps()
    
    interp = scipy.interpolate.interp1d(np.arange(0., L, dX), dtrw.Xs[0][:,:,-1])
    print "Square err:", ((interp(x_fit).flatten() - y_fit) * (interp(x_fit).flatten() - y_fit)).sum()
    return ((interp(x_fit).flatten() - y_fit) * (interp(x_fit).flatten() - y_fit)).sum()

    fit = np.log(interp(x_fit).flatten())
    goal = np.log(y_fit)

    #return ((interp(x_fit).flatten() - y_fit) * (interp(x_fit).flatten() - y_fit)).sum()
    return ((fit - goal) * (fit - goal)).sum()



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

#depth_hist, depth_bins = np.histogram(nz_depth, bins=np.linspace(nz_depth.min(), nz_depth.max(), num_depth_bins, endpoint=True))
depth_hist, depth_bins = np.histogram(nz_depth, num_depth_bins, density=True)
bin_cent = (depth_bins[1:]+depth_bins[:-1])/2.0

T = 10.0
L = nz_depth.max()
dX = L / 50.0

D_alpha = 1.0
alpha = 0.7
init_params = [D_alpha, alpha, depth_hist[0]]

#fit = scipy.optimize.leastsq(produce_soln, init_params, args=(T, L, dX, (depth_bins[1:]+depth_bins[:-1])/2.0, depth_hist), options={'disp': True})
fit = scipy.optimize.fmin_slsqp(produce_soln, init_params, args=(T, L, dX, bin_cent, depth_hist), \
                                bounds=[(0.0, 100.0),(0.55, 1.0), (0.0, 10.0)], epsilon = 1.0e-5, acc=1.0e-9)

diff_init_params = [2.0*D_alpha, depth_hist[0]]
diff_fit = scipy.optimize.fmin_slsqp(produce_diff_soln, diff_init_params, args=(T, L, dX, bin_cent, depth_hist), \
                                bounds=[(0.0, 100.0), (0.0, 10.0)], epsilon = 1.0e-5, acc=1.0e-9)


D_alpha, alpha, LHS = fit

dT = pow((dX * dX / (2.0 * D_alpha)), 1./alpha)
N = int(math.floor(T / dT))
# Calculate r for diffusive case so as to get the *same* dT as the subdiffusive case
r = dT / (dX * dX / (2.0 * D_alpha))
X_init = np.zeros(np.floor(L / dX))
xs = np.arange(0.0, L, dX)

dtrw_sub = DTRW_subdiffusive(X_init, N, alpha, history_length=N, boundary_condition=BC_Dirichelet([LHS, 0.0]))
dtrw_sub.solve_all_steps()

slope, offset = np.linalg.lstsq(np.vstack([bin_cent, np.ones(len(bin_cent))]).T, np.log(depth_hist).T)[0]
exp_fit = np.exp(offset + xs * slope)
 
D_alpha, LHS = diff_fit

dT = dX * dX / (2.0 * D_alpha)
N = int(math.floor(T / dT))

# Calculate r for diffusive case so as to get the *same* dT as the subdiffusive case
r = dT / (dX * dX / (2.0 * D_alpha))
X_init = np.zeros(np.floor(L / dX))
    
dtrw = DTRW_diffusive(X_init, N, r, history_length=N, boundary_condition=BC_Dirichelet([LHS, 0.0]))
dtrw.solve_all_steps()

diff_analytic_soln = LHS * (1.0 - scipy.special.erf(xs / math.sqrt(4. * D_alpha * T)))
    
fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(1, 2, 1)
bar1 = ax1.bar(bin_cent, depth_hist)
line1, = ax1.plot(xs, dtrw_sub.Xs[0][:,:,-1].T, 'r.-')
line2, = ax1.plot(xs, dtrw.Xs[0][:,:,-1].T, 'g.-')
line4, = ax1.plot(xs, diff_analytic_soln, 'gx')
line3, = ax1.plot(xs, exp_fit, 'b')

ax2 = fig.add_subplot(1, 2, 2)
ax2.semilogy(bin_cent, depth_hist, 'o')
ax2.semilogy(xs, dtrw_sub.Xs[0][:,:,-1].T, 'r.-')
ax2.semilogy(xs, dtrw.Xs[0][:,:,-1].T, 'g.-')
ax2.semilogy(xs, diff_analytic_soln, 'gx')
ax2.semilogy(xs, exp_fit, 'b')

plt.legend([bar1, line1, line2, line3], ["Viral Penetration Hist", "Subdiffusion fit", "Diffusion fit", "Exponential fit"])
plt.show()

