
#! /usr/bin/env python

# Libraries are in parent directory
import sys
sys.path.append('../')

import numpy as np
import scipy
import time, csv, math
from dtrw import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages

import pdb
pp = PdfPages('Viral_Penetration_Plots.pdf')

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

def produce_subdiff_soln(params, T, L, dX):
    
    D_alpha, alpha = params
    LHS = 1.0

    dT_pre = pow((dX * dX / (2.0 * D_alpha)), 1./alpha)
    N = int(math.ceil(T / dT_pre))
    
    # Now find r to get an *exact* D_alpha
    dT = T / N

    # Calculate r for diffusive case so as to get the *same* dT as the subdiffusive case
    r = pow(dT, alpha) / (dX * dX / (2.0 * D_alpha))
    xs = np.arange(0.0, L+dX, dX)
    X_init = np.zeros(xs.shape)
    
    print 'N:', N, 'r:', r, 
    dtrw_sub = DTRW_subdiffusive(X_init, N, alpha, r = r, history_length=N, boundary_condition=BC_Dirichelet_LHS([LHS]))
    dtrw_sub.solve_all_steps()
    
    return dtrw_sub.Xs[0][:,:,-1]

def produce_subdiff_soln_survival(params, T, L, dX):
    
    D_alpha, alpha = params
    LHS = 1.0

    soln = produce_subdiff_soln(params, T, L, dX)
    return 1.0 - np.array([np.trapz(soln.flatten()[:i], dx=dX) for i in range(1,soln.size+1)]) / np.trapz(soln.flatten(), dx=dX)

def produce_diff_soln(params, T, L, dX, xs):
    
    D_alpha = params
    D_alpha = max(D_alpha, 0.0)
    LHS = 1.0

    return LHS * (1.0 - scipy.special.erf(xs / math.sqrt(4. * D_alpha * T)))

def produce_diff_soln_survival(params, T, L, dX, xs):
    
    D_alpha = params
    D_alpha = max(D_alpha, 0.0)
    LHS = 1.0
    
    integral = LHS * (xs - xs * scipy.special.erf(xs / math.sqrt(4. * D_alpha * T)) - math.sqrt(4. * D_alpha * T / math.pi) * (np.exp(-xs * xs / (4. * D_alpha * T)) - 1.))
    full_integral = LHS * math.sqrt(4. * D_alpha * T / math.pi)
    
    return 1.0 - integral / full_integral 

def lsq_subdiff(params, T, L, dX, x_fit, y_fit):

    print "Subdiff params: ", params[0], params[1]
    #soln = produce_subdiff_soln(params, T, L, dX)
    # If we're fitting to a survival function, we must fit the partial integral of the solution...
    soln = produce_subdiff_soln_survival(params, T, L, dX)

    interp = scipy.interpolate.interp1d(np.arange(0., L+dX, dX), soln)
    
    fit = (interp(x_fit).flatten())
    goal = (y_fit)
    sq_err = ((fit - goal) * (fit - goal)).sum()
    print "Square err:", sq_err
    return sq_err

def lsq_diff(params, T, L, dX, x_fit, y_fit):
    fit = (produce_diff_soln_survival(params, T, L, dX, x_fit))
    goal = (y_fit)
    sq_err = ((fit - goal) * (fit - goal)).sum()

    return sq_err

