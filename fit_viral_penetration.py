#! /usr/bin/env python

import numpy as np
import scipy.interpolate
import time, csv
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

    #alpha = min(1.0, alpha)
    #alpha = max(0.0, alpha)
    #D_alpha = max(0.0, D_alpha)
    #LHS = max(0.0, LHS)
    
    print D_alpha, alpha, LHS

    dT = pow((dX * dX / (2.0 * D_alpha)), 1./alpha)
    N = int(math.floor(T / dT))

    # Calculate r for diffusive case so as to get the *same* dT as the subdiffusive case
    r = dT / (dX * dX / (2.0 * D_alpha))

    X_init = np.zeros(np.floor(L / dX))

    dtrw_sub = DTRW_subdiffusive(X_init, N, alpha, history_length=N, boundary_condition=BC_Dirichelet([LHS, 0.0]))
    dtrw_sub.solve_all_steps()
    
    interp = scipy.interpolate.interp1d(np.arange(0., L, dX), dtrw_sub.Xs[0][:,:,-1])
    
    return ((interp(x_fit).flatten() - y_fit) * (interp(x_fit).flatten() - y_fit)).sum()


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

T = 1.0
L = nz_depth.max()
dX = L / 400.0

D_alpha = 2.0
alpha = 0.8
init_params = [D_alpha, alpha, depth_hist[0]]

#fit = scipy.optimize.leastsq(produce_soln, init_params, args=(T, L, dX, (depth_bins[1:]+depth_bins[:-1])/2.0, depth_hist), options={'disp': True})
fit = scipy.optimize.fmin_slsqp(produce_soln, init_params, args=(T, L, dX, (depth_bins[1:]+depth_bins[:-1])/2.0, depth_hist), bounds=[(0.0, np.Inf),(0.0, 1.0), (0.0, np.Inf)])

pdb.set_trace()

D_alpha, alpha, LHS = fit

dT = pow((dX * dX / (2.0 * D_alpha)), 1./alpha)
N = int(math.floor(T / dT))
# Calculate r for diffusive case so as to get the *same* dT as the subdiffusive case
r = dT / (dX * dX / (2.0 * D_alpha))
X_init = np.zeros(np.floor(L / dX))
xs = np.arange(0.0, L, dX)

dtrw_sub = DTRW_subdiffusive(X_init, N, alpha, history_length=N, boundary_condition=BC_Dirichelet([LHS, 0.0]))
dtrw_sub.solve_all_steps()

alpha = 0.75
D_alpha = 4.0
dT = pow((dX * dX / (2.0 * D_alpha)), 1./alpha)
N = int(math.floor(T / dT))
dtrw_test = DTRW_subdiffusive(X_init, N, alpha, history_length=N, boundary_condition=BC_Dirichelet([LHS, 0.0]))
dtrw_test.solve_all_steps()
pdb.set_trace()
plt.bar((depth_bins[1:]+depth_bins[:-1])/2.0, depth_hist)
plt.plot(xs, dtrw_sub.Xs[0][:,:,-1].T, xs, dtrw_test.Xs[0][:,:,-1].T)
plt.plot(xs, dtrw_test.Xs[0][:,:,-1].T)
plt.show()

n_points = 100
T = 2.0

alpha = 0.75
D_alpha = 0.1

dT = pow((dX * dX / (2.0 * D_alpha)), 1./alpha)
N = int(math.floor(T / dT))

L = 5.0
dX = L / n_points

X_init = np.zeros(n_points)
X_init[n_points / 2] = 1.0

T = 2.0

alpha = 0.75
D_alpha = 0.1

dT = pow((dX * dX / (2.0 * D_alpha)), 1./alpha)
N = int(math.floor(T / dT))

# Calculate r for diffusive case so as to get the *same* dT as the subdiffusive case
r = dT / (dX * dX / (2.0 * D_alpha))

print "Diffusive sim with dT =", dT, "N =", N, "alpha =", alpha, "diffusion matching r =", r

dtrw = DTRW_diffusive(X_init, N, r)
dtrw_sub = DTRW_subdiffusive(X_init, N, alpha, history_length=N)

print "Left jump probs: ", dtrw.lam[:,:,0]
print "Right jump probs: ", dtrw.lam[:,:,1]

dtrw.solve_all_steps()
dtrw_sub.solve_all_steps()

print "Solutions computed, now creating animation..."

xs = np.linspace(0., L, n_points, endpoint=False)

fig = plt.figure(figsize=(8,8))
plt.xlim(L/4,3*L/4)
plt.ylim(0,0.1)
plt.xlabel('x')
line1, = plt.plot([],[],'r-')
line2, = plt.plot([],[],'g-')
line3, = plt.plot([],[],'bx')
line4, = plt.plot([],[],'kx')
plt.legend([line1, line2, line3, line4], ["Analytic diffusion", "Analytic subdiffusion, alpha=3/4", "DTRW diffusion", "DTRW Sub-diffusion, alpha=3/4"])

def update(i, line1, line2, line3, line4):
    line1.set_data(xs,dtrw.Xs[0][:,:,i])
    line2.set_data(xs,dtrw_sub.Xs[0][:,:,i])
    if i == 0:
        analytic_soln = X_init
    else:
        analytic_soln = (1./math.sqrt(4. * math.pi * float(i) * dT * D_alpha)) * np.exp( - (xs - 2.5) * (xs - 2.5) / (4. * D_alpha * float(i) * dT)) * dX
    line3.set_data(xs, analytic_soln)
    line4.set_data(xs,dtrw_sub.Xs[0][:,:,i])
    return line1, line2, line3, line4

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, update, 
        frames=N, fargs=(line1, line2, line3, line4), interval=10)

import inspect, os, subprocess
exec_name =  os.path.splitext(os.path.basename(inspect.getfile(inspect.currentframe())))[0]
git_tag = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).replace('\n', '')

file_name = '{0}_{1}.mp4'.format(exec_name, git_tag)
print "Saving animation to", file_name

anim.save(file_name, fps=24)
plt.show()

