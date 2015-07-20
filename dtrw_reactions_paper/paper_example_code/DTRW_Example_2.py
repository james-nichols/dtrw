#! /usr/bin/env python

import math
import numpy as np

""" 
This script can recreate the results from Example 2 of the paper 
"From stochastic processes to numerical methods: A new scheme for solving 
reaction subdiffusion fractional partial differential equations".
It is not advised to use this code for the calculations with small dX, as
this code is much more inneficient than production code used in producing
the graphs in the paper. For more efficient code and latest improvements, see 

github.com/james-nichols/dtrw/

James Nichols, 2015
"""

#############
# PARAMETERS
#############

T = 0.1         # End time of simulation

alpha = 0.5     # Anomalous exponent
D_alpha = 1.    # Diffusion coefficient
r = 1.0         
k = 1.0         # Removal rate parameter

dX = 0.2       # Delta x, the spatial grid size

# Calculate Delta t as in Eq. (58) of the paper
dT = math.pow(r * dX * dX / (2.0 * D_alpha), 1.0 / alpha)

# x coordinates vector
xs = np.arange(-1., 1.+dX, dX)

# Number of spatial...
L = len(xs)
# ...and time points
N = int(round(T / dT))+1

##################
# DTRW SIMULATION
##################

# U_a and U_b are the solutions
U_a = np.zeros([L, N])
U_b = np.zeros([L, N])

# We set the initial condition as a delta function at the origin
U_a[L/2, 0] = float(L)/2.0 

# The memory kernel
K = np.zeros(N+1)

# Calculate the Sibuya memory kernel
K[0] = 0.0
K[1] = alpha 
K[2] = alpha * 0.5 * (alpha - 1.0)
for i in range(3,N+1):
    K[i] = (float(i) + alpha - 2.0) * K[i-1] / float(i)

# Right and left jump probabilities
p_r = 0.5
p_l = 1 - p_r

print "Calculating U for", N, "time steps, on", L, "spatial points..."

# The main loop! 
for n in range(1, N):
        
    for i in range(0, L):

        # The annihilation probability
        A = 1.0 - math.exp(-k * dT)
        
        flux_a = 0.0
        flux_b = 0.0
        for m in range(0, n):
            flux_a += U_a[i, m] * math.exp(-(n - m) * k * dT) * K[n - m]
            flux_b += U_b[i, m] * math.exp(-(n - m) * k * dT) * K[n - m]
    
        U_a[i, n] += U_a[i, n-1] - r * flux_a - A * U_a[i, n-1] + A * U_b[i, n-1]
        U_b[i, n] += U_b[i, n-1] - r * flux_b - A * U_b[i, n-1] + A * U_a[i, n-1]
        
        if i < L-1:
            U_a[i+1, n] += p_r * r * flux_a
            U_b[i+1, n] += p_r * r * flux_b
        else:
            # These steps apply the zero-flux boundary at the RHS, by
            # re-directing escaping flux back to that point
            U_a[i, n] += p_r * r * flux_a
            U_b[i, n] += p_r * r * flux_b
        
        if i > 0:
            U_a[i-1, n] += p_l * r * flux_a
            U_b[i-1, n] += p_l * r * flux_b
        else:
            # This is applying the zero-flux boundary at the LHS, by
            # re-directing flux back to the point
            U_a[i, n] += p_l * r * flux_a
            U_b[i, n] += p_l * r * flux_b
       
##############
# SAVE OUTPUT
##############

dtrw_file_name = "DTRW_ex2_Ua_soln_{0:f}_{1:f}_{2:f}_{3:f}.csv".format(alpha, T, k, dX)
np.savetxt(dtrw_file_name, U_a[:,-1], delimiter=',')
dtrw_file_name = "DTRW_ex2_Ub_soln_{0:f}_{1:f}_{2:f}_{3:f}.csv".format(alpha, T, k, dX)
np.savetxt(dtrw_file_name, U_b[:,-1], delimiter=',')

