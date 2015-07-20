#! /usr/bin/env python

import math
import numpy as np

""" 
This script can recreate the results from Example 1 of the paper 
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

l = 5.0        # Length of the spatial domain
T = 2.0         # End time of simulation

alpha = 0.9     # Anomalous exponent
D_alpha = 1.    # Diffusion coefficient
r = 1.0         
g = 1.
k = 10.0         # Removal rate parameter

dX = 0.5       # Delta x, the spatial grid size

# Calculate Delta t as in Eq. (58) of the paper
dT = math.pow(r * dX * dX / (2.0 * D_alpha), 1.0 / alpha)

# Number of spatial...
L = int(math.floor(l / dX))
# ...and time points
N = int(math.floor(T / dT))

# x coordinates vector
xs = np.linspace(0., l, L, endpoint=False)

####################
# ANALYTIC SOLUTION
####################

# Analytic solution parameters
mu = 2. - alpha
g_star = g / (math.sqrt(D_alpha * pow(k, 2.-alpha)))
pst_0 = pow(g_star * math.sqrt((mu+2.)/(2.*mu)), 2. / (mu+2.))
x_0 = (2. * mu / (2. - mu)) * pow(g_star, -(2.-mu)/(mu+2.)) * pow((mu+2.)/(2.*mu), mu/(mu+2.)) * math.sqrt(D_alpha/pow(k,alpha))

analytic_soln = pst_0 * pow((1 + xs / x_0), -2./alpha)

##################
# DTRW SIMULATION
##################

# U is the solution
U = np.zeros([L, N])

# We set the initial condition
#U[0, 0] = g 
U[:,0] = analytic_soln

# Theta is the survival probability between (i,m) and the current time step
Theta = np.ones([L, N])
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
        A = 1.0 - math.exp(-k * dT * U[i,n-1])
        
        # Update the survival probabilities
        Theta[i, :n] *= (1.0 - A)
            
        flux = 0.0
        for m in range(0, n):
            flux += U[i, m] * Theta[i, m] * K[n - m]
        
        U[i, n] += U[i, n-1] - r * flux - A * U[i, n-1]
        
        if i < L-1:
            U[i+1, n] += p_r * r * flux
        
        if i > 0:
            U[i-1, n] += p_l * r * flux
        else:
            # This is applying the zero-flux boundary at the LHS, by
            # re-directing flux back to the point
            U[i, n] += p_l * r * flux
        
        # All particles that are removed are created again at the origin
        U[0, n] += A * U[i, n-1]
        
        if i==L-1:
            # All particles jumping out of the domain are re-directed to the origin
            U[0,n] += p_r * r * flux

##############
# SAVE OUTPUT
##############

dtrw_file_name = "DTRW_soln_{0:f}_{1:f}_{2:f}_{3:f}.csv".format(alpha, T, k, dX)
np.savetxt(dtrw_file_name, U[:,-1], delimiter=',')

analytic_file_name = "Analytic_soln_{0:f}_{1:f}_{2:f}_{3:f}.csv".format(alpha, T, k, dX)
np.savetxt(analytic_file_name, analytic_soln, delimiter=',')

