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

alphas = np.linspace(0.1, 1.0, 10, endpoint=True)
N = 100
alpha = 0.8
X_init = np.zeros(10)

Vs = np.linspace(0.1, 1.0, 10, endpoint=True)
for V in Vs:
    dtrw = DTRW_subdiffusive(X_init, N, alpha, history_length=N)

    xi = np.zeros(N)
    corrections = np.zeros(N)
    for i in range(0,N):
        corrections[i] = sum([ dtrw.K[j+1] / xi[i-j:i].prod() for j in range(i+1)]) 
        xi[i] = 1. - V * ( sum([ dtrw.K[j+1] / xi[i-j:i].prod() for j in range(i+1)]) )

    print "At V=", V, "xi is", xi
    print "At V=", V, "corr is", corrections
    plt.plot(xi)
plt.show()
