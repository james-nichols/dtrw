#! /usr/bin/env python

import numpy as np
import time
from dtrw import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
import pdb

X_init = np.zeros((5,5))
X_init[2,2] = 1.0

N = 10
history_length = N
dT = 0.5
alpha = 0.5
omega = 0.0 #0.05

beta = 1.0
potentials = np.ones((5,5))

dtrw = DTRW_diffusive(X_init, N, omega, potential=potentials)
dtrw_sub = DTRW_subdiffusive(X_init, N, alpha, r=omega, potential=potentials)
dtrw.solve_all_steps()
dtrw_sub.solve_all_steps()


