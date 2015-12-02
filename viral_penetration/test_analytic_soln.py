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
pdf_name = "dtrw_vs_analytic.pdf"
pp = PdfPages(pdf_name)

T = 1.0
L = 1.0
dX = 0.1

D_alpha = 1.0
alpha = 0.5

xs = np.arange(0.0, L+dX, dX)

dX_hires = dX
xs_hires = np.arange(0.0, L+dX_hires, dX_hires)

dtrw_sub_soln = produce_subdiff_soln([D_alpha, alpha], T, 2.0*L, dX)
dtrw_sub_soln_hires = produce_subdiff_soln([D_alpha, alpha], T, 4.0*L, dX_hires)
anal_sub_soln = produce_subdiff_analytic_soln([D_alpha], T, xs)

#
# PLOT IT ALL
#
fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(1, 2, 1)
line1, = ax1.plot(xs, anal_sub_soln, 'y.-')
line2, = ax1.plot(xs, dtrw_sub_soln.T[:xs.size], 'r.-')
line3, = ax1.plot(xs_hires, dtrw_sub_soln_hires.T[:xs_hires.size], 'b.-')

ax2 = fig.add_subplot(1, 2, 2)
ax2.semilogy(xs, anal_sub_soln, 'y.-')
ax2.semilogy(xs, dtrw_sub_soln.T[:xs.size], 'r.-')
ax2.semilogy(xs_hires, dtrw_sub_soln_hires.T[:xs_hires.size], 'b.-')

plt.legend([line1, line2, line3], ["Analytic Soln", "DTRW Soln", "DTRW Soln double LHS"], loc=3)

pp.savefig()
pp.close()

print "Result saved in ", pdf_name

#plt.show()

