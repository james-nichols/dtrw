#! /usr/bin/env python

import numpy as np
import time
from dtrw import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm

n_points = 40 
L = 1.0
dX = L / n_points

V_1_init = np.zeros([n_points, n_points])
V_1_init[n_points/2,n_points/2] = 1.0 / (dX * dX)

V_2_init = np.zeros([n_points, n_points])

S_init = 2.*np.ones([n_points, n_points])
I_init = np.zeros([n_points, n_points])

T = 0.5

alpha = 0.75
D_alpha = 0.05

dT = pow((dX * dX / (2.0 * D_alpha)), 1./alpha)
N = int(math.floor(T / dT))
history_length = N

# Calculate r for diffusive case so as to get the *same* dT as the subdiffusive case
r = dT / (dX * dX / (2.0 * D_alpha))

k_1 = dT * 5.
k_2 = dT * 0.5
infection_rate = dT * 2.
clearance_rate = 0.

dtrw_sub = DTRW_subdiffusive_with_transition([V_1_init, V_2_init, S_init, I_init], N, alpha, k_1, k_2, clearance_rate, infection_rate, history_length, boundary_condition = BC_periodic())
dtrw = DTRW_diffusive_with_transition([V_1_init, V_2_init, S_init, I_init], N, r, k_1, k_2, clearance_rate, infection_rate, history_length, boundary_condition = BC_periodic())

print "Solving for", N, "steps, dT =", dT, ", diffusion matching gives r =", r

start = time.clock()
dtrw.solve_all_steps()
dtrw_sub.solve_all_steps()
end = time.clock()

print "Time for solution: ", end - start

xs = np.linspace(0, 1, n_points)
ys = np.linspace(0, 1, n_points)
Xs, Ys = np.meshgrid(xs, ys)

fig = plt.figure(figsize=(32,16))
#ax = Axes3D(fig)

ax1 = fig.add_subplot(2, 4, 1, projection='3d')
wframe = ax1.plot_surface(Xs, Ys, V_1_init, rstride=5, cstride=5)
ax1.set_zlim(0.,10.)

ax2 = fig.add_subplot(2, 4, 2, projection='3d')
wframe2 = ax2.plot_surface(Xs, Ys, V_1_init, rstride=5, cstride=5)
ax2.set_zlim(0.,10.)

ax3 = fig.add_subplot(2, 4, 3, projection='3d')
wframe3 = ax3.plot_surface(Xs, Ys, S_init, rstride=5, cstride=5)
ax3.set_zlim(0.,2.)

ax4 = fig.add_subplot(2, 4, 4, projection='3d')
wframe4 = ax4.plot_surface(Xs, Ys, I_init, rstride=5, cstride=5)
ax4.set_zlim(0.,2,)

ax5 = fig.add_subplot(2, 4, 5, projection='3d')
wframe5 = ax5.plot_surface(Xs, Ys, V_1_init, rstride=5, cstride=5)
ax5.set_zlim(0.,10.)

ax6 = fig.add_subplot(2, 4, 6, projection='3d')
wframe6 = ax6.plot_surface(Xs, Ys, V_1_init, rstride=5, cstride=5)
ax6.set_zlim(0.,10.)

ax7 = fig.add_subplot(2, 4, 7, projection='3d')
wframe7 = ax7.plot_surface(Xs, Ys, S_init, rstride=5, cstride=5)
ax7.set_zlim(0.,2.)

ax8 = fig.add_subplot(2, 4, 8, projection='3d')
wframe8 = ax8.plot_surface(Xs, Ys, I_init, rstride=5, cstride=5)
ax8.set_zlim(0.,2.)

V_1 = dtrw.Xs[0]
V_2 = dtrw.Xs[1]
S = dtrw.Xs[2]
I = dtrw.Xs[3]
V_1_sub = dtrw_sub.Xs[0]
V_2_sub = dtrw_sub.Xs[1]
S_sub = dtrw_sub.Xs[2]
I_sub = dtrw_sub.Xs[3]

def update(i, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, fig):
    
    ax1.cla()
    wframe = ax1.plot_surface(Xs, Ys, V_1[:,:,i], rstride=5, cstride=5, color='Blue', alpha=0.2)
    ax1.set_zlim(0.,10.)

    ax2.cla()
    wframe2 = ax2.plot_surface(Xs, Ys, V_2[:,:,i], rstride=5, cstride=5, color='Red', alpha=0.2)
    ax2.set_zlim(0.,10.)

    ax3.cla()
    wframe3 = ax3.plot_surface(Xs, Ys, S[:,:,i], rstride=5, cstride=5, color='Purple', alpha=0.2)
    ax3.set_zlim(0.,2.)

    ax4.cla()
    wframe4 = ax4.plot_surface(Xs, Ys, I[:,:,i], rstride=5, cstride=5, color='Green', alpha=0.2)
    ax4.set_zlim(0.,2.)
    
    ax5.cla()
    wframe5 = ax5.plot_surface(Xs, Ys, V_1_sub[:,:,i], rstride=5, cstride=5, color='Blue', alpha=0.2)
    ax5.set_zlim(0.,10.)

    ax6.cla()
    wframe6 = ax6.plot_surface(Xs, Ys, V_2_sub[:,:,i], rstride=5, cstride=5, color='Red', alpha=0.2)
    ax6.set_zlim(0.,10.)

    ax7.cla()
    wframe7 = ax7.plot_surface(Xs, Ys, S_sub[:,:,i], rstride=5, cstride=5, color='Purple', alpha=0.2)
    ax7.set_zlim(0.,2.)

    ax8.cla()
    wframe8 = ax8.plot_surface(Xs, Ys, I_sub[:,:,i], rstride=5, cstride=5, color='Green', alpha=0.2)
    ax8.set_zlim(0.,2.)
    
 
    #cset = ax.contour(Xs, Ys, X[:,:,i], zdir='z', offset=plot_min, cmap=cm.coolwarm)
    #cset = ax.contour(Xs, Ys, X[:,:,i], zdir='x', offset=0., cmap=cm.coolwarm)
    #cset = ax.contour(Xs, Ys, X[:,:,i], zdir='y', offset=1., cmap=cm.coolwarm)
    #ax.set_zlim(-0.1 * plot_max,1.1 * plot_max)
  
    #print X[47:53,47:53,i], X[:,:,i].sum()
    #print X_Q[47:53,47:53,i], X_Q[:,:,i].sum()
    #print X[:,:,i].sum(), X[:,:,i].sum()

    return wframe, wframe2, wframe3, wframe4, wframe5, wframe6, wframe7, wframe8

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, update, 
        frames=N, 
        fargs=(ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, fig), interval=100)

import inspect, os, subprocess
exec_name =  os.path.splitext(os.path.basename(inspect.getfile(inspect.currentframe())))[0]
git_tag = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).replace('\n', '')

file_name = '{0}_{1}.mp4'.format(exec_name, git_tag)
print "Saving animation to", file_name

anim.save(file_name, fps=24)#, extra_args=['-vcodec', 'libx264'])
#plt.show()
