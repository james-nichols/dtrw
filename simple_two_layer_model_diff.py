#! /usr/bin/env python

import numpy as np
import time
from dtrw import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm

n_points = 100 
L = 1.0
dX = L / n_points

V_1_init = np.zeros([n_points, n_points])
V_1_init[50,50] = 1.0 / (dX * dX)

V_2_init = np.zeros([n_points, n_points])

S_init = np.ones([n_points, n_points])
I_init = np.zeros([n_points, n_points])

T = 0.05

r = 0.2
D_alpha = 0.05

dT = r * dX * dX / (2.0 * D_alpha)
N = int(math.floor(T / dT))
history_length = N

k_1 = dT * 5.
k_2 = dT * 0.5
infection_rate = dT * 100.
clearance_rate = 0.

dtrw = DTRW_diffusive_with_transition([V_1_init, V_2_init, S_init, I_init], N, r, k_1, k_2, clearance_rate, infection_rate, history_length, is_periodic=True)

print "Solving for", N, "steps."

start = time.clock()
dtrw.solve_all_steps()
end = time.clock()

print "Time for solution: ", end - start

xs = np.linspace(0, 1, n_points)
ys = np.linspace(0, 1, n_points)
Xs, Ys = np.meshgrid(xs, ys)

fig = plt.figure(figsize=(32,8))
#ax = Axes3D(fig)
ax1 = fig.add_subplot(1, 4, 1, projection='3d')
wframe = ax1.plot_surface(Xs, Ys, V_1_init, rstride=5, cstride=5)
ax1.set_zlim(-0.1,1)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('Particle density')

ax2 = fig.add_subplot(1, 4, 2, projection='3d')
wframe2 = ax2.plot_surface(Xs, Ys, V_1_init, rstride=5, cstride=5)
ax2.set_zlim(-0.1,1)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('Particle density')

ax3 = fig.add_subplot(1, 4, 3, projection='3d')
wframe3 = ax3.plot_surface(Xs, Ys, S_init, rstride=5, cstride=5)
ax3.set_zlim(-0.1,1)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('Particle density')

ax4 = fig.add_subplot(1, 4, 4, projection='3d')
wframe4 = ax4.plot_surface(Xs, Ys, I_init, rstride=5, cstride=5)
ax4.set_zlim(-0.1,1)
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_zlabel('Particle density')

V_1 = dtrw.Xs[0]
V_2 = dtrw.Xs[1]
S = dtrw.Xs[2]
I = dtrw.Xs[3]

def update(i, ax1, ax2, ax3, ax4, fig):
    
    #ax = fig.get_axes(1, 3, 1)
    ax1.cla()
    wframe = ax1.plot_surface(Xs, Ys, V_1[:,:,i], rstride=5, cstride=5, color='Blue', alpha=0.2)
    ax1.set_zlim(-0.1 * V_1[:,:,i].max(), 1.1 * V_1[:,:,i].max())

    #ax = fig.get_axes(1, 3, 2) 
    ax2.cla()
    wframe2 = ax2.plot_surface(Xs, Ys, V_2[:,:,i], rstride=5, cstride=5, color='Red', alpha=0.2)
    ax2.set_zlim(-0.1 * V_2[:,:,i].max(), 1.1 * V_2[:,:,i].max())

    #ax = fig.get_axes(1, 3, 3) 
    ax3.cla()
    wframe3 = ax3.plot_surface(Xs, Ys, S[:,:,i], rstride=5, cstride=5, color='Purple', alpha=0.2)
    ax3.set_zlim(-0.1 * S[:,:,i].max(), 1.1 * S[:,:,i].max())

    #ax = fig.get_axes(1, 3, 3) 
    ax4.cla()
    wframe4 = ax4.plot_surface(Xs, Ys, I[:,:,i], rstride=5, cstride=5, color='Green', alpha=0.2)
    ax4.set_zlim(-0.1 * I[:,:,i].max(), 1.1 * I[:,:,i].max())
    
    #cset = ax.contour(Xs, Ys, X[:,:,i], zdir='z', offset=plot_min, cmap=cm.coolwarm)
    #cset = ax.contour(Xs, Ys, X[:,:,i], zdir='x', offset=0., cmap=cm.coolwarm)
    #cset = ax.contour(Xs, Ys, X[:,:,i], zdir='y', offset=1., cmap=cm.coolwarm)
    #ax.set_zlim(-0.1 * plot_max,1.1 * plot_max)
  
    #print X[47:53,47:53,i], X[:,:,i].sum()
    #print X_Q[47:53,47:53,i], X_Q[:,:,i].sum()
    #print X[:,:,i].sum(), X[:,:,i].sum()

    return wframe, wframe2, wframe3, wframe4

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, update, 
        frames=N, 
        fargs=(ax1, ax2, ax3, ax4, fig), interval=100)

import inspect, os, subprocess
exec_name =  os.path.splitext(os.path.basename(inspect.getfile(inspect.currentframe())))[0]
git_tag = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).replace('\n', '')

file_name = '{0}_{1}.mp4'.format(exec_name, git_tag)
print "Saving animation to", file_name

anim.save(file_name, fps=24)
plt.show()
