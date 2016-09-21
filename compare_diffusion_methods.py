#! /usr/bin/env python

import numpy as np
from dtrw import * 

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm

n_points = 100
L = 1.0
dX = L / n_points

# Some arbitrary initial conditions
X_init = np.zeros([n_points, n_points])
X_init[50,50] = 1.0
X_init[50,10] = 1.0
X_init[80,85] = 1.0

r = 0.8
T = 0.1

D_alpha = 0.01
dT = r * dX * dX / (2.0 * D_alpha)
N = int(math.floor(T / dT))

dtrw = DTRW_diffusive(X_init, N, r)
dtrw_X = DTRW_diffusive(X_init, N, r, history_length=N)

print(dtrw.psi, dtrw.psi.sum())
print(dtrw.Phi)
print(dtrw.K)
print("solving for", N, "time steps")

dtrw.solve_all_steps()
# Note that for the other solver we solve using the arrival densities, rather than the memory
# kernel. This means, that as the default memory length is 2, that this method will give us 
# faulty results.
dtrw_X.solve_all_steps_with_Q()

xs = np.linspace(0, 1, X_init.shape[0])
ys = np.linspace(0, 1, X_init.shape[1])
Xs, Ys = np.meshgrid(xs, ys)

fig = plt.figure(figsize=(24,8))
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
wframe = ax1.plot_surface(Xs, Ys, X_init, rstride=5, cstride=5)
ax1.set_zlim(-0.1,1)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('Particle density')

ax2 = fig.add_subplot(1, 3, 2, projection='3d')
wframe2 = ax2.plot_surface(Xs, Ys, X_init, rstride=5, cstride=5)
ax2.set_zlim(-0.1,1)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('Particle density')

ax3 = fig.add_subplot(1, 3, 3, projection='3d')
wframe3 = ax3.plot_surface(Xs, Ys, X_init - X_init, rstride=5, cstride=5)
ax3.set_zlim(-0.1,1)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('Particle density')
#title = ax.text(.5, 1.05, '', transform = ax.transAxes, va='center')
#title = ax.text(0.5, 0.5, 0., 'Hi', zdir=None)

X = dtrw.Xs[0]
X_Q = dtrw_X.Xs[0]

def update(i, ax1, ax2, ax3, fig):
    
    #ax = fig.get_axes(1, 3, 1)
    ax1.cla()
    wframe = ax1.plot_surface(Xs, Ys, X[:,:,i], rstride=5, cstride=5, color='Blue', alpha=0.2)
    ax1.set_zlim(-0.1 * X[:,:,i].max(), 1.1 * X[:,:,i].max())

    #ax = fig.get_axes(1, 3, 2) 
    ax2.cla()
    wframe2 = ax2.plot_surface(Xs, Ys, X_Q[:,:,i], rstride=5, cstride=5, color='Red', alpha=0.2)
    ax2.set_zlim(-0.1 * X_Q[:,:,i].max(), 1.1 * X_Q[:,:,i].max())

    #ax = fig.get_axes(1, 3, 3) 
    ax3.cla()
    wframe3 = ax3.plot_surface(Xs, Ys, X[:,:,i] - X_Q[:,:,i], rstride=5, cstride=5, color='Purple', alpha=0.2)
    plot_max = (X[:,:,i] - X_Q[:,:,i]).max()
    plot_min = (X[:,:,i] - X_Q[:,:,i]).min()
    ax3.set_zlim(1.1 * plot_min, 1.1 * plot_max)

    #cset = ax.contour(Xs, Ys, dtrw.X[:,:,i], zdir='z', offset=plot_min, cmap=cm.coolwarm)
    #cset = ax.contour(Xs, Ys, dtrw.X[:,:,i], zdir='x', offset=0., cmap=cm.coolwarm)
    #cset = ax.contour(Xs, Ys, dtrw.X[:,:,i], zdir='y', offset=1., cmap=cm.coolwarm)
    
    return wframe, wframe2, wframe3

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, update, 
        frames=N, 
        fargs=(ax1, ax2, ax3, fig), interval=10)

import inspect, os, subprocess
exec_name =  os.path.splitext(os.path.basename(inspect.getfile(inspect.currentframe())))[0]
git_tag = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).replace('\n', '')

file_name = '{0}_{1}.mp4'.format(exec_name, git_tag)
print("Saving animation to", file_name)

anim.save(file_name, fps=24)
plt.show()

