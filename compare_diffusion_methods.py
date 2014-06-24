#! /usr/bin/env python

import numpy as np
from dtrw import * 

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm

X_init = np.zeros([100, 100])
X_init[50,50] = 1.0
X_init[50,10] = 1.0
X_init[80,85] = 1.0

N = 1000
dT = 0.5
tau = 0.5
alpha = 0.5
tau2 = 0.2
omega = 0.0 #0.05
nu = 0.0 #0.0005

history_length = 40

dtrw = DTRW_diffusive(X_init, N, dT, tau, omega, nu, history_length)
dtrw_X = DTRW_diffusive(X_init, N, dT, tau, omega, nu, history_length)

print dtrw.psi, dtrw.psi.sum()
print dtrw.Phi
print dtrw.K

dtrw.solve_all_steps()
dtrw_X.solve_all_steps_with_K()

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

def update(i, ax1, ax2, ax3, fig):
    
    #ax = fig.get_axes(1, 3, 1)
    ax1.cla()
    wframe = ax1.plot_surface(Xs, Ys, dtrw.X[:,:,i], rstride=5, cstride=5, color='Blue', alpha=0.2)
    ax1.set_zlim(-0.1 * dtrw.X[:,:,i].max(), 1.1 * dtrw.X[:,:,i].max())

    #ax = fig.get_axes(1, 3, 2) 
    ax2.cla()
    wframe2 = ax2.plot_surface(Xs, Ys, dtrw_X.X[:,:,i], rstride=5, cstride=5, color='Red', alpha=0.2)
    ax2.set_zlim(-0.1 * dtrw_X.X[:,:,i].max(), 1.1 * dtrw_X.X[:,:,i].max())

    #ax = fig.get_axes(1, 3, 3) 
    ax3.cla()
    wframe3 = ax3.plot_surface(Xs, Ys, dtrw.X[:,:,i] - dtrw_X.X[:,:,i], rstride=5, cstride=5, color='Purple', alpha=0.2)
    plot_max = (dtrw.X[:,:,i] - dtrw_X.X[:,:,i]).max()
    plot_min = (dtrw.X[:,:,i] - dtrw_X.X[:,:,i]).min()
    ax3.set_zlim(1.1 * plot_min, 1.1 * plot_max)

    #cset = ax.contour(Xs, Ys, dtrw.X[:,:,i], zdir='z', offset=plot_min, cmap=cm.coolwarm)
    #cset = ax.contour(Xs, Ys, dtrw.X[:,:,i], zdir='x', offset=0., cmap=cm.coolwarm)
    #cset = ax.contour(Xs, Ys, dtrw.X[:,:,i], zdir='y', offset=1., cmap=cm.coolwarm)
    
    return wframe, wframe2, wframe3

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, update, 
        frames=N, 
        fargs=(ax1, ax2, ax3, fig), interval=10)

anim.save('compare_diffusion_methods.mp4', fps=24)
plt.show()

