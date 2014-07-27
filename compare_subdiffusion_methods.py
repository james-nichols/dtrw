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

X_init = np.zeros([n_points, n_points])
X_init[50,50] = 1.0
X_init[50,10] = 1.0
X_init[80,85] = 1.0

T = 0.01

alpha = 0.75
D_alpha = 0.05

r = 0.8

dT = pow((dX * dX / (2.0 * D_alpha)), 1./alpha)

N = int(math.floor(T / dT))
history_length = N

omega = 0.0 #0.05
nu = 0.0 #0.0005

dtrw = DTRW_subdiffusive(X_init, N, alpha, omega, nu, history_length)
dtrw_Q = DTRW_subdiffusive(X_init, N, alpha, omega, nu, history_length)

print dtrw.psi, dtrw.psi.sum()
print dtrw.Phi
print dtrw.K
print "Solving for", N, "steps."

start = time.clock()
dtrw.solve_all_steps()
mid = time.clock()
dtrw_Q.solve_all_steps_with_Q()
end = time.clock()

print "Time for K method: ", mid - start, ", time for Q method: ", end - mid

xs = np.linspace(0, 1, X_init.shape[0])
ys = np.linspace(0, 1, X_init.shape[1])
Xs, Ys = np.meshgrid(xs, ys)

fig = plt.figure(figsize=(24,8))
#ax = Axes3D(fig)
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

X = dtrw.Xs[0]
X_Q = dtrw_Q.Xs[0]

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

    #cset = ax.contour(Xs, Ys, X[:,:,i], zdir='z', offset=plot_min, cmap=cm.coolwarm)
    #cset = ax.contour(Xs, Ys, X[:,:,i], zdir='x', offset=0., cmap=cm.coolwarm)
    #cset = ax.contour(Xs, Ys, X[:,:,i], zdir='y', offset=1., cmap=cm.coolwarm)
    #ax.set_zlim(-0.1 * plot_max,1.1 * plot_max)
  
    #print X[47:53,47:53,i], X[:,:,i].sum()
    #print X_Q[47:53,47:53,i], X_Q[:,:,i].sum()
    #print X[:,:,i].sum(), X[:,:,i].sum()

    return wframe, wframe2, wframe3

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, update, 
        frames=N, 
        fargs=(ax1, ax2, ax3, fig), interval=100)

anim.save('compare_subdiffusion_methods.mp4', fps=24)
plt.show()

