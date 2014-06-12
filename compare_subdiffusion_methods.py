#! /usr/bin/env python

import numpy as np
from dtrw import *

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

dtrw = DTRW(X_init, N, dT, tau, omega, nu, history_length)
dtrw_X = DTRW(X_init, N, dT, tau, omega, nu, history_length)
dtrw_sub = DTRW_subdiffusive(X_init, N, dT, alpha, omega, nu, history_length)
dtrw_sub_X = DTRW_subdiffusive(X_init, N, dT, alpha, omega, nu, history_length)

print dtrw.psi, dtrw.psi.sum()
print dtrw_sub.psi, dtrw_sub.psi.sum()
print dtrw.Phi
print dtrw_sub.Phi

#dtrw.solve_all_steps()
#dtrw_X.solve_all_steps_with_K()
dtrw_sub.solve_all_steps()
dtrw_sub_X.solve_all_steps_with_K()

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
#title = ax.text(.5, 1.05, '', transform = ax.transAxes, va='center')
#title = ax.text(0.5, 0.5, 0., 'Hi', zdir=None)

def update(i, ax1, ax2, ax3, fig):
    
    #ax = fig.get_axes(1, 3, 1)
    ax1.cla()
    wframe = ax1.plot_surface(Xs, Ys, dtrw_sub.X[:,:,i], rstride=5, cstride=5, color='Blue', alpha=0.2)
    ax1.set_zlim(-0.1 * dtrw_sub.X[:,:,i].max(), 1.1 * dtrw_sub.X[:,:,i].max())

    #ax = fig.get_axes(1, 3, 2) 
    ax2.cla()
    wframe2 = ax2.plot_surface(Xs, Ys, dtrw_sub_X.X[:,:,i], rstride=5, cstride=5, color='Red', alpha=0.2)
    ax2.set_zlim(-0.1 * dtrw_sub_X.X[:,:,i].max(), 1.1 * dtrw_sub_X.X[:,:,i].max())
    #wframe2 = ax2.plot_surface(Xs, Ys, dtrw_sub.X[:,:,i], rstride=5, cstride=5, color='Red', alpha=0.2)
    #ax2.set_zlim(-0.1 * dtrw_sub.X[:,:,i].max(), 1.1 * dtrw_sub.X[:,:,i].max())

    #ax = fig.get_axes(1, 3, 3) 
    ax3.cla()
    wframe3 = ax3.plot_surface(Xs, Ys, dtrw_sub.X[:,:,i] - dtrw_sub_X.X[:,:,i], rstride=5, cstride=5, color='Purple', alpha=0.2)
    plot_max = (dtrw_sub.X[:,:,i] - dtrw_sub_X.X[:,:,i]).max()
    plot_min = (dtrw_sub.X[:,:,i] - dtrw_sub_X.X[:,:,i]).min()
    #wframe3 = ax3.plot_surface(Xs, Ys, dtrw.X[:,:,i] - dtrw_sub.X[:,:,i], rstride=5, cstride=5, color='Purple', alpha=0.2)
    #plot_max = (dtrw.X[:,:,i] - dtrw_sub.X[:,:,i]).max()
    #plot_min = (dtrw.X[:,:,i] - dtrw_sub.X[:,:,i]).min()
    ax3.set_zlim(1.1 * plot_min, 1.1 * plot_max)

    #cset = ax.contour(Xs, Ys, dtrw.X[:,:,i], zdir='z', offset=plot_min, cmap=cm.coolwarm)
    #cset = ax.contour(Xs, Ys, dtrw.X[:,:,i], zdir='x', offset=0., cmap=cm.coolwarm)
    #cset = ax.contour(Xs, Ys, dtrw.X[:,:,i], zdir='y', offset=1., cmap=cm.coolwarm)
    #ax.set_zlim(-0.1 * plot_max,1.1 * plot_max)
   
    #print dtrw.X[:,:,i].sum(), dtrw_sub.X[:,:,i].sum()

    return wframe, wframe2, wframe3

def update2(i, ax, fig):
    ax.cla()
    wframe = ax.plot_wireframe(Xs, Ys, dtrw.X[:,:,i] - dtrw_sub.X[:,:,i], rstride=5, cstride=5, color='Red')
    ax.set_zlim(-0.1, 1.1 * (dtrw.X[:,:,i]-dtrw_sub.X[:,:,i]).max())
    return wframe,

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, update, 
        frames=N, 
        fargs=(ax1, ax2, ax3, fig), interval=100)

anim.save('basic_animation.mp4', fps=30)
plt.show()

