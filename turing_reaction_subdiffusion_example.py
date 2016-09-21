#!/usr/local/bin/python3

import numpy as np
import time
from dtrw import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm


class DTRW_diffusive_turing_example(DTRW_diffusive):
    
    def __init__(self, X_inits, N, omega, a, b, c, d, r = 1., history_length=2, boltz_beta = 0., potential = np.array([]), boundary_condition=BC_periodic()):
        
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
        super(DTRW_diffusive_turing_example, self).__init__(X_inits, N, omega, r, history_length, boltz_beta, potential, boundary_condition)
   
    def calc_omega(self):
        """ Probability of death between n and n+1"""
        if self.omegas == None:
            self.omegas = [np.zeros((X.shape[0], X.shape[1])) for X in self.Xs]
            
            self.omegas[0][:,:] = 1. - np.exp(-self.a)
            self.omegas[1][:,:] = 1. - np.exp(-self.c)

    def calc_nu(self):
        """Likelihood of birth happening at i"""
        if self.nus == None:
            self.nus = [np.zeros((X.shape[0], X.shape[1])) for X in self.Xs]
    
        self.nus[0][:,:] = (1. - np.exp(-self.b)) * self.Xs[1][:,:,self.n]
        self.nus[1][:,:] = (1. - np.exp(-self.d)) * self.Xs[0][:,:,self.n]


class DTRW_subdiffusive_turing_example(DTRW_subdiffusive):
    
    def __init__(self, X_inits, N, alpha, a, b, c, d, r = 1., history_length=0, boltz_beta = 0., potential = np.array([]), boundary_condition=BC_periodic()):
        
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
        super(DTRW_subdiffusive_turing_example, self).__init__(X_inits, N, alpha, r, history_length, boltz_beta, potential, boundary_condition)
   
    def calc_omega(self):
        """ Probability of death between n and n+1"""
        if self.omegas == None:
            self.omegas = [np.zeros((X.shape[0], X.shape[1])) for X in self.Xs]
            
        self.omegas[0][:,:] = 1. - np.exp(-self.a)
        self.omegas[1][:,:] = 1. - np.exp(-self.c)

    def calc_nu(self):
        """Likelihood of birth happening at i"""
        if self.nus == None:
            self.nus = [np.zeros((X.shape[0], X.shape[1])) for X in self.Xs]
    
        self.nus[0][:,:] = (1. - np.exp(-self.b)) * self.Xs[1][:,:,self.n]
        self.nus[1][:,:] = (1. - np.exp(-self.d)) * self.Xs[0][:,:,self.n]

n_points = 40 
L = 1.0
dX = L / n_points

X_init = np.random.random(n_points)
Y_init = np.random.random(n_points)

T = 0.1

alpha = 0.75
D_alpha = 0.2

dT = pow((dX * dX / (2.0 * D_alpha)), 1./alpha)
N = int(math.floor(T / dT))
history_length = N

# Calculate r for diffusive case so as to get the *same* dT as the subdiffusive case
r = dT / (dX * dX / (2.0 * D_alpha))

a=-2.5 * dT
b=1.5 * dT
c=-1.5 * dT
d=2.5 * dT
mu=0.4 
nu=0.2
r_diff = [mu, nu]
dtrw_sub = DTRW_subdiffusive_turing_example([X_init, Y_init], N, alpha, a,b,c,d, r=r_diff)
dtrw = DTRW_diffusive_turing_example([X_init, Y_init], N, r, a,b,c,d, r=r_diff)

print("Solving for", N, "steps, dT =", dT, ", diffusion matching gives r =", r)

start = time.clock()
dtrw.solve_all_steps()
dtrw_sub.solve_all_steps()
end = time.clock()

print("Time for solution: ", end - start)

xs = np.linspace(0., L, n_points)
ys = np.linspace(0., L, n_points)

fig = plt.figure(figsize=(16,8))
#ax = Axes3D(fig)

ax1 = fig.add_subplot(1, 2, 1)

ax1.set_xlim([0, L])
ax1.set_ylim([0.2 + dtrw.Xs[0].min(), dtrw.Xs[0].max()])
ax1.set_title('Reaction diffusion system')
ax1.set_xlabel('x')
ax1.set_ylabel('Morphogen concentration')

line1, = ax1.plot([],[],'r-', label=r'Morphogen $\rho_1$')
line2, = ax1.plot([],[],'g-', label=r'Morphogen $\rho_2$')
#line3, = ax1.plot([],[],'r:', label='Morphogen A')
#line4, = ax1.plot([],[],'g:', label='Morphogen B')
ax1.legend()

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlim([0, L])
ax2.set_ylim([0.2 + dtrw.Xs[0].min(), dtrw.Xs[0].max()])

ax2.set_title('Reaction subdiffusion system with alpha={0}'.format(alpha))
ax2.set_xlabel('x')
ax2.set_ylabel('Morphogen concentration')

line3, = ax2.plot([],[],'r-', label=r'Morphogen $\rho_1$')
line4, = ax2.plot([],[],'g-', label=r'Morphogen $\rho_2$')
ax2.legend()

X_diff = dtrw.Xs[0]
Y_diff = dtrw.Xs[1]
X_sub = dtrw_sub.Xs[0]
Y_sub = dtrw_sub.Xs[1]

def update(i, line1, line2, line3, line4):
    
    line1.set_data(xs,dtrw.Xs[0][:,:,i])
    line2.set_data(xs,dtrw.Xs[1][:,:,i])
    line3.set_data(xs,dtrw_sub.Xs[0][:,:,i]) 
    line4.set_data(xs,dtrw_sub.Xs[1][:,:,i]) 

    return 

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, update, 
        frames=N, fargs=(line1, line2, line3, line4), interval=20)

import inspect, os, subprocess
exec_name =  os.path.splitext(os.path.basename(inspect.getfile(inspect.currentframe())))[0]
git_tag = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).replace('\n', '')

file_name = '{0}_{1}.mp4'.format(exec_name, git_tag)
print("Saving animation to", file_name)

anim.save(file_name, fps=24)#, extra_args=['-vcodec', 'libx264'])
plt.show()
