#!/usr/local/bin/python3

import numpy as np
import time
import math

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm

import pdb



def calc_sibuya(N, alpha):

    sibuya_phi = np.zeros(N)
    sibuya_K = np.zeros(N)
    comp = np.array(list(range(1,N+1)))
    
    sibuya_phi[0] = 1.-alpha
    sibuya_phi[1] = (1.-alpha) * (1. - alpha / 2.)
    sibuya_K[0] = alpha
    sibuya_K[1] = (1.-alpha) * alpha / 2.
    
    upper = 1. / (pow(comp, alpha)) 
    lower = (1.-alpha) / (pow(comp, alpha))
    
    K_upper = 1. / (math.gamma(alpha) * pow(comp, 2.-alpha)) 
    K_lower = (1.-alpha) / (math.gamma(alpha) * pow(comp, 2.-alpha))
    
    for i in range(2,N):
        sibuya_phi[i] = sibuya_phi[i-1] * (1. - alpha / float(i+1))
        sibuya_K[i] = sibuya_K[i-1] * (float(i+1) + alpha - 2.)/float(i+1)

    return sibuya_phi, sibuya_K, upper, lower, K_upper, K_lower 

N = 50
N_alpha = 100
xs = np.array(list(range(N)))

fig = plt.figure(figsize=(8,8))
plt.xlim(0,N)
plt.ylim(0,1.0)
plt.xlabel('x')
line1, = plt.plot([],[],'r-')
line2, = plt.plot([],[],'g-')
line3, = plt.plot([],[],'b-')
line4, = plt.plot([],[],'y-')
line5, = plt.plot([],[],'p-')
line6, = plt.plot([],[],'k-')
plt.legend([line1, line2, line3, line4], ["Phi", "K", "Upper bound", "lower bound"])

def update(i, line1, line2, line3, line4, line5, line6):
    alpha = float(i+1) / float(N_alpha+1)
    phi, K, upper, lower, K_upper, K_lower = calc_sibuya(N, alpha)
    line1.set_data(xs, phi)
    line2.set_data(xs, K)
    line3.set_data(xs, upper)
    line4.set_data(xs, lower)
    line5.set_data(xs, K_upper)
    line6.set_data(xs, K_lower)
   
    print(alpha)
    print((K_upper - K)[:10])
    print((K - K_lower)[:10])
    return line1, line2, line3, line4, line5, line6

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, update, 
        frames=N_alpha, fargs=(line1, line2, line3, line4, line5, line6), interval=100)

import inspect, os, subprocess
exec_name =  os.path.splitext(os.path.basename(inspect.getfile(inspect.currentframe())))[0]
#git_tag = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).replace('\n', '')
git_tag = "no_git"

file_name = '{0}_{1}.mp4'.format(exec_name, git_tag)
print("Saving animation to", file_name)

anim.save(file_name, fps=24)
plt.show()

