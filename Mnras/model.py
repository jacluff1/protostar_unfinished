import initialConditions as ic
import kernel as kernel
import smoothingLength as sl
import timeStep as ts
import functions as func

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation

import djak.gen as dg

from params import par
for key,val in par.items():
    exec(key + '=val')

def plot_initial_placement(show=True,save=True):

    X       =   ic.placement_uniform_sphere()
    xmax    =   -r0*1.2 , r0*1.2

    fig =   plt.figure()
    ax  =   fig.gca(projection='3d')

    ax.plot(X[:,0],X[:,1],X[:,2], 'go')
    ax.set_aspect(1)
    ax.set_xlim(*xmax)
    ax.set_ylim(*xmax)
    ax.set_zlim(*xmax)

    if save:
        dg.directory_checker('plots')
        fig.savefig('plots/initialPlacement.png')

    if show:
        plt.show()
    else:
        plt.close()

    return X,xmax

def plot_smoothingL(show=True,save=True):

    positions,xmax  =   plot_initial_placement(show=False)
    smoothingL      =   func.find_smoothingL(positions)

    return smoothingL

def plot_initial_density(size=100,b=.25,show=True,save=True):

    X,xmax  =   plot_initial_placement(show=False)
    h       =   sl.find_smoothing_length(X,1)
    axis    =   np.linspace(*xmax,size)
    z1      =   np.zeros( (size,size) )
    z2      =   np.zeros_like(z1)

    for i,x1 in enumerate(axis):
        for j,x2 in enumerate(axis):
            pos1    =   np.array([x1,x2,0])
            pos2    =   np.array([0,x1,x2])
            z1[i,j] =   func.find_density(pos1,X,b)
            z2[i,j] =   func.find_density(pos2,X,b)

    fig,ax  =   plt.subplots(1,2,figsize=(20,10))
    ax[0].contourf(axis,axis,z1,100,cmap=cm.hot)
    ax[1].contourf(axis,axis,z2,100,cmap=cm.hot)
    ax[0].set_title('xy-plane',fontsize=20)
    ax[1].set_title('yz-plane',fontsize=20)

    if save:
        dg.directory_checker('plots')
        fig.savefig('plots/initialDensity.png')

    if show:
        plt.show()
    else:
        plt.close()
