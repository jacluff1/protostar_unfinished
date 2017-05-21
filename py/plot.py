import numpy as np
import auxillary as aux
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import griddata
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as ani
import matplotlib.colors as colors
import matplotlib.cm as cm
import djak.constants as dc
from djak.constants import SI, solarM, pc, Myr, gr
import pdb

#===============================================================================
""" plotting parameters """
#-------------------------------------------------------------------------------

pp  =   {'size':    (15,15),    # figsize
         'fs':      20,         # fontsize
         'cmap':    cm.magma,   # color map
         'N':       100         # length of plotting arrays
         }

#===============================================================================
""" misc """
#-------------------------------------------------------------------------------

def map_grid(X,Y,RHO):
    """ sets 2D image to grid
    https://matplotlib.org/examples/pylab_examples/griddata_demo.html

    Parameters
    ----------
    X:  1D x-axis data
    Y:  1D y-axis data
    M:  1D particle mass data

    Returns
    -------
    tuple of X,Y,Z arrays
    """

    rmax    =   max( np.max(X) , np.max(Y) )
    Xp,Yp   =   [ np.linspace(-rmax,rmax,100) for i in range(2) ]
    Zp      =   griddata(X, Y, RHO, Xp, Yp, interp='linear')

    return Xp,Yp,Zp

#===============================================================================
""" Profiles """
#-------------------------------------------------------------------------------

def profile_density(Mc,Np,Rt,h):
    """ calculate density profiles for X, Y, and Z
    arrays from SPH particle positions at time t

    Parameters
    ----------
    Mc:     total mass of cloud
    Np:     number of SPH particles
    Rt:     SPH particle positions at time t
    h:      smoothing length

    Returns
    -------
    panda Series of X, Y, Z, rhoX , rhoY, and rhoZ
    """

    Xd,Yd,Zd    =   [ R[:,e] for e in range(3) ]
    rmax        =   np.max(R)
    E           =   np.linspace(-rmax,rmax,pp['N'])
    X           =   np.array([ np.array([e,0,0]) for e in E ])
    Y           =   np.array([ np.array([0,e,0]) for e in E ])
    Z           =   np.array([ np.array([0,0,e]) for e in E ])
    rhoX        =   aux.density(Mc,Np,Rt,X,h)
    rhoY        =   aux.density(Mc,Np,Rt,Y,h)
    rhoZ        =   aux.density(Mc,Np,Rt,Z,h)

    d           =   {'X':X, 'rhoX':rhoX,
                     'Y':Y, 'rhoY':rhoY,
                     'Z':Z, 'rhoZ':rhoZ}
    return pd.Series(d)

#===============================================================================
""" plotting functions """
#-------------------------------------------------------------------------------

def scatter_3D(model,t,saveA=True):
    """ make a 3D scatter plot of
    particles at time t

    Parameters
    ----------
    model:      panda Series containing all model data
    t:          time index
    """

    # model data
    Np      =   model['Np']
    Rt      =   model['R'][t,:,:] * pc.value
    time    =   model['TIME'][t] * Myr.value
    Mc      =   model['Mc'] * solarM.value
    RHO     =   model['rho'][t]

    # analysis
    X,Y,Z   =   [ Rt[:,e] for e in range(3) ]
    limits  =   np.min(RHO),np.max(RHO)

    # figure
    plt.close('all')
    fig     =   plt.figure(figsize=pp['size'])
    ax      =   fig.add_subplot(111, projection='3d')
    ax.set_title("%s M$_\odot$ cloud, t = %s" % (Mc,time), fontsize=pp['fs']+2)
    ax.set_xlabel("x [%s]" % pc.units, fontsize=pp['fs'])
    ax.set_ylabel("y [%s]" % pc.units, fontsize=pp['fs'])
    ax.set_zlabel("z [%s]" % pc.units, fontsize=pp['fs'])
    ax.set_aspect(1)

    levels  =   MaxNLocator(nbins=100).tick_values(*limits)
    cmap    =   pp['cmap']
    norm    =   BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    im      =   ax.scatter(X,Y,Z, c=RHO, cmap=cmap)
    cbar    =   fig.colorbar(im, ax=ax, pad=0)
    cbar.set_label('$\\rho$ [ %s / %s$^2$ ]' % (dc.kg.units,dc.m.units), fontsize=pp['fs'])

    plt.tight_layout()

    if saveA:
        fig.savefig('../plots/scatter_%.2f_%.2f.png' % (Mc,time))
        plt.close()

    else:
        plt.show()

def contourf_2D(model,t,saveA=True):
    """ make a 2D contour plot of
    particles at time t
    http://stackoverflow.com/a/23907866

    Parameters
    ----------
    model:      panda Series containing all model data
    t:          time index
    """

    # data
    time        =   model['TIME'][t] * Myr.value
    Mc          =   model['Mc'] * solarM.value

    R           =   model['R'][t] * pc.value
    X,Y         =   [ R[:,e] for e in range(2) ]
    rho         =   model['rho'][t]
    Xp,Yp,Zp    =   map_grid(X,Y,rho)
    limits      =   0,np.max(Zp)

    # figure
    plt.close('all')
    fig     =   plt.figure(figsize=pp['size'])
    ax      =   fig.add_subplot(111)
    ax.patch.set_facecolor('black')
    ax.set_title("%s M$_\odot$ cloud, t = %s" % (Mc,time), fontsize=pp['fs']+2)
    ax.set_xlabel("x [%s]" % pc.units, fontsize=pp['fs'])
    ax.set_ylabel("y [%s]" % pc.units, fontsize=pp['fs'])
    ax.set_aspect(1)

    levels  =   MaxNLocator(nbins=100).tick_values(*limits)
    cmap    =   pp['cmap']
    norm    =   BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    im      =   ax.contourf(Xp,Yp,Zp, levels=levels, cmap=cmap)
    cbar    =   fig.colorbar(im, ax=ax, pad=0)
    cbar.set_label('$\\rho$ [ %s / %s$^2$ ]' % (dc.kg.units,dc.m.units), fontsize=pp['fs'])

    plt.tight_layout()

    if saveA:
        fig.savefig('../plots/contourf_%.2f_%.2f.png' % (Mc,time))
        plt.close()
    else:
        plt.show()

def density_profile(model,t,saveA=True):
    """ plot density profile at time t

    Parameters
    ----------
    model:      panda Series containing all model data
    t:          time index
    """
