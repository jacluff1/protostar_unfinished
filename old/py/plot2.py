import numpy as np
import auxillary2 as aux
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import griddata
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as ani
import matplotlib.colors as colors
import matplotlib.cm as cm
import pdb

#===============================================================================
""" plotting parameters """
#-------------------------------------------------------------------------------

pp  =   {}
pp['size']      =   (15,15)     # figsize
pp['fs']        =   20          # fontsize
pp['cmap']      =   cm.magma    # color map
pp['N']         =   100         # length of plotting arrays
pp['interval']  =   200         # interval
pp['dpi']       =   400         # resolution
pp['Nx']        =   1000        # number image bins on single axis
pp['logscale']  =   True        # logscale images (or not)

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

def scatter_3D(t,model,saveA=True):
    """ make a 3D scatter plot of
    particles at time t

    Parameters
    ----------
    model:      panda Series containing all model data
    t:          time index
    """

    # model data
    Np      =   model['Np']
    Rt      =   model['pos'][t,:,:]
    # time    =   model['time'][t]
    Mc      =   model['M']
    RHO     =   model['rho'][t,:]

    # analysis
    X,Y,Z   =   [ Rt[:,e] for e in range(3) ]
    limits  =   np.min(RHO),np.max(RHO)

    # figure
    plt.close('all')
    fig     =   plt.figure(figsize=pp['size'])
    ax      =   plt.subplot(111, projection='3d')
    ax.set_title("%s M$_\odot$ cloud" % Mc, fontsize=pp['fs']+2)
    ax.set_xlabel("x", fontsize=pp['fs'])
    ax.set_ylabel("y", fontsize=pp['fs'])
    ax.set_zlabel("z", fontsize=pp['fs'])
    ax.set_aspect(1)

    levels  =   MaxNLocator(nbins=100).tick_values(*limits)
    cmap    =   pp['cmap']
    norm    =   BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    im      =   ax.scatter(X,Y,Z, c=RHO, cmap=cmap)
    cbar    =   fig.colorbar(im, ax=ax, pad=0)
    cbar.set_label('$\\rho$', fontsize=pp['fs'])

    plt.tight_layout()

    if saveA:
        fig.savefig('../plots/scatter_%.2f_%s.png' % (Mc,t) )
        plt.close()

    else:
        return ax

def contourf_2D(t,model,saveA=True):
    """ make a 2D contour plot of
    particles at time t
    http://stackoverflow.com/a/23907866

    Parameters
    ----------
    model:      panda Series containing all model data
    t:          time index
    """

    # data
    # time        =   model['time'][t]
    Mc          =   model['M']
    R           =   model['pos'][t,:,:]
    Np          =   model['Np']
    Nt          =   model['Nt']

    try:
        images      =   np.load('../data/images_Np%s_Nt%s.npy' % (Np,Nt) )
        Xp          =   np.load('../data/Xp_Np%s_Nt%s.npy' % (Np,Nt) )
        Yp          =   np.load('../data/Yp_Np%s_Nt%s.npy' % (Np,Nt) )
        Zp          =   images[t,:,:]
    except:
        rlim        =   np.max( np.array([ np.linalg.norm( R[i,:] ) for i in range(Np) ]) )
        X,Y         =   [ np.linspace(-rlim,rlim,1000) for i in range(2) ]
        Xp,Yp       =   np.meshgrid(X,Y)

        Zp          =   aux.density_xy(Xp,Yp,t,model)
        # Zp          =   np.log10(Zp)

    limits      =   0,np.max(Zp)

    # figure
    plt.close('all')
    fig     =   plt.figure(figsize=pp['size'])
    ax      =   plt.subplot(111)
    ax.patch.set_facecolor('black')
    ax.set_title("%s M$_\odot$ cloud" % Mc, fontsize=pp['fs']+2)
    ax.set_xlabel("x", fontsize=pp['fs'])
    ax.set_ylabel("y", fontsize=pp['fs'])
    ax.set_aspect(1)

    levels  =   MaxNLocator(nbins=100).tick_values(*limits)
    cmap    =   pp['cmap']
    norm    =   BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    im      =   ax.contourf(Xp,Yp,Zp, levels=levels, cmap=cmap)
    cbar    =   fig.colorbar(im, ax=ax, pad=0)
    cbar.set_label('$\\rho$', fontsize=pp['fs'])

    plt.tight_layout()

    if saveA:
        fig.savefig('../plots/contourf_%.2f_%s.png' % (Mc,t) )
        plt.close()
    else:
        plt.show()
        return ax

# def density_profile(model,t,saveA=True):
#     """ plot density profile at time t
#
#     Parameters
#     ----------
#     model:      panda Series containing all model data
#     t:          time index
#     """

#===============================================================================
""" movie """
#-------------------------------------------------------------------------------

def movie_contourf(model,saveA=True,overWrite=False):

    Nt          =   model['Nt']
    Np          =   model['Np']
    M           =   model['M']

    if overWrite:   aux.generate_contour_images(pp,model)

    try:
        print("\nloading arrays...")
        images          =   np.load('../data/images_Np%s_Nt%s.npy' % (Np,Nt) )
        Xp              =   np.load('../data/Xp_Np%s_Nt%s.npy' % (Np,Nt) )
        Yp              =   np.load('../data/Yp_Np%s_Nt%s.npy' % (Np,Nt) )
    except:
        print("\ngenerating arrays...")
        Xp,Yp,images    =   aux.generate_contour_images(pp,model)

    limits  =   0,np.max(images)

    fig     =   plt.figure(figsize=pp['size'])
    ax      =   fig.add_subplot(111)
    ax.patch.set_facecolor('black')
    ax.set_title("%s M$_\odot$ cloud" % M, fontsize=pp['fs']+2)
    ax.set_xlabel("x", fontsize=pp['fs'])
    ax.set_ylabel("y", fontsize=pp['fs'])
    ax.set_aspect(1)

    levels  =   MaxNLocator(nbins=100).tick_values(*limits)
    cmap    =   pp['cmap']
    norm    =   BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    im      =   ax.contourf(Xp,Yp,images[0,:,:], levels=levels, cmap=cmap)
    cbar    =   fig.colorbar(im, ax=ax, pad=0)
    cbar.set_label('$\\rho$', fontsize=pp['fs'])

    if saveA == False: plt.show()

    def animate(t):

        ax.cla()
        ax.patch.set_facecolor('black')
        im          =   ax.contourf(Xp,Yp,images[t,:,:], level=levels, cmap=cmap)

        return ax

    frames	=	int(Nt)
    anim    =   ani.FuncAnimation(fig, animate, frames=frames, blit=False, interval=pp['interval'])
    anim.save('../plots/star.mp4', writer='ffmpeg', dpi=pp['dpi'])
