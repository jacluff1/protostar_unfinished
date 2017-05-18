import numpy as np
import djak.constants as dc
from djak.constants import SI, solarM, pc, Myr, gr
import djak.astro.cloud as dac
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import griddata
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as ani
import matplotlib.colors as colors
import matplotlib.cm as cm
import pandas as pd

#===============================================================================
""" Misc """
#-------------------------------------------------------------------------------

def r_vec(ri,rj):
    """ find the relative position vector

    Parameters
    ----------
    r0:     object particle position
    ri:     ajent particle position

    Returns
    -------
    tuple - rmag,rhat
    """

    r       =   ri - rj
    rmag    =   np.linalg.norm(r)
    assert rmag > 0, "rmag = 0, ri != rj"
    rhat    =   r / rmag
    return rmag,rhat

def average_particle_mass(frac_H,frac_He):
    """ find the average particle mass
    from particle populations

    Parameters
    ----------
    frac_H:     fraction of hydrogen
    frac_He:    fraction of Helium

    Returns
    -------
    scalar
    """

    return frac_H*SI['m_H'].value + frac_He*SI['m_He'].value

#===============================================================================
""" Kernal Smoothing """
#-------------------------------------------------------------------------------

def kernal_gauss(ri,rj,h):
    """ Gaussian method for kernal Smoothing

    Parameters
    ----------
    ri:     object particle position
    rj:     agent particle position
    h:      smoothing length

    Returns
    -------
    scalar
    """
    rmag,rhat   =   r_vec(ri,rj)

    return ( h * np.sqrt(2) )**(-3) * np.exp( -rmag**2 / h**2 )

def gradient_kernal_gauss(ri,rj,h):
    """ returns the gradient of the
    gaussian kernal smoothing function

    Parameters
    ----------
    ri:     object particle position
    rj:     agent particle position
    h:      kernal smoothing length

    Returns
    -------
    vector array
    """

    W       =   kernal_gauss(ri,rj,h)
    return (2 / h**2) * W * rj

#===============================================================================
""" Cloud Physics """
#-------------------------------------------------------------------------------

def random_particle_placement(Np,r_jean):
    """ generate randome particle positions
    for all particles

    Parameters
    ----------
    Np:     number of particles
    r_jean: Jeans radius of cloud
    d:      number of dimenetions

    Returns
    -------
    tuple of 3 1D arrays
    """

    # set up random positions in spherical coordinates
    radius  =   np.random.rand(Np) * r_jean
    theta   =   np.random.rand(Np) * np.pi
    phi     =   np.random.rand(Np) * 2*np.pi

    # convert spherical coordinates to cartesian
    X       =   radius * np.sin(theta) * np.cos(phi)
    Y       =   radius * np.sin(theta) * np.sin(phi)
    Z       =   radius * np.cos(theta)

    return X,Y,Z

def pressure(k,rho,n):
    """ find the pressure of polytropic fluid

    Parameters
    ----------
    k:      constant
    rho:    density
    n:      polytropic index

    Returns
    -------
    scalar - pressure
    """

    return k * rho**( 1 + 1/n )

def density(M,Rt,i,h):
    """ find density of SPH

    Parameters
    ----------
    M:
    Rt:
    i:
    h:

    Returns
    -------
    scalar
    """

    Np      =   len(M)
    rho     =   0
    ri      =   Rt[i,:]
    for j in range(Np):
        if j != i:
            rj      =   Rt[j,:]
            mj      =   M[j]
            Wj      =   kernal_gauss(ri,rj,h)
            rho     +=  mj * Wj
    return rho

#===============================================================================
""" Equations of Motion """
#-------------------------------------------------------------------------------

def acc_gravity(ri,rj,mj):
    """ find acceleration due to gravity between
    two particles

    Parameters
    ----------
    r0:     object particle position
    ri:     agent particle position
    mi:     agent particle mass

    Returns
    -------
    vector array
    """

    rmag,rhat   =   r_vec(ri,rj)
    return - SI['G'] * mj / rmag**2 * rhat

def acc_particle_external(M,Rt,i):
    """ find acceleration of particle
    due to external forces

    Parameters
    ----------
    M:      1D array of particles masses
    Rt:     2D array of particle positions at time t
    i:      index of particle

    Returns
    -------
    vector array
    """

    # particle information
    Np      =   len(M)
    ri      =   Rt[i,:]

    # interactions with other particles
    total   =   0
    for j in range(Np):
        if j!= i:
            rj      =   Rt[j,:]
            mj      =   M[j]
            total   +=  acc_gravity(ri,rj,mj)

    return total

def acc_particle_total(M,Rt,RHOt,Pt,i,h,k,n,nu):
    """ returns total acceleration of particle

    Parameters
    ----------
    M:      1D array of particle masses
    Rt:     2D array of particle posiions at time t
    RHOt:   1D array of particle densities at time t
    Pt:     1D array of particle polytropic pressures at time t
    i:      index of particle
    h:      smoothing length
    k:      pressure constant
    n:      polytropic index
    nu:     damping constant

    Returns
    -------
    vector array
    """

    Np      =   len(M)

    # particle properties
    ri      =   Rt[i,:]
    rhoi    =   RHOt[i]
    Pi      =   Pt[i]
    bi      =   acc_particle_external(M,Rt,i)

    # interactions with other particles
    total   =   0
    for j in range(Np):
        if j != i:
            mj      =   M[j]
            rj      =   Rt[j,:]
            rhoj    =   RHOt[j]
            Pj      =   Pt[j]
            dWj     =   gradient_kernal_gauss(ri,rj,h)
            total   -=  mj * ( (Pi/rhoi**2) + (Pj/rhoj**2) ) * dWj

    return total + bi

#===============================================================================
""" Choosing dt and h """
#-------------------------------------------------------------------------------

def choose_h(Rt,alpha=1):
    """ choose smoothing length
    another method could be to choose h
    for each particle such that there are
    k variables enclosed by radius h around
    each particle

    Parameters
    ----------
    Rt:     2D array of particle positions at time t
    alpha:  ** factor to multiply result by

    Returns
    -------
    scalar
    """

    r1      =   np.average( Rt**2 )
    r2      =   np.average( Rt )**2
    return alpha * np.sqrt( r1 - r2 )

def choose_dt(h,V,F,alpha=1):
    """ choose time step

    Parameters
    ----------
    h:      smoothing length
    V:      1D array of particle velocities
    F:      1D array of forces on particle
    alpha:  ** factor to multiply result by, default = 1
    """

    vmax    =   np.max(V)
    fmax    =   np.max(F)

    t1      =   h / vmax
    t2      =   np.sqrt( h / fmax )
    tmin    =   np.min( t1 , t2 )
    return alpha * tmin

#===============================================================================
""" Setting up Model """
#-------------------------------------------------------------------------------

def model_params(Np,Mc,T,frac_H,frac_He,Nt=1000):
    """ set up model by creating a dictionary
    of model values

    Parameters
    ----------
    d:      number of dimentions
    Np:     number of particles
    Mc:     Mass of cloud [solar mass]

    Returns
    -------
    dictionary
    """

    # need to figure out
    k           =   1
    n           =   1

    # construct arrays
    M           =   np.zeros(Np)
    R           =   np.zeros((Nt,Np,3))
    dt          =   np.zeros(Nt)
    TIME        =   np.zeros_like(dt)
    h           =   np.zeros_like(dt)
    rho         =   np.zeros((Nt,Np))
    P           =   np.zeros_like(rho)

    # scalar values
    num_H       =   int(Np * frac_H)
    num_He      =   int(Np - num_H)
    mass        =   Mc / solarM.value
    mu          =   average_particle_mass(frac_H,frac_He)
    r_jean      =   dac.Jean_radius_M(mass,mu,T)

    # initialize arrays
    M[:num_H]   =   SI['m_H'].value
    M[num_H:]   =   SI['m_He'].value
    X,Y,Z       =   random_particle_placement(Np,r_jean)
    R[0,:,0]    =   X
    R[0,:,1]    =   Y
    R[0,:,2]    =   Z
    # dt[0]       =   NotImplemented
    TIME[0]     =   0
    h[0]        =   choose_h(R[0,:,:])
    rho[0]      =   np.array([ density(M,R[0],i,h[0]) for i in range(Np) ])
    P[0]        =   pressure(k,rho[0],n)

    dic      =   {'Np':      Np,
                 'temp0':   T,
                 'frac_H':  frac_H,
                 'frac_He': frac_He,
                 'num_H':   num_H,
                 'num_He':  num_He,
                 'Mc':      mass,
                 'mu':      mu,
                 'M':       M,
                 'r_jean':  r_jean,
                 'R':       R,
                 'dt':      dt,
                 'TIME':    TIME,
                 'h':       h,
                 'rho':     rho,
                 'P':       P}

    # convert dictionary to panda Series
    model   =   pd.Series(dic)
    return model

#===============================================================================
""" plotting """
#-------------------------------------------------------------------------------

# plotting parameters
pp  =   {'size':    (15,15),    # figsize
         'fs':      20,         # fontsize
         'cmap':    cm.magma    # color map
         }

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

def plot_3D_scatter(model,t):
    """ make a 3D scatter plot of
    particles at time t

    Parameters
    ----------
    model:      panda Series containing all model data
    t:          time index
    """

    # data
    R       =   model['R'][t] * pc.value
    X,Y,Z   =   [ R[:,e] for e in range(3) ]
    time    =   model['TIME'][t] * Myr.value
    Mc      =   model['Mc'] * solarM.value
    rho     =   model['rho'][t]
    limits  =   0,np.max(rho)

    # figure
    fig     =   plt.figure(figsize=pp['size'])
    ax      =   fig.add_subplot(111, projection='3d')
    ax.set_title("%s M$_\odot$ cloud, t = %s" % (Mc,time), fontsize=pp['fs']+2)
    ax.set_xlabel("x [%s]" % pc.units, fontsize=pp['fs'])
    ax.set_ylabel("y [%s]" % pc.units, fontsize=pp['fs'])
    ax.set_zlabel("z [%s]" % pc.units, fontsize=pp['fs'])

    levels  =   MaxNLocator(nbins=100).tick_values(*limits)
    cmap    =   pp['cmap']
    norm    =   BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    im      =   ax.scatter(X,Y,Z, c=rho, cmap=cmap)
    cbar    =   fig.colorbar(im, ax=ax, pad=0)
    cbar.set_label('$\\rho$ [ %s / %s$^2$ ]' % (dc.kg.units,dc.m.units), fontsize=pp['fs'])

    plt.tight_layout()
    plt.show()

def plot_2D_contourf(model,t):
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
    fig     =   plt.figure(figsize=pp['size'])
    ax      =   fig.add_subplot(111)
    ax.patch.set_facecolor('black')
    ax.set_title("%s M$_\odot$ cloud, t = %s" % (Mc,time), fontsize=pp['fs']+2)
    ax.set_xlabel("x [%s]" % pc.units, fontsize=pp['fs'])
    ax.set_ylabel("y [%s]" % pc.units, fontsize=pp['fs'])

    levels  =   MaxNLocator(nbins=100).tick_values(*limits)
    cmap    =   pp['cmap']
    norm    =   BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    im      =   ax.contourf(Xp,Yp,Zp, levels=levels, cmap=cmap)
    cbar    =   fig.colorbar(im, ax=ax, pad=0)
    cbar.set_label('$\\rho$ [ %s / %s$^2$ ]' % (dc.kg.units,dc.m.units), fontsize=pp['fs'])

    plt.tight_layout()
    plt.show()
