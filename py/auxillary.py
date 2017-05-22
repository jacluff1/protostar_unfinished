import numpy as np
import djak.constants as dc
from djak.constants import SI, solarM, pc, Myr, gr
import djak.astro.cloud as dac
import pdb
import pandas as pd

#===============================================================================
""" Initializing Functions """
#-------------------------------------------------------------------------------

def average_particle_mass(p):
    """ find the average particle mass
    from particle populations

    args
    ----
    p:  dictionary  -   assumed values

    returns
    -------
    scalar
    """

    frac_H  =   p['frac_H']
    frac_He =   p['frac_H']

    return frac_H*SI['m_H'].value + frac_He*SI['m_He'].value

def initial_rotation_vectors(model):
    """ find initial angular frequency
    http://www.am.ub.edu/~robert/master/03-clouds.pdf

    args
    ----
    model:  Series   -   model data
    """

    Np      =   model['Np']
    Rj      =   model['Rj'] * pc.value

    omag    =   Rj**(-.56) * 10**(.34) * 1e-14
    ohats   =   np.zeros(( Np , 3 ))

    Omega0  =   omag * ohats

    model['Omega'][0,:,:]   =   Omega0

def initial_SP_placement(model):
    """ generate randome particle positions
    for all particles

    args
    ----
    model:  Series   -   model data
    """

    Np      =   model['Np']
    Rj      =   model['Rj']

    # set up random positions in spherical coordinates
    radius  =   np.random.rand(Np) * Rj
    theta   =   np.random.rand(Np) * np.pi
    phi     =   np.random.rand(Np) * 2*np.pi

    # convert spherical coordinates to cartesian
    x       =   radius * np.sin(theta) * np.cos(phi)
    y       =   radius * np.sin(theta) * np.sin(phi)
    z       =   radius * np.cos(theta)

    r0      =   np.vstack((x,y,z)).T

    model['r'][0,:,:]   =   r0

def initial_SP_velocities(model):
    """ find initial velocities
    1) random velocities with some average dispersion
    2) motion includes a net rotation around z-axis

    args
    ----
    model:  Series  -   model data
    """

    Np      =   model['Np']
    disp    =   model['disp']
    r0      =   model['r'][0,:,:]
    Omega0  =   model['Omega'][0,:,:]

    vrot    =   np.array([ np.cross( Omega0[i] , r0[i] ) for i in range(Np) ])
    vrotx   =   np.array([ np.dot( vrot[i] , np.array([ 1 , 0 , 0 ]) ) for i in range(Np) ])
    vroty   =   np.array([ np.dot( vrot[i] , np.array([ 0 , 1 , 0 ]) ) for i in range(Np) ])

    vx      =   np.random.rand(Np) * disp + vrotx
    vy      =   np.random.rand(Np) * disp + vroty
    vz      =   np.random.rand(Np) * disp

    v0      =   np.vstack((vx,vy,vz)).T

    model['v'][0,:,:]   =   v0

def polytropic_pressure_constant(model):
    """ find the polytropic pressure constant
    1) use PV = (M/mu) * k T
    2) solve for P
    3) P = K rho^( 1 + 1/n )
    4) solve for K

    args
    ----
    model:  Series  -   model data
    """

    k       =   SI['k'].value

    Mc      =   model['Mc']
    T0      =   model['T0']
    Rj      =   model['Rj']
    n       =   model['n']
    mu      =   model['mu']
    rho0    =   model['rho'][0,:]

    V       =   (4/3) * np.pi * Rj**3

    K       =  ( Mc / mu) * ( k * T0 / V ) * 1/rho0**( 1 + 1/n )

    model['K']  =   K

#===============================================================================
""" Misc """
#-------------------------------------------------------------------------------

def r_vec(ri,rj):
    """ find the relative position vector

    args
    ----
    r0:     vector  -   particle position
    ri:     vector  -   particle position

    returns
    -------
    tuple - rmag,rhat
    """

    r       =   ri - rj
    rmag    =   np.linalg.norm(r)
    assert rmag > 0, "rmag = 0, ri != rj"
    rhat    =   r / rmag
    return rmag,rhat

# def gradient():
#     """ find the gradient of a vector
#
#     Parameters
#     ----------
#
#
#     Returns
#     -------
#     vector array
#     """

#===============================================================================
""" Kernal Smoothing """
#-------------------------------------------------------------------------------

def kernal_gauss(ri,rj,h):
    """ Gaussian method for kernal Smoothing

    args
    ----
    ri:     vector  -   object particle position
    rj:     vector  -   agent particle position
    h:      scalar  -   smoothing length

    returns
    -------
    scalar
    """

    rmag = np.linalg.norm( ri - rj )

    return ( h * np.sqrt(np.pi) )**(-3) * np.exp( -rmag**2 / h**2 )

def gradient_kernal_gauss(ri,rj,h):
    """ returns the gradient of the
    gaussian kernal smoothing function

    args
    ----
    ri:     vector  -   object particle position
    rj:     vector  -   agent particle position
    h:      scalar  -   kernal smoothing length

    returns
    -------
    vector
    """

    W       =   kernal_gauss(ri,rj,h)
    return (2 / h**2) * W * rj

#===============================================================================
""" Cloud Physics """
#-------------------------------------------------------------------------------

def density(r,rdot,t,i,model):
    """ find density of SPH

    args
    ----
    r:      vector  -   position of interest
    t:      int     -   time index
    i:      int     -   particle index
    model:  Series  -   model data

    returns
    -------
    scalar
    """

    Mc      =   model['Mc']
    Np      =   model['Np']
    rt      =   model['r'][t,:,:]
    h       =   model['h'][t]

    # mnras181-0375.pdf
    total   =   0
    for j in range(Np):
        rj      =   rt[j,:]
        total   +=  kernal_gauss(r,rj,h)

    rho     =   (Mc/Np) * total
    return rho

def pressure(r,rdot,t,i,model):
    """ find the pressure of polytropic fluid

    args
    ----
    r:      vector  -   position of interest
    t:      int     -   time index
    i:      int     -   particle index
    model:  Series  -   model data

    returns
    -------
    scalar
    """

    K       =   model['K'][i]
    n       =   model['n']
    rho     =   model['rho'][t,i]

    return K * rho**( 1 + 1/n )

def gradient_density(r,rdot,t,i,model):
    """ find the density gradient
    calculated by hand
    multipy rho by -(2/h^2) and sum uj vectors

    args
    ----
    r:      vector  -   position of interest
    t:      int     -   time index
    i:      int     -   particle index
    model:  Series  -   model data

    returns
    -------
    vector
    """

    Np      =   model['Np']
    rt      =   model['r'][t,:,:]
    h       =   model['h'][t]
    rho     =   model['rho'][t,i]

    total   =   0
    for j in range(Np):
        rj      =   rt[j,:]
        uj      =   r - rj
        total   +=  uj

    return - ( 2 / h**2 ) * rho * total

#===============================================================================
""" Equations of Motion """
#-------------------------------------------------------------------------------

def acc_gravity(r,rdot,t,i,model):
    """ find acceleration from Gaussian smoothing
    gravitational potential

    args
    ----
    r:      vector  -   position of interest
    t:      int     -   time index
    i:      int     -   particle index
    model:  Series  -   model data

    returns
    -------
    vector
    """

    G       =   SI['G']
    rt      =   model['r'][t,:,:]
    Mc      =   model['Mc']
    Np      =   model['Np']
    h       =   model['h'][t]
    f       =   1 / h**2

    total   =   0
    for j in range(Np):
        u           =   r - rt[j]
        uj,ujhat    =   r_vec(u)
        one         =   ( 2 / uj ) * np.sqrt( f / np.pi )
        two_1       =   np.exp( -f * uj**2 )
        two_2       =   np.sqrt( np.pi / f ) / ( 2 * uj ) * erf( np.sqrt(f) * uj )
        two         =   two_1 - two_2
        total       +=  one * two * ujhat

    return - ( G * Mc / Np ) * total

def acc_polytrope_pressure(r,rdot,t,i,model):
    """ acceleration from polytropic pressure

    args
    ----
    r:      vector  -   position of interest
    t:      int     -   time index
    i:      int     -   particle index
    model:  Series  -   model data

    returns
    -------
    vector
    """

    n           =   model['n']
    rho         =   model['rho'][t,i]
    grad_rho    =   model['grad_rho'][t,i,:]
    K           =   model['K'][i]

    return K * rho**( 1/n - 1 ) * ( grad_rho / n ) * ( 1 + n )

def acc_centrifugal(r,rdot,t,i,model):
    """ acceleration from centrifugal force

    args
    ----
    r:      vector  -   position of interest
    t:      int     -   time index
    i:      int     -   particle index
    model:  Series  -   model data

    returns
    -------
    vector
    """

    Omega   =   model['Omega'][t,i,:]

    return np.cross( Omega , np.cross(Omega,r) )

def acc_coriolis(r,rdot,t,i,model):
    """ acceleration from coriolis force

    args
    ----
    r:      vector  -   position of interest
    t:      int     -   time index
    i:      int     -   particle index
    model:  Series  -   model data

    returns
    -------
    vector
    """

    Omega   =   model['Omega'][t,i,:]

    return 2 * np.cross( rdot , Omega )

def acc_damping(r,rdot,t,i,model):
    """ acceleration from internal friction

    args
    ----
    r:      vector  -   position of interest
    t:      int     -   time index
    i:      int     -   particle index
    model:  Series  -   model data

    returns
    -------
    vector
    """

    Lambda      =   model['Lambda']

    return Lambda * rdot

# def acc_magnetic(J,B,rho):
#     """ acceleration from magnetic force
#
#     Parameters
#     ----------
#     J:      current density
#     B:      magnetic field
#     rho:    density?
#
#     Returns
#     -------
#     vector array
#     """
#     raise NotImplementedError
#     return np.cross( J , B ) / rho

def acc_total(r,rdot,t,i,model):
    """ returns total acceleration of particle

    args
    ----
    r:      vector  -   position of interest
    t:      int     -   time index
    i:      int     -   particle index
    model:  Series  -   model data

    returns
    -------
    vector
    """

    damping     =   acc_damping(r,rdot,t,i,model)
    pressure    =   acc_polytrope_pressure(r,rdot,t,i,model)
    gravity     =   acc_gravity(r,rdot,t,i,model)
    centrifugal =   acc_centrifugal(r,rdot,t,i,model)
    coriolis    =   acc_coriolis(r,rdot,t,i,model)
    magnetic    =   0

    return - (damping + pressure + gravity + centrifugal + coriolis) + magnetic

#===============================================================================
""" Choosing dt and h """
#-------------------------------------------------------------------------------

def choose_h(r,rdot,t,i,model):
    """ choose smoothing length
    another method could be to choose h
    for each particle such that there are
    k variables enclosed by radius h around
    each particle

    args
    ----
    r:      vector  -   position of interest
    t:      int     -   time index
    i:      int     -   particle index
    model:  Series  -   model data

    returns
    -------
    vector
    """

    r       =   model['r'][t,:,:]
    Np      =   model['Np']
    alpha   =   model['alpha']

    r1      =   np.average( np.array([ np.linalg.norm(r[j,:])**2 for j in range(Np) ]) )
    r2      =   np.average( np.array([ np.linalg.norm(r[j,:]) for j in range(Np) ]) )**2

    return alpha * np.sqrt( r1 - r2 )

def choose_dt(h,V,F,par,alpha=.1):
    """ choose time step

    args
    ----
    r:      vector  -   position of interest
    t:      int     -   time index
    i:      int     -   particle index
    model:  Series  -   model data

    returns
    -------
    vector
    """

    Np      =   model['Np']
    h       =   model['h'][t]
    v       =   model['v'][t,:,:]
    F       =   model['a'][t,:,:] * model['mi']

    vmags   =   np.array([ np.linalg.norm(v[j,:]) for j in range(Np) ])
    Fmags   =   np.array([ np.linalg.norm(F[j,:]) for j in range(Np) ])

    vmax    =   np.max( vmags )
    fmax    =   np.max( Fmags )

    t1      =   h / vmax
    t2      =   np.sqrt( h / fmax )
    tmin    =   np.min( t1 , t2 )

    return alpha * tmin
