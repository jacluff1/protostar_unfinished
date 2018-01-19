import numpy as np
import djak.constants as dc
from djak.constants import SI, solarM, pc, Myr, gr
import djak.astro.cloud as dac
import pdb
import pandas as pd
from scipy.special import erf

#===============================================================================
"""
Initializing Functions
f(p) & f(model)
no returns, only initilizes model
"""
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

    returns
    -------
    None - initializes model
    """

    Np      =   model['Np']
    Rj      =   model['Rj'] * pc.value

    omag    =   Rj**(-.56) * 10**(.34) * 1e-14
    ohats   =   np.zeros(( Np , 3 ))
    ohats[:,2] = 1

    Omega0  =   omag * ohats

    model['Omega'][0,:,:]   =   Omega0

def initial_SP_placement(model):
    """ generate randome particle positions
    for all particles

    args
    ----
    model:  Series   -   model data

    returns
    -------
    None - initializes model
    """

    Np      =   model['Np']
    Rj      =   model['Rj']

    # set up random positions in spherical coordinates
    radius  =   np.random.rand(Np) * Rj
    theta   =   np.random.rand(Np) * np.pi
    phi     =   np.random.rand(Np) * 2*np.pi

    # convert spherical coordinates to cartesian
    X       =   radius * np.sin(theta) * np.cos(phi)
    Y       =   radius * np.sin(theta) * np.sin(phi)
    Z       =   radius * np.cos(theta)

    R0      =   np.vstack((X,Y,Z)).T

    model['R'][0,:,:]   =   R0

def initial_SP_velocities(model):
    """ find initial velocities
    1) random velocities with some average dispersion
    2) motion includes a net rotation around z-axis

    args
    ----
    model:  Series  -   model data

    returns
    -------
    None - initializes model
    """

    Np      =   model['Np']
    disp    =   model['disp']
    R0      =   model['R'][0,:,:]
    Omega0  =   model['Omega'][0,:,:]

    Vrot    =   np.array([ np.cross( Omega0[i] , R0[i] ) for i in range(Np) ])
    Vrotx   =   np.array([ np.dot( Vrot[i] , np.array([ 1 , 0 , 0 ]) ) for i in range(Np) ])
    Vroty   =   np.array([ np.dot( Vrot[i] , np.array([ 0 , 1 , 0 ]) ) for i in range(Np) ])

    Vx      =   np.random.rand(Np) * disp + Vrotx
    Vy      =   np.random.rand(Np) * disp + Vroty
    Vz      =   np.random.rand(Np) * disp

    V0      =   np.vstack((Vx,Vy,Vz)).T

    model['V'][0,:,:]   =   V0

def polytropic_pressure_constant(model):
    """ find the polytropic pressure constant
    1) use PV = (M/mu) * k T
    2) solve for P
    3) P = K rho^( 1 + 1/n )
    4) solve for K

    args
    ----
    model:  Series  -   model data

    returns
    -------
    None - initializes model
    """

    k       =   SI['k'].value

    Mc      =   model['Mc']
    T0      =   model['T0']
    Rj      =   model['Rj']
    n       =   model['n']
    mu      =   model['mu']
    rho0    =   model['rho'][0,:]
    Np      =   model['Np']

    V       =   (4/3) * np.pi * Rj**3

    K       =   np.array([ ( Mc / mu) * ( k * T0 / V ) * 1/rho0[i]**( 1 + 1/n ) for i in range(Np) ])

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

def density_r(r,t,i,model):
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

    mi      =   model['mi']
    Rt      =   model['R'][t,:,:]
    h       =   model['h'][t]
    Np      =   model['Np']

    # mnras181-0375.pdf (2.5)
    total   =   0
    for j in range(Np):
        rj      =   Rt[j,:]
        total   +=  kernal_gauss(r,rj,h)

    rho     =   mi * total
    return rho

#===============================================================================
"""
Kernal Smoothing
f(ri,rj,h)
smoothing method for model - Gaussian
"""
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

    # mnras181-0375.pdf (2.10 i)
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

    # derived from mnras181-0375.pdf (2.10 i)
    uj,ujhat=   r_vec(ri,rj)
    W       =   kernal_gauss(ri,rj,h)
    return (2 / h**2) * W * ujhat

#===============================================================================
"""
Cloud Physics
f(t,i,model)
"""
#-------------------------------------------------------------------------------

def density(t,model):
    """ find density of SPH

    args
    ----
    t:      int     -   time index
    i:      int     -   particle index
    model:  Series  -   model data

    returns
    -------
    None -  fills in model at time t
    """

    mi      =   model['mi']
    Rt      =   model['R'][t,:,:]
    h       =   model['h'][t]
    Np      =   model['Np']

    for i in range(Np):
        ri      =   Rt[i,:]
        # mnras181-0375.pdf (2.5)
        total   =   0
        for j in range(Np):
            rj      =   Rt[j,:]
            total   +=  kernal_gauss(ri,rj,h)

        model['rho'][t,i]   =   mi * total

def gradient_density(t,model):
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
    None - fills in model at time t
    """

    Np      =   model['Np']
    Rt      =   model['R'][t,:,:]
    h       =   model['h'][t]
    mi      =   model['mi']

    # derived from mnras181-0375.pdf (2.5)
    for i in range(Np):
        ri      =   Rt[i,:]
        total   =   0
        for j in range(Np):
            if j != i:
                rj      =   Rt[j,:]
                total   +=  gradient_kernal_gauss(ri,rj,h)
        model['grad_rho'][t,i,:]    =   mi * total

def pressure(t,model):
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

    # pmocz_sph.pdf (2)
    K       =   model['K'][:]
    n       =   model['n']
    rhot    =   model['rho'][t,:]

    model['P'][t,:]   =   K * rhot**( 1 + 1/n )

#===============================================================================
"""
Equations of Motion
f(t,i,model)
"""
#-------------------------------------------------------------------------------

def acc_gravity(t,i,model):
    """ find acceleration from Gaussian smoothing
    gravitational potential

    args
    ----
    t:      int     -   time index
    i:      int     -   particle index
    model:  Series  -   model data

    returns
    -------
    vector
    """

    G       =   SI['G'].value
    Rt      =   model['R'][t,:,:]
    Mc      =   model['Mc']
    Np      =   model['Np']
    h       =   model['h'][t]

    # derived from mnras181-0375.pdf (2.21)
    f       =   1 / h**2
    ri      =   Rt[i,:]
    total   =   0
    for j in range(Np):
        if j != i:
            rj          =   Rt[j,:]
            uj,ujhat    =   r_vec(ri,rj)
            one         =   ( 2 / uj ) * np.sqrt( f / np.pi )
            two_1       =   np.exp( -f * uj**2 )
            two_2       =   np.sqrt( np.pi / f ) / ( 2 * uj ) * erf( np.sqrt(f) * uj )
            two         =   two_1 - two_2
            total       +=  one * two * ujhat

    return - ( G * Mc / Np ) * total

def acc_polytrope_pressure(t,i,model):
    """ acceleration from polytropic pressure

    args
    ----
    t:      int     -   time index
    i:      int     -   particle index
    model:  Series  -   model data

    returns
    -------
    vector
    """

    # mnras181-0375.pdf (3.1)
    n           =   model['n']
    rhoi        =   model['rho'][t,i]
    grad_rhoi   =   model['grad_rho'][t,i,:]
    K           =   model['K'][i]

    return K * rhoi**( 1/n - 1 ) * ( grad_rhoi / n ) * ( 1 + n )

def acc_centrifugal(t,i,model):
    """ acceleration from centrifugal force

    args
    ----
    t:      int     -   time index
    i:      int     -   particle index
    model:  Series  -   model data

    returns
    -------
    vector
    """

    # mnras181-0375.pdf (3.1)
    Np      =   model['Np']
    Omegati =   model['Omega'][t,i,:]
    Rti     =   model['R'][t,i,:]

    return np.cross( Omegati , np.cross(Omegati,Rti) )

def acc_coriolis(t,i,model):
    """ acceleration from coriolis force

    args
    ----
    t:      int     -   time index
    i:      int     -   particle index
    model:  Series  -   model data

    returns
    -------
    vector
    """

    # mnras181-0375.pdf (3.1)
    Np      =   model['Np']
    Omegati =   model['Omega'][t,i,:]
    Vti     =   model['V'][t,i,:]

    return 2 * np.cross( Vti , Omegati )

def acc_damping(t,i,model):
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

    # mnras181-0375.pdf (3.1)
    Np          =   model['Np']
    Lambda      =   model['Lambda']
    Vti         =   model['V'][t,i,:]

    return Lambda * Vti

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

def acc_total(t,model):
    """ finds total acceleration of particle

    args
    ----
    r:      vector  -   position of interest
    t:      int     -   time index
    i:      int     -   particle index
    model:  Series  -   model data

    returns
    -------
    None - updates model at time t
    """

    Np      =   model['Np']
    for i in range(Np):

        # damping     =   acc_damping(t,i,model)
        pressure    =   acc_polytrope_pressure(t,i,model)
        gravity     =   acc_gravity(t,i,model)
        centrifugal =   acc_centrifugal(t,i,model)
        coriolis    =   acc_coriolis(t,i,model)
        magnetic    =   0

        # A           =   - (damping + pressure + gravity + centrifugal + coriolis) + magnetic
        # A           =   - (pressure + gravity + centrifugal + coriolis) + magnetic
        A           =   - gravity
        model['A'][t,i,:]       =   A
        model['a_grav'][t,i,:]  =   gravity
        model['a_pres'][t,i,:]  =   pressure
        model['a_cent'][t,i,:]  =   centrifugal
        model['a_cori'][t,i,:]  =   coriolis
        model['a_magn'][t,i,:]  =   magnetic

#===============================================================================
"""
Choosing dt and h
f(t,model)
h and dt are consistent for all particles in time slice """
#-------------------------------------------------------------------------------

def choose_h(t,model):
    """ choose smoothing length
    another method could be to choose h
    for each particle such that there are
    k variables enclosed by radius h around
    each particle

    args
    ----
    t:      int     -   time index
    model:  Series  -   model data

    returns
    -------
    None - updates model
    """

    # pmocz_sph.pdf (20)
    R       =   model['R'][t,:,:]
    Np      =   model['Np']
    alpha   =   model['alpha']

    r1      =   np.average( np.array([ np.linalg.norm(R[j,:])**2 for j in range(Np) ]) )
    r2      =   np.average( np.array([ np.linalg.norm(R[j,:]) for j in range(Np) ]) )**2

    h       =   alpha * np.sqrt( r1 - r2 )
    model['h'][t]  =   h

def choose_dt(t,model):
    """ choose time step

    args
    ----
    t:      int     -   time index
    model:  Series  -   model data

    returns
    -------
    None - updates model
    """

    # pmocz_sph.pdf (19)
    Np      =   model['Np']
    h       =   model['h'][t]
    V       =   model['V'][t,:,:]
    F       =   model['A'][t,:,:] * model['mi']
    alpha   =   model['alpha']

    vmags   =   np.array([ np.linalg.norm(V[j,:]) for j in range(Np) ])
    fmags   =   np.array([ np.linalg.norm(F[j,:]) for j in range(Np) ])

    vmax    =   np.max( vmags )
    fmax    =   np.max( fmags )

    t1      =   h / vmax
    t2      =   np.sqrt( h / fmax )
    tmin    =   min( t1 , t2 )

    dt      =   alpha * tmin
    model['dt'][t]  =   dt
    print(t1,t2,dt)
