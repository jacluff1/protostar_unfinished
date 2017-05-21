import numpy as np
import djak.constants as dc
from djak.constants import SI, solarM, pc, Myr, gr
import djak.astro.cloud as dac
import pdb
import pandas as pd

#===============================================================================
""" Initializing Functions """
#-------------------------------------------------------------------------------

def average_particle_mass(par):
    """ find the average particle mass
    from particle populations

    Parameters
    ----------
    par:    dictionary of model parameters

    Returns
    -------
    scalar
    """
    frac_H  =   par['frac_H']
    frac_He =   par['frac_He']

    return frac_H*SI['m_H'].value + frac_He*SI['m_He'].value

def polytropic_pressure_constant(par):
    """ find the polytropic pressure constant
    1) use PV = (M/mu) * k T
    2) solve for P
    3) P = K rho^( 1 + 1/n )
    4) solve for K

    Parameters
    ----------
    par:    dictionary of model parameters

    Returns
    -------
    scalar
    """

    Mc      =   par['Mc']
    mu      =   par['mu']
    k       =   SI['k']
    T0      =   par['T0']
    Rj      =   par['Rj']
    n       =   par['n']

    V       =   (4/3) * np.pi * Rj**3

    return ( Mc / mu) * ( k * T0 / V ) * 1/rho0**( 1 + 1/n )

def initial_angular_frequency(par):
    """ find initial angular frequency
    http://www.am.ub.edu/~robert/master/03-clouds.pdf

    Parameters
    ----------
    par:    dictionary of model parameters

    Returns
    -------
    scalar
    """

    Rj      =   par['Rj'] * pc.value

    return Rj**(-.56) * 10**(.34) * 1e-14

def initial_SP_placement(par):
    """ generate randome particle positions
    for all particles

    Parameters
    ----------
    par:    dictionary of model parameters

    Returns
    -------
    2D array
    """

    Rj     =   par['Rj']

    # set up random positions in spherical coordinates
    radius  =   np.random.rand(Np) * Rj
    theta   =   np.random.rand(Np) * np.pi
    phi     =   np.random.rand(Np) * 2*np.pi

    # convert spherical coordinates to cartesian
    X       =   radius * np.sin(theta) * np.cos(phi)
    Y       =   radius * np.sin(theta) * np.sin(phi)
    Z       =   radius * np.cos(theta)

    R0      =   np.vstack((X,Y,Z)).T

    return R0

def initial_SP_velocities(par):
    """ find initial velocities
    1) random velocities with some average dispersion
    2) motion includes a net rotation around z-axis

    Parameters
    ----------
    par:    dictionary of model parameters

    Returns
    -------
    tuple of 3 1D arrays
    """

    R0      =   par['R0']
    Omega0  =   par['Omega0'] * np.array([ 0 , 0 , 1 ])
    Np      =   par['Np']
    disp    =   par['disp']

    Z0      =   R0[:,2]
    Rz      =   R0 - Z0
    Rz_mag  =   np.array([ np.linalg.norm( Rz[i,:] ) for i in range(Np) ])
    Vrot    =   np.cross( Omega0 , Rz )
    Vrotx   =   np.dot( Vrot , np.array([ 1 , 0 , 0 ]))
    Vroty   =   np.dot( Vrot , np.array([ 0 , 1 , 0 ]))

    Vx      =   np.random.rand(Np) * disp + Vrotx
    Vy      =   np.random.rand(Np) * disp + Vroty
    Vz      =   np.random.rand(Np) * disp

    V0      =   np.vstack((Vx,Vy,Vz)).T
    return V0

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

    Parameters
    ----------
    ri:     object particle position
    rj:     agent particle position
    h:      smoothing length

    Returns
    -------
    scalar
    """
    # rmag,rhat   =   r_vec(ri,rj)
    rmag = np.linalg.norm( ri - rj )

    return ( h * np.sqrt(np.pi) )**(-3) * np.exp( -rmag**2 / h**2 )

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

def pressure(rho,par):
    """ find the pressure of polytropic fluid

    Parameters
    ----------
    rho:    density
    par:    dictionary of model parameters

    Returns
    -------
    scalar - pressure
    """

    Kj      =   par['Kj']
    n       =   par['n']

    return Kj * rho**( 1 + 1/n )

def density(Rt,r,h,par):
    """ find density of SPH

    Parameters
    ----------
    Rt:     2D SPH particle positions at time t
    r:      position vector of interest
    h:      smoothing length
    par:    dictionary of model parameters

    Returns
    -------
    scalar
    """

    Mc      =   par['Mc']
    Np      =   par['Np']

    # mnras181-0375.pdf
    total   =   0
    for j in range(Np):
        rj      =   Rt[j,:]
        total   +=  kernal_gauss(r,rj,h)

    rho     =   (Mc/Np) * total
    return rho

def gradient_density(Rt,r,h,par):
    """ find the density gradient
    calculated by hand
    multipy rho by -(2/h^2) and sum uj vectors

    Parameters
    ----------
    Rt:     2D SPH positions at time t
    r:      position vector
    h:      smoothing length
    par:    dictionary of model parameters

    returns
    vector array
    """

    Np      =   par['Np']

    rho     =   density(Rt,r,h,par)
    total   =   0
    for j in range(Np):
        rj      =   Rt[j]
        uj      =   r - rj
        total   +=  uj

    return - ( 2 / h**2 ) * rho * total

#===============================================================================
""" Equations of Motion """
#-------------------------------------------------------------------------------

def acc_gravity(Rt,r,h,par):
    """ find acceleration from Gaussian smoothing
    gravitational potential

    Parameters
    ----------
    Rt:     SPH particle positions (2D array)
    r:      position vector
    h:      smoothing length
    par:    dictionar of model parameters

    Returns
    -------
    vector array
    """

    f       =   1 / h**2
    G       =   SI['G']
    Mc      =   par['Mc']
    Np      =   par['Np']

    total   =   0
    for j in range(Np):
        u           =   r - Rt[j]
        uj,ujhat    =   r_vec(u)
        one         =   ( 2 / uj ) * np.sqrt( f / np.pi )
        two_1       =   np.exp( -f * uj**2 )
        two_2       =   np.sqrt( np.pi / f ) / ( 2 * uj ) * erf( np.sqrt(f) * uj )
        two         =   two_1 - two_2
        total       +=  one * two * ujhat

    return - ( G * Mc / Np ) * total

def acc_polytrope_pressure(rho,grad_rho,par):
    """ acceleration from polytropic pressure

    Parameters
    ----------
    K:          polytropic pressure constant
    n:          polytropic index
    rho:        density
    grad_rho:   gradient of density

    Returns
    -------
    vector array
    """

    Kj      =   par['Kj']
    n       =   par['n']

    return - K * rho**( 1/n - 1 ) * ( grad_rho / n ) * ( 1 + n )

def acc_centrifugal(Omega,rj):
    """ acceleration from centrifugal force

    Parameters
    ----------
    Omega:      angular velocity
    rj:         position of SPH particle

    Returns
    -------
    velocity array
    """

    return np.cross( np.cross(Omega,rj) , Omega )

def acc_coriolis(Omega,rj_dot):
    """ acceleration from coriolis force

    Parameters
    ----------
    Omega:      angular velocity
    r:          position (of SPH particle?)

    Returns
    -------
    vector array
    """

    return 2 * np.cross( rj_dot , Omega )

def acc_damping(rj_dot,par):
    """ acceleration from internal friction

    Parameters
    ----------
    rj_dot: velociy of SPH particle j
    par:    dictionary of model parameters

    Returns
    -------
    vector array
    """

    Lambda      =   par['Lambda']

    return - Lambda * rj_dot

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

# def acc_particle_external(M,Rt,i):
#     """ find acceleration of particle
#     due to external forces
#
#     Parameters
#     ----------
#     M:      1D array of particles masses
#     Rt:     2D array of particle positions at time t
#     i:      index of particle
#
#     Returns
#     -------
#     vector array
#     """
#
#     # # particle information
#     # Np      =   len(M)
#     # ri      =   Rt[i,:]
#     #
#     # # interactions with other particles
#     # total   =   0
#     # for j in range(Np):
#     #     if j!= i:
#     #         rj      =   Rt[j,:]
#     #         mj      =   M[j]
#     #         total   +=  acc_gravity(ri,rj,mj)
#     #
#     # return

def acc_particle(Rt,Vt,rhot,rho_gradt,i,h,par):
    # """ returns total acceleration of particle
    #
    # Parameters
    # ----------
    # Rt:     2D array of SPH particle posiions at time t
    # Vt:     2D array of SPH particle velocities at time t
    # i:      index of SPH particle
    # h:      smoothing length
    # par:    dictionar of model parameters
    #
    # Returns
    # -------
    # vector array
    # """

    ri          =   Rt[i]
    rho         =   density(Rt,ri,h,par)


    gravity     =   acc_gravity(Rt,ri,h,par)
    pressure    =   acc_polytrope_pressure(rho,grad_rho,par)




    return 

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

# #===============================================================================
# """ Setting up Model """
# #-------------------------------------------------------------------------------
#
# def model_params(Np,Mc,T,frac_H,frac_He,Nt=1000):
#     """ set up model by creating a dictionary
#     of model values
#
#     Parameters
#     ----------
#     d:      number of dimentions
#     Np:     number of particles
#     Mc:     Mass of cloud [solar mass]
#
#     Returns
#     -------
#     dictionary
#     """
#
#     # need to figure out
#     k           =   1
#     n           =   1
#
#     # construct arrays
#     R           =   np.zeros((Nt,Np,3))
#     dt          =   np.zeros(Nt)
#     TIME        =   np.zeros_like(dt)
#     h           =   np.zeros_like(dt)
#     rho         =   np.zeros((Nt,Np))
#     P           =   np.zeros_like(rho)
#
#     # scalar values
#     Mc          /=  solarM.value
#     mu          =   average_particle_mass(frac_H,frac_He)
#     mi          =   Mc/Np
#     r_jean      =   dac.Jean_radius_M(Mc,mu,T)
#
#     # initialize arrays
#     X,Y,Z       =   random_particle_placement(Np,r_jean)
#     R[0,:,0]    =   X
#     R[0,:,1]    =   Y
#     R[0,:,2]    =   Z
#     # # dt[0]       =   NotImplemented
#     TIME[0]     =   0
#     h[0]        =   choose_h(R[0,:,:])
#     rho[0]      =   np.array([ density(Mc,Np,R[0,:,:],R[0,i,:],h[0]) for i in range(Np) ])
#     P[0]        =   pressure(k,rho[0],n)
#
#     dic      =   {'Np':     Np,
#                  'temp0':   T,
#                  'frac_H':  frac_H,
#                  'frac_He': frac_He,
#                  'Mc':      Mc,
#                  'mu':      mu,
#                  'mi':      mi,
#                  'r_jean':  r_jean,
#                  'R':       R,
#                 #  'dt':      dt,
#                  'TIME':    TIME,
#                  'h':       h,
#                  'rho':     rho,
#                  'P':       P}
#
#     # convert dictionary to panda Series
#     model   =   pd.Series(dic)
#     return model
