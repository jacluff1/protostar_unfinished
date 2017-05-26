import numpy as np
import pandas as pd
import auxillary as aux
import plot as plot
from djak.constants import solarM
import djak.astro.cloud as dac

#===============================================================================
""" Model Parameters """
#-------------------------------------------------------------------------------

# Assumed values
Np      =   100             # number of SPH particles
Nt      =   200             # number of time steps
Mc      =   1/solarM.value  # mass of cloud [solar mass -> kg]
T0      =   15              # temp of cloud [K]
frac_H  =   .75             # fraction of particles that are hydrogen
frac_He =   .25             # fraction of particles that are helium
n       =   1               # polytropic index
Lambda  =   1               # static damping coefficient
disp    =   2400            # velocity dispersion [ km / s ]
alpha   =   .1              # smoothing and time step coefficient

#===============================================================================
"""
1) Gather Model Parameters
2) Calculate Initial Conditions
"""
#-------------------------------------------------------------------------------

def construct_model():

    # assumed values
    d               =   {}
    d['Np']         =   Np
    d['Nt']         =   Nt
    d['Mc']         =   Mc
    d['T0']         =   T0
    d['frac_H']     =   frac_H
    d['frac_He']    =   frac_He
    d['n']          =   n
    d['Lambda']     =   Lambda
    d['disp']       =   disp
    d['alpha']      =   alpha

    # calculated scalar constants
    mu      =   aux.average_particle_mass(d)    # average particle mass
    mi      =   Mc/Np                           # SPH particle mass
    Rj      =   dac.Jean_radius_M(Mc,mu,T0)     # Jean's radius

    # create empty arrays
    # time , particle , dimention
    R       =   np.zeros(( Nt , Np , 3 ))       # SPH particle posisitons
    V       =   np.zeros_like(R)                # SPH particle velocities
    A       =   np.zeros_like(R)                # SPH particle accelerations
    a_grav  =   np.zeros_like(R)
    a_pres  =   np.zeros_like(R)
    a_cent  =   np.zeros_like(R)
    a_cori  =   np.zeros_like(R)
    a_magn  =   np.zeros_like(R)
    Omega   =   np.zeros_like(R)                # SPH particle rotation vectors
    grad_rho=   np.zeros_like(R)                # SPH density gradients

    # time , particle
    rho     =   np.zeros(( Nt , Np ))           # SPH particle mass densities
    P       =   np.zeros_like(rho)              # SPH particle pressure

    # time
    time    =   np.zeros( Nt )                  # time
    h       =   np.zeros_like(time)             # smoothing length
    dt      =   np.zeros_like(time)             # time step

    # particle
    K       =   np.zeros( Np )                  # SPH particle pressure constant

    # enter calculated scalars and arrays into dictionary
    d['mu']         =   mu
    d['mi']         =   mi
    d['Rj']         =   Rj
    d['R']          =   R
    d['V']          =   V
    d['A']          =   A
    d['a_grav']     =   a_grav
    d['a_pres']     =   a_pres
    d['a_cent']     =   a_cent
    d['a_cori']     =   a_cori
    d['a_magn']     =   a_magn
    d['Omega']      =   Omega
    d['grad_rho']   =   grad_rho
    d['rho']        =   rho
    d['P']          =   P
    d['time']       =   time
    d['h']          =   h
    d['dt']         =   dt
    d['K']          =   K

    # convert dictionary to panda Series
    model           =   pd.Series(d)

    return model

def initialize_model(model):

    Np      =   model['Np']

    # Initialize Functions
    aux.initial_rotation_vectors(model)
    aux.initial_SP_placement(model)
    aux.initial_SP_velocities(model)

    # Choosing dt and h
    aux.choose_h(0,model)

    # Cloud Physics
    aux.density(0,model)
    aux.gradient_density(0,model)
    aux.polytropic_pressure_constant(model)
    aux.pressure(0,model)

    # Equations of Motion
    aux.acc_total(0,model)

    # Choosing dt and h
    aux.choose_dt(0,model)

#===============================================================================
""" Integrate Constructed Model over Time """
#-------------------------------------------------------------------------------

def integrate_model():

    model       =   construct_model()
    initialize_model(model)

    return model

#===============================================================================
""" Print Test Results """
#-------------------------------------------------------------------------------

def acc_test(model):

    Np      =   model['Np']

    def test_key(key):
        A       =   model[key][0,:,:]
        A1      =   np.array([ np.linalg.norm( A[i,:]) for i in range(Np) ])
        return np.average(A1)

    print("grav",test_key('a_grav'))
    print("pres",test_key('a_pres'))
    print("cent",test_key('a_cent'))
    print("cori",test_key('a_cori'))
