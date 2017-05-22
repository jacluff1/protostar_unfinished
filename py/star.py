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
Np      =   1000            # number of SPH particles
Nt      =   1000            # number of time steps
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
    r       =   np.zeros(( Nt , Np , 3 ))       # SPH particle posisitons
    v       =   np.zeros_like(r)                # SPH particle velocities
    a       =   np.zeros_like(r)                # SPH particle accelerations
    Omega   =   np.zeros_like(r)                # SPH particle rotation vectors
    grad_rho=   np.zeros_like(r)                # SPH density gradients

    # time , particle
    rho     =   np.zeros(( Nt , Np ))           # SPH particle mass densities
    p       =   np.zeros_like(rho)              # SPH particle pressure

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
    d['r']          =   r
    d['v']          =   v
    d['a']          =   a 
    d['Omega']      =   Omega
    d['grad_rho']   =   grad_rho
    d['rho']        =   rho
    d['p']          =   p
    d['time']       =   time
    d['h']          =   h
    d['dt']         =   dt
    d['K']          =   K

    # convert dictionary to panda Series
    model           =   pd.Series(d)

    return model

def initialize_model(model):

    aux.initial_rotation_vectors(model)
    aux.initial_SP_placement(model)
    aux.initial_SP_velocities(model)





    # # 2D            -   initial rotation vectors
    # Omega0          =   aux.initial_rotation_vectors(d)
    # d['Omega0']     =   Omega0
    #
    # # 2D            -   initial SP placement
    # R0              =   aux.initial_SP_placement(d)
    # d['R0']         =   R0
    #
    # # 2D            -   initial SP speed
    # V0              =   aux.initial_SP_velocities(par)
    # d['V0']         =   V0
    #
    # # scalar        -   initial smoothing length
    # h0              =   aux.choose_h(R0)
    # par['h0']       =   h0
    #
    # # 1D            -   initial mass density
    # rho0            =   np.array([ aux.density(R0,R0[j],h0,par) for j in range(Np) ])
    # par['rho0']     =   rho0
    #
    # # 2D            -   initial density gradients
    # grad_rho0       =   np.array([ aux.gradient_density(R0,R0[j],h0,par) for j in range(Np) ])
    # par['grad_rho0']=   grad_rho0
    #
    # # 1D            -   polytropic pressure constant
    # Ki              =   aux.polytropic_pressure_constant(par)
    # par['Ki']       =   Ki
    #
    # # 1D            -   initial politripic pressure
    # P0              =   aux.pressure(rho0,par)
    # par['P0']       =   P0
    #
    # # 2D            -   inital forces on SPH particles
    # F0              =   np.array([ aux.acc_particle(R0,V0,rho0,grad_rho0,Omega0,h0,j,par) for j in range(Np) ])
    # par['F0']       =   F0
    #
    # # scalar        -   initial time step
    # dt0             =   aux.choose_dt(h0,V0,F0)
    # par['dt0']      =   dt0
    #
    return

#===============================================================================
""" Integrate Constructed Model over Time """
#-------------------------------------------------------------------------------

def integrate_model(par):
    # """ carry out time evelution of model
    #
    # Parameters
    # ----------
    # par:    dictionary of model parmaeters
    #
    # Returns
    # -------
    # panda Series of important data values and arrays
    # """

    return
