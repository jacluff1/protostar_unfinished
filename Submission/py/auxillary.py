#===============================================================================
""" Import Modules"""
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import units as units
import pdb

#===============================================================================
""" Imported Constants"""
#-------------------------------------------------------------------------------

C           =   units.C
SI          =   units.SI

#===============================================================================
""" Auxillary Functions """
#-------------------------------------------------------------------------------

def Jeans_radius(M):
    return (1/5) * (C['G'] * C['mu'] * M) / (C['k'] * C['T'])

def volume_cloud(R):
    return (4/3) * np.pi * R**3

def surface_area_cloud(R):
    return 4 * np.pi * R**2

def FreeFallTime(rho0):
    return np.sqrt( (3 * np.pi) / (32 * C['G'] * rho0) )

def vt_const(V0):
    return C['T'] * V0**(C['gamma']-1)

def temperature_cloud(V,vt):
    return vt / V**(C['gamma']-1)

def flux_density_cloud(T):
    return C['sigma'] * T**4

def potential_energy_cloud(M,R):
    return -(3/5) * C['G'] * M**2 / R

def terminator(A,i_terminate,axis=None):
    """ rise of the machines"""
    if axis == None:
        A   =   np.delete(A, np.s_[i_terminate:])
        return A
    else:
        assert type(axis) == int, "axis must be 'None' or 'int'"
        A   =   np.delete(A, np.s_[i_terminate:],axis)
        return A

def Heaviside(x):
    return (0.5 * (np.sign(x) + 1))

def MakeColorMap(x,y,rcloud,phi_cloud):
    r = np.sqrt(x**2 + y**2)
    a = 4 * np.pi * r**2
    # return Heaviside(rcloud-r) * phi_cloud
    return Heaviside(rcloud-r) * phi_cloud * a

#===============================================================================
""" Acceleration Functions """
#-------------------------------------------------------------------------------

def acc_pressure_gas(M,R,T):
    A   =   surface_area_cloud(R)
    V   =   volume_cloud(R)
    P   =   ( M/C['mu'] ) * C['k'] * T / V
    return A * P / M

def acc_gravity(M,R):
    return - C['G'] * M / R**2

def acc_total(M,R):
    return acc_gravity(M,R)
    # return acc_gravity(M,R) + acc_pressure_gas(M,R,T)

#===============================================================================
""" Integration Functions """
#-------------------------------------------------------------------------------

def rk4(M,R_prev,dt):
    k1  =   dt * acc_total(M,R_prev)
    k2  =   dt * acc_total(M,R_prev + k1/2)
    k3  =   dt * acc_total(M,R_prev + k2/2)
    k4  =   dt * acc_total(M,R_prev + k3)
    return R_prev + (1/3)*(k1/2 + k2 + k3 + k4/2)

def integrate(M,R_star,N_time,saveA=True):
    """ Assume homologous collapse:
    1) uniform density and spherical symmetry
    2) density will increase uniformly through cloud
    3) uniform temperature through cloud"""

    # data dictionary
    d       =   {}

    # calculate cloud constants and initial conditions
    R_j         =   Jeans_radius(M)                 #   jeans radius
    V0          =   volume_cloud(R_j)               #   initial volume of cloud
    vt          =   vt_const(V0)                    #   adiabadic constant of cloud
    rho0        =   M/V0                            #   initial density of cloud
    tff         =   FreeFallTime(rho0)              #   cloud free fall time
    dt          =   tff/N_time                      #   time incremenet
    T0          =   C['T']                          #   initial temperature of cloud
    Tc          =   C['T_c']                        #   quantum corrected critical temperature

    # update data dictionary
    d['M']      =   M
    d['R_star'] =   R_star
    d['N_time'] =   N_time
    d['R_j']    =   R_j
    d['V0']     =   V0
    d['vt']     =   vt
    d['rho0']   =   rho0
    d['tff']    =   tff
    d['T0']     =   T0
    d['Tc']     =   Tc

    # assign needed arrays
    TIME        =   np.array([])
    R           =   np.array([])
    T           =   np.array([])

    # assign termination values
    time        =   0
    r           =   R_j
    temp        =   T0
    dt          =   tff/N_time

    # update arrays
    while temp < Tc:
        print("\nstarting loop:\n\
        time:      %s\n\
        temp:      %s\n\
        radius:    %s\n"\
        % (time,temp,r) )

        # set up new arrays
        TIME1       =   np.linspace(time+dt,tff,N_time)
        R1          =   np.zeros_like(TIME1)
        T1          =   np.zeros_like(TIME1)

        R1[0]       =   r
        T1[0]       =   temp

        # update new arrays
        for i in np.arange(1,N_time):

            R1[i]   =   rk4(M, R1[i-1], dt)
            V_i     =   volume_cloud(R1[i])
            T1[i]   =   temperature_cloud(V_i,vt)

            # cut off iterating new arrays if clout radius goes negative
            if R1[i] <= 0:
                TIME1   =   terminator(TIME1,i)
                R1      =   terminator(R1,i)
                T1      =   terminator(T1,i)
                break

        # add new arrays to pre-existing arrays
        TIME        =   np.hstack((TIME,TIME1))
        R           =   np.hstack((R,R1))
        T           =   np.hstack((T,T1))

        # update termination values
        time        =   TIME[-1]
        print("loop travel distance:    %s" % (r - R[-1]) )
        r           =   R[-1]
        temp        =   T[-1 ]
        dt          /=  10

        # turn on conditions
        if temp >= Tc:
            d['t_on']   =   time
            print("star turned on at %s %s" % ( time , C['time'] ) )

    # check that arrays are same length
    assert len(R) == len(T) == len(TIME), "arrays must be same size (%s cloud)" % M

    # update data dictionary
    d['R']      =   R
    d['T']      =   T
    d['TIME']   =   TIME


    data        =   pd.Series(d)
    if saveA:   data.to_pickle('../data/cloud_%s' % M)
    return data
