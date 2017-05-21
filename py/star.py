import numpy as np
import auxillary as aux
import plot as plot

#===============================================================================
""" Model Parameters """
#-------------------------------------------------------------------------------

# Assumed values
Np      =   1000            # scalar    - number of SPH particles
Mc      =   1/solarM.value  # scalar    - mass of cloud [solar mass -> kg]
T0      =   15              # scalar    - temp of cloud [K]
frac_H  =   .75             # scalar    - fraction of particles that are hydrogen
frac_He =   .25             # scalar    - fraction of particles that are helium
n       =   1               # scalar    - polytropic index
Lambda  =   1               # scalar    - static damping coefficient
disp    =   2400            # scalar    - velocity dispersion [ km / s ]

par     =   {'Np':      Np,
             'Mc':      Mc,
             'T0':      T0,
             'frac_H':  frac_H,
             'frac_He': frac_He,
             'n':       n,
             'Lambda':  Lambda}

# calculated values
par['mu']       =   aux.average_particle_mass(par)                                      # scalar    - average particle mass
par['mi']       =   Mc/Np                                                               # scalar    - average SPH particle mass
par['Kj']       =   aux.polytropic_pressure_constant(par)                               # 1D        - polytropic pressure constant
par['Rj']       =   dac.Jean_radius_M(par)                                              # scalar    - Jean's radius
par['Omega0']   =   aux.initial_angular_frequency(par)                                  # scalar    - initial angular frequency ( 0 , 0 , Omega0 )
par['R0']       =   aux.initial_SP_placement(par)                                       # 2D        - initial SP placement
par['V0']       =   aux.initial_SP_velocities(par)                                      # 2D        - initial SP speed
par['rho0']     =   np.array([ aux.initial_density(R0,R0[j],par) for j in range(Np) ])  # 1D        - initial mass density

#===============================================================================
""" Make Plots """
#-------------------------------------------------------------------------------

# plot.scatter_3D(model,0)
# plot.contourf_2D(model,0)
