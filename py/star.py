import numpy as np
import auxillary as aux

#===============================================================================
""" 1) Assign Model Parameters
    2) Constuct Model """
#-------------------------------------------------------------------------------

Np      =   1000    # number of particles
Mc      =   1       # mass of cloud [solar mass]
T0      =   15      # temp of cloud [K]
frac_H  =   .75     # fraction of particles that are hydrogen
frac_He =   .25     # fraction of particles that are helium

model   =   aux.model_params(Np,Mc,T0,frac_H,frac_He)

#===============================================================================
""" Make Plots """
#-------------------------------------------------------------------------------

# plot_scatter    =   aux.plot_3D_scatter(model,0)
