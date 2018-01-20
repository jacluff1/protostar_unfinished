import numpy as np
import kernel as kernel

from params import par
for key,val in par.items():
    exec(key + '=val')

#===============================================================================
"Auxillary Functions"
#-------------------------------------------------------------------------------

def find_distances(positions):
    return np.sqrt( (positions*positions).sum(axis=1) )

#===============================================================================
"Gas Equations"
#-------------------------------------------------------------------------------

def find_density(position,ppositions,smoothingL,kernel=kernel.kernel_gaussian):
    total   =   0
    for j in range(N):
        total   +=  kernel(position-ppositions[j,:],smoothingL)
    return m * total

#===============================================================================
"Equations of Motion"
#-------------------------------------------------------------------------------

# def find_acceleration_total(positions,velocities):
