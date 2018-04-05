import numpy as np
from sklearn.neighbors import KDTree,DistanceMetric

import kernel as kernel

from params import par
for key,val in par.items():
    exec(key + '=val')

#===============================================================================
"Auxillary Functions"
#-------------------------------------------------------------------------------

def find_distances_pairwise(positions,grid=None):
    dist    =   DistanceMetric.get_metric('euclidean')
    if grid == None:
        return dist.pairwise(positions)
    if grid != None:
        return dist.pairwise(grid,positions)

def find_distance_origin(positions):
    return np.sqrt( (positions**2).sum(axis=1) )

#===============================================================================
"Grid Equations"
#-------------------------------------------------------------------------------

def find_density_grid(grid,positions,smoothingL,kernel=kernel.kernel_gaus):
    # find grid
    # differneces = grid-positions
    # W = kernel(differences,smoothingL)
    # sum W along appropriate axis 
    return NotImplemented

#===============================================================================
"SPH Equations"
#-------------------------------------------------------------------------------

def find_density(positions,smoothingL,kernel=kernel.kernel_gaus):
    return NotImplemented



#===============================================================================
"Equations of Motion"
#-------------------------------------------------------------------------------

# def find_acceleration_total(positions,velocities):
