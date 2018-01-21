import functions as func

import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV

from params import par
for key,val in par.items():
    exec(key + '=val')

def find_smoothingL(positions):
    R       =   func.find_distances(positions)
    grid    =   GridSearchCV( KernelDensity(), {'bandwidth':np.linspace(0, 1.0, 50)}, cv=20)
    grid.fit(R[:,None])
    return grid.best_params_['bandwidth']

# def find_smoothing_length_constant(positions,b_factor):
#     r2_av   =   (positions*positions).sum() / N
#     rav     =   positions.sum(axis=0) / N
#     rav_2   =   rav.dot(rav)
#     return b_factor * np.sqrt( r2_av - rav_2 )

# def find_optimal_N_neighbors(positions):
#
# def find_smoothing_length_variable(positions,neighbors):
