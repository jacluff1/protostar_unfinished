import numpy as np

from params import par
for key,val in par.items():
    exec(key + '=val')

def find_smoothing_length(positions,b_factor):
    r2_av   =   (positions*positions).sum() / N
    rav     =   positions.sum(axis=0) / N
    rav_2   =   rav.dot(rav)
    return b_factor * np.sqrt( r2_av - rav_2 )
