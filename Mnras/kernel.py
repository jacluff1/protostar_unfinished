import numpy as np
# from sklearn import pairwise_distances

from params import par
for key,val in par.items():
    exec(key + '=val')

def kernel_gaussian(relativePosition,smoothingL):

    W           =   ( 1 / (np.pi*smoothingL**2) )**(3/2) * np.exp(-relativePosition.dot(relativePosition)/smoothingL**2)
    # gradientW   =   W * (-2*relativePosition/smoothingL**2)

    return W
