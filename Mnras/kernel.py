import numpy as np
# from sklearn import pairwise_distances

from params import par
for key,val in par.items():
    exec(key + '=val')

# def kernel_gaussian(relativePosition,smoothingL):
#     return  ( 1 / (np.pi*smoothingL**2) )**(3/2) * np.exp(-relativePosition.dot(relativePosition)/smoothingL**2)

def kernel_gaussian(relativePosition,smoothingL):
    return  ( 1 / (np.pi*smoothingL**2) )**(3/2) * np.exp(-(relativePosition/smoothingL)**2)

def kernel_gradient_gaussian(relativePosition,smoothingL):
    return  (-2*relativePosition/smoothingL**2) * kernel_gaussian(relativePosition,smoothingL)
