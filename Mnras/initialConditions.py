import numpy as np
import matplotlib.pyplot as plt

from params import par
for key,val in par.items():
    exec(key + '=val')

def placement_uniform_sphere():
    """
    make a random distribution of 'N' particles
    inside a sphere.

    returns
    -------
    2D position array

    thanks
    ------
    https://stackoverflow.com/a/5408843
    """

    # set up 1D position arrays in SPC
    U           =   np.random.uniform(0,1,N)
    COSTHETA    =   np.random.uniform(-1,1,N)

    R       =   r0 * U**(1/3)
    THETA   =   np.arccos(COSTHETA)
    PHI     =   np.random.uniform(0,2*np.pi,N)

    # set up 2D position array
    X       =   np.zeros( (N,3) )
    # convert SPC to CC
    X[:,0]  =   R * np.sin(THETA) * np.cos(PHI)
    X[:,1]  =   R * np.sin(THETA) * np.sin(PHI)
    X[:,2]  =   R * np.cos(THETA)

    return X

def velocity_zero():

    V   =   np.zeros( (N,3) )

    return V

def velocity_random_motion():

    V   =   np.zeros( (N,3) )

    V[:,0]  =   np.random.normal(0,disp0,N)
    V[:,1]  =   np.random.normal(0,disp0,N)
    V[:,2]  =   np.random.normal(0,disp0,N)

    return V

def velocity_rotation_uniform(R):

    V   =   np.zeros( (N,3) )

    for i in range(N):
        V[i,:]  =   np.cross(omega0,R[i,:])

    return V
