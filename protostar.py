import djak.phys.SPH.placement as placement
import djak.phys.SPH.incompressible as sph


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm as cm
import pandas as pd

#===============================================================================
""" parameters """
#-------------------------------------------------------------------------------

# number of SPH particles
N       =   100

# inital cloud size
radius  =   2

# COM of cloud
R_cloud =   np.zeros(3)

#===============================================================================
""" initial conditions """
#-------------------------------------------------------------------------------

# particle placement
# x,y,z   =   placement.sphere_uniformish(N,radius,R=R_cloud)
x,y,z   =   placement.sphere_random(N,radius,R_cloud)

# re-assign N
N       =   len(x)

# particle positions array
rA      =   np.zeros((N,3))
rA[:,0] =   x
rA[:,1] =   y
rA[:,2] =   z

# particle velocities array
vA      =   np.zeros((N,3))

# particle smoothing lengths
hA      =   np.ones(N)

# particle masses
mA      =   np.ones(N)

# particle densities array
rhoA    =   np.zeros(N)


# particle pressures array
PA      =   np.zeros(N)

# particle timesteps array
dtA     =   np.ones(N)

# particle accelerations array
aA      =   np.zeros((N,3))

#===============================================================================
""" advancing particles """
#-------------------------------------------------------------------------------

# for j in range(N):
#     hA[j]    =   1

for j in range(N):
    rhoA[j] =   sph.rho(j,rA,mA,hA)
    PA[j]   =   sph.P_bol(j,rhoA)




#===============================================================================
""" visuals """
#-------------------------------------------------------------------------------

def plot(rA=rA,vA=vA,mA=mA,rhoA=rhoA,PA=PA,hA=hA):

    plt.close('all')
    fig     =   plt.figure(figsize=(30,15))
    ax      =   fig.add_subplot(121, projection='3d')

    pcm     =   ax.scatter(rA[:,0],rA[:,1],rA[:,2], c=rhoA, cmap=cm.magma)

    for j in range(N):
        aA[j,:] =   sph.acc_total(j,rA,vA,mA,rhoA,PA,hA)
    rA_half =   rA + np.multiply( vA.T, dtA/2 ).T
    vA_half =   vA + np.multiply( aA.T, dtA/2 ).T
    for j in range(N):
        rhoA[j] =   sph.rho(j,rA_half,mA,hA)
        PA[j]   =   sph.P_bol(j,rhoA)
    for j in range(N):
        aA[j]   =   sph.acc_total(j,rA_half,vA_half,mA,rhoA,PA,hA)
    rA      +=  np.multiply( vA_half.T, dtA ).T
    vA      +=  np.multiply( aA.T, dtA ).T

    ax1     =   fig.add_subplot(122, projection='3d')
    pcm1    =   ax1.scatter(rA[:,0],rA[:,1],rA[:,2], c=rhoA, cmap=cm.magma)
    plt.show()

    # d       =   {'r':rA, 'r1':rA_half, 'v':vA, 'v1':vA_half, 'a':aA, 'rho':rhoA, 'P':PA}
    # return pd.Series(d)
    # return d
