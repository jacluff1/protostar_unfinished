import aux as aux
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as ani
import scipy.special as special
import pdb

class model:

    def __init__(self,**kwargs):

        print("setting up SPH model...")

        # handle default attributes
        values  =   dict(M=2, N=400, n=1, k=0.1, nu=1, alpha=1, beta=2, kernel='gaussian', n_neighbors=7, initial_distribution='uniform', initial_dispersion=0, initial_omega=0, initial_radius=0.75, initial_temp=5)
        values  =   aux.update_values(values,kwargs)

        # add initial attributes
        for key in values: setattr(self,key,values[key])
        self.m  =   self.M / self.N

        # handle other parameters
        particle_preview    =   True
        if 'particle_preview' in kwargs: particle_preview = kwargs['particle_preview']

        # particle positions
        if self.initial_distribution == 'uniform':
            self.positions  =   self.__uniform_sphere()

        # particle velocities
        V       =   self.__zero()
        if self.initial_dispersion > 0: V += self.__random()
        if abs(self.initial_omega) > 0: V += self.__rotation()
        self.velocities =   V

        # model constants
        self.__add_constant_lambda()

        #  initialize time
        self.time   =   0

        # set default plotting params for session
        aux.set_plot_defaults()

        # preview position
        if particle_preview: self.particle_preview()

    def __add_constant_lambda(self):
        self.lam    =   2*self.k*(1+self.n)*np.pi**(-3/(2*self.n)) * ((self.M*special.gamma((5/2)+self.n))/(self.initial_radius**3 * special.gamma(1+self.n)) )**(1/self.n) / self.initial_radius**2

    #===========================================================================
    # initial position options
    #===========================================================================

    def __uniform_sphere(self):
        """ uniformly distributes particles in a sphere scaled so that 0 < r < 1. """

        print("placing particles in sphere with uniform distribution...")

        # set up 1D position arrays in SPC
        U           =   np.random.uniform(0,self.initial_radius,self.N)
        COSTHETA    =   np.random.uniform(-1,1,self.N)

        R       =   U**(1/3)
        THETA   =   np.arccos(COSTHETA)
        PHI     =   np.random.uniform(0,2*np.pi,self.N)

        # set up 2D position array in CC
        X       =   np.zeros( (self.N,3) )

        # convert SPC to CC
        X[:,0]  =   R * np.sin(THETA) * np.cos(PHI)
        X[:,1]  =   R * np.sin(THETA) * np.sin(PHI)
        X[:,2]  =   R * np.cos(THETA)

        return X

    #===========================================================================
    # initial particle velocity options
    #===========================================================================

    def __zero(self):
        return np.zeros( (self.N,3) )

    def __random(self):
        return np.random.normal(0,self.initial_dispersion,self.N*3).reshape([self.N,3])

    def __rotation(self):
        Omega   =   np.array([0,0,self.initial_omega])
        return np.cross(Omega,self.positions)

    #===========================================================================
    # preview options
    #===========================================================================

    def particle_preview(self):

        # get particle positions
        X   =   self.positions

        # clear previous plots
        plt.close('all')

        fig =   plt.figure()
        ax  =   fig.add_subplot(111, projection='3d')
        ax.plot(X[:,0], X[:,1], X[:,2], 'bo', ms=1)
        ax.set_aspect(1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title('particle positions')

        plt.show()

    #===========================================================================
    # kernel options
    #===========================================================================

    def __W_gauss(self,distance_ij,smoothing_length_j):
        return (np.pi*smoothing_length_j**2)**(-3/2) * np.exp(-(distance_ij/smoothing_length_j)**2)

    def __dW_gauss(self,distance_ij,vector_ij,smoothing_length_j):
        W_ij    =   self.__W_gauss(distance_ij,smoothing_length_j)
        return -(2*vector_ij/smoothing_length_j**2) * W_ij

    # add other kernel options

    #===========================================================================
    # general/standard equations
    #===========================================================================

    def __update_nearest_neighbors(self):

        try:

            # get nearest neighbors (since nearest neighbor is itself, look for desired n_neighbors+1)
            nbrs = NearestNeighbors(n_neighbors=self.n_neighbors+1).fit(self.positions)
            distances, indices = nbrs.kneighbors(self.positions)

            # add attributes (since nearest neighbor is itself, each particle ignores itself)
            self.distances  =   distances[:,1:]
            self.indices    =   indices[:,1:]

        except:
            print("error occured, check positions.")
            print("time: ", self.time)

    def __update_smoothing_lengths(self):
        # Hernquist & Katz (1989) -> have each particle fetch the distance to its furthest neighbor and divide by two.
        self.smoothing_lengths  =   self.distances[:,-1] / 2

    def __update_density(self):
        Array   =   np.zeros(self.N)
        for i in range(self.N):
            indices     =   self.indices[i,:]
            # get distances
            r           =   self.distances[i,:]
            # get smoothing lengths
            h           =   self.smoothing_lengths[indices]
            for j in range(self.n_neighbors):
                Array[i]    +=  self.__W_gauss(r[j],h[j])
        self.densities  =   self.m * Array

    def __update_pressure(self):
        # use polytropic pressure
        self.pressures  =   self.k * self.densities**(1 + (1/self.n))

    def __update_acceleration(self):

        # update dependencies
        self.__update_nearest_neighbors()
        self.__update_smoothing_lengths()
        self.__update_density()
        self.__update_pressure()

        # create empty array to dump calculations into
        Array   =   np.zeros( (self.N,3) )

        # loop through every particle
        for i in range(self.N):

            # select nearest neighbor subset data
            indices             =   self.indices[i,:]
            positions           =   self.positions[indices,:]
            smoothing_lengths   =   self.smoothing_lengths[indices]
            densities           =   self.densities[indices]
            pressures           =   self.pressures[indices]

            # iterate over nearest neighbors
            for j in range(self.n_neighbors):

                # get kernel gradient
                distance_ij         =   self.distances[i,j]
                vector_ij           =   self.positions[i,:] - positions[j,:]
                smoothing_length_j  =   smoothing_lengths[j]
                dW_ij               =   self.__dW_gauss(distance_ij,vector_ij,smoothing_length_j)

                # get rest of pressure gradient factor
                factor_ij   =   (self.pressures[i]/self.densities[i]**2) + (pressures[j]/densities[j]**2)

                # subtract pressure contribution of nearest neighbor
                Array   -=  factor_ij * dW_ij

            # multiply total pressure contributions by particle mass
            Array[i,:]  *=  self.m

            # subtract basic artificial viscosity
            Array[i,:]  -=  self.nu * self.velocities[i,:]

            # subtract position (gravity?) term
            Array[i,:]  -=  self.lam * self.positions[i,:]

        # update attribute
        self.accelerations  =   Array

    def __update_dt(self):
        options =   np.zeros(2)
        try:
            options[0]  =   (self.smoothing_lengths / np.sqrt((self.velocities**2).sum(axis=1)) ).min()
        except:
            print("max speed = 0")
        try:
            options[1]  =   np.sqrt( (self.smoothing_lengths / (self.m*np.sqrt((self.accelerations**2).sum(axis=1)))).min() )
        except:
            print("max force = 0")
        self.dt =   0.25 * options[ options > 0 ].min()

    def __update_temperature(self):
        NotImplemented

    #===========================================================================
    # make movies
    #===========================================================================

    def cloud_movie(self,**kwargs):

        # handle default parameters
        # values  =   dict(name='SPH_star.mp4', title='SPH Star Formation', fps=15, dpi=100, N_frames=100)
        # values  =   aux.update_values(values,kwargs)
        # for key in values:
        #     exec("%s=%s" % (key,values[key]) in globals(), locals())

        name    =   'SPH_star.mp4'
        title   =   'SPH Star Formation'
        fps     =   15
        dpi     =   100
        Nframes =   100

        # set up figure
        fig =   plt.figure()
        ax  =   fig.gca(projection='3d')
        ax.set_aspect(1)
        ax.plot(self.positions[:,0],self.positions[:,1],self.positions[:,2], 'go')

        # set up movie writer
        FFMpegWriter    =   ani.writers['ffmpeg']
        metadata        =   dict(title=title, artist='Matplotlib')
        writer          =   FFMpegWriter(fps=fps, metadata=metadata)

        # set up data for initial frame
        self.__update_acceleration()
        self.__update_dt()

        # write movie
        with writer.saving(fig, name, dpi):
            # loop through all frames
            for t in range(Nframes):
                print("time: ", self.time)

                # clear previous frame
                ax.clear()

                # update data for frame
                self.time       =   self.time + self.dt
                v_half          =   self.velocities + self.accelerations * self.dt
                self.positions  =   self.positions + v_half * self.dt
                self.velocities =   0.5 * (self.velocities + v_half)

                # update data for next frame
                self.__update_acceleration()
                self.__update_dt

                # make frame
                ax.plot(self.positions[:,0],self.positions[:,1],self.positions[:,2], 'go')
                ax.set_aspect(1)
                rmax    =   1
                ax.set_xlim([-rmax,rmax])
                ax.set_ylim([-rmax,rmax])
                ax.set_zlim([-rmax,rmax])
                writer.grab_frame()

        plt.close()
