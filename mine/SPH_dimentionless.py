import aux as aux
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as ani
import pdb

#===============================================================================
# units
#===============================================================================

pc2meter    =   3.086e16            # pc -> meter
sm2kg       =   1.99e30             # solar mass -> kilogram
myr2s       =   60*60*24*365.25*1e6 # mega year -> second

# m^3 kg^-1 s^-2
G_const     =   6.67e-11

# pc^3 solarMass^-1 Myr^-2
G_model     =   G_const * (pc2meter)**(-3) * sm2kg * myr2s**2

#===============================================================================

class model:

    def __init__(self,**kwargs):

        print("setting up SPH model...")

        # handle default attributes
        values  =   dict(M=10, N=500, n=3/2, k=0.1, nu=1, kernel='gaussian', n_neighbors=7, initial_distribution='uniform', initial_dispersion=0, initial_omega=0, initial_radius=50, initial_temp=5)
        values  =   aux.update_values(values,kwargs)

        # add initial attributes
        for key in values: setattr(self,key,values[key])
        self.m  =   self.M / self.N

        # update G_model
        self.G  =   G_model / self.initial_radius**3

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
        self.lam    =   self.__get_constant_lambda()

        #  initialize time
        self.time   =   0

        # set default plotting params for session
        aux.set_plot_defaults()

        # preview position
        if particle_preview: self.particle_preview()

    #===========================================================================
    # particle placement options
    #===========================================================================

    def __uniform_sphere(self):
        """ uniformly distributes particles in a sphere scaled so that 0 < r < 1. """

        print("placing particles in sphere with uniform distribution...")

        # set up 1D position arrays in SPC
        U           =   np.random.uniform(0,1,self.N)
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

    def __W_gauss(self,r,h):
        # r is magnitute of difference vector
        return (np.pi*h**2)**(-3/2) * np.exp(-(r/h)**2)

    def __W_gradient_gauss(self,r,h,W):
        # r is difference vector
        return -(2*r/h**2) * W

    # add other kernel options

    #===========================================================================
    # general/standard equations
    #===========================================================================

    # improve nn_relative_positions? get piecewise positions?
    def __update_nearest_neighbors(self):

        try:

            # get nearest neighbors (since nearest neighbor is itself, look for desired n_neighbors+1)
            nbrs = NearestNeighbors(n_neighbors=self.n_neighbors+1).fit(self.positions)
            distances, indices = nbrs.kneighbors(self.positions)

            # get nearest neighbor relative positions
            pos     =   np.zeros( (self.N, self.n_neighbors, 3) )
            for i in range(self.N):
                mask    =   indices[i,:]
                pos1    =   self.positions[mask]
                pos_i   =   pos1[0,:]
                for j,pos_j in enumerate(pos1[1:,:]):
                    pos[i,j,:]  =   pos_i - pos_j

            # add attributes (since nearest neighbor is itself, each particle ignores itself)
            self.nn_distances           =   distances[:,1:]
            self.nn_indices             =   indices[:,1:]
            self.nn_relative_positions  =   pos

        except:
            print("error occured, check positions.")
            print("time: ", self.time)

    def __update_smoothing_lengths(self):
        # Hernquist & Katz (1989) -> have each particle fetch the distance to its furthest neighbor and divide by two.
        self.smoothing_lengths  =   self.nn_distances[:,-1] / 2

    def __update_density(self):
        NotImplemented

    def __update_pressure(self):
        NotImplemented

    # def __update_temperature(self):
    #     NotImplemented

    #===========================================================================
    # model constants
    #===========================================================================

    def __get_constant_lambda(self):
        NotImplemented

    #===========================================================================
    # acceleration terms
    #===========================================================================

    def __gravity(self):
        A   =   np.zeros( (self.N,3) )
        for i in range(self.N):
            for j in range(self.n_neighbors):
                A[i,:]  +=  self.nn_relative_positions[i,j,:] / self.nn_distances[i,j]**3
        return -self.G * self.m**2 * A

    #===========================================================================
    # time step
    #===========================================================================

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

    def update_cloud(self):

        # general/standard
        self.__update_nearest_neighbors()
        self.__update_smoothing_lengths()

        # acceleration
        A                   =   self.__gravity()
        self.accelerations  =   A
        # print(A[:10,:])

        # update dt
        self.__update_dt()
        print("time: ", self.time)
        self.time   =   self.time + self.dt

    #===========================================================================
    # make movies
    #===========================================================================

    def cloud_movie(self):

        fig =   plt.figure()
        ax  =   fig.gca(projection='3d')
        ax.set_aspect(1)
        ax.plot(self.positions[:,0],self.positions[:,1],self.positions[:,2], 'go')

        FFMpegWriter    =   ani.writers['ffmpeg']
        metadata        =   dict(title='SPH Star Formation', artist='Matplotlib')
        writer          =   FFMpegWriter(fps=15, metadata=metadata)

        with writer.saving(fig,"SPH.mp4", 100):
            for t in range(100):

                ax.clear()

                self.update_cloud()

                V_new   =   self.velocities + self.accelerations * self.dt
                X_new   =   self.positions + self.velocities * self.dt

                self.velocities =   V_new
                self.positions  =   X_new

                # positions_half = positions + velocities*(dt/2)
                # velocities_half = velocities + accelerations*(dt/2)
                # densities_half = np.array([ find_density(positions_half,i) for i in range(N) ])
                # pressures_half = find_pressure(densities_half)
                # accelerations_half = np.array([ find_acceleration(positions_half,velocities_half,densities_half,pressures_half,i) for i in range(N) ])
                #
                # positions += velocities_half*dt
                # velocities += accelerations_half*dt

                ax.plot(self.positions[:,0],self.positions[:,1],self.positions[:,2], 'go')
                ax.set_aspect(1)
                rmax    =   1
                ax.set_xlim([-rmax,rmax])
                ax.set_ylim([-rmax,rmax])
                ax.set_zlim([-rmax,rmax])
                writer.grab_frame()

        plt.close()
