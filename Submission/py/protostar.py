#===============================================================================
"""Import Modules"""
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors
import matplotlib.cm as cm
import auxillary as aux
import units as units
import pdb

#===============================================================================
"""Known Constants and parameters"""
#-------------------------------------------------------------------------------

C           =   units.C
SI          =   units.SI

# mass of model clouds: solar mass -> kg
M_clouds    =   np.array([.7,   .8,     1,      1.5,    2,      3,      5,      9,      15,     25,     60])
# M_clouds    =   M_clouds / units.solar_mass

# collapse time of model clouds: Myr -> s
T_bench     =   np.array([100,  68.4,   38.9,   35.4,   23.4,   7.24,   1.15,   .288,   .117,   .0708,  .0282])
# T_bench     =   T_bench / units.Myr

# radii of model stars: solar radius -> m
R_star      =   np.array([.8,   .93,    1,      1.2,    1.8,    2.2,    2.9,    4.1,    5.2,    10,     13.4])
R_star      =   R_star * C['r_sun']

N_clouds    =   len(M_clouds)

# M_clouds    =   M_clouds[::-1]
# T_bench     =   T_bench[::-1]
# R_star      =   R_star[::-1]

# plotting parameters
p           =   {'figsize':(20,15),
                 'polar':(15,15),
                 'fs':20,
                 'style':'-r',
                 'lw':2}

m            =   {'N_grid':1000,
                  'writer':'ffmpeg',
                  'interval':int(200),
                  'dpi':int(400),
                  'cmap':cm.hot}

#===============================================================================
""" Main Problem """
#-------------------------------------------------------------------------------

def single_cloud_collapse(i, N_time=1000):
    print("\ncalculating and collecting 'cloud_%s'..." % M_clouds[i])
    return aux.integrate(M_clouds[i],R_star[i],N_time)

def write_clouds(N_time=1000):
    print("\nstarting cloud writing and compilation sequence...")
    for i in range(N_clouds):
        single_cloud_collapse(i, N_time=N_time)

def plot_protostars(N_time=1000,saveA=True):
    """ plot core temperature vs time for all clouds
    M_clouds:   numpy array of cloud masses
    p:          dictionary of plotting parameters
    """

    def plot_axis(i):
        try:
            print("\nloading 'cloud_%s'..." % M_clouds[i])
            data    =   pd.read_pickle('../data/cloud_%s' % M_clouds[i])
        except:
            print("\ndid not find 'cloud_%s'. Calculating..." % M_clouds[i])
            data    =   single_cloud_collapse(i, N_time=N_time)

        X       =   data['TIME']
        Y       =   np.log(data['T'])

        fig = plt.figure(figsize=p['figsize'])
        # ax      =   plt.subplot(4,3,i+1)
        ax      =   plt.subplot(111)
        ax.set_title("%s M$_\odot$ Cloud" % M_clouds[i], fontsize=p['fs']+2)
        ax.set_xlabel("Time [ %s ]" % C['time'], fontsize=p['fs'])
        ax.set_ylabel("ln ( Temp [ %s ] )" % C['temp'], fontsize=p['fs'])
        ax.set_xlim([min(X),max(X)])
        ax.plot(X,Y,p['style'], lw=p['lw'])
        if saveA:
            fig.savefig('../figures/temp_vs_time_%s.png' % M_clouds[i])
            plt.close('all')
        # return ax

    print("\nstarting plotting sequence...")
    # fig = plt.figure(figsize=p['figsize'])
    for i in range(N_clouds):
        plot_axis(i)
    # plt.tight_layout()

    # save and return
    # if saveA:
    #     print("\nsaving 'temp_vs_time' in 'figures' folder.")
    #     fig.savefig('../figures/temp_vs_time.png')
    #     plt.close()
    # else:
    #     plt.show()

def single_cloud_movie(i, N_x=1000,N_time=1000,saveA=True):
    """ acknowledgements:
    http://matplotlib.org/examples/images_contours_and_fields/pcolormesh_levels.html
    https://matplotlib.org/users/colormapnorms.html"""

    print("\nstarting movie sequence for %s..." % M_clouds[i])
    try:
        print("\nloading 'cloud_%s'..." % M_clouds[i])
        data    =   pd.read_pickle('../data/cloud_%s' % M_clouds[i])
    except:
        print("\ndid not find 'cloud_%s'. Calculating..." % M_clouds[i])
        data    =   single_cloud_collapse(i, N_time=N_time)

    # take useful information from cloud data
    T           =   data['T']
    R           =   data['R']
    TIME        =   data['TIME']
    M           =   data['M']

    i_test      =   -1
    TIME        =   TIME[i_test]
    R           =   R[i_test]
    T           =   T[i_test]

    # create flux density in solar luminosity/pc^2
    # PHI         =   SI['sigma'] * T**4 * ( units.solar_lum / units.unit_length**2 )
    PHI         =   (C['sigma'] * T**4 / units.unit_power) * units.solar_lum     # flux density [solar luminosity / pc^2]

    # write luminosity array
    print("\nwriting luminosity array...")
    # L           =   np.zeros(( len(T) , N_x , N_x ))
    L           =   np.zeros(( N_x , N_x ))

    # for i_time,time in enumerate(T):
    rcloud      =   R
    phi         =   PHI
    X           =   np.linspace( -rcloud , rcloud , N_x )
    Y           =   np.linspace( -rcloud , rcloud , N_x )
    X,Y         =   np.meshgrid( X , Y )
    L[:,:]      =   aux.MakeColorMap(X,Y,rcloud,phi)
    # print("done")
    # L           =   L[0]
    limits      =   np.min(L),np.max(L)
    print(limits)

    # set up movie figure
    plt.close('all')
    fig         =   plt.figure(figsize=p['figsize'])
    ax          =   plt.subplot(111)
    ax.set_title('%s M$_\odot$: radius = %.3f %s , temp = %.0f %s , time = %.3f %s'\
    % ( M , rcloud*1e5 , ' x 10$^{-5}$ pc' , T , C['temp'] , TIME , C['time'] ),\
    fontsize=p['fs']+2)
    ax.set_xlabel('x [ %s ]' % '10$^{-6}$ pc', fontsize=p['fs'])
    ax.set_ylabel('y [ %s ]' % '10$^{-6}$ pc', fontsize=p['fs'])

    levels      =   MaxNLocator(nbins=100).tick_values(*limits)
    cmap        =   m['cmap']
    norm        =   BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    rcloud      =   R*1e6
    X           =   np.linspace( -rcloud , rcloud , N_x )
    Y           =   np.linspace( -rcloud , rcloud , N_x )
    # Z           =   L[0]
    Z           =   L


    xlim        =   -rcloud,rcloud
    ax.set_xlim([*xlim])
    ax.set_ylim([*xlim])
    ax.set_aspect(1)
    im          =   ax.contourf(X,Y,Z, levels=levels, cmap=cmap)
    cbar        =   fig.colorbar(im, ax=ax, pad=0)
    cbar.set_label('L$_\odot$', fontsize=p['fs'])
    if saveA:   fig.savefig('../figures/still_image_%s.png' % M)

    def animate_cloud(t):
        ax.clear()

        rcloud  =   R[t]
        X       =   np.linspace( -rcloud , rcloud , N_x )
        Y       =   np.linspace( -rcloud , rcloud , N_x )
        Z       =   L[t]

        ax.set_title('%s M$_\odot$: radius = %.3f %s , temp = %.3f %s , time = %.3f %s'\
        % ( M , rcloud , C['length'] , T[t] , C['temp'] , TIME[t] , C['time'] ),\
        fontsize=p['fs']+2)
        ax.set_xlabel('x [ %s ]' % C['length'], fontsize=p['fs'])
        ax.set_ylabel('y [ %s ]' % C['length'], fontsize=p['fs'])
        xlim        =   -rcloud,rcloud
        ax.set_xlim([*xlim])
        ax.set_ylim([*xlim])
        ax.set_aspect(1)
        im      =   ax.contourf(X,Y,Z, levels=levels, cmap=cmap)

    # # run movie
    # print("\n writing movie...")
    # movie_anim  =   animation.FuncAnimation(fig, animate_cloud, frames=len(TIME), blit=False, interval=m['interval'])
    # print("done")
    # if saveA:
    #     # movie_anim.save('../figures/movie_%s.mp4' % M, dpi=m['dpi'])
    #     movie_anim.save('../figures/movie_%s.mp4' % M, writer=m['writer'], dpi=m['dpi'])
    # else:
    #     plt.show()

def write_protostar_movies(saveA=True):
    N_clouds    =   len(M_clouds)
    print("\nstarting movie sequence...")
    for i in range(N_clouds):
        single_cloud_movie(i)


def printClouds():
    print("\ninitial temp = %s K, critical temp = %s K" % (C['T'],C['T_c']))
    for i in range(N_clouds):
         d = pd.read_pickle('../data/cloud_%s' % M_clouds[i])
         print("\nCloud         %s\n\
         theory R_star:         %s\n\
         Jean radius:           %s\n\
         initial volume:        %s\n\
         adiabadic constant:    %s\n\
         initial density:       %s\n\
         free fall time:        %s\n\
         ending radius:         %s\n\
         ending temperature:    %s\n\
         ending time:           %s\n"\
         % (d['M'],\
         d['R_star'],\
         d['R_j'],\
         d['V0'],\
         d['vt'],\
         d['rho0'],\
         d['tff'],\
         d['R'][-1],\
         d['T'][-1],\
         d['TIME'][-1]) )
