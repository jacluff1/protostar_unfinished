import numpy as np
from scipy.special import gamma
import pdb

#===============================================================================
""" misc """
#-------------------------------------------------------------------------------

def r_vec(ri,rj):
    """ find the relative position vector

    args
    ----
    r0:     vector  -   particle position
    ri:     vector  -   particle position

    returns
    -------
    tuple - rmag,rhat
    """

    r       =   ri - rj
    rmag    =   np.linalg.norm(r)
    assert rmag > 0, "rmag = 0, ri != rj"
    rhat    =   r / rmag
    return rmag,rhat

def u_vec(uj):

    umag    =   np.linalg.norm(uj)
    uhat    =   uj / umag
    return umag,uhat

#===============================================================================
""" initialize arrays """
#-------------------------------------------------------------------------------

def initial_SP_placement(model):

    Np      =   model['Np']
    radius  =   model['radius']

    # set up random positions in spherical coordinates
    R       =   np.random.rand(Np) * radius
    THETA   =   np.random.rand(Np) * np.pi
    PHI     =   np.random.rand(Np) * 2*np.pi

    # convert spherical coordinates to cartesian
    X       =   R * np.sin(THETA) * np.cos(PHI)
    Y       =   R * np.sin(THETA) * np.sin(PHI)
    Z       =   R * np.cos(THETA)

    pos0    =   np.vstack((X,Y,Z)).T

    model['pos'][0,:,:]   =   pos0

def Lambda(model):
    """ pmocz_sph.pdf (26) """

    k   =   model['k']
    n   =   model['n']
    M   =   model['M']
    R   =   model['radius']

    one     =   2 * k * (1+n) * np.pi**(-3/(2*n))
    two_1   =   M * gamma( (5/2) + n )
    two_2   =   R**3 * gamma( 1 + n )
    two     =   ( two_1 / two_2 )**(1/n)
    three   =   1 / R**2

    model['Lambda'] =   one * two * three

#===============================================================================
""" smoothing """
#-------------------------------------------------------------------------------

def kernal_gauss(uj,h):
    """ mnras181-0375.pdf (2.10 i) """

    umag    =   np.linalg.norm(uj)

    return ( h * np.sqrt(np.pi) )**(-3) * np.exp( -umag**2 / h**2 )

def gradient_kernal_gauss(uj,h):
    """ derived from mnras181-0375.pdf (2.10 i) """

    u,uhat  =   u_vec(uj)
    W       =   kernal_gauss(uj,h)
    return (2 / h**2) * W * uhat

#===============================================================================
""" cloud t """
#-------------------------------------------------------------------------------

def density(t,model):
    """ pmocz_sph.pdf (15) """

    Np      =   model['Np']
    post    =   model['pos'][t,:,:]
    rhot    =   model['rho'][t,:]
    mi      =   model['mi']
    h       =   model['h'][t]

    for i in range(Np):
        rhot[i]     +=  mi * kernal_gauss( np.zeros(3) , h )
        for j in range(Np):
            uij     =   post[i,:] - post[j,:]
            rhot[i] +=  mi * kernal_gauss( uij , h )

    # for i in np.arange(1,Np):
    #     rhot[i]     +=  mi * kernal_gauss( np.zeros(3) , h )
    #     for j in np.arange(i+1,Np):
    #         uij     =   post[i,:] - post[j,:]
    #         rho_ij  =   mi * kernal_gauss( uij , h )
    #         rhot[i] +=  rho_ij
    #         rhot[j] +=  rho_ij

def pressure(t,model):
    """ pmocz_sph.pdf (2) """

    k       =   model['k']
    rho     =   model['rho'][t,:]
    n       =   model['n']
    Np      =   model['Np']
    P       =   model['P'][t,:]

    for i in range(Np):
        P[i]    =   k * rho[i]**( 1 + 1/n )

def acceleration(t,model):

    x       =   model['pos'][t,:,:]
    v       =   model['vel'][t,:,:]
    a       =   model['acc'][t,:,:]
    rho     =   model['rho'][t,:]
    P       =   model['P'][t,:]
    mi      =   model['mi']
    nu      =   model['nu']
    lam     =   model['Lambda']
    h       =   model['h'][t]
    Np      =   model['Np']

    # for i in range(Np):
    #     a[i,:]  +=  - nu * v[i,:] - lam * x[i,:]
    #     for j in range(Np):
    #         uij     =   x[i,:] - x[j,:]
    #         gW      =   gradient_kernal_gauss( uij , h )
    #         a[i,:]  +=  - mi * ( P[i]/rho[i]**2 + P[j]/rho[j]**2 ) * gW

    for i in np.arange(1,Np):
        a[i,:]  +=  -nu * v[i,:] - lam * x[i,:]
    for i in np.arange(1,Np):
        for j in np.arange(i+1,Np):
            uij     =   x[i,:] - x[j,:]
            gW      =   gradient_kernal_gauss( uij , h )
            p_a     =   -mi * ( P[i]/rho[i]**2 + P[j]/rho[j]**2 ) * gW
            a[i,:]  +=  p_a
            a[j,:]  +=  -p_a

#===============================================================================
"""
Choosing dt and h
f(t,model)
h and dt are consistent for all particles in time slice """
#-------------------------------------------------------------------------------

def choose_h(t,model):
    """ choose smoothing length
    another method could be to choose h
    for each particle such that there are
    k variables enclosed by radius h around
    each particle

    args
    ----
    t:      int     -   time index
    model:  Series  -   model data

    returns
    -------
    None - updates model
    """

    # pmocz_sph.pdf (20)
    R       =   model['pos'][t,:,:]
    Np      =   model['Np']
    alpha   =   model['alpha']
    h0      =   model['h'][0]

    r1      =   np.average( np.array([ np.linalg.norm(R[j,:])**2 for j in range(Np) ]) )
    r2      =   np.average( np.array([ np.linalg.norm(R[j,:]) for j in range(Np) ]) )**2

    h       =   alpha * np.sqrt( r1 - r2 )
    model['h'][t]  =   h
    # model['h'][t]  =   h0

def choose_dt(t,model):
    """ choose time step

    args
    ----
    t:      int     -   time index
    model:  Series  -   model data

    returns
    -------
    None - updates model
    """

    # pmocz_sph.pdf (19)
    Np      =   model['Np']
    h       =   model['h'][t]
    V       =   model['vel'][t,:,:]
    F       =   model['acc'][t,:,:] * model['mi']
    alpha   =   model['alpha']
    dt0     =   model['dt'][0]

    vmags   =   np.array([ np.linalg.norm(V[j,:]) for j in range(Np) ])
    fmags   =   np.array([ np.linalg.norm(F[j,:]) for j in range(Np) ])

    vmax    =   np.max( vmags )
    fmax    =   np.max( fmags )

    t1      =   h / vmax
    t2      =   np.sqrt( h / fmax )

    if t == 0:
        tmin    =   t2
    else:
        tmin    =   min( t1 , t2 )

    dt      =   alpha * tmin
    model['dt'][t]  =   dt
    # model['dt'][t]  =   dt0

#===============================================================================
""" integration """
#-------------------------------------------------------------------------------

def leap_frog(model):

    Nt      =   model['Nt']
    x       =   model['pos']
    v       =   model['vel']
    a       =   model['acc']
    dt      =   model['dt']
    h       =   model['h']

    t           =   1
    h1          =   h[t-1]
    dt1         =   dt[t-1]

    a[t,:,:]    =   a[t-1,:,:]
    v[t,:,:]    =   a[t,:,:] * ( dt1 / 2)
    x[t,:,:]    =   x[t-1,:,:] + v[t,:,:] * dt1
    density(t,model)
    pressure(t,model)
    acceleration(t,model)
    v[t,:,:]    =   a[t,:,:] * ( dt1 / 2)

    choose_h(t,model)
    choose_dt(t,model)

    # for t in np.arange(1,Nt):
    #
    #     choose_h(t,model)
    #     density(t,model)
    #     pressure(t,model)
    #
    #     a[t,:,:]    =   a[t-1,:,:]
    #     v[t,:,:]    =   a[t,:,:] * ( h[t] / 2)
    #     x[t,:,:]    =   x[t-1,:,:] + v[t,:,:] * h[t]
    #     acceleration(t,model)
    #     v[t,:,:]    =   a[t,:,:] * ( h[t] / 2)
    #
    #     choose_dt(t,model)

#===============================================================================
""" auxillary plotting functions """
#-------------------------------------------------------------------------------

def density_xy(x,y,t,model):
    """ pmocz_sph.pdf (15) """

    Np      =   model['Np']
    Rt      =   model['pos'][t,:,:]
    mi      =   model['mi']
    h       =   model['h'][t]

    r       =   np.array([ x , y , 0 ])

    W       =   0
    for j in range(Np):
        rj  =   Rt[j,:]
        if np.array_equal(r,rj) == False:
            uj      =   r - rj
            W       +=  kernal_gauss(uj,h)
    return mi * W

def generate_contour_images(pp,model):

    # pull model data (model)
    Nt          =   model['Nt']
    Np          =   model['Np']
    R           =   model['pos']

    # pull plot parameters (pp)
    Nx          =   pp['Nx']
    logscale    =   pp['logscale']

    # create grid
    rlim        =   np.max( np.array([ np.linalg.norm( R[0,i,:] ) for i in range(Np) ]) )
    X,Y         =   [ np.linspace(-rlim,rlim,Nx) for i in range(2) ]
    X,Y         =   np.meshgrid(X,Y)

    # create empty 3D array for images
    images      =   np.zeros(( Nt , Nx , Nx ))

    # fill in images array
    for t in range(Nt):
        Z               =   density_xy(X,Y,t,model)
        if logscale:    Z   =   np.log10(Z)
        images[t,:,:]   =   Z

    np.save('../data/images_Np%s_Nt%s.npy' % (Np,Nt), images)
    np.save('../data/Xp_Np%s_Nt%s.npy' % (Np,Nt), X)
    np.save('../data/Yp_Np%s_Nt%s.npy' % (Np,Nt), Y)
    return Xp,Yp,images

# def map_grid(X,Y,RHO,alpha=1.5,Ng=1000):
#     """ sets 2D image to grid
#     https://matplotlib.org/examples/pylab_examples/griddata_demo.html
#
#     Parameters
#     ----------
#     X:  1D x-axis data
#     Y:  1D y-axis data
#     M:  1D particle mass data
#
#     Returns
#     -------
#     tuple of X,Y,Z arrays
#     """
#
#     rmax    =   max( np.max(X) , np.max(Y) ) * alpha
#     Xp,Yp   =   [ np.linspace(-rmax,rmax,Ng) for i in range(2) ]
#     Zp      =   griddata(X, Y, RHO, Xp, Yp, interp='linear')
#
#     return Xp,Yp,Zp

#===============================================================================
""" test function """
#-------------------------------------------------------------------------------

def rho_test(model):

    rho = model['rho']
    print("\nrho0", rho[0])
    print("\nrho1", rho[1])

def pos_test(model):

    R = model['pos']
    print("\nR0 - R1", R[0] - R[1])
