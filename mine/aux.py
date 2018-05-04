import matplotlib as mpl
from matplotlib import cm

def update_values(values,kwargs):
    """ update default dictionary of values with provided kwargs. """
    for key in values:
        if key in kwargs: values[key] = kwargs[key]
    return values

def set_plot_defaults():

    # axes
    mpl.rcParams['axes.labelpad']=20
    mpl.rcParams['axes.labelweight']='bold'
    mpl.rcParams['axes.labelsize']=20
    mpl.rcParams['axes.titlepad']=15
    mpl.rcParams['axes.titleweight']='bold'
    mpl.rcParams['axes.titlesize']=24

    # figure
    mpl.rcParams['figure.figsize']=(15,15)
    mpl.rcParams['figure.titlesize']=30
    mpl.rcParams['figure.titleweight']='bold'

    # image
    # mpl.rcParams['image.cmap']=cm.hot

# maybe include these later
# #===============================================================================
# # units
# #===============================================================================
#
# pc2meter    =   3.086e16            # pc -> meter
# sm2kg       =   1.99e30             # solar mass -> kilogram
# myr2s       =   60*60*24*365.25*1e6 # mega year -> second
#
# # m^3 kg^-1 s^-2
# G_const     =   6.67e-11
#
# # pc^3 solarMass^-1 Myr^-2
# G_model     =   G_const * (pc2meter)**(-3) * sm2kg * myr2s**2
#
# #===============================================================================
