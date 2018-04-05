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
