import numpy as np
import pandas as pd
import auxillary2 as aux

def construct_model():

    d           =   {}

    # parameters
    d['Np']     =   100
    d['Nt']     =   400
    d['M']      =   2
    d['mi']     =   d['M'] / d['Np']
    d['radius'] =   .75
    # d['h']      =   .04 / np.sqrt( d['Np']/1000 )
    # d['dt']     =   .04
    d['nu']     =   1
    d['k']      =   0.1
    d['n']      =   1
    d['alpha']  =   .01
    # d['Lambda'] =   0

    # arrays
    d['pos']    =   np.zeros(( d['Nt'] , d['Np'], 3 ))
    d['vel']    =   np.zeros_like( d['pos'] )
    d['acc']    =   np.zeros_like( d['pos'] )

    d['rho']    =   np.zeros(( d['Nt'] , d['Np'] ))
    d['P']      =   np.zeros_like( d['rho'] )

    # d['time']   =   np.arange( 0 , (d['Nt']+1) * d['dt'] , d['dt'])
    d['h']      =   np.zeros( d['Nt'] )
    d['dt']     =   np.zeros_like( d['h'] )

    # convert to Series
    model       =   pd.Series(d)

    # initialize
    model['h'][0]   =   .04 / np.sqrt( d['Np']/1000 )
    model['dt'][0]  =   .04

    aux.Lambda(model)
    aux.initial_SP_placement(model)
    aux.density(0,model)
    aux.pressure(0,model)
    aux.acceleration(0,model)

    # integrate
    aux.leap_frog(model)
    model.to_pickle('model')

    return model
