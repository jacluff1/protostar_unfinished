import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb

import aux as aux

class sph:

    def __init__(self,ptypes=['container', 'water'], smoothing_lengths=[0.1, 0.05], colors=['r', 'b'], alphas=[.5, .3], sizes=[15, 8], placement_shape=['rectangle', 'rectangle'], placement_kwargs=[dict(), dict(smoothing_l=0.05, length=2, height=2, x0=3, y0=5, filled=True)]):

        # handle metadata
        colors          =   [c+'o' for c in colors]
        metadata        =   dict(smoothing_l=smoothing_lengths, color=colors , ms=sizes, alpha=alphas)
        self.metadata   =   pd.DataFrame(metadata, index=ptypes)
        self.ptypes     =   ptypes

        # set up data
        data            =   {}
        for i,ptype in enumerate(ptypes):
            args            =   placement_kwargs[i]
            if placement_shape[i] == 'rectangle':
                positions       =   self.place_rectangle(**args)
            data_ptype      =   dict(positions=positions)
            data[ptype]     =   pd.Series(data_ptype)
        self.data       =   pd.Series(data)

    # 2D shapes
    #---------------------------------------------------------------------------

    def place_rectangle(self, smoothing_l=0.1, length=10, height=5, x0=0, y0=0, top=False, filled=False):

        # make arrays for particle positions along length and height
        dx      =   smoothing_l*2
        L       =   np.arange(x0, x0+length+dx, dx)
        H       =   np.arange(y0, y0+height+dx, dx)
        sizeL   =   L.shape[0]
        sizeH   =   H.shape[0]

        # determine size of positions array
        if filled:
            size    =   sizeL * sizeH
            X       =   np.zeros( (size,2) )
            # pdb.set_trace()
            for i,y in enumerate(H):
                i_0             =   i * sizeL
                i_f             =   (i+1) * sizeL
                X[i_0:i_f,0]    =   L
                X[i_0:i_f,1]    =   np.ones(sizeL)*y
        else:
            size    =   sizeL + 2*(sizeH-1)
            if top: size += sizeL - 2
            X       =   np.zeros( (size,2) )

            # fill in X
            i_f1            =   sizeL
            i_f2            =   i_f1 + sizeH-1
            i_f3            =   i_f2 + sizeH-1
            X[:i_f1,0]      =   L
            X[:i_f1,1]      =   np.ones(sizeL)*y0
            # pdb.set_trace()
            X[i_f1:i_f2,0]  =   np.ones(sizeH-1)*x0
            X[i_f1:i_f2,1]  =   H[1:]
            X[i_f2:i_f3,0]  =   np.ones(sizeH-1)*(x0+length)
            X[i_f2:i_f3,1]  =   H[1:]
            if top:
                i_f4            =   i_f3 + sizeL-2
                X[i_f3:i_f4,0]  =   L[1:-1]
                X[i_f3:i_f4,1]  =   np.ones(sizeL-2)*(y0+height)

        # data    =   dict(positions=X)
        # return pd.Series(data)
        return X

        # X   =   []
        #
        # # make arrays for particle positions along length and height
        # L   =   np.arange(x0, x0+length+smoothing_l, smoothing_l*2)
        # H   =   np.arange(y0+smoothing_l*2, y0+height+smoothing_l, smoothing_l*2)
        #
        # if filled:
        #     for x in L:
        #         for y in H:
        #             X.append([x,y])
        # else:
        #     if bottom:
        #         for x in L: X.append([x,y0])
        #     if top:
        #         for x in L: X.append([x,y0+height])
        #     if left_side:
        #         for y in H: X.append([x0,y])
        #     if right_side:
        #         for y in H: X.append([x0+length,y])
        #
        # data    =   dict(positions=X, smoothing_l=smoothing_l)
        # return pd.Series(data)

    # SPH fundamentals
    #---------------------------------------------------------------------------

    # def kernel(distance, smoothing_l):
    #
    #     q   =   distance/smoothing_l
    #
    #     f0  =   10/(7*np.pi) / smoothing_l**self.dimention
    #     f1  =   1 - (3/2)*q**2 + (3/4)*q**3
    #     f2  =   (1/4)*(2-q)**3
    #     f3  =   0
    #
    #     h1  =   np.heavside(q,.5) - np.heavside(q-1,.5)
    #     h2  =   np.heavside(q-1,.5) - np.heavside(q-2,.5)
    #     h3  =   np.heaviside(q-2,.5)
    #
    #     return f0*(h1*f1 + h2*f2 + h3*f3)
    #
    # def kernel_gradient(pos_vector, smoothing_l):
    #     NotImplemented

    def kernel(distance,smoothing_l):


    def density(self):
        NotImplemented

    # acceleration
    #---------------------------------------------------------------------------



    # figures and animation
    #---------------------------------------------------------------------------

    def make_image(self,fig,ax):

        ax.clear()
        ax.set_aspect(1)

        meta    =   self.metadata
        data    =   self.data
        ax.plot(*data['positions'], meta['color'], ms=meta['ms'], alpha=meta['alpha'])
