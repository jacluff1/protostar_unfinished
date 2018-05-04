import numpy as np

class kernel_gauss:

    def __init__(self,dimention):
        self.dimention  =   dimention

    def W(distance,smoothing_l):
        NotImplemented

    def gradient_W(pos_vector,smoothing_l):
        NotImplemented

class kernel_spline:

    def __init__(self,dimention):

        self.dimention  =   dimention

        if dimention == 1:
            self.norm_const =   2/3
        elif dimention == 2:
            self.norm_const =   10/(7*np.pi)
        elif:
            self.norm_const =   1/np.pi
        else:
            raise ValueError, "dimention must be 0 < integer < 4"

    def W(distance,smoothing_l):
        q   =   distance/smoothing_l

        f0  =   self.norm_const / smoothing_l**self.dimention
        f1  =   1 - (3/2)*q**2 + (3/4)*q**3
        f2  =   (1/4)*(2-q)**3
        f3  =   0

        h1  =   np.heavside(q,.5) - np.heavside(q-1,.5)
        h2  =   np.heavside(q-1,.5) - np.heavside(q-2,.5)
        h3  =   np.heaviside(q-2,.5)

        return f0*(h1*f1 + h2*f2 + h3*f3)
