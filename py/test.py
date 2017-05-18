import numpy as np

import pytest
from numpy.testing import assert_array_almost_equal

def test_import_star():
    try:
        import star
    except ImportError:
        raise AssertionError("can't import 'star.py'")

def test_import_auxillary():
    try:
        import auxillary
    except ImportError:
        raise AssertionError("can't import 'auxillary.py'")

import star as star
import auxillary as aux

def test_model_setup():

    # test that random particle placement is inside cloud
    model   =   star.model
    r_jean  =   model['r_jean']
    R       =   model['R'][0]
    X       =   R[:,0]
    Y       =   R[:,1]
    Z       =   R[:,2]
    radii   =   np.sqrt( X**2 + Y**2 + Z**2 )
    assert r_jean >= np.max(radii), "particles should be contained inside Jean's radius of cloud"

    # test that the smoothing lengths are smaller than reans radius
    h       =   model['h'][0]
    assert r_jean > h, "smoothing lengths must be smaller than cloud"

    # test that the mass of the cloud is consistent after SPH
    Mc      =   model['Mc']
    rho     =   model['rho'][0]
    vol     =   (4/3) * np.pi * h**3
    M       =   np.sum(rho * vol)
    assert all(( M/Mc > .9 , M/Mc < 1.1 )), "mass is inconsistent after SPH"
