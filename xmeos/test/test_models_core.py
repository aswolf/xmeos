from __future__ import absolute_import, print_function, division
import numpy as np
from models import core

import pytest
import matplotlib.pyplot as plt
import matplotlib as mpl
from abc import ABCMeta, abstractmethod
import copy
#====================================================================
# Define "slow" tests
#  - indicated by @slow decorator
#  - slow tests are run only if using --runslow cmd line arg
#====================================================================
slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)


#====================================================================
# SEC:3 Test Admin Funcs
#====================================================================
def test_shift_poly():
    TOL = 1e-6

    V0 = 0.408031
    coefs = np.array([127.116,-3503.98,20724.4,-60212.,86060.5,-48520.4])
    shift_coefs = core.shift_poly(coefs, xscale=V0)

    undo_coefs = core.unshift_poly(shift_coefs, xscale=V0)

    assert np.all(np.abs(coefs-undo_coefs) < TOL), \
        'Shifted and unshifted polynomial coefs must match originals within TOL everywhere.'

    # shift_coefs = np.array([-105.88087217, -1.97201769, 4.4888164,  -36.1310988 ,
    #    -358.36482008, -548.76975936])

    V = V0*np.linspace(0.6,1.2,101)

    dev = np.polyval(shift_coefs[::-1], V/V0-1) - np.polyval(coefs[::-1], V)

    assert np.all(np.abs(dev) < TOL), \
        'Shifted polynomial curve must match original to within TOL everywhere.'
#====================================================================
