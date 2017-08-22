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
def test_simplify_poly():
    TOL = 1e-6
    shifted_coefs = np.random.rand(6)
    abs_coefs = core.simplify_poly(shifted_coefs)

    x = np.linspace(-20,20,100)

    dev = np.polyval(shifted_coefs, x-1) - np.polyval(abs_coefs, x)

    assert np.all(np.abs(dev) < TOL), \
        'Simplified polynomial must match to within TOL everywhere.'

#====================================================================
