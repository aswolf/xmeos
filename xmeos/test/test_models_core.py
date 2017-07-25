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
class TestControl(object):
    def test_get_array_params(self):
        TOL = 1e-6
        eos_d, acoef_a = self.init_params()
        param_a = core.get_array_params('acoef',eos_d)

        assert np.all(np.abs(param_a-acoef_a)<TOL), 'Stored and retrieved parameter array do not match within TOL'

        param_a = core.get_array_params('V0',eos_d)
        assert param_a.size==0, 'non-array parameter should not be retrievable with get_array_params()'

        pass

    def test_set_array_params(self):
        TOL = 1e-6
        eos_d = {}
        # Set model parameter values
        E0 = 0.0 # eV/atom
        V0 = 38.0 # 1e-5 m^3 / kg
        K0 = 25.0 # GPa
        KP0 = 9.0 # 1
        acoef_a = np.array([1.3,-.23,9.99,-88])

        param_key_a = ['V0','K0','KP0','E0']
        param_val_a = np.array([ V0, K0, KP0, E0 ])
        core.set_params( param_key_a, param_val_a, eos_d )

        core.set_array_params( 'acoef', acoef_a, eos_d )
        core.set_consts( [], [], eos_d )

        param_a =  core.get_array_params( 'acoef', eos_d )

        assert np.all(np.abs(param_a-acoef_a)<TOL), 'Stored and retrieved parameter array do not match within TOL'

        pass

    def init_params(self):
        eos_d = {}
        # Set model parameter values
        E0 = 0.0 # eV/atom
        V0 = 38.0 # 1e-5 m^3 / kg
        K0 = 25.0 # GPa
        KP0 = 9.0 # 1
        acoef = np.array([1.3,-.23,9.99,-88])

        param_key_a = ['V0','K0','KP0','E0','acoef_0','acoef_1','acoef_2','acoef_3']
        param_val_a = np.array([ V0, K0, KP0, E0, acoef[0], acoef[1], acoef[2], acoef[3] ])

        core.set_consts( [], [], eos_d )

        core.set_params( param_key_a, param_val_a, eos_d )
        return eos_d, acoef
#====================================================================
