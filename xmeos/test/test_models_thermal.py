from __future__ import absolute_import, print_function, division, with_statement
from builtins import object
import numpy as np
import xmeos
from xmeos import models
from xmeos.models import core

import pytest
import matplotlib.pyplot as plt
import matplotlib as mpl
from abc import ABCMeta, abstractmethod
import copy

import test_models

try:
   import cPickle as pickle
except:
   import pickle


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
class BaseTestThermalEnergyEos(test_models.BaseTestEos):
    def test_heat_capacity(self):
        TOL = 1e-3

        Nsamp = 10001
        eos_mod = self.load_eos()

        Tmod_a = np.linspace(300.0, 3000.0, Nsamp)
        dT = Tmod_a[1] - Tmod_a[0]

        energy_a = eos_mod.energy(Tmod_a)
        heat_capacity_a = eos_mod.heat_capacity(Tmod_a)

        abs_err, rel_err, range_err = self.numerical_deriv(
            Tmod_a, energy_a, heat_capacity_a, scale=1)

        assert rel_err < TOL, 'rel-error in Cv, ' + np.str(rel_err) + \
            ', must be less than TOL, ' + np.str(TOL)
#====================================================================

#====================================================================
# SEC:2 Implimented Test Clases
#====================================================================
class TestDebye(BaseTestThermalEnergyEos):
    def load_eos(self):
        eos_mod = models.ThermalEnergyEos(
            kind='Debye', level_const=100)
        return eos_mod
#====================================================================

# 2.2: ThermalPathMod Tests
#====================================================================
# class TestGenRosenfeldTaranzona(BaseTestThermalPathMod):
#     def load_thermal_path_mod(self, eos_d):
#         thermal_path_mod = thermal.GenRosenfeldTaranzona(path_const='V')
#         core.set_modtypes( ['ThermalPathMod'], [thermal_path_mod], eos_d )
#
#         pass
#
#     def init_params(self,eos_d):
#         # Set model parameter values
#         acoef = -158.2
#         bcoef = .042
#         mexp = 3/5
#         lognfac = 0.0
#         T0 = 5000.0
#
#         param_key_a = ['acoef','bcoef','mexp','lognfac','T0']
#         param_val_a = np.array([acoef,bcoef,mexp,lognfac,T0])
#
#         core.set_consts( [], [], eos_d )
#         self.load_thermal_path_mod( eos_d )
#
#         core.set_params( param_key_a, param_val_a, eos_d )
#
#         return eos_d
#====================================================================
