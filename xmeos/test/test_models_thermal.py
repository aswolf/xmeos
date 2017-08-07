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
class BaseTestThermalEos(test_models.BaseTestEos):
    def test_heat_capacity_0K(self):
        self.calc_test_heat_capacity(T0=0)

    def test_heat_capacity_300K(self):
        self.calc_test_heat_capacity(T0=300)

    def test_entropy_0K(self):
        self.calc_test_entropy(T0=0)

    def test_entropy_300K(self):
        self.calc_test_entropy(T0=300)

    def calc_test_heat_capacity(self, T0=0):
        TOL = 1e-3

        Nsamp = 10001
        eos_mod = self.load_eos(T0=T0)

        Tmod_a = np.linspace(300.0, 3000.0, Nsamp)
        dT = Tmod_a[1] - Tmod_a[0]

        energy_a = eos_mod.energy(Tmod_a)
        heat_capacity_a = eos_mod.heat_capacity(Tmod_a)

        abs_err, rel_err, range_err = self.numerical_deriv(
            Tmod_a, energy_a, heat_capacity_a, scale=1)

        assert rel_err < TOL, 'rel-error in Cv, ' + np.str(rel_err) + \
            ', must be less than TOL, ' + np.str(TOL)

    def calc_test_entropy(self, T0=0):
        TOL = 1e-3

        Nsamp = 10001
        eos_mod = self.load_eos(T0=T0)

        Tmod_a = np.linspace(300.0, 3000.0, Nsamp)
        dT = Tmod_a[1] - Tmod_a[0]

        entropy_a = eos_mod.entropy(Tmod_a)
        heat_capacity_a = eos_mod.heat_capacity(Tmod_a)

        abs_err, rel_err, range_err = self.numerical_deriv(
            Tmod_a, entropy_a, heat_capacity_a, scale=Tmod_a)

        assert rel_err < TOL, 'rel-error in Cv, ' + np.str(rel_err) + \
            ', must be less than TOL, ' + np.str(TOL)
#====================================================================

#====================================================================
# SEC:2 Implimented Test Clases
#====================================================================
class TestDebye(BaseTestThermalEos):
    def load_eos(self, T0=0):
        # add T0
        eos_mod = models.ThermalEos(kind='Debye')
        eos_mod.set_param_values(param_names=['T0'],param_values=[T0])
        return eos_mod
#====================================================================
class TestEinstein(BaseTestThermalEos):
    def load_eos(self, T0=0):
        eos_mod = models.ThermalEos(kind='Einstein')
        eos_mod.set_param_values(param_names=['T0'],param_values=[T0])
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
