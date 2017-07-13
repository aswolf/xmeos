import numpy as np
from models import thermal
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
class BaseTestThermalPathMod(object):
    @abstractmethod
    def load_thermal_path_mod(self, eos_d):
        assert False, 'must implement load_thermal_path_mod()'

    @abstractmethod
    def init_params(self,eos_d):
        assert False, 'must implement init_params()'
        return eos_d

    def test_heat_capacity(self):
        Nsamp = 10001
        eos_d = self.init_params({})

        param_d = eos_d['param_d']
        Tmod_a = np.linspace(.7,1.3,Nsamp)*param_d['T0']
        dT = Tmod_a[1] - Tmod_a[0]

        # print eos_d['modtype_d']
        thermal_path_mod = eos_d['modtype_d']['ThermalPathMod']

        heat_capacity_a = thermal_path_mod.heat_capacity(Tmod_a,eos_d)
        energy_a = thermal_path_mod.energy(Tmod_a,eos_d)

        heat_capacity_num_a = np.gradient(energy_a,dT)

        E_range = np.max(energy_a)-np.min(energy_a)
        T_range = Tmod_a[-1]-Tmod_a[0]
        Cv_scl = E_range/T_range
        # Cv_range = np.max(heat_capacity_a)-np.min(heat_capacity_a)

        Cv_diff_a = heat_capacity_num_a-heat_capacity_a
        # Cverr =  np.max(np.abs(Cv_diff_a/Cv_range))
        Cverr =  np.max(np.abs(Cv_diff_a/Cv_scl))
        CVTOL = 1.0/Nsamp

        # print self
        # print PTOL*Prange


        # def plot_press_mismatch(Tmod_a,press_a,press_num_a):
        #     plt.figure()
        #     plt.ion()
        #     plt.clf()
        #     plt.plot(Tmod_a,press_num_a,'bx',Tmod_a,press_a,'r-')
        #     from IPython import embed; embed(); import ipdb; ipdb.set_trace()

        # plot_press_mismatch(Tmod_a,press_a,press_num_a)

        assert np.abs(Cverr) < CVTOL, '(Cv error)/Cv_scl, ' + np.str(Cverr) + \
            ', must be less than CVTOL, ' + np.str(CVTOL)
#====================================================================
class BaseTestThermalMod(object):
    @abstractmethod
    def load_thermal_mod(self, eos_d):
        assert False, 'must implement load_thermal_mod()'

    @abstractmethod
    def init_params(self,eos_d):
        assert False, 'must implement init_params()'
        return eos_d

    def test_heat_capacity_isochore(self):
        Nsamp = 10001
        eos_d = self.init_params({})

        param_d = eos_d['param_d']
        Viso = 0.7*param_d['V0']
        Tmod_a = np.linspace(.7,1.3,Nsamp)*param_d['T0']
        dT = Tmod_a[1] - Tmod_a[0]

        # print eos_d['modtype_d']
        thermal_mod = eos_d['modtype_d']['ThermalMod']

        heat_capacity_a = thermal_mod.heat_capacity(Viso,Tmod_a,eos_d)
        energy_a = np.squeeze( thermal_mod.energy(Viso,Tmod_a,eos_d) )

        heat_capacity_num_a = np.gradient(energy_a,dT)

        E_range = np.max(energy_a)-np.min(energy_a)
        T_range = Tmod_a[-1]-Tmod_a[0]
        Cv_scl = E_range/T_range
        # Cv_range = np.max(heat_capacity_a)-np.min(heat_capacity_a)

        Cv_diff_a = heat_capacity_num_a-heat_capacity_a
        # Cverr =  np.max(np.abs(Cv_diff_a/Cv_range))
        Cverr =  np.max(np.abs(Cv_diff_a/Cv_scl))
        CVTOL = 1.0/Nsamp

        # print self
        # print PTOL*Prange


        # def plot_press_mismatch(Tmod_a,press_a,press_num_a):
        #     plt.figure()
        #     plt.ion()
        #     plt.clf()
        #     plt.plot(Tmod_a,press_num_a,'bx',Tmod_a,press_a,'r-')
        #     from IPython import embed; embed(); import ipdb; ipdb.set_trace()

        # plot_press_mismatch(Tmod_a,press_a,press_num_a)

        assert np.abs(Cverr) < CVTOL, '(Cv error)/Cv_scl, ' + np.str(Cverr) + \
            ', must be less than CVTOL, ' + np.str(CVTOL)
#====================================================================

#====================================================================
# SEC:2 Implimented Test Clases
#====================================================================
# 2.2: ThermalPathMod Tests
#====================================================================
class TestGenRosenfeldTaranzona(BaseTestThermalPathMod):
    def load_thermal_path_mod(self, eos_d):
        thermal_path_mod = thermal.GenRosenfeldTaranzona(path_const='V')
        core.set_modtypes( ['ThermalPathMod'], [thermal_path_mod], eos_d )

        pass

    def init_params(self,eos_d):
        # Set model parameter values
        acoef = -158.2
        bcoef = .042
        mexp = 3.0/5
        lognfac = 0.0
        T0 = 5000.0

        param_key_a = ['acoef','bcoef','mexp','lognfac','T0']
        param_val_a = np.array([acoef,bcoef,mexp,lognfac,T0])

        core.set_consts( [], [], eos_d )
        self.load_thermal_path_mod( eos_d )

        core.set_params( param_key_a, param_val_a, eos_d )

        return eos_d
#====================================================================
