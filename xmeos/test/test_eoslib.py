import numpy as np
import xmeos
from xmeos import models
from xmeos import eoslib
from xmeos.models import core

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
class TestThomas2013():
    def test_chondrite_melt(self):
        meltcomp = {'En':62, 'Fo':24, 'Fa':8, 'An':4, 'Di':2}
        eos_mod = eoslib.CMASF_melt_Thomas2013(meltcomp, kind='endmem')
        P_a = np.linspace(5,135,51)
        # P_a = np.array([5,25,50,75,100,125])
        T = 4000
        V_a = eos_mod.volume(P_a, T)
        # plt.ion()
        # plt.figure()
        # plt.plot(P_a,V_a,'k')
        # assert False

    def _test_endmem_melt(self, eos_mod):
        """
        NOTE These easily become unphysical at low temps
        """
        P_a = np.array([5,25,50,75,100,125])
        T = 4000
        V_a = eos_mod.volume(P_a, T)

        # print(V_a)

    def test_endmem_En(self):
        eos_mod,comp_d = eoslib.CMASF_melt_Thomas2013.get_endmem_eos(endmem='En')
        self._test_endmem_melt(eos_mod)

    def test_endmem_Fo(self):
        eos_mod,comp_d = eoslib.CMASF_melt_Thomas2013.get_endmem_eos(endmem='Fo')
        self._test_endmem_melt(eos_mod)

    def test_endmem_Fa(self):
        eos_mod,comp_d = eoslib.CMASF_melt_Thomas2013.get_endmem_eos(endmem='Fa')
        self._test_endmem_melt(eos_mod)

    def test_endmem_An(self):
        eos_mod,comp_d = eoslib.CMASF_melt_Thomas2013.get_endmem_eos(endmem='An')
        self._test_endmem_melt(eos_mod)

    def test_endmem_Di(self):
        eos_mod,comp_d = eoslib.CMASF_melt_Thomas2013.get_endmem_eos(endmem='Di')
        self._test_endmem_melt(eos_mod)
#====================================================================
class TestMgSiO3RTPress():
    def test_basic_isotherms(self):
        eos_mod = eoslib.MgSiO3_RTPress()
        refstate_calc = eos_mod.calculators['refstate']

        V0  = refstate_calc.ref_volume()
        T0  = refstate_calc.ref_temp()
        # V0, T0 = eos_mod.get_param_values(param_names=['V0','T0'])
        V_a = V0*np.linspace(.4,1.15,1001)
        # P_a = np.linspace(5,135,51)
        # P_a = np.array([5,25,50,75,100,125])
        P2500_a = eos_mod.press(V_a, 2500)
        P3000_a = eos_mod.press(V_a, 3000)
        P5000_a = eos_mod.press(V_a, 5000)
        E2500_a = eos_mod.internal_energy(V_a, 2500)
        E3000_a = eos_mod.internal_energy(V_a, 3000)
        E5000_a = eos_mod.internal_energy(V_a, 5000)
        # T = 4000
        # assert False, 'stop'
        # V_a = eos_mod.volume(P_a, T)

        plt.ion()
        plt.figure()
        plt.plot(V_a/V0,P2500_a,'b-')
        plt.plot(V_a/V0,P3000_a,'k-')
        plt.plot(V_a/V0,P5000_a,'r-')
        plt.ylim(-1,179)
        plt.xlim(0.4,1.13)

        plt.ion()
        plt.figure()
        plt.plot(V_a/V0,E2500_a,'b-')
        plt.plot(V_a/V0,E3000_a,'k-')
        plt.plot(V_a/V0,E5000_a,'r-')
        plt.xlim(0.4,1.13)



        eos_mod.thermal_press(V_a, 5000)
        # assert False

#====================================================================
