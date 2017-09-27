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

        print(V_a)

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
