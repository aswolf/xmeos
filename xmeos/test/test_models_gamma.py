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
class BaseTestGammaEos(test_models.BaseTestEos):
    def test_gamma(self):
        TOL = 1e-3

        Nsamp = 10001
        eos_mod = self.load_eos()

        V0, gamma0 = eos_mod.get_param_values(param_names=['V0','gamma0'])
        Vmod_a = np.linspace(.6,1.1,Nsamp)*V0
        dV = Vmod_a[1] - Vmod_a[0]

        temp_a = eos_mod.temp(Vmod_a)
        gamma_a = eos_mod.gamma(Vmod_a)

        abs_err, rel_err, range_err = self.numerical_deriv(
            Vmod_a, temp_a, gamma_a, scale=-Vmod_a/temp_a)

        assert abs_err/gamma0 < TOL, (
            'range error in Press, ' + np.str(range_err) +
            ', must be less than TOL, ' + np.str(TOL))

    # def test_gamma_deriv(self):

    #     TOL = 1e-3

    #     Nsamp = 10001
    #     eos_mod = self.load_eos()

    #     V0, = eos_mod.get_param_values(param_names='V0')
    #     Vmod_a = np.linspace(.7,1.2,Nsamp)*V0
    #     dV = Vmod_a[1] - Vmod_a[0]

    #     press_a = eos_mod.press(Vmod_a)
    #     energy_a = eos_mod.energy(Vmod_a)

    #     abs_err, rel_err, range_err = self.numerical_deriv(
    #         Vmod_a, energy_a, press_a, scale=-core.CONSTS['PV_ratio'])

    #     assert range_err < TOL, 'range error in Press, ' + np.str(range_err) + \
    #         ', must be less than TOL, ' + np.str(TOL)
#====================================================================

#====================================================================
class TestGammaPowLaw(BaseTestGammaEos):
    def load_eos(self):
        eos_mod = models.GammaEos(kind='GammaPowLaw')
        return eos_mod
#====================================================================


#====================================================================
# class TestGammaComparison():
#     def init_params(self,eos_d):
#         VR = 1.0
#         gammaR = 1.0
#         gammapR = -1.0
#         qR = gammapR/gammaR
#         # qR = +1.0
#         # qR = +0.5
#
#         param_key_a = ['VR','gammaR','gammapR','qR']
#         param_val_a = np.array([VR,gammaR,gammapR,qR])
#         core.set_params( param_key_a, param_val_a, eos_d )
#
#         return eos_d
#
#     def load_gamma_mod(self, eos_d):
#         gamma_mod = gamma.GammaPowLaw()
#         core.set_modtypes( ['GammaMod'], [gamma_mod], eos_d )
#
#         pass
#
#     def test_gamma(self):
#         eos_d = self.init_params({})
#         VR = eos_d['param_d']['VR']
#         TR = 1000.0
#
#         eos_pow_d = copy.deepcopy(eos_d)
#         eos_str_d = copy.deepcopy(eos_d)
#
#
#         core.set_modtypes( ['GammaMod'], [gamma.GammaPowLaw],
#                                     eos_pow_d )
#         core.set_modtypes( ['GammaMod'], [gamma.GammaFiniteStrain],
#                                     eos_str_d )
#
#         gammaR = eos_d['param_d']['gammaR']
#         qR = eos_d['param_d']['qR']
#
#
#         N = 1001
#         V_a = VR*np.linspace(0.4,1.3,N)
#         dV = V_a[1]-V_a[0]
#
#         gam_pow_mod = eos_pow_d['modtype_d']['GammaMod'](V0ref=False )
#         gam_str_mod = eos_str_d['modtype_d']['GammaMod'](V0ref=False )
#
#
#         gam_pow_a = gam_pow_mod.gamma(V_a,eos_pow_d)
#         gam_str_a = gam_str_mod.gamma(V_a,eos_str_d)
#
#         temp_pow_a = gam_pow_mod.temp(V_a,TR,eos_pow_d)
#         temp_str_a = gam_str_mod.temp(V_a,TR,eos_str_d)
#
#         q_pow_a = V_a/gam_pow_a*np.gradient(gam_pow_a,dV)
#         q_str_a = V_a/gam_str_a*np.gradient(gam_str_a,dV)
#
#
#         # mpl.rcParams(fontsize=16)
#         plt.ion()
#         plt.figure()
#
#         plt.clf()
#         hleg = plt.plot(V_a,q_pow_a,'k--',V_a,q_str_a,'r-',lw=2)
#         plt.legend(hleg,['Power-Law','Finite Strain'], loc='upper right',fontsize=16)
#         plt.xlabel('$V / V_0$',fontsize=16)
#         plt.ylabel('$q$',fontsize=16)
#         plt.text(.9,1.1*qR,'$(\gamma_0,q_0) = ('+np.str(gammaR)+','+np.str(qR)+')$',fontsize=20)
#
#         plt.savefig('test/figs/gamma-q-comparison.png',dpi=450)
#
#         # from IPython import embed; embed(); import ipdb; ipdb.set_trace()
#
#         # plt.clf()
#         # hleg = plt.plot(1.0/V_a,gam_str_a,'r-',lw=2)
#
#         # eos_str_d['param_d']['gammapR'] = -0.5
#         # eos_str_d['param_d']['gammapR'] = -2
#
#         # eos_str_d['param_d']['gammapR'] = -1.0
#         # eos_str_d['param_d']['gammaR'] = 0.5
#         # eos_str_d['param_d']['gammapR'] = -2.0
#         # eos_str_d['param_d']['gammapR'] = -10.0
#
#         # eos_str_d['param_d']['gammaR'] = 0.75
#         # eos_str_d['param_d']['gammapR'] = -10.0
#         # eos_str_d['param_d']['gammapR'] = -30.0
#
#         gam_str_a = gam_str_mod.gamma(V_a,eos_str_d)
#         eos_str_d['param_d']['gammapR'] = -0.5
#
#         plt.clf()
#         hleg = plt.plot(V_a,gam_pow_a,'k--',V_a,gam_str_a,'r-',lw=2)
#         plt.legend(hleg,['Power-Law','Finite Strain'], loc='upper right',fontsize=16)
#         plt.xlabel('$V / V_0$',fontsize=16)
#         plt.ylabel('$\gamma$',fontsize=16)
#
#         plt.text(.9,1.1*gammaR,'$(\gamma_0,q_0) = ('+np.str(gammaR)+','+np.str(qR)+')$',fontsize=20)
#
#         plt.savefig('test/figs/gamma-comparison.png',dpi=450)
#
#
#
#
#         plt.clf()
#         hleg = plt.plot(V_a,temp_pow_a,'k--',V_a,temp_str_a,'r-',lw=2)
#         plt.legend(hleg,['Power-Law','Finite Strain'], loc='upper right',
#                    fontsize=16)
#         plt.xlabel('$V / V_0$',fontsize=16)
#         plt.ylabel('$T\; [K]$',fontsize=16)
#         plt.text(.9,1.1*TR,'$(\gamma_0,q_0) = ('+np.str(gammaR)+','+np.str(qR)+')$',fontsize=20)
#         plt.savefig('test/figs/gamma-temp-comparison.png',dpi=450)
#====================================================================
