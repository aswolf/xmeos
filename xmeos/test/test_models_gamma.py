import numpy as np
from models import core
from models import gamma

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
#
#    NOTE: This only produces figures for gamma models
#
#====================================================================

#====================================================================
class TestGammaComparison():
    def init_params(self,eos_d):
        VR = 1.0
        gammaR = 1.0
        gammapR = -1.0
        qR = gammapR/gammaR
        # qR = +1.0
        # qR = +0.5

        param_key_a = ['VR','gammaR','gammapR','qR']
        param_val_a = np.array([VR,gammaR,gammapR,qR])
        core.set_params( param_key_a, param_val_a, eos_d )

        return eos_d

    def load_gamma_mod(self, eos_d):
        gamma_mod = gamma.GammaPowLaw()
        core.set_modtypes( ['GammaMod'], [gamma_mod], eos_d )

        pass

    def test_gamma(self):
        eos_d = self.init_params({})
        VR = eos_d['param_d']['VR']
        TR = 1000.0

        eos_pow_d = copy.deepcopy(eos_d)
        eos_str_d = copy.deepcopy(eos_d)


        core.set_modtypes( ['GammaMod'], [gamma.GammaPowLaw],
                                    eos_pow_d )
        core.set_modtypes( ['GammaMod'], [gamma.GammaFiniteStrain],
                                    eos_str_d )

        gammaR = eos_d['param_d']['gammaR']
        qR = eos_d['param_d']['qR']


        N = 1001
        V_a = VR*np.linspace(0.4,1.3,N)
        dV = V_a[1]-V_a[0]

        gam_pow_mod = eos_pow_d['modtype_d']['GammaMod'](V0ref=False )
        gam_str_mod = eos_str_d['modtype_d']['GammaMod'](V0ref=False )


        gam_pow_a = gam_pow_mod.gamma(V_a,eos_pow_d)
        gam_str_a = gam_str_mod.gamma(V_a,eos_str_d)

        temp_pow_a = gam_pow_mod.temp(V_a,TR,eos_pow_d)
        temp_str_a = gam_str_mod.temp(V_a,TR,eos_str_d)

        q_pow_a = V_a/gam_pow_a*np.gradient(gam_pow_a,dV)
        q_str_a = V_a/gam_str_a*np.gradient(gam_str_a,dV)


        # mpl.rcParams(fontsize=16)
        plt.ion()
        plt.figure()

        plt.clf()
        hleg = plt.plot(V_a,q_pow_a,'k--',V_a,q_str_a,'r-',lw=2)
        plt.legend(hleg,['Power-Law','Finite Strain'], loc='upper right',fontsize=16)
        plt.xlabel('$V / V_0$',fontsize=16)
        plt.ylabel('$q$',fontsize=16)
        plt.text(.9,1.1*qR,'$(\gamma_0,q_0) = ('+np.str(gammaR)+','+np.str(qR)+')$',fontsize=20)

        plt.savefig('test/figs/gamma-q-comparison.png',dpi=450)

        # from IPython import embed; embed(); import ipdb; ipdb.set_trace()

        # plt.clf()
        # hleg = plt.plot(1.0/V_a,gam_str_a,'r-',lw=2)

        # eos_str_d['param_d']['gammapR'] = -0.5
        # eos_str_d['param_d']['gammapR'] = -2

        # eos_str_d['param_d']['gammapR'] = -1.0
        # eos_str_d['param_d']['gammaR'] = 0.5
        # eos_str_d['param_d']['gammapR'] = -2.0
        # eos_str_d['param_d']['gammapR'] = -10.0

        # eos_str_d['param_d']['gammaR'] = 0.75
        # eos_str_d['param_d']['gammapR'] = -10.0
        # eos_str_d['param_d']['gammapR'] = -30.0

        gam_str_a = gam_str_mod.gamma(V_a,eos_str_d)
        eos_str_d['param_d']['gammapR'] = -0.5

        plt.clf()
        hleg = plt.plot(V_a,gam_pow_a,'k--',V_a,gam_str_a,'r-',lw=2)
        plt.legend(hleg,['Power-Law','Finite Strain'], loc='upper right',fontsize=16)
        plt.xlabel('$V / V_0$',fontsize=16)
        plt.ylabel('$\gamma$',fontsize=16)

        plt.text(.9,1.1*gammaR,'$(\gamma_0,q_0) = ('+np.str(gammaR)+','+np.str(qR)+')$',fontsize=20)

        plt.savefig('test/figs/gamma-comparison.png',dpi=450)




        plt.clf()
        hleg = plt.plot(V_a,temp_pow_a,'k--',V_a,temp_str_a,'r-',lw=2)
        plt.legend(hleg,['Power-Law','Finite Strain'], loc='upper right',
                   fontsize=16)
        plt.xlabel('$V / V_0$',fontsize=16)
        plt.ylabel('$T\; [K]$',fontsize=16)
        plt.text(.9,1.1*TR,'$(\gamma_0,q_0) = ('+np.str(gammaR)+','+np.str(qR)+')$',fontsize=20)
        plt.savefig('test/figs/gamma-temp-comparison.png',dpi=450)
#====================================================================
