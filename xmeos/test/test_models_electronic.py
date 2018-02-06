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

class TestCvPowLaw(test_models.BaseTestEos):
    def load_eos(self):
        eos_mod = models.ElectronicEos(kind='CvPowLaw')
        return eos_mod

    def test_heat_capacity_T(self):
        TOL = 1e-3
        Nsamp = 100001
        eos_mod = self.load_eos()

        Tmod_a = np.linspace(2000.0, 6000.0, Nsamp)

        V0, = eos_mod.get_param_values(param_names=['V0'])
        # Vmod = V0*(0.6+.5*np.random.rand(Nsamp))
        Vmod = V0*0.7

        energy_a = eos_mod.energy(Vmod, Tmod_a)
        heat_capacity_a = eos_mod.heat_capacity(Vmod, Tmod_a)

        abs_err, rel_err, range_err = self.numerical_deriv(
            Tmod_a, energy_a, heat_capacity_a, scale=1)

        assert abs_err < TOL, 'rel-error in Cv, ' + np.str(abs_err) + \
            ', must be less than TOL, ' + np.str(TOL)

    def test_press_simple(self):
        TOL = 1e-3
        Nsamp = 10001
        eos_mod = self.load_eos()

        V0, = eos_mod.get_param_values(param_names=['V0'])
        Vmod_a = np.linspace(.7,1.2,Nsamp)*V0
        T = 10000
        dV = Vmod_a[1] - Vmod_a[0]

        P_a = eos_mod.press(Vmod_a, T)
        F_a = eos_mod.helmholtz_energy(Vmod_a, T)
        abs_err, rel_err, range_err = self.numerical_deriv(
              Vmod_a, F_a, P_a, scale=-core.CONSTS['PV_ratio'])

        # Pdiff = -xmeos.models.CONSTS['PV_ratio']*np.gradient(F_a,dV)

        # calc = eos_mod.calculators['electronic']

        # V_a = Vmod_a
        # T_a = T
        # V_a, T_a = core.fill_array(V_a, T_a)

        # dV = V_a[1]-V_a[0]
        # PV_ratio, = core.get_consts(['PV_ratio'])

        # CvFac = calc._calc_CvFac(V_a)
        # CvFac_deriv = calc._calc_CvFac(V_a, deriv=1)

        # Tel = calc._calc_Tel(V_a)
        # Tel_deriv = calc._calc_Tel(V_a, deriv=1)

        # F1 = -CvFac*(.5*(T_a**2 - Tel**2))
        # F2 = +CvFac*T_a*Tel*np.log(T_a/Tel)

        # P1 = -PV_ratio*(
        #     -CvFac_deriv*(.5*(T_a**2 - Tel**2))
        #     +CvFac*Tel*Tel_deriv
        #     )
        # P2 = -PV_ratio*(
        #     +CvFac_deriv*T_a*Tel*np.log(T_a/Tel)
        #     +CvFac*T_a*Tel_deriv*np.log(T_a/Tel)
        #     -CvFac*T_a*Tel_deriv
        #     )

        assert abs_err < TOL, ('abs error in Press, ' + np.str(abs_err) +
                                 ', must be less than TOL, ' + np.str(TOL))

        # plt.plot(V_a, P1,'k-', V_a, -PV_ratio*np.gradient(F1,dV),'r--')
        # plt.plot(V_a, P2,'k-', V_a, -PV_ratio*np.gradient(F2,dV),'r--')

        # plt.plot(V_a, P1+P2,'k-',
        #          V_a, -PV_ratio*np.gradient(F1+F2,dV),'r--',lw=2)

        # plt.plot(V_a, P_a, 'gx')
        # plt.plot(V_a, -PV_ratio*np.gradient(F_a,dV), 'm-')

        # plt.clf()
        # plt.plot(V_a, F_a, 'k-', V_a, F1+F2,'r--')

        # plt.clf()
        # plt.plot(V_a, P_a, 'k-', V_a, P1+P2,'r--')


        # plt.plot(Vmod_a,P_a,'k-',Vmod_a,Pdiff,'r--')

    def test_entropy(self):
        TOL = 1e-3
        Nsamp = 100001
        eos_mod = self.load_eos()

        Tmod_a = np.linspace(2000.0, 6000.0, Nsamp)
        dT = Tmod_a[1] - Tmod_a[0]
        V0, = eos_mod.get_param_values(param_names=['V0'])

        # Vmod = V0*(0.6+.5*np.random.rand(Nsamp))
        Vmod = V0*0.7

        S_a = eos_mod.entropy(Vmod, Tmod_a)
        F_a = eos_mod.helmholtz_energy(Vmod, Tmod_a)

        abs_err, rel_err, range_err = self.numerical_deriv(
              Tmod_a, F_a, S_a, scale=-1)


        assert abs_err < TOL, ('abs error in entropy, ' + np.str(abs_err) +
                                 ', must be less than TOL, ' + np.str(TOL))
