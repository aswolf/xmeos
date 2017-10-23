# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from future.utils import with_metaclass
import numpy as np
import scipy as sp
from abc import ABCMeta, abstractmethod
from scipy import integrate, optimize
import scipy.interpolate as interpolate

from . import core
from . import compress
from . import thermal
from . import gamma

__all__ = ['CompositeEos','MieGruneisenEos','RTPolyEos','RTPressEos']

# class RTPolyEos(with_metaclass(ABCMeta, core.Eos)):
# class RTPressEos(with_metaclass(ABCMeta, core.Eos)):
# class CompressPolyEos(with_metaclass(ABCMeta, core.Eos)):

#====================================================================
class CompositeEos(with_metaclass(ABCMeta, core.Eos)):
    # Meta class

    # MUST be used
    # def __init__():
    # def __repr__():

    # def thermal_energy(self, V_a, T_a):
    # def compress_energy(self, V_a, T_a):

    def entropy(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)
        S0, = self.get_param_values(param_names=['S0'])
        S_compress_a = self.compress_entropy(V_a)
        S_therm_a = self.thermal_entropy(V_a, T_a)

        entropy_a = S0 + S_compress_a + S_therm_a
        return entropy_a

    def press(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)
        compress_calc = self.calculators['compress']
        P_ref_a = compress_calc._calc_press(V_a)
        P_therm_a = self.thermal_press(V_a, T_a)

        press_a = P_ref_a + P_therm_a
        return press_a

    def helmholtz_energy(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)

        E_a = self.internal_energy(V_a, T_a)
        S_a = self.entropy(V_a, T_a)
        F_a = E_a - T_a*S_a

        return F_a

    def internal_energy(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)
        compress_calc = self.calculators['compress']
        compress_path_const = compress_calc.path_const

        if   compress_path_const=='T':
            F0, T0, S0 = self.get_param_values(param_names=['F0','T0','S0'])
            E0 = F0 + T0*S0

        elif (compress_path_const=='S')|(compress_path_const=='0K'):
            E0, = self.get_param_values(param_names=['E0'])

        else:
            raise NotImplementedError(
                'path_const '+path_const+' is not valid for CompressEos.')

        E_compress_a = self.compress_energy(V_a)
        E_therm_a = self.thermal_energy(V_a, T_a)

        internal_energy_a = E0 + E_compress_a + E_therm_a
        return internal_energy_a

    def bulk_modulus(self, V_a, T_a, TOL=1e-4):
        P_lo_a = self.press(V_a*np.exp(-TOL/2), T_a)
        P_hi_a = self.press(V_a*np.exp(+TOL/2), T_a)
        K_a = -(P_hi_a-P_lo_a)/TOL

        return K_a

    def volume(self, P_a, T_a, TOL=1e-3, step=0.1, Kmin=1):
        V0, K0, KP0 = self.get_param_values(param_names=['V0','K0','KP0'])

        Kapprox = K0 + KP0*P_a

        Kscl = 0.5*(K0+Kapprox)
        iV_a = V0*np.exp(-step*P_a/Kscl)
        # iV_a = V0*np.exp(-step*P_a/Kapprox)

        # from IPython import embed;embed();import ipdb as pdb;pdb.set_trace()
        while True:
            #print('V = ', iV_a)
            iK_a = self.bulk_modulus(iV_a, T_a)
            # print('K = ', iK_a)
            iP_a = self.press(iV_a, T_a)
            # print('P = ', iP_a)
            idelP = P_a-iP_a
            # print(idelP/iK_a)
            #print(P_a)

            Kapprox = iK_a + .3*0.5*KP0*idelP

            # iKscl = np.maximum(iK_a,Kmin)
            idelV = np.exp(-idelP/Kapprox)
            # print('idelV = ',idelV)
            iV_a = iV_a*idelV
            if np.all(np.abs(idelV-1) < TOL):
                break

        V_a = iV_a
        return V_a
#====================================================================
class MieGruneisenEos(CompositeEos):
    _kind_thermal_opts = ['Debye','Einstein','ConstHeatCap']
    _kind_gamma_opts = ['GammaPowLaw','GammaShiftedPowLaw','GammaFiniteStrain']
    _kind_compress_opts = ['Vinet','BirchMurn3','BirchMurn4',
                           'GenFiniteStrain','Tait']

    def __init__(self, kind_thermal='Debye', kind_gamma='GammaPowLaw',
                 kind_compress='Vinet', compress_path_const='T',
                 natom=1, model_state={}):
        self._pre_init(natom=natom)

        compress.set_calculator(self, kind_compress, self._kind_compress_opts,
                                path_const=compress_path_const)
        gamma.set_calculator(self, kind_gamma, self._kind_gamma_opts)
        thermal.set_calculator(self, kind_thermal, self._kind_thermal_opts)

        self._set_ref_state()

        self._post_init(model_state=model_state)
        pass

    def __repr__(self):
        calc_thermal = self.calculators['thermal']
        calc_compress = self.calculators['compress']
        calc_gamma = self.calculators['gamma']

        return ("ThermalEos(kind_thermal={kind_thermal}, "
                "kind_gamma={kind_gamma}, "
                "kind_compress={kind_compress}, "
                "compress_path_const={compress_path_const}, "
                "natom={natom}, "
                "model_state={model_state}, "
                ")"
                .format(kind_thermal=repr(calc_thermal.name),
                        kind_gamma=repr(calc_gamma.name),
                        kind_compress=repr(calc_compress.name),
                        compress_path_const=repr(calc_compress.path_const),
                        natom=repr(self.natom),
                        model_state=self.model_state
                        )
                )

    def _set_ref_state(self):
        compress_calc = self.calculators['compress']
        compress_path_const = compress_calc.path_const

        V0, K0 = compress_calc.get_param_defaults(['V0','K0'])
        # redundant T0 declaration
        T0_scale = 300
        Cv_scale = 3*self.natom*core.CONSTS['kboltz']
        S0_scale = Cv_scale*T0_scale
        energy_scale = np.round(V0*K0/core.CONSTS['PV_ratio'],decimals=2)

        if   compress_path_const=='T':
            param_ref_names = ['T0', 'F0', 'S0']
            param_ref_units = ['K', 'eV', 'eV/K']
            param_ref_defaults = [T0_scale, 0.0, S0_scale]
            param_ref_scales = [T0_scale, energy_scale, S0_scale]

        elif compress_path_const=='S':
            param_ref_names = ['T0', 'E0', 'S0']
            param_ref_units = ['K', 'eV', 'eV/K']
            param_ref_defaults = [300, 0.0, S0_scale]
            param_ref_scales = [300, energy_scale, S0_scale]

        elif compress_path_const=='0K':
            param_ref_names = []
            param_ref_units = []
            param_ref_defaults = []
            param_ref_scales = []

        else:
            raise NotImplementedError(
                'path_const '+path_const+' is not valid for CompressEos.')

        self._param_ref_names = param_ref_names
        self._param_ref_units = param_ref_units
        self._param_ref_defaults = param_ref_defaults
        self._param_ref_scales = param_ref_scales
        pass

    def _calc_theta(self, V_a):
        gamma_calc = self.calculators['gamma']

        theta0, = self.get_param_values(['theta0'])
        theta_a = gamma_calc._calc_temp(V_a, T0=theta0)
        return theta_a

    def ref_temp_path(self, V_a):
        T0, theta0 = self.get_param_values(param_names=['T0','theta0'])

        gamma_calc = self.calculators['gamma']
        compress_calc = self.calculators['compress']
        compress_path_const = compress_calc.path_const

        # Tref_path = gamma_calc._calc_temp(V_a)
        theta_ref = self._calc_theta(V_a)

        if   compress_path_const=='T':
            Tref_path = T0

        elif compress_path_const=='S':
            Tref_path = gamma_calc._calc_temp(V_a)

        elif compress_path_const=='0K':
            Tref_path = 0

        else:
            raise NotImplementedError(
                'path_const '+path_const+' is not valid for CompressEos.')

        Tref_path, V_a = core.fill_array(Tref_path, V_a)
        Tref_path, theta_ref = core.fill_array(Tref_path, theta_ref)

        return Tref_path, theta_ref

    def compress_energy(self, V_a):
        V_a = core.fill_array(V_a)
        compress_calc = self.calculators['compress']
        compress_path_const = compress_calc.path_const

        T0, = self.get_param_values(param_names=['T0'])
        if   compress_path_const=='T':
            F_compress = compress_calc._calc_energy(V_a)
            S_compress = self.compress_entropy(V_a)
            E_compress = F_compress + T0*S_compress

        elif (compress_path_const=='S')|(compress_path_const=='0K'):
            E_compress = compress_calc._calc_energy(V_a)

        else:
            raise NotImplementedError(
                'path_const '+path_const+' is not valid for CompressEos.')

        return E_compress

    def compress_entropy(self, V_a):
        V_a = core.fill_array(V_a)

        T0, theta0 = self.get_param_values(param_names=['T0','theta0'])
        Tref_path, theta_ref = self.ref_temp_path(V_a)
        thermal_calc = self.calculators['thermal']

        S_compress = thermal_calc._calc_entropy(Tref_path, theta=theta_ref,
                                                T0=T0, theta0=theta0)
        return S_compress

    def thermal_energy(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)
        Tref_path, theta_ref = self.ref_temp_path(V_a)
        thermal_calc = self.calculators['thermal']

        E_therm_a = thermal_calc._calc_energy(T_a, theta=theta_ref,
                                              T0=Tref_path)
        return E_therm_a

    def thermal_entropy(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)
        Tref_path, theta_ref = self.ref_temp_path(V_a)
        thermal_calc = self.calculators['thermal']

        S_therm_a = thermal_calc._calc_entropy(T_a, theta=theta_ref,
                                               T0=Tref_path, theta0=theta_ref)
        return S_therm_a

    def thermal_press(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)
        gamma_calc = self.calculators['gamma']

        PV_ratio, = core.get_consts(['PV_ratio'])
        gamma_a = gamma_calc._calc_gamma(V_a)
        E_therm_a = self.thermal_energy(V_a, T_a)
        P_therm_a = PV_ratio*gamma_a/V_a*E_therm_a
        return P_therm_a

    def heat_capacity(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)
        thermal_calc = self.calculators['thermal']
        Tref_path, theta_ref = self.ref_temp_path(V_a)

        heat_capacity_a = thermal_calc._calc_heat_capacity(T_a, theta=theta_ref)
        return heat_capacity_a

    def press(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)
        compress_calc = self.calculators['compress']
        P_ref_a = compress_calc._calc_press(V_a)
        P_therm_a = self.thermal_press(V_a, T_a)

        press_a = P_ref_a + P_therm_a
        return press_a

    def entropy(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)
        S0, = self.get_param_values(param_names=['S0'])
        S_compress_a = self.compress_entropy(V_a)
        S_therm_a = self.thermal_entropy(V_a, T_a)

        entropy_a = S0 + S_compress_a + S_therm_a
        return entropy_a

    def helmholtz_energy(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)

        E_a = self.internal_energy(V_a, T_a)
        S_a = self.entropy(V_a, T_a)
        F_a = E_a - T_a*S_a

        return F_a

    def internal_energy(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)
        compress_calc = self.calculators['compress']
        compress_path_const = compress_calc.path_const

        if   compress_path_const=='T':
            F0, T0, S0 = self.get_param_values(param_names=['F0','T0','S0'])
            E0 = F0 + T0*S0

        elif (compress_path_const=='S')|(compress_path_const=='0K'):
            E0, = self.get_param_values(param_names=['E0'])

        else:
            raise NotImplementedError(
                'path_const '+path_const+' is not valid for CompressEos.')

        E_compress_a = self.compress_energy(V_a)
        E_therm_a = self.thermal_energy(V_a, T_a)

        internal_energy_a = E0 + E_compress_a + E_therm_a
        return internal_energy_a
#====================================================================
    # def bulk_mod(self, V_a, T_a):
    #     calculator = self.calculators['thermal']
    #     energy_a =  calculator._calc_energy(T_a)
    #     return energy_a
class RTPolyEos(with_metaclass(ABCMeta, core.Eos)):
    _kind_thermal_opts = ['GenRosenfeldTarazona']
    _kind_compress_opts = ['Vinet','BirchMurn3','BirchMurn4',
                           'GenFiniteStrain','Tait','PolyRho']

    def __init__(self, kind_compress='Vinet', compress_order=None,
                 compress_path_const='T', kind_RTpoly='V', RTpoly_order=5,
                 natom=1, model_state={}):

        kind_thermal = 'GenRosenfeldTarazona'
        self._pre_init(natom=natom)

        compress.set_calculator(self, kind_compress, self._kind_compress_opts,
                                path_const=compress_path_const)
        thermal.set_calculator(self, kind_thermal, self._kind_thermal_opts)

        self._set_poly_calculators(kind_RTpoly, RTpoly_order)

        self._set_ref_state()

        self._post_init(model_state=model_state)
        pass

    def __repr__(self):
        calc_compress = self.calculators['compress']

        # kind_compress='Vinet', compress_order=None,
        #          compress_path_const='T', kind_RTpoly='V', RTpoly_order=5,
        #          natom=1, model_state={}):
        return ("ThermalEos(kind_compress={kind_compress}, "
                "compress_order={compress_order}, "
                "compress_path_const={compress_path_const}, "
                "kind_RTpoly={kind_RTpoly}, "
                "RTpoly_order={RTpoly_order}, "
                "natom={natom}, "
                "model_state={model_state}, "
                ")"
                .format(kind_compress=repr(calc_compress.name),
                        compress_order=repr(calc_compress.order),
                        kind_RTpoly=repr(self._kind_RTpoly),
                        RTpoly_order=repr(self._RTpoly_order),
                        compress_path_const=repr(calc_compress.path_const),
                        natom=repr(self.natom),
                        model_state=self.model_state
                        )
                )

    def ref_temp_path(self, V_a):
        T0, theta0 = self.get_param_values(param_names=['T0','theta0'])

        gamma_calc = self.calculators['gamma']
        compress_calc = self.calculators['compress']
        compress_path_const = compress_calc.path_const

        # Tref_path = gamma_calc._calc_temp(V_a)
        theta_ref = self._calc_theta(V_a)

        if   compress_path_const=='T':
            Tref_path = T0

        elif compress_path_const=='S':
            Tref_path = gamma_calc._calc_temp(V_a)

        elif compress_path_const=='0K':
            Tref_path = 0

        else:
            raise NotImplementedError(
                'path_const '+path_const+' is not valid for CompressEos.')

        Tref_path, V_a = core.fill_array(Tref_path, V_a)
        Tref_path, theta_ref = core.fill_array(Tref_path, theta_ref)

        return Tref_path, theta_ref

    def thermal_energy(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)

        thermal_calc = self._calculators['thermal']
        a_V, b_V = self._calc_RTcoefs(V_a)

        # Tref_path, theta_ref = self.ref_temp_path(V_a)

        thermal_energy_a = a_V + thermal_calc._calc_energy(T_a, bcoef=b_V)
        return thermal_energy_a

    def internal_energy(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)
        compress_calc = self.calculators['compress']
        compress_path_const = compress_calc.path_const

        if   compress_path_const=='T':
            F0, T0, S0 = self.get_param_values(param_names=['F0','T0','S0'])
            E0 = F0 + T0*S0

        elif (compress_path_const=='S')|(compress_path_const=='0K'):
            E0, = self.get_param_values(param_names=['E0'])

        else:
            raise NotImplementedError(
                'path_const '+path_const+' is not valid for CompressEos.')

        E_compress_a = self.compress_energy(V_a)
        E_therm_a = self.thermal_energy(V_a, T_a)

        internal_energy_a = E0 + E_compress_a + E_therm_a
        return internal_energy_a

    def heat_capacity(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)

        thermal_calc = self.calculators['thermal']
        a_V, b_V = self._calc_RTcoefs(V_a)

        heat_capacity_a = thermal_calc._calc_heat_capacity(T_a, bcoef=b_V)
        return heat_capacity_a

    def press(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)
        compress_calc = self.calculators['compress']
        P_ref_a = compress_calc._calc_press(V_a)
        P_therm_a = self.thermal_press(V_a, T_a)

        press_a = P_ref_a + P_therm_a
        return press_a

    def _set_poly_calculators(self, kind_RTpoly, RTpoly_order):
        bcoef_calc = _RTPolyCalc(self, order=RTpoly_order, kind=kind_RTpoly,
                                 coef_basename='bcoef')
        acoef_calc = _RTPolyCalc(self, order=RTpoly_order, kind=kind_RTpoly,
                                 coef_basename='acoef')

        self._add_calculator(bcoef_calc, calc_type='bcoef')
        self._add_calculator(acoef_calc, calc_type='acoef')
        self._kind_RTpoly = kind_RTpoly
        self._RTpoly_order = RTpoly_order

    def _set_ref_state(self):
        compress_calc = self.calculators['compress']
        compress_path_const = compress_calc.path_const

        V0, K0 = compress_calc.get_param_defaults(['V0','K0'])
        # redundant T0 declaration
        T0_scale = 300
        Cv_scale = 3*self.natom*core.CONSTS['kboltz']
        S0_scale = Cv_scale*T0_scale
        energy_scale = np.round(V0*K0/core.CONSTS['PV_ratio'],decimals=2)

        if   compress_path_const=='T':
            param_ref_names = ['T0', 'F0', 'S0']
            param_ref_units = ['K', 'eV', 'eV/K']
            param_ref_defaults = [T0_scale, 0.0, S0_scale]
            param_ref_scales = [T0_scale, energy_scale, S0_scale]

        elif compress_path_const=='S':
            param_ref_names = ['T0', 'E0', 'S0']
            param_ref_units = ['K', 'eV', 'eV/K']
            param_ref_defaults = [300, 0.0, S0_scale]
            param_ref_scales = [300, energy_scale, S0_scale]

        else:
            raise NotImplementedError(
                'path_const '+path_const+' is not valid for CompressEos.')

        self._param_ref_names = param_ref_names
        self._param_ref_units = param_ref_units
        self._param_ref_defaults = param_ref_defaults
        self._param_ref_scales = param_ref_scales
        pass

    def _calc_RTcoefs(self, V_a):
        bcoef_calc = self._calculators['bcoef']
        acoef_calc = self._calculators['acoef']

        a_V = acoef_calc.calc_coef(V_a)
        b_V = bcoef_calc.calc_coef(V_a)
        return a_V, b_V

    def _calc_RTcoefs_deriv(self, V_a):
        bcoef_calc = self._calculators['bcoef']
        acoef_calc = self._calculators['acoef']

        a_deriv_V = acoef_calc.calc_coef_deriv(V_a)
        b_deriv_V = bcoef_calc.calc_coef_deriv(V_a)
        return a_deriv_V, b_deriv_V
#====================================================================
    # def bulk_mod(self, V_a, T_a):
    #     calculator = self.calculators['thermal']
    #     energy_a =  calculator._calc_energy(T_a)
    #     return energy_a
class RTPressEos(with_metaclass(ABCMeta, core.Eos)):
    _kind_thermal_opts = ['GenRosenfeldTarazona']
    _kind_gamma_opts = ['GammaPowLaw','GammaShiftedPowLaw','GammaFiniteStrain']
    _kind_compress_opts = ['Vinet','BirchMurn3','BirchMurn4',
                           'GenFiniteStrain','Tait','PolyRho']

    def __init__(self, kind_compress='Vinet', compress_path_const='T',
                 kind_gamma='GammaFiniteStrain', kind_RTpoly='V',
                 RTpoly_order=5, natom=1, model_state={}):

        assert compress_path_const=='T', (
            'Only isothermal compress models supported now.')

        kind_thermal = 'GenRosenfeldTarazona'
        self._pre_init(natom=natom)

        compress.set_calculator(self, kind_compress, self._kind_compress_opts,
                                path_const=compress_path_const)
        gamma.set_calculator(self, kind_gamma, self._kind_gamma_opts)
        thermal.set_calculator(self, kind_thermal, self._kind_thermal_opts)

        self._set_poly_calculators(kind_RTpoly, RTpoly_order)

        self._set_ref_state()

        self._post_init(model_state=model_state)
        pass

    def __repr__(self):
        calc_compress = self.calculators['compress']

        # kind_compress='Vinet', compress_order=None,
        #          compress_path_const='T', kind_RTpoly='V', RTpoly_order=5,
        #          natom=1, model_state={}):
        return ("ThermalEos(kind_compress={kind_compress}, "
                "compress_order={compress_order}, "
                "compress_path_const={compress_path_const}, "
                "kind_RTpoly={kind_RTpoly}, "
                "RTpoly_order={RTpoly_order}, "
                "natom={natom}, "
                "model_state={model_state}, "
                ")"
                .format(kind_compress=repr(calc_compress.name),
                        compress_order=repr(calc_compress.order),
                        kind_RTpoly=repr(self._kind_RTpoly),
                        RTpoly_order=repr(self._RTpoly_order),
                        compress_path_const=repr(calc_compress.path_const),
                        natom=repr(self.natom),
                        model_state=self.model_state
                        )
                )

    def heat_capacity(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)

        thermal_calc = self.calculators['thermal']
        b_V = self._calc_RTcoefs(V_a)

        heat_capacity_a = thermal_calc._calc_heat_capacity(T_a, bcoef=b_V)
        return heat_capacity_a

    def ref_temp_adiabat(self, V_a):
        """
        Calculate reference adiabatic temperature path

        Parameters
        ==========
        V_a: array of volumes

        Returns
        =======
        Tref_adiabat:  array of temperature values

        """

        T0, = self.get_param_values(param_names=['T0',])

        gamma_calc = self.calculators['gamma']
        Tref_adiabat = gamma_calc._calc_temp(V_a)
        Tref_adiabat, V_a = core.fill_array(Tref_adiabat, V_a)

        return Tref_adiabat

    def _calc_thermal_press_S(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)

        thermal_calc = self._calculators['thermal']
        gamma_calc = self.calculators['gamma']

        PV_ratio, = core.get_consts(['PV_ratio'])
        T0, mexp = self.get_param_values(param_names=['T0','mexp'])

        b_V = self._calc_RTcoefs(V_a)
        b_deriv_V = self._calc_RTcoefs_deriv(V_a)

        Tref_adiabat = self.ref_temp_adiabat(V_a)

        heat_capacity_0S_a = thermal_calc._calc_heat_capacity(
            Tref_adiabat, bcoef=b_V)

        dtherm_dev_deriv_T = (
            thermal_calc._calc_therm_dev_deriv(T_a)
            - thermal_calc._calc_therm_dev_deriv(Tref_adiabat) )
        dtherm_dev_deriv_T0 = (
            thermal_calc._calc_therm_dev_deriv(T0)
            - thermal_calc._calc_therm_dev_deriv(Tref_adiabat) )

        gamma_a = gamma_calc._calc_gamma(V_a)

        P_therm_S = +PV_ratio*(
            b_deriv_V/(mexp-1)*(T_a*dtherm_dev_deriv_T
                                -T0*dtherm_dev_deriv_T0) +
            gamma_a/V_a*(T_a-T0)*heat_capacity_0S_a
            )

        return P_therm_S

    def _calc_thermal_press_E(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)

        thermal_calc = self._calculators['thermal']

        PV_ratio, = core.get_consts(['PV_ratio'])
        T0, mexp = self.get_param_values(param_names=['T0','mexp'])

        b_deriv_V = self._calc_RTcoefs_deriv(V_a)

        dtherm_dev = (
            thermal_calc._calc_therm_dev(T_a)
            - thermal_calc._calc_therm_dev(T0) )

        P_therm_E = -PV_ratio*b_deriv_V*dtherm_dev

        return P_therm_E

    def thermal_press(self, V_a, T_a):
        P_therm_E = self._calc_thermal_press_E(V_a, T_a)
        P_therm_S = self._calc_thermal_press_S(V_a, T_a)
        P_therm_a = P_therm_E + P_therm_S
        return P_therm_a

    def thermal_energy(self, V_a, T_a):
        """
        Internal Energy difference from T=T0
        """

        V_a, T_a = core.fill_array(V_a, T_a)

        b_V = self._calc_RTcoefs(V_a)
        thermal_calc = self.calculators['thermal']
        T0, = self.get_param_values(param_names=['T0',])

        thermal_energy_a = thermal_calc._calc_energy(T_a, bcoef=b_V, Tref=T0)
        return  thermal_energy_a

    def thermal_entropy(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)

        Tref_adiabat = self.ref_temp_adiabat(V_a)
        thermal_calc = self._calculators['thermal']
        b_V = self._calc_RTcoefs(V_a)

        thermal_entropy_a = thermal_calc._calc_entropy(
            T_a, bcoef=b_V, Tref=Tref_adiabat)
        return  thermal_entropy_a

    def press(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)

        P_compress_a = self.compress_press(V_a)
        P_therm_a = self.thermal_press(V_a, T_a)

        press_a = P_compress_a + P_therm_a
        return press_a

    def compress_press(self, V_a):
        V_a = core.fill_array(V_a)

        compress_calc = self.calculators['compress']
        P_compress_a = compress_calc._calc_press(V_a)
        return P_compress_a

    def compress_energy(self, V_a):
        V_a = core.fill_array(V_a)
        compress_calc = self.calculators['compress']
        F_compress = compress_calc._calc_energy(V_a)
        return F_compress

    def internal_energy(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)
        compress_calc = self.calculators['compress']
        compress_path_const = compress_calc.path_const
        assert compress_path_const=='T', (
            'Only isothermal compress models supported now.')

        S0, T0 = self.get_param_values(param_names=['S0','T0'])

        if   compress_path_const=='T':
            F0, T0, S0 = self.get_param_values(param_names=['F0','T0','S0'])
            E0 = F0 + T0*S0

        elif (compress_path_const=='S')|(compress_path_const=='0K'):
            E0, = self.get_param_values(param_names=['E0'])

        else:
            raise NotImplementedError(
                'path_const '+path_const+' is not valid for CompressEos.')

        F_compress = self.compress_energy(V_a)
        # Sref = S0 + self.thermal_entropy(V_a, T0)
        Sref = self.entropy(V_a, T0)
        E_compress = F_compress + T0*Sref

        thermal_energy_a = self.thermal_energy(V_a, T_a)

        internal_energy_a = E0 + E_compress + thermal_energy_a
        return internal_energy_a

    def helmholtz_energy(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)

        E_a = self.internal_energy(V_a, T_a)
        S_a = self.entropy(V_a, T_a)
        F_a = E_a - T_a*S_a

        return F_a

    def entropy(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)
        S0, = self.get_param_values(param_names=['S0'])

        thermal_entropy_a = self.thermal_entropy(V_a, T_a)

        entropy_a = S0 + thermal_entropy_a
        return entropy_a

    def _set_poly_calculators(self, kind_RTpoly, RTpoly_order):
        bcoef_calc = _RTPolyCalc(self, order=RTpoly_order, kind=kind_RTpoly,
                                 coef_basename='bcoef')

        self._add_calculator(bcoef_calc, calc_type='bcoef')
        self._kind_RTpoly = kind_RTpoly
        self._RTpoly_order = RTpoly_order

    def _set_ref_state(self):
        compress_calc = self.calculators['compress']
        compress_path_const = compress_calc.path_const

        V0, K0 = compress_calc.get_param_defaults(['V0','K0'])
        # redundant T0 declaration
        T0_scale = 300
        Cv_scale = 3*self.natom*core.CONSTS['kboltz']
        S0_scale = Cv_scale*T0_scale
        energy_scale = np.round(V0*K0/core.CONSTS['PV_ratio'],decimals=2)

        if   compress_path_const=='T':
            param_ref_names = ['T0', 'F0', 'S0']
            param_ref_units = ['K', 'eV', 'eV/K']
            param_ref_defaults = [T0_scale, 0.0, S0_scale]
            param_ref_scales = [T0_scale, energy_scale, S0_scale]

        elif compress_path_const=='S':
            param_ref_names = ['T0', 'E0', 'S0']
            param_ref_units = ['K', 'eV', 'eV/K']
            param_ref_defaults = [300, 0.0, S0_scale]
            param_ref_scales = [300, energy_scale, S0_scale]

        else:
            raise NotImplementedError(
                'path_const '+path_const+' is not valid for CompressEos.')

        self._param_ref_names = param_ref_names
        self._param_ref_units = param_ref_units
        self._param_ref_defaults = param_ref_defaults
        self._param_ref_scales = param_ref_scales
        pass

    def _calc_RTcoefs(self, V_a):
        bcoef_calc = self._calculators['bcoef']

        b_V = bcoef_calc.calc_coef(V_a)
        return b_V

    def _calc_RTcoefs_deriv(self, V_a):
        bcoef_calc = self._calculators['bcoef']

        b_deriv_V = bcoef_calc.calc_coef_deriv(V_a)
        return b_deriv_V
#====================================================================

#     def thermal_press(self, V_a, T_a):
#         V_a, T_a = core.fill_array(V_a, T_a)
#
#         thermal_calc = self._calculators['thermal']
#         gamma_calc = self.calculators['gamma']
#
#         PV_ratio, = core.get_consts(['PV_ratio'])
#         T0, mexp = self.get_param_values(param_names=['T0','mexp'])
#
#         b_V = self._calc_RTcoefs(V_a)
#         b_deriv_V = self._calc_RTcoefs_deriv(V_a)
#
#         Tref_adiabat = self.ref_temp_adiabat(V_a)
#
#         heat_capacity_0S_a = thermal_calc._calc_heat_capacity(
#             Tref_adiabat, bcoef=b_V)
#
#         dtherm_dev = (
#             thermal_calc._calc_therm_dev(T_a)
#             - thermal_calc._calc_therm_dev(T0) )
#         dtherm_dev_deriv_T = (
#             thermal_calc._calc_therm_dev_deriv(T_a)
#             - thermal_calc._calc_therm_dev_deriv(Tref_adiabat) )
#         dtherm_dev_deriv_T0 = (
#             thermal_calc._calc_therm_dev_deriv(T0)
#             - thermal_calc._calc_therm_dev_deriv(Tref_adiabat) )
#
#         gamma_a = gamma_calc._calc_gamma(V_a)
#
#         P_therm_E = -b_deriv_V*dtherm_dev
#         P_therm_S = (
#             b_deriv_V/(mexp-1)*(T_a*dtherm_dev_deriv_T
#                                 -T0*dtherm_dev_deriv_T0) +
#             PV_ratio*gamma_a/V_a*(T_a-T0)*heat_capacity_0S_a
#             )
#         P_therm_a = P_therm_E + P_therm_S
#
#         return P_therm_a
#     def compress_entropy(self, V_a):
#         """
#         Calc entropy along compression path
#
#         NOTE: currently assumes isothermal compression path
#
#         Parameters
#         ==========
#         V_a : array of volume values
#
#         Returns
#         =======
#         S_compress : array of entropy values
#
#         """
#
#         V_a = core.fill_array(V_a)
#
#
#         thermal_calc = self._calculators['thermal']
#         compress_calc = self._calculators['compress']
#
#         assert compress_calc.path_const=='T', (
#             'Only isothermal compress models supported now.')
#
#         b_V = self._calc_RTcoefs(V_a)
#
#         Tref_adiabat = self.ref_temp_adiabat(V_a)
#         T0, = self.get_param_values(param_names=['T0',])
#
#         V_a, T0 = core.fill_array(V_a, T0)
#
#         # def _calc_entropy(self, T_a, bcoef=None, Tref=None):
#         S_compress = thermal_calc._calc_entropy(
#             T0, bcoef=b_V, Tref=Tref_adiabat)
#         return S_compress
#
#     def thermal_entropy(self, V_a, T_a):
#         """
#         Calc thermal entropy deviation from reference compression path
#
#         NOTE: currently assumes isothermal compression path
#
#         Parameters
#         ==========
#         V_a : array of volume values
#
#         Returns
#         =======
#         S_thermal : array of entropy values
#
#         """
#
#         V_a, T_a = core.fill_array(V_a, T_a)
#
#         thermal_calc = self._calculators['thermal']
#         compress_calc = self._calculators['compress']
#
#         assert compress_calc.path_const=='T', (
#             'Only isothermal compress models supported now.')
#         b_V = self._calc_RTcoefs(V_a)
#
#         T0, = self.get_param_values(param_names=['T0',])
#
#         V_a, T0 = core.fill_array(V_a, T0)
#
#         # def _calc_entropy(self, T_a, bcoef=None, Tref=None):
#         S_thermal = thermal_calc._calc_entropy(
#             T_a, bcoef=b_V, Tref=T0)
#
#         return S_thermal
#
#     def thermal_energy(self, V_a, T_a):
#         """
#         Calc thermal energy deviation from ref compression path
#
#         NOTE: currently assumes isothermal compression path
#
#         Parameters
#         ==========
#         V_a : array of volume values
#
#         Returns
#         =======
#         E_compress : array of internal energy values
#
#         """
#
#         V_a, T_a = core.fill_array(V_a, T_a)
#
#
#         thermal_calc = self._calculators['thermal']
#         compress_calc = self._calculators['compress']
#
#         assert compress_calc.path_const=='T', (
#             'Only isothermal compress models supported now.')
#         b_V = self._calc_RTcoefs(V_a)
#
#         # Tref_path = self.ref_temp_adiabat(V_a)
#         T0, = self.get_param_values(param_names=['T0',])
#
#         thermal_energy_a = thermal_calc._calc_energy(T_a, bcoef=b_V, Tref=T0)
#
#         return thermal_energy_a
#     def compress_energy(self, V_a):
#         """
#         Calc internal energy along compression path
#
#         NOTE: currently assumes isothermal compression path
#
#         Parameters
#         ==========
#         V_a : array of volume values
#
#         Returns
#         =======
#         E_compress : array of internal energy values
#
#         """
#
#         V_a = core.fill_array(V_a)
#         compress_calc = self.calculators['compress']
#         compress_path_const = compress_calc.path_const
#         assert compress_path_const=='T', (
#             'Only isothermal compress models supported now.')
#
#         T0, = self.get_param_values(param_names=['T0'])
#         if   compress_path_const=='T':
#             F_compress = compress_calc._calc_energy(V_a)
#             S_compress = self.compress_entropy(V_a)
#             E_compress = F_compress + T0*S_compress
#
#         elif (compress_path_const=='S')|(compress_path_const=='0K'):
#             E_compress = compress_calc._calc_energy(V_a)
#
#         else:
#             raise NotImplementedError(
#                 'path_const '+path_const+' is not valid for CompressEos.')
#
#         return E_compress

#====================================================================
class _RTPolyCalc(with_metaclass(ABCMeta, core.Calculator)):
    _kind_opts = ['V','rho','logV']

    def __init__(self, eos_mod, order=6, kind='logV', coef_basename='bcoef'):

        if kind not in self._kind_opts:
            raise NotImplementedError(
                'kind '+kind+' is not valid for RTPolyCalc.')

        if ((not np.isscalar(order)) | (order < 0) | (np.mod(order,0) !=0)):
            raise ValueError(
                'order ' + str(order) +' is not valid for RTPolyCalc. '+
                'It must be a positive integer.')

        self._eos_mod = eos_mod
        self._coef_basename = coef_basename
        self._init_params(order, kind)
        self._required_calculators = None

        self._kind = kind

    def _init_params(self, order, kind):

        # NOTE switch units cc/g -> ang3,  kJ/g -> eV
        V0 = 0.408031

        coef_basename = self._coef_basename
        param_names = core.make_array_param_names(coef_basename, order,
                                                  skipzero=False)

        if coef_basename == 'bcoef':
            bcoefs_mgsio3 = np.array([-.371466, 7.09542, -45.7362, 139.020,
                                      -201.487, 112.513])

            shift_coefs_mgsio3 = core.shift_poly(bcoefs_mgsio3, xscale=V0)

        elif coef_basename == 'acoef':
            acoefs_mgsio3 = np.array([127.116, -3503.98, 20724.4, -60212.0,
                                      86060.5, -48520.4])

            shift_coefs_mgsio3 = core.shift_poly(acoefs_mgsio3, xscale=V0)
        else:
            raise NotImplemented('This is not a valid RTcoef type')

        param_defaults = [0 for ind in range(0,order+1)]
        if order>5:
            param_defaults[0:6] = shift_coefs_mgsio3
        else:
            param_defaults[0:order+1] = shift_coefs_mgsio3[0:order+1]

        param_scales = [1 for ind in range(0,order+1)]
        param_units = core.make_array_param_units(param_names, base_unit='kJ/g',
                                                  deriv_unit='(cc/g)')

        param_names.append('V0')
        param_scales.append(V0)
        param_units.append('cc/g')
        param_defaults.append(V0)

        self._set_params(param_names, param_units,
                         param_defaults, param_scales, order=order)

    def _get_polyval_coef(self):
        coef_basename = self._coef_basename

        param_names = self.eos_mod.get_array_param_names(coef_basename)
        param_values = self.eos_mod.get_param_values(param_names=param_names)
        V0, = self.eos_mod.get_param_values(param_names=['V0'])

        coef_index = core.get_array_param_index(param_names)
        order = np.max(coef_index)+1
        param_full = np.zeros(order)
        param_full[coef_index] = param_values

        coefs_a = param_full[::-1]  # Reverse array for np.polyval
        return coefs_a, V0

    def calc_coef(self, V_a):
        coefs_a, V0 = self._get_polyval_coef()
        coef_V = np.polyval(coefs_a, V_a/V0-1)
        return coef_V

    def calc_coef_deriv(self, V_a):
        coefs_a, V0 = self._get_polyval_coef()
        order = coefs_a.size-1
        coefs_deriv_a = np.polyder(coefs_a)/V0
        coef_deriv_V = np.polyval(coefs_deriv_a, V_a/V0-1)
        return coef_deriv_V
#====================================================================


# class CompressedThermalEnergyCalc(with_metaclass(ABCMeta, core.Calculator)):
#     """
#     Abstract Equation of State class for a reference Thermal Energy Path
#
#     Path can either be isothermal (T=const) or adiabatic (S=const)
#
#     For this restricted path, thermodyn properties depend only on volume
#
#     """
#
#     def __init__(self, eos_mod):
#         self._eos_mod = eos_mod
#         self._init_params()
#         self._required_calculators = None
#         pass
#
#     ####################
#     # Required Methods #
#     ####################
#     @abstractmethod
#     def _init_params(self):
#         """Initialize list of calculator parameter names."""
#         pass
#
#     @abstractmethod
#     def _init_required_calculators(self):
#         """Initialize list of other required calculators."""
#         pass
#
#     @abstractmethod
#     def _calc_heat_capacity(self, V_a, T_a):
#         """Returns heat capacity as a function of temperature."""
#         pass
#
#     @abstractmethod
#     def _calc_energy(self, V_a, T_a):
#         """Returns thermal energy as a function of temperature."""
#         pass
#====================================================================


#====================================================================
# Base Class
#====================================================================
# class CompositeMod(EosMod):
#     """
#     Abstract Equation of State class for Full Model (combines all EOS terms)
#     """
#     __metaclass__ = ABCMeta
#
#     # Standard methods must be overridden (as needed) by implimentation model
#     def press( self, V_a, T_a, eos_d ):
#         """Returns Total Press."""
#         raise NotImplementedError("'press' function not implimented for this model")
#
#     def energy( self, V_a, T_a, eos_d ):
#         """Returns Toal Energy."""
#         raise NotImplementedError("'energy' function not implimented for this model")
#
#     def therm_exp( self, V_a, T_a, eos_d ):
#         TOL = 1e-4
#         V_a, T_a = fill_array( V_a, T_a )
#
#         dlogV = 1e-4
#         S_a = self.entropy( V_a, T_a, eos_d )
#         KT_a = self.bulk_modulus( V_a, T_a, eos_d )
#
#         S_hi_a = self.entropy( np.exp(dlogV)*V_a, T_a, eos_d )
#         S_lo_a = self.entropy( np.exp(-dlogV)*V_a, T_a, eos_d )
#
#         dSdlogV_a = (S_hi_a-S_lo_a) / (2*dlogV)
#         alpha_a = dSdlogV_a/(KT_a*V_a*eos_d['const_d']['PV_ratio'])
#
#         return alpha_a
#
#
#     def bulk_mod( self, V_a, T_a, eos_d ):
#         """Returns Total Bulk Modulus."""
#         raise NotImplementedError("'bulk_mod' function not implimented for this model")
# #====================================================================
#
# #====================================================================
# # Implementations
# #====================================================================
# class ThermalPressMod(CompositeMod):
#     # Need to impliment get_param_scale_sub
#
#     def press( self, V_a, T_a, eos_d ):
#         """Returns Press variation along compression curve."""
#         V_a, T_a = fill_array( V_a, T_a )
#         # compress_path_mod, thermal_mod = Control.get_modtypes( ['CompressPathMod', 'ThermalMod'],
#         #                                        eos_d )
#         # press_a = np.squeeze( compress_path_mod.press( V_a, eos_d )
#         #                      + thermal_mod.press( V_a, T_a, eos_d ) )
#         # return press_a
#
#         TOL = 1e-4
#         PV_ratio, = Control.get_consts( ['PV_ratio'], eos_d )
#
#         F_mod_a = self.free_energy(V_a,T_a,eos_d)
#         F_hi_mod_a = self.free_energy(V_a*(1.0+TOL),T_a,eos_d)
#         P_mod_a = -PV_ratio*(F_hi_mod_a-F_mod_a)/(V_a*TOL)
#         return P_mod_a
#
#     def energy( self, V_a, T_a, eos_d ):
#         """Returns Internal Energy."""
#         V_a, T_a = fill_array( V_a, T_a )
#         compress_path_mod, thermal_mod = Control.get_modtypes( ['CompressPathMod', 'ThermalMod'],
#                                                eos_d )
#
#         energy_compress_a = compress_path_mod.energy( V_a, eos_d )
#         if compress_path_mod.path_const=='T':
#             """
#             Convert free energy to internal energy
#             """
#             free_energy_compress_a = energy_compress_a
#             T0, = Control.get_params(['T0'],eos_d)
#
#             # wrong
#             # S_a = thermal_mod.entropy( V_a, T_a, eos_d )
#             # dF_a = thermal_mod.calc_free_energy( V_a, T_a, eos_d )
#             # Ftot_a = free_energy_compress_a + dF_a
#             # energy_a = Ftot_a+T_a*S_a
#
#
#             # correct
#             # energy_a = thermal_mod.calc_energy_adiabat_ref(V_a,eos_d)\
#             #     +thermal_mod.calc_energy(V_a,T_a,eos_d)
#
#             S_T0_a = thermal_mod.entropy( V_a, T0, eos_d )
#             energy_T0_a = free_energy_compress_a + T0*S_T0_a
#             energy_S0_a = energy_T0_a - thermal_mod.calc_energy(V_a,T0,eos_d)
#
#             energy_a = energy_S0_a + thermal_mod.calc_energy(V_a,T_a,eos_d)
#
#
#
#         else:
#             energy_a = np.squeeze( energy_compress_a
#                                   + thermal_mod.energy( V_a, T_a, eos_d ) )
#
#         return energy_a
#
#     def bulk_modulus( self, V_a, T_a, eos_d ):
#         TOL = 1e-4
#
#         P_lo_a = self.press( V_a*(1.0-TOL/2), T_a, eos_d )
#         P_hi_a = self.press( V_a*(1.0+TOL/2), T_a, eos_d )
#         K_a = -V_a*(P_hi_a-P_lo_a)/(V_a*TOL)
#
#         return K_a
#
#     def dPdT( self, V_a, T_a, eos_d, Tscl=1000.0 ):
#         TOL = 1e-4
#
#         P_lo_a = self.press( V_a, T_a*(1.0-TOL/2), eos_d )
#         P_hi_a = self.press( V_a, T_a*(1.0+TOL/2), eos_d )
#         dPdT_a = (P_hi_a-P_lo_a)/(T_a*TOL)*Tscl
#
#         # # By a maxwell relation dPdT_V = dSdV_T
#         # S_lo_a = self.entropy( V_a*(1.0-TOL/2), T_a, eos_d )
#         # S_hi_a = self.entropy( V_a*(1.0+TOL/2), T_a, eos_d )
#         # dSdV_a = (S_hi_a-S_lo_a)/(V_a*TOL)
#         # dPdT_a = dSdV_a*Tscl
#
#         return dPdT_a
#
#     def free_energy( self, V_a, T_a, eos_d ):
#         """Returns Free Energy."""
#         V_a, T_a = fill_array( V_a, T_a )
#         compress_path_mod, thermal_mod = Control.get_modtypes( ['CompressPathMod', 'ThermalMod'],
#                                                eos_d )
#         T0,S0 = Control.get_params(['T0','S0'],eos_d)
#
#         energy_compress_a = compress_path_mod.energy( V_a, eos_d )
#         if compress_path_mod.path_const=='T':
#             free_energy_compress_a = energy_compress_a
#
#             S_T0_a = thermal_mod.entropy( V_a, T0, eos_d )
#             # wrong
#             # S_a = thermal_mod.entropy( V_a, T_a, eos_d )
#             # dF_a = thermal_mod.calc_free_energy( V_a, T_a, eos_d )
#             # Ftot_a = free_energy_compress_a + dF_a
#             # energy_a = Ftot_a+T_a*S_a
#
#
#             # correct
#             # energy_a = thermal_mod.calc_energy_adiabat_ref(V_a,eos_d)\
#             #     +thermal_mod.calc_energy(V_a,T_a,eos_d)
#
#             energy_T0_a = free_energy_compress_a + T0*S_T0_a
#             energy_S0_a = energy_T0_a - thermal_mod.calc_energy(V_a,T0,eos_d)
#
#         else:
#             energy_S0_a = energy_compress_a
#
#         Tref_a = thermal_mod.calc_temp_path(V_a,eos_d)
#
#         free_energy_S0_a = energy_S0_a - Tref_a*S0
#
#         # Fix bug for nonzero ref entropy values, need to subtract off reference
#         dF_a = thermal_mod.calc_free_energy( V_a, T_a, eos_d ) \
#             - thermal_mod.calc_free_energy( V_a, Tref_a, eos_d )
#         free_energy_a = free_energy_S0_a + dF_a
#
#         return free_energy_a
#
#     def entropy( self, V_a, T_a, eos_d ):
#         """Returns Free Energy."""
#         V_a, T_a = fill_array( V_a, T_a )
#         thermal_mod, = Control.get_modtypes( ['ThermalMod'], eos_d )
#         S_a = np.squeeze( thermal_mod.entropy( V_a, T_a, eos_d ) )
#
#         return S_a
#
#     def heat_capacity( self, V_a, T_a, eos_d ):
#         """Returns Free Energy."""
#         V_a, T_a = fill_array( V_a, T_a )
#         thermal_mod, = Control.get_modtypes( ['ThermalMod'], eos_d )
#         Cv_a = np.squeeze( thermal_mod.heat_capacity( V_a, T_a, eos_d ) )
#
#         return Cv_a
# #====================================================================
#
# #====================================================================
# class RosenfeldTaranzonaShiftedAdiabat(CompressPath):
#     def get_param_scale_sub( self, eos_d):
#         """Return scale values for each parameter"""
#         V0, K0, KP0 = core.get_params( ['V0','K0','KP0'], eos_d )
#         PV_ratio, = core.get_consts( ['PV_ratio'], eos_d )
#
#         paramkey_a = np.array(['V0','K0','KP0','E0'])
#         scale_a = np.array([V0,K0,KP0,K0*V0/PV_ratio])
#
#         return scale_a, paramkey_a
#
#     def _calc_press( self, V_a, eos_d ):
#         PV_ratio, = core.get_consts( ['PV_ratio'], eos_d )
#         fac = 1e-3
#         Vhi_a = V_a*(1.0 + 0.5*fac)
#         Vlo_a = V_a*(1.0 - 0.5*fac)
#
#         dV_a = Vhi_a-Vlo_a
#
#
#         E0S_hi_a = self._calc_energy(Vhi_a, eos_d)
#         E0S_lo_a = self._calc_energy(Vlo_a, eos_d)
#
#         P0S_a = -PV_ratio*(E0S_hi_a - E0S_lo_a)/dV_a
#         return P0S_a
#
#     def _calc_energy( self, V_a, eos_d ):
#         V0, T0, mexp  = core.get_params( ['V0','T0','mexp'], eos_d )
#         kB, = core.get_consts( ['kboltz'], eos_d )
#
#         poly_blogcoef_a = core.get_array_params( 'blogcoef', eos_d )
#
#
#         compress_path_mod, thermal_mod, gamma_mod = \
#             core.get_modtypes( ['CompressPath', 'ThermalMod', 'GammaMod'],
#                                  eos_d )
#
#         free_energy_isotherm_a = compress_path_mod.energy(V_a,eos_d)
#
#         T0S_a = gamma_mod.temp(V_a,T0,eos_d)
#
#
#         bV_a = np.polyval(poly_blogcoef_a,np.log(V_a/V0))
#
#         dS_a = -mexp/(mexp-1)*bV_a/T0*((T0S_a/T0)**(mexp-1)-1)\
#             -3./2*kB*np.log(T0S_a/T0)
#
#
#         energy_isotherm_a = free_energy_isotherm_a + T0*dS_a
#         E0S_a = energy_isotherm_a + bV_a*((T0S_a/T0)**mexp-1)\
#             +3./2*kB*(T0S_a-T0)
#
#         return E0S_a
# #====================================================================
