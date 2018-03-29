# -*- coding: utf-8 -*-
# from __future__ import absolute_import, print_function, division
from future.utils import with_metaclass
import numpy as np
import scipy as sp
from abc import ABCMeta, abstractmethod
from scipy import integrate, optimize
import scipy.interpolate as interpolate
from collections import OrderedDict

from . import core
from . import refstate
from . import compress
from . import thermal
from . import electronic
from . import gamma

__all__ = ['CompositeEos','MieGruneisenEos','PThermPolyEos',
           'RTPolyEos','RTPressEos']

# class RTPolyEos(with_metaclass(ABCMeta, core.Eos)):
# class RTPressEos(with_metaclass(ABCMeta, core.Eos)):
# class CompressPolyEos(with_metaclass(ABCMeta, core.Eos)):

#====================================================================
class CompositeEos(with_metaclass(ABCMeta, core.Eos)):
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

        E0 = self.refstate.ref_internal_energy()
        # if   compress_path_const=='T':
        #     F0, T0, S0 = self.get_param_values(param_names=['F0','T0','S0'])
        #     E0 = F0 + T0*S0

        # elif (compress_path_const=='S')|(compress_path_const=='0K'):
        #     E0, = self.get_param_values(param_names=['E0'])

        # else:
        #     raise NotImplementedError(
        #         'path_const '+path_const+' is not valid for CompressEos.')

        E_compress_a = self.compress_energy(V_a)
        E_therm_a = self.thermal_energy(V_a, T_a)

        internal_energy_a = E0 + E_compress_a + E_therm_a
        return internal_energy_a

    def bulk_mod(self, V_a, T_a, TOL=1e-4):
        P_lo_a = self.press(V_a*np.exp(-TOL/2), T_a)
        P_hi_a = self.press(V_a*np.exp(+TOL/2), T_a)
        K_a = -(P_hi_a-P_lo_a)/TOL

        return K_a

    def volume(self, P, T, Vinit=None, TOL=1e-3, bounds_error=True):
        if Vinit is None:
            V0, = self.get_param_values(param_names='V0')
            Vinit = 0.8*V0

        assert np.isscalar(Vinit), 'Vinit must be a scalar val.'

        def press_diff(V, P=P, T=T):
            return self.press(V, T) - P

        def press_diff_sqr(V, P=P, T=T):
            return (self.press(V, T) - P)**2

        def press_deriv(V, P=P, T=T):
            K = self.bulk_mod(V, T)
            dPdV = -K/V
            return dPdV

        # V = sp.optimize.fsolve(press_diff, Vinit, fprime=press_deriv)
        V_min = sp.optimize.fmin(press_diff, Vinit, disp=False)

        if +press_diff(V_min) > +TOL:
            if bounds_error:
                raise ValueError(
                    'The EOS is being sampled at an unphysical '
                    'location! The target pressure is not accesible '
                    'at this temperature.'
                )
            V = np.nan

        else:
            output = sp.optimize.minimize(
                press_diff_sqr, [0.8*V_min], bounds=[(None, V_min)],
                options={'disp':False})
            V, = output.x

        return V

    def gamma(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)

        gamma_calc = self.calculators['gamma']
        gamma0S = gamma_calc._calc_gamma(V_a)
        gamma = gamma0S
        return gamma

    def thermal_exp(self, V_a, T_a):
        gamma_a = self.gamma(V_a, T_a)
        KT_a = self.bulk_mod(V_a, T_a)
        CV_a = self.heat_capacity(V_a, T_a)

        alpha_a = core.CONSTS['PV_ratio']*gamma_a/V_a * CV_a/KT_a
        return alpha_a

    def material_properties(self, Pref, Tref, Vref=None):
        if Vref is None:
            Vref = self.volume(Pref, Tref, Vinit=12.8)[0]

        KT = self.bulk_mod(Vref,Tref)[0]
        CV = self.heat_capacity(Vref,Tref)
        alpha =  self.thermal_exp(Vref,Tref)[0]
        gamma =  self.gamma(Vref,Tref)[0]
        KS = KT*(1+alpha*gamma*Tref)
        props = OrderedDict()
        props['P'] = Pref
        props['T'] = Tref
        props['V'] = Vref
        props['KT'] = KT
        props['KS'] =  KS
        props['Cv'] = CV/core.CONSTS['kboltz']
        props['therm_exp'] = alpha
        props['gamma'] = gamma
        return props

    def adiabatic_path(self, Tfoot, P_a):
        Pfoot = P_a[0]
        Vfoot = self.volume(Pfoot, Tfoot, bounds_error=True)

        VTfoot = [Vfoot, Tfoot]
        soln = integrate.odeint(self._calc_adiabatic_derivs_fun,
                                VTfoot, P_a)
        # yvals = soln[0]
        V_adiabat = soln[:,0]
        T_adiabat = soln[:,1]
        return V_adiabat, T_adiabat

    def adiabatic_path_grid(self, Tfoot_grid, Pgrid):
        Vgrid = np.zeros((len(Tfoot_grid), len(Pgrid)))
        Tgrid = np.zeros((len(Tfoot_grid), len(Pgrid)))

        for ind, Tfoot in enumerate(Tfoot_grid):
            iVad, iTad = self.adiabatic_path(Tfoot, Pgrid)
            Vgrid[ind] = iVad
            Tgrid[ind] = iTad

        return Vgrid, Tgrid

    def material_properties(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)

        S_a = self.entropy(V_a,T_a)
        P_a = self.press(V_a,T_a)
        Cv = self.heat_capacity(V_a,T_a)
        alpha = self.thermal_exp(V_a,T_a)
        gamma = self.gamma(V_a, T_a)
        KT = self.bulk_mod(V_a, T_a)

        Cp = Cv*(1+alpha*gamma*T_a)
        KS = KT*(1+alpha*gamma*T_a)
        dTdPs = gamma*T_a/KS
        rho = (self.molar_mass/V_a*
               core.CONSTS['ang3percc']/core.CONSTS['Nmol'])

        props = OrderedDict()
        props['V'] = V_a
        props['T'] = T_a
        props['rho'] = rho
        props['alpha'] = alpha
        props['gamma'] = gamma
        props['S'] = S_a
        props['P'] = P_a
        props['C_V'] = Cv
        props['C_P'] = Cp
        props['K_T'] = KT
        props['K_S'] = KS
        props['dTdP_S'] = dTdPs

        return props

    def hugoniot(self, rhofaclims, rhoinit, Tinit, Etrans=0, Ttrans=None,
                 isobar_trans=True, Nsamp=30, Pshift=0):

        rho_a = rhoinit*np.linspace(rhofaclims[0],rhofaclims[1],Nsamp)
        T_a = np.zeros(rho_a.shape)
        P_a = np.zeros(rho_a.shape)
        E_a = np.zeros(rho_a.shape)
        V_a = np.zeros(rho_a.shape)

        hugoniot_d = {}
        hugoniot_d['rhoinit'] = rhoinit
        hugoniot_d['Tinit'] = Tinit
        hugoniot_d['Etrans'] = Etrans

        for ind,rho in enumerate(rho_a):
            hugoniot_dev_d = self._calc_hugoniot_dev(
                rho, rhoinit, Tinit, Etrans=Etrans, Ttrans=Ttrans,
                isobar_trans=isobar_trans, Pshift=Pshift)

            P_a[ind] = hugoniot_dev_d['Phug']
            T_a[ind] = hugoniot_dev_d['Thug']
            E_a[ind] = hugoniot_dev_d['Ehug']
            V_a[ind] = hugoniot_dev_d['V']

        hugoniot_d['P_a'] = P_a
        hugoniot_d['T_a'] = T_a
        hugoniot_d['E_a'] = E_a
        hugoniot_d['V_a'] = V_a
        hugoniot_d['rho_a'] = rho_a

        return hugoniot_d

    def _calc_hugoniot_dev(self, rho, rhoinit, Tinit, Etrans=0, Ttrans=None,
                           isobar_trans=True, logTfaclim=[-1.5,1.5],
                           Nsamp=3001, Pshift=0):

        PV_ratio = core.CONSTS['PV_ratio']
        molar_mass = self.molar_mass

        Pinit=0
        # rho=1/V*eos_d['param_d']['mass_avg']/const_d['Nmol']*const_d['ang3percc']
        # rhoinit=1/Vinit*eos_d['param_d']['mass_avg']/const_d['Nmol']*const_d['ang3percc']

        V=1/rho*molar_mass/core.CONSTS['Nmol']*core.CONSTS['ang3percc']
        Vinit=1/rhoinit*molar_mass/core.CONSTS['Nmol']*core.CONSTS['ang3percc']

        T_a = Tinit*np.logspace(logTfaclim[0],logTfaclim[1],Nsamp)

        if Ttrans is None:
            Ttrans = Tinit

        if isobar_trans:
            V0 = self.get_param_values(param_names='V0')
            logVfac = optimize.fsolve(lambda logVfac: self.press(np.exp(logVfac)*V0,Ttrans)+Pshift-Pinit,0.0)
            Vtrans = np.exp(logVfac)*V0
        else:
            Vtrans = Vinit

        Einit = self.internal_energy(Vtrans, Ttrans)-Etrans
        # Pinit = full_mod.press(Vinit, Tinit, eos_d)

        E_a = self.internal_energy(V, T_a)
        delE_a = E_a-Einit

        P_a = self.press(V, T_a)+Pshift
        Pavg_a = 0.5*(P_a+Pinit)

        delV = V-Vinit

        Eshock_a = -delV*Pavg_a/PV_ratio
        Edev_a = delE_a-Eshock_a

        indmin = np.argmin(np.abs(Edev_a))

        Ploc_a = P_a[indmin-3:indmin+3]
        Tloc_a = T_a[indmin-3:indmin+3]
        Eloc_a = Eshock_a[indmin-3:indmin+3]
        Edevloc_a = Edev_a[indmin-3:indmin+3]

        Thug = interpolate.interp1d(Edevloc_a, Tloc_a, kind='cubic')(0)
        Phug = interpolate.interp1d(Edevloc_a, Ploc_a, kind='cubic')(0)
        Ehug = interpolate.interp1d(Edevloc_a, Eloc_a, kind='cubic')(0)

        # optimize.fsolve(

        # Phug = P_a[indmin]
        # Thug = T_a[indmin]
        # return delE_a-Eshock_a

        hugoniot_dev_d = {}
        hugoniot_dev_d['rho'] = rho
        hugoniot_dev_d['rhoinit'] = rhoinit
        hugoniot_dev_d['V'] = V
        hugoniot_dev_d['Vinit'] = Vinit
        hugoniot_dev_d['Tinit'] = Tinit
        hugoniot_dev_d['Einit'] = Einit
        hugoniot_dev_d['Etrans'] = Etrans
        hugoniot_dev_d['T_a'] = T_a
        hugoniot_dev_d['P_a'] = P_a
        hugoniot_dev_d['E_a'] = E_a
        hugoniot_dev_d['delE_a'] = delE_a
        hugoniot_dev_d['Eshock_a'] = Eshock_a
        hugoniot_dev_d['Phug'] = Phug
        hugoniot_dev_d['Thug'] = Thug
        hugoniot_dev_d['Ehug'] = Ehug
        hugoniot_dev_d['indmin'] = indmin

        return hugoniot_dev_d

    def _calc_adiabatic_derivs_fun(self, VT, P):
        """
        Calculate adiabatic derivatives (dTdP_S, dVdP_S)
        """

        V, T = VT
        gamma = self.gamma(V, T)
        KT = self.bulk_mod(V, T)
        CV = self.heat_capacity(V, T)

        alpha = core.CONSTS['PV_ratio']*gamma/V * CV/KT
        KS = KT*(1+alpha*gamma*T)

        dVdP_S = -V/KS
        # dTdP_S = 1/(alpha*KS)
        dTdP_S = T/KS*gamma

        return [np.squeeze(dVdP_S), np.squeeze(dTdP_S)]
#====================================================================
class MieGruneisenEos(CompositeEos):
    _kind_thermal_opts = ['Debye','Einstein','PThermPoly','ConstHeatCap']
    _kind_gamma_opts = ['GammaPowLaw','GammaShiftedPowLaw','GammaFiniteStrain']
    _kind_compress_opts = ['Vinet','BirchMurn3','BirchMurn4',
                           'GenFiniteStrain','Tait']

    def __init__(self, kind_thermal='Debye', kind_gamma='GammaPowLaw',
                 kind_compress='Vinet', compress_path_const='T',
                 ref_energy_type='F0', natom=1, molar_mass=20,
                 model_state={}):

        self._pre_init(natom=natom)

        ref_compress_state='P0'
        ref_thermal_state='T0'

        compress.set_calculator(self, kind_compress, self._kind_compress_opts,
                                path_const=compress_path_const)
        gamma.set_calculator(self, kind_gamma, self._kind_gamma_opts)
        thermal.set_calculator(self, kind_thermal, self._kind_thermal_opts)
        refstate.set_calculator(self, ref_compress_state=ref_compress_state,
                                ref_thermal_state=ref_thermal_state,
                                ref_energy_type=ref_energy_type)
        # self._set_ref_state()

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

    def _calc_theta(self, V_a):
        gamma_calc = self.calculators['gamma']

        theta0, = self.get_param_values(['theta0'])
        theta_a = gamma_calc._calc_temp(V_a, T0=theta0)
        return theta_a

    def ref_temp_path(self, V_a):
        T0 = self.refstate.ref_temp()
        theta0 = self.get_param_values(param_names=['theta0'])

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

        T0 = self.refstate.ref_temp()
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

        T0 = self.refstate.ref_temp()

        theta0 = self.get_param_values(param_names=['theta0'])

        Tref_path, theta_ref = self.ref_temp_path(V_a)
        thermal_calc = self.calculators['thermal']

        opt_args = {'theta':theta_ref, 'T0':T0, 'theta0':theta0}
        S_compress = thermal_calc._calc_entropy(Tref_path, opt_args=opt_args)
        return S_compress

    def thermal_energy(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)
        Tref_path, theta_ref = self.ref_temp_path(V_a)
        thermal_calc = self.calculators['thermal']

        opt_args = {'theta':theta_ref, 'T0':Tref_path}
        E_therm_a = thermal_calc._calc_energy(T_a, opt_args=opt_args)
        return E_therm_a

    def thermal_entropy(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)
        Tref_path, theta_ref = self.ref_temp_path(V_a)
        thermal_calc = self.calculators['thermal']

        opt_args = {'theta':theta_ref, 'T0':Tref_path, 'theta0':theta_ref}
        S_therm_a = thermal_calc._calc_entropy(T_a, opt_args=opt_args)
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

        heat_capacity_a = thermal_calc._calc_heat_capacity(
            T_a, opt_args={'theta':theta_ref})
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

        T0 = self.refstate.ref_temp()
        E0 = self.refstate.ref_internal_energy()

        # if   compress_path_const=='T':
        #     F0, T0, S0 = self.get_param_values(param_names=['F0','T0','S0'])
        #     E0 = F0 + T0*S0

        # elif (compress_path_const=='S')|(compress_path_const=='0K'):
        #     E0, = self.get_param_values(param_names=['E0'])

        # else:
        #     raise NotImplementedError(
        #         'path_const '+path_const+' is not valid for CompressEos.')

        E_compress_a = self.compress_energy(V_a)
        E_therm_a = self.thermal_energy(V_a, T_a)

        internal_energy_a = E0 + E_compress_a + E_therm_a
        return internal_energy_a
#====================================================================
class PThermPolyEos(CompositeEos):
    _kind_thermal_opts = ['PTherm']
    _kind_gamma_opts = ['GammaPowLaw','GammaShiftedPowLaw','GammaFiniteStrain']
    _kind_compress_opts = ['Vinet','BirchMurn3','BirchMurn4',
                           'GenFiniteStrain','Tait']

    def __init__(self, logscale=True, kind_Pth='V', Pth_order=5,
                 kind_gamma='GammaPowLaw', kind_compress='Vinet', natom=1,
                 ref_energy_type='E0', molar_mass=20, model_state={}):

        kind_thermal='PTherm'
        compress_path_const='T'
        # _logPThermPolyCalc
        # _logPThermPolyCalc(self, order=5, kind='V')

        ref_compress_state='P0'
        ref_thermal_state='T0'

        self._pre_init(natom=natom, molar_mass=molar_mass)

        compress.set_calculator(self, kind_compress, self._kind_compress_opts,
                                path_const=compress_path_const)
        refstate.set_calculator(self, ref_compress_state=ref_compress_state,
                                ref_thermal_state=ref_thermal_state,
                                ref_energy_type=ref_energy_type)
        gamma.set_calculator(self, kind_gamma, self._kind_gamma_opts)
        thermal.set_calculator(self, kind_thermal, self._kind_thermal_opts)

        self.logscale = logscale

        # set poly calcs
        Pth_calc = _logPThermPolyCalc(self, order=Pth_order, kind=kind_Pth)

        self._add_calculator(Pth_calc, calc_type='Pth_coef')
        self._kind_Pth = kind_Pth
        self._Pth_order = Pth_order

        self._post_init(model_state=model_state)
        pass

    def __repr__(self):
        calc_compress = self.calculators['compress']
        calc_gamma = self.calculators['gamma']
        calc_thermal = self.calculators['thermal']
        calc_refstate = self.calculators['refstate']
        # kind_compress='Vinet', compress_order=None,
        #          compress_path_const='T', kind_RTpoly='V', RTpoly_order=5,
        #          natom=1, model_state={}):
        return ("ThermalEos(kind_gamma={kind_gamma}, "
                "kind_compress={kind_compress}, "
                "ref_energy_type={ref_energy_type}, "
                "logscale={logscale}, "
                "kind_Pth={kind_Pth}, "
                "Pth_order={Pth_order}, "
                "natom={natom}, "
                "molar_mass={molar_mass}, "
                "model_state={model_state}, "
                ")"
                .format(kind_gamma=repr(calc_gamma.name),
                        kind_compress=repr(calc_compress.name),
                        ref_energy_type=repr(calc_refstate.ref_energy_type),
                        logscale=repr(self.logscale),
                        kind_Pth=repr(self._kind_Pth),
                        Pth_order=repr(self._Pth_order),
                        natom=repr(self.natom),
                        molar_mass=repr(self.molar_mass),
                        model_state=self.model_state
                        )
                )

    def press(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)

        compress_calc = self.calculators['compress']

        P_therm_a = self.thermal_press(V_a, T_a)
        P_ref_a = compress_calc._calc_press(V_a)

        press_a = P_ref_a + P_therm_a
        return press_a

    def thermal_press(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)

        compress_calc = self.calculators['compress']
        thermal_calc = self.calculators['thermal']

        Pth_coef = self.calc_Pth_coefs(V_a)

        P_therm_a = thermal_calc._calc_press(T_a, Pth=Pth_coef)
        return P_therm_a

    def thermal_energy(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)

        compress_calc = self.calculators['compress']
        gamma_calc = self.calculators['gamma']
        thermal_calc = self.calculators['thermal']

        Pth_coef = self.calc_Pth_coefs(V_a)
        gamma_a = gamma_calc.gamma(V_a)

        opt_args = {'gamma':gamma_a, 'Pth':Pth_coef}
        E_therm_a = thermal_calc._calc_energy(T_a, opt_args=opt_args)
        return E_therm_a

    def compress_energy(self, V_a):
        V_a = core.fill_array(V_a)
        compress_calc = self.calculators['compress']
        F_compress = compress_calc._calc_energy(V_a)


        return F_compress

    def internal_energy(self, V_a, T_a):
        compress_calc = self.calculators['compress']
        compress_path_const = compress_calc.path_const

        try:
            E0, = self.get_param_values(param_names=['E0'])

        except:
            raise NotImplementedError('path const must be E0.')


        E_compress_a = self.compress_energy(V_a)
        E_therm_a = self.thermal_energy(V_a, T_a)

        internal_energy_a = E0 + E_compress_a + E_therm_a
        return internal_energy_a

    # def heat_capacity(self, V_a, T_a):
    #     V_a, T_a = core.fill_array(V_a, T_a)

    #     thermal_calc = self.calculators['thermal']
    #     a_V, b_V = self.calc_RTcoefs(V_a)

    #     heat_capacity_a = thermal_calc._calc_heat_capacity(T_a, bcoef=b_V)
    #     return heat_capacity_a


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

    def calc_Pth_coefs(self, V_a):
        Pth_calc = self._calculators['Pth_coef']
        logPth_V = Pth_calc.calc_coef(V_a)
        Pth_V = np.exp(logPth_V)
        return Pth_V
#====================================================================
class RTPolyEos(CompositeEos):
    _kind_thermal_opts = ['GenRosenfeldTarazona']
    _kind_compress_opts = ['Vinet','BirchMurn3','BirchMurn4',
                           'GenFiniteStrain','Tait','PolyRho']

    def __init__(self, kind_compress='Vinet', compress_order=None,
                 compress_path_const='T', kind_RTpoly='V', RTpoly_order=5,
                 natom=1, ref_energy_type='E0', model_state={}):

        kind_thermal = 'GenRosenfeldTarazona'
        ref_compress_state='P0'
        ref_thermal_state='T0'

        self._pre_init(natom=natom)

        compress.set_calculator(self, kind_compress, self._kind_compress_opts,
                                path_const=compress_path_const)
        thermal.set_calculator(self, kind_thermal, self._kind_thermal_opts)
        refstate.set_calculator(self, ref_compress_state=ref_compress_state,
                                ref_thermal_state=ref_thermal_state,
                                ref_energy_type=ref_energy_type)

        self._set_poly_calculators(kind_RTpoly, RTpoly_order)

        # self._set_ref_state()

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
        T0 = self.refstate.ref_temp()
        theta0 = self.get_param_values(param_names=['theta0'])

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
        a_V, b_V = self.calc_RTcoefs(V_a)

        # Tref_path, theta_ref = self.ref_temp_path(V_a)

        opt_args = {'bcoef':b_V}
        thermal_energy_a = a_V + thermal_calc._calc_energy(
            T_a, opt_args=opt_args)
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
        a_V, b_V = self.calc_RTcoefs(V_a)

        heat_capacity_a = thermal_calc._calc_heat_capacity(
            T_a, opt_args={'bcoef':b_V})
        return heat_capacity_a

    # def heat_capacity(self, V_a, T_a):
    #     V_a, T_a = core.fill_array(V_a, T_a)
    #     thermal_calc = self.calculators['thermal']
    #     Tref_path, theta_ref = self.ref_temp_path(V_a)

    #     heat_capacity_a = thermal_calc._calc_heat_capacity(T_a, theta=theta_ref)
    #     return heat_capacity_a

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

    def calc_RTcoefs(self, V_a):
        bcoef_calc = self._calculators['bcoef']
        acoef_calc = self._calculators['acoef']

        a_V = acoef_calc.calc_coef(V_a)
        b_V = bcoef_calc.calc_coef(V_a)
        return a_V, b_V

    def calc_RTcoefs_deriv(self, V_a):
        bcoef_calc = self._calculators['bcoef']
        acoef_calc = self._calculators['acoef']

        a_deriv_V = acoef_calc.calc_coef_deriv(V_a)
        b_deriv_V = bcoef_calc.calc_coef_deriv(V_a)
        return a_deriv_V, b_deriv_V
#====================================================================
class RTPressEos(CompositeEos):
    _kind_thermal_opts = ['GenRosenfeldTarazona']
    _kind_gamma_opts = ['GammaPowLaw','GammaShiftedPowLaw','GammaFiniteStrain']
    _kind_compress_opts =['Vinet','BirchMurn3','BirchMurn4',
                           'GenFiniteStrain','Tait','PolyRho']
    _kind_electronic_opts = ['None','CvPowLaw']

    def __init__(self, kind_compress='Vinet', compress_path_const='T',
                 kind_gamma='GammaFiniteStrain', kind_electronic='None',
                 apply_electronic=False, kind_RTpoly='V', RTpoly_order=5,
                 ref_energy_type='F0', natom=1, molar_mass=20, model_state={}):

        assert compress_path_const=='T', (
            'Only isothermal compress models supported now.')

        kind_thermal = 'GenRosenfeldTarazona'
        ref_compress_state='P0'
        ref_thermal_state='T0'

        self._pre_init(natom=natom, molar_mass=molar_mass)

        compress.set_calculator(self, kind_compress, self._kind_compress_opts,
                                path_const=compress_path_const)
        gamma.set_calculator(self, kind_gamma, self._kind_gamma_opts)
        thermal.set_calculator(self, kind_thermal, self._kind_thermal_opts,
                               external_bcoef=True)
        electronic.set_calculator(self, kind_electronic,
                                  self._kind_electronic_opts,
                                  apply_correction=apply_electronic)

        refstate.set_calculator(self, ref_compress_state=ref_compress_state,
                                ref_thermal_state=ref_thermal_state,
                                ref_energy_type=ref_energy_type)

        self._set_poly_calculators(kind_RTpoly, RTpoly_order)

        # self._set_ref_state()

        self._post_init(model_state=model_state)
        pass

    def __repr__(self):
        calc_compress = self.calculators['compress']
        calc_gamma = self.calculators['gamma']
        calc_electronic = self.calculators['electronic']
        return ("RTPressEos(kind_compress={kind_compress}, "
                "compress_path_const={compress_path_const}, "
                "kind_gamma={kind_gamma}, "
                "kind_electronic={kind_electronic}, "
                "apply_electronic={apply_electronic}, "
                "kind_RTpoly={kind_RTpoly}, "
                "RTpoly_order={RTpoly_order}, "
                "natom={natom}, "
                "molar_mass={molar_mass}, "
                "model_state={model_state}, "
                ")"
                .format(kind_compress=repr(calc_compress.name),
                        compress_path_const=repr(calc_compress.path_const),
                        kind_gamma=repr(calc_gamma.name),
                        kind_electronic=repr(calc_electronic.name),
                        apply_electronic=repr(calc_electronic.apply_correction),
                        kind_RTpoly=repr(self._kind_RTpoly),
                        RTpoly_order=repr(self._RTpoly_order),
                        natom=repr(self.natom),
                        molar_mass=repr(self.molar_mass),
                        model_state=self.model_state
                        )
                )

    @property
    def apply_electronic(self):
        calc = self.calculators['electronic']
        return calc.apply_correction

    @apply_electronic.setter
    def apply_electronic(self, apply_electronic):
        calc = self.calculators['electronic']
        calc.apply_correction = apply_electronic

    def heat_capacity(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)

        electronic_calc = self.calculators['electronic']
        thermal_calc = self.calculators['thermal']
        b_V = self.calc_RTcoefs(V_a)

        heat_capacity_a = (
            +thermal_calc._calc_heat_capacity(T_a, opt_args={'bcoef':b_V})
            +electronic_calc._calc_heat_capacity(V_a, T_a)
            )

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

        gamma_calc = self.calculators['gamma']
        Tref_adiabat = gamma_calc._calc_temp(V_a)
        Tref_adiabat, V_a = core.fill_array(Tref_adiabat, V_a)

        return Tref_adiabat

    def _calc_adiabatic_derivs_fun(self, VT, P):
        """
        Calculate adiabatic derivatives (dTdP_S, dVdP_S)
        """

        V, T = VT
        gamma = self.gamma(V, T)
        KT = self.bulk_mod(V, T)
        CV = self.heat_capacity(V, T)

        alpha = core.CONSTS['PV_ratio']*gamma/V * CV/KT
        KS = KT*(1+alpha*gamma*T)

        dVdP_S = -V/KS
        # dTdP_S = 1/(alpha*KS)
        dTdP_S = T/KS*gamma

        return [np.squeeze(dVdP_S), np.squeeze(dTdP_S)]

    def _calc_thermal_press_S(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)

        thermal_calc = self._calculators['thermal']
        gamma_calc = self.calculators['gamma']
        T0 = self.refstate.ref_temp()

        PV_ratio, = core.get_consts(['PV_ratio'])
        mexp = self.get_param_values(param_names='mexp')

        b_V = self.calc_RTcoefs(V_a)
        b_deriv_V = self.calc_RTcoefs_deriv(V_a)

        Tref_adiabat = self.ref_temp_adiabat(V_a)

        heat_capacity_0S_a = thermal_calc._calc_heat_capacity(
            Tref_adiabat, opt_args={'bcoef':b_V})

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
        T0 = self.refstate.ref_temp()

        PV_ratio, = core.get_consts(['PV_ratio'])
        mexp = self.get_param_values(param_names='mexp')

        b_deriv_V = self.calc_RTcoefs_deriv(V_a)

        dtherm_dev = (
            thermal_calc._calc_therm_dev(T_a)
            - thermal_calc._calc_therm_dev(T0) )

        P_therm_E = -PV_ratio*b_deriv_V*dtherm_dev

        return P_therm_E

    def thermal_press(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)
        electronic_calc = self.calculators['electronic']

        P_therm_elec = electronic_calc._calc_press(V_a, T_a)
        P_therm_E = self._calc_thermal_press_E(V_a, T_a)
        P_therm_S = self._calc_thermal_press_S(V_a, T_a)
        P_therm_a = P_therm_E + P_therm_S + P_therm_elec
        return P_therm_a

    def thermal_energy(self, V_a, T_a):
        """
        Internal Energy difference from T=T0
        """

        V_a, T_a = core.fill_array(V_a, T_a)
        T0 = self.refstate.ref_temp()

        b_V = self.calc_RTcoefs(V_a)
        thermal_calc = self.calculators['thermal']
        electronic_calc = self.calculators['electronic']
        # T0, = self.get_param_values(param_names=['T0',])

        opt_args = {'bcoef':b_V, 'Tref':T0}
        thermal_energy_a = (
            thermal_calc._calc_energy(T_a, opt_args=opt_args) +
            electronic_calc._calc_energy(V_a, T_a) )

        return  thermal_energy_a

    def thermal_entropy(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)

        electronic_calc = self.calculators['electronic']

        Tref_adiabat = self.ref_temp_adiabat(V_a)
        thermal_calc = self._calculators['thermal']
        b_V = self.calc_RTcoefs(V_a)

        opt_args = {'bcoef':b_V, 'Tref':Tref_adiabat}
        thermal_entropy_a = (
            thermal_calc._calc_entropy(T_a, opt_args=opt_args) +
            electronic_calc._calc_entropy(V_a, T_a) )

        return  thermal_entropy_a

    def press(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)
        P0 = self.refstate.ref_press()

        P_compress_a = self.compress_press(V_a)
        P_therm_a = self.thermal_press(V_a, T_a)

        press_a = P0 + P_compress_a + P_therm_a
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
        # electronic_calc = self.calculators['electronic']

        compress_path_const = compress_calc.path_const
        assert compress_path_const=='T', (
            'Only isothermal compress models supported now.')

        T0 = self.refstate.ref_temp()
        S0 = self.refstate.ref_entropy()
        E0 = self.refstate.ref_internal_energy()

        # S0, T0 = self.get_param_values(param_names=['S0','T0'])

        # if   compress_path_const=='T':
        #     F0, T0, S0 = self.get_param_values(param_names=['F0','T0','S0'])
        #     E0 = F0 + T0*S0

        # elif (compress_path_const=='S')|(compress_path_const=='0K'):
        #     E0, = self.get_param_values(param_names=['E0'])

        # else:
        #     raise NotImplementedError(
        #       'path_const '+path_const+' is not valid for CompressEos.')

        F_compress = self.compress_energy(V_a)
        # Sref = S0 + self.thermal_entropy(V_a, T0)
        Sref = self.entropy(V_a, T0)
        E_compress = F_compress + T0*Sref

        # E_electronic_a = electronic_calc._calc_energy(V_a, T_a)

        thermal_energy_a = self.thermal_energy(V_a, T_a)

        # internal_energy_a = E0 + E_compress + thermal_energy_a + E_electronic_a
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
        S0 = self.refstate.ref_entropy()

        thermal_entropy_a = self.thermal_entropy(V_a, T_a)

        entropy_a = (S0 + thermal_entropy_a)
        return entropy_a

    def gamma(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)

        gamma_calc = self.calculators['gamma']
        thermal_calc = self._calculators['thermal']
        electronic_calc = self.calculators['electronic']

        T0S = self.ref_temp_adiabat(V_a)

        CV = self.heat_capacity(V_a, T_a)
        CV_0S = self.heat_capacity(V_a, T0S)

        gamma0S = gamma_calc._calc_gamma(V_a)

        bcoef = self.calc_RTcoefs(V_a)
        bcoef_deriv = self.calc_RTcoefs_deriv(V_a)

        entropy_pot_a = thermal_calc._calc_entropy_pot(
            T_a, opt_args={'bcoef':bcoef, 'Tref':T0S})

        term1 = gamma0S/V_a*CV_0S
        term2 = bcoef_deriv/bcoef*entropy_pot_a
        # dSdV_T =  gamma0S/V_a*CV_0S + bcoef/bcoef_deriv*entropy_pot_a
        term_electronic = electronic_calc._calc_dSdV_T(V_a, T_a)
        dSdV_T =  term1 + term2 + term_electronic

        # This ONLY works if T is const
        # dV = np.diff(V_a)[0]
        # S_a = self.entropy(V_a,T_a)
        # dSdV_num = np.gradient(S_a,dV)
        # dSdV_num-dSdV_T

        gamma = V_a/CV*dSdV_T
        return gamma

    def _set_poly_calculators(self, kind_RTpoly, RTpoly_order):
        bcoef_calc = _RTPolyCalc(self, order=RTpoly_order, kind=kind_RTpoly,
                                 coef_basename='bcoef', RTpress=True)

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

    def calc_RT_vol_dev(self, V_a):
        bcoef_calc = self._calculators['bcoef']
        vol_dev = bcoef_calc._calc_vol_dev(V_a)
        return vol_dev

    def calc_RTcoefs(self, V_a):
        bcoef_calc = self._calculators['bcoef']

        b_V = bcoef_calc.calc_coef(V_a)
        return b_V

    def calc_RTcoefs_deriv(self, V_a):
        bcoef_calc = self._calculators['bcoef']

        b_deriv_V = bcoef_calc.calc_coef_deriv(V_a)
        return b_deriv_V
#====================================================================
class PolyRegressEos(CompositeEos):
    _kind_thermal_opts = ['']
    _kind_compress_opts = ['soundspeed']

    def __init__(self, kind_compress='Vinet', compress_path_const='T',
                 kind_gamma='GammaFiniteStrain', kind_RTpoly='V',
                 RTpoly_order=5, natom=1, ref_energy_type='F0',
                 model_state={}):

        assert compress_path_const=='T', (
            'Only isothermal compress models supported now.')

        kind_thermal = 'GenRosenfeldTarazona'
        ref_compress_state='P0'
        ref_thermal_state='T0'

        self._pre_init(natom=natom)

        compress.set_calculator(self, kind_compress, self._kind_compress_opts,
                                path_const=compress_path_const)
        gamma.set_calculator(self, kind_gamma, self._kind_gamma_opts)
        thermal.set_calculator(self, kind_thermal, self._kind_thermal_opts,
                               external_bcoef=True)

        refstate.set_calculator(self, ref_compress_state=ref_compress_state,
                                ref_thermal_state=ref_thermal_state,
                                ref_energy_type=ref_energy_type)

        self._set_poly_calculators(kind_RTpoly, RTpoly_order)

        # self._set_ref_state()

        self._post_init(model_state=model_state)
        pass

    def __repr__(self):
        calc_compress = self.calculators['compress']

        # kind_compress='Vinet', compress_order=None,
        #          compress_path_const='T', kind_RTpoly='V', RTpoly_order=5,
        #          natom=1, model_state={}):
        return ("RTPressEos(kind_compress={kind_compress}, "
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
        b_V = self.calc_RTcoefs(V_a)

        heat_capacity_a = thermal_calc._calc_heat_capacity(
            T_a, opt_args={'bcoef':b_V})
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

        gamma_calc = self.calculators['gamma']
        Tref_adiabat = gamma_calc._calc_temp(V_a)
        Tref_adiabat, V_a = core.fill_array(Tref_adiabat, V_a)

        return Tref_adiabat

    def adiabatic_path(self, Tfoot, P_a):
        Pfoot = P_a[0]
        Vfoot = self.volume(Pfoot, Tfoot)

        VTfoot = [Vfoot, Tfoot]
        soln = integrate.odeint(self._calc_adiabatic_derivs_fun,
                                VTfoot, P_a)
        # yvals = soln[0]
        V_adiabat = soln[:,0]
        T_adiabat = soln[:,1]
        return V_adiabat, T_adiabat

    def adiabatic_path_grid(self, Tfoot_grid, Pgrid):
        Vgrid = np.zeros((len(Tfoot_grid), len(Pgrid)))
        Tgrid = np.zeros((len(Tfoot_grid), len(Pgrid)))

        for ind, Tfoot in enumerate(Tfoot_grid):
            iVad, iTad = self.adiabatic_path(Tfoot, Pgrid)
            Vgrid[ind] = iVad
            Tgrid[ind] = iTad

        return Vgrid, Tgrid

    def thermal_exp(self, V_a, T_a):
        gamma_a = self.gamma(V_a, T_a)
        KT_a = self.bulk_mod(V_a, T_a)
        CV_a = self.heat_capacity(V_a, T_a)

        alpha_a = core.CONSTS['PV_ratio']*gamma_a/V_a * CV_a/KT_a
        return alpha_a

    def _calc_adiabatic_derivs_fun(self, VT, P):
        """
        Calculate adiabatic derivatives (dTdP_S, dVdP_S)
        """

        V, T = VT
        gamma = self.gamma(V, T)
        KT = self.bulk_mod(V, T)
        CV = self.heat_capacity(V, T)

        alpha = core.CONSTS['PV_ratio']*gamma/V * CV/KT
        KS = KT*(1+alpha*gamma*T)

        dVdP_S = -V/KS
        # dTdP_S = 1/(alpha*KS)
        dTdP_S = T/KS*gamma

        return [np.squeeze(dVdP_S), np.squeeze(dTdP_S)]

    def _calc_thermal_press_S(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)

        thermal_calc = self._calculators['thermal']
        gamma_calc = self.calculators['gamma']
        T0 = self.refstate.ref_temp()

        PV_ratio, = core.get_consts(['PV_ratio'])
        mexp = self.get_param_values(param_names='mexp')

        b_V = self.calc_RTcoefs(V_a)
        b_deriv_V = self.calc_RTcoefs_deriv(V_a)

        Tref_adiabat = self.ref_temp_adiabat(V_a)

        heat_capacity_0S_a = thermal_calc._calc_heat_capacity(
            Tref_adiabat, opt_args={'bcoef':b_V})

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
        T0 = self.refstate.ref_temp()

        PV_ratio, = core.get_consts(['PV_ratio'])
        mexp = self.get_param_values(param_names='mexp')

        b_deriv_V = self.calc_RTcoefs_deriv(V_a)

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
        T0 = self.refstate.ref_temp()

        b_V = self.calc_RTcoefs(V_a)
        thermal_calc = self.calculators['thermal']
        # T0, = self.get_param_values(param_names=['T0',])

        opt_args = {'bcoef':b_V, 'Tref':T0}
        thermal_energy_a = thermal_calc._calc_energy(T_a, opt_args=opt_args)
        return  thermal_energy_a

    def thermal_entropy(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)

        Tref_adiabat = self.ref_temp_adiabat(V_a)
        thermal_calc = self._calculators['thermal']
        b_V = self.calc_RTcoefs(V_a)

        opt_args = {'bcoef':b_V, 'Tref':Tref_adiabat}
        thermal_entropy_a = thermal_calc._calc_entropy(T_a, opt_args=opt_args)
        return  thermal_entropy_a

    def press(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)
        P0 = self.refstate.ref_press()

        P_compress_a = self.compress_press(V_a)
        P_therm_a = self.thermal_press(V_a, T_a)

        press_a = P0 + P_compress_a + P_therm_a
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

        T0 = self.refstate.ref_temp()
        S0 = self.refstate.ref_entropy()
        E0 = self.refstate.ref_internal_energy()

        # S0, T0 = self.get_param_values(param_names=['S0','T0'])

        # if   compress_path_const=='T':
        #     F0, T0, S0 = self.get_param_values(param_names=['F0','T0','S0'])
        #     E0 = F0 + T0*S0

        # elif (compress_path_const=='S')|(compress_path_const=='0K'):
        #     E0, = self.get_param_values(param_names=['E0'])

        # else:
        #     raise NotImplementedError(
        #       'path_const '+path_const+' is not valid for CompressEos.')

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
        S0 = self.refstate.ref_entropy()

        thermal_entropy_a = self.thermal_entropy(V_a, T_a)

        entropy_a = S0 + thermal_entropy_a
        return entropy_a

    def gamma(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)

        gamma_calc = self.calculators['gamma']
        thermal_calc = self._calculators['thermal']

        T0S = self.ref_temp_adiabat(V_a)

        CV = self.heat_capacity(V_a, T_a)
        CV_0S = self.heat_capacity(V_a, T0S)

        gamma0S = gamma_calc._calc_gamma(V_a)

        bcoef = self.calc_RTcoefs(V_a)
        bcoef_deriv = self.calc_RTcoefs_deriv(V_a)

        entropy_pot_a = thermal_calc._calc_entropy_pot(
            T_a, opt_args={'bcoef':bcoef, 'Tref':T0S})

        term1 = gamma0S/V_a*CV_0S
        term2 = bcoef_deriv/bcoef*entropy_pot_a
        # dSdV_T =  gamma0S/V_a*CV_0S + bcoef/bcoef_deriv*entropy_pot_a
        dSdV_T =  term1 + term2

        # This ONLY works if T is const
        # dV = np.diff(V_a)[0]
        # S_a = self.entropy(V_a,T_a)
        # dSdV_num = np.gradient(S_a,dV)
        # dSdV_num-dSdV_T

        gamma = V_a/CV*dSdV_T
        return gamma

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

    def calc_RT_vol_dev(self, V_a):
        bcoef_calc = self._calculators['bcoef']
        vol_dev = bcoef_calc._calc_vol_dev(V_a)
        return vol_dev

    def calc_RTcoefs(self, V_a):
        bcoef_calc = self._calculators['bcoef']

        b_V = bcoef_calc.calc_coef(V_a)
        return b_V

    def calc_RTcoefs_deriv(self, V_a):
        bcoef_calc = self._calculators['bcoef']

        b_deriv_V = bcoef_calc.calc_coef_deriv(V_a)
        return b_deriv_V
#====================================================================



#====================================================================
# class RTPolyHeatCap(CompositeEos):
#     _kind_thermal_opts = ['GenRosenfeldTarazona']
#
#     def __init__(self, kind_compress='Vinet', compress_path_const='T',
#                  kind_gamma='GammaFiniteStrain', kind_electronic='None',
#                  apply_electronic=False, kind_RTpoly='V', RTpoly_order=5,
#                  ref_energy_type='F0', natom=1, molar_mass=20, model_state={}):
#
#         assert compress_path_const=='T', (
#             'Only isothermal compress models supported now.')
#
#         kind_thermal = 'GenRosenfeldTarazona'
#         ref_compress_state='P0'
#         ref_thermal_state='T0'
#
#         self._pre_init(natom=natom, molar_mass=molar_mass)
#
#         compress.set_calculator(self, kind_compress, self._kind_compress_opts,
#                                 path_const=compress_path_const)
#         gamma.set_calculator(self, kind_gamma, self._kind_gamma_opts)
#         thermal.set_calculator(self, kind_thermal, self._kind_thermal_opts,
#                                external_bcoef=True)
#         electronic.set_calculator(self, kind_electronic,
#                                   self._kind_electronic_opts,
#                                   apply_correction=apply_electronic)
#
#         refstate.set_calculator(self, ref_compress_state=ref_compress_state,
#                                 ref_thermal_state=ref_thermal_state,
#                                 ref_energy_type=ref_energy_type)
#
#         self._set_poly_calculators(kind_RTpoly, RTpoly_order)
#
#         # self._set_ref_state()
#
#         self._post_init(model_state=model_state)
#         pass
# #====================================================================
# class PolyRosenfeldTarazona(_PolyRosenfeldTarazona):
#     def __init__(self, eos_mod, kind='V', poly_order=5):
#         super(_PolyRosenfeldTarazona, self).__init__(
#             eos_mod, external_bcoef=True)
#
#         # thermal.set_calculator(self, kind_thermal, self._kind_thermal_opts,
#         #                        external_bcoef=True)
#
#         self._set_poly_calculators(kind, poly_order)
#         pass
#
#     def _set_poly_calculators(self, kind, poly_order):
#         bcoef_calc = _RTPolyCalc(self, order=poly_order, kind=kind,
#                                  coef_basename='bcoef', RTpress=True)
#
#         self._add_calculator(bcoef_calc, calc_type='bcoef')
#         self._kind_RTpoly = kind_RTpoly
#         self._RTpoly_order = RTpoly_order
#
#     def _calc_heat_capacity(self, T_a, opt_args=None):
#     def heat_capacity(self, V_a, T_a):
#         V_a, T_a = core.fill_array(V_a, T_a)
#
#         electronic_calc = self.calculators['electronic']
#         thermal_calc = self.calculators['thermal']
#         b_V = self.calc_RTcoefs(V_a)
#
#         heat_capacity_a = (
#             +thermal_calc._calc_heat_capacity(T_a, opt_args={'bcoef':b_V})
#             +electronic_calc._calc_heat_capacity(V_a, T_a)
#             )
#
#         return heat_capacity_a
#
#     def _calc_adiabatic_derivs_fun(self, VT, P):
#         """
#         Calculate adiabatic derivatives (dTdP_S, dVdP_S)
#         """
#
#         V, T = VT
#         gamma = self.gamma(V, T)
#         KT = self.bulk_mod(V, T)
#         CV = self.heat_capacity(V, T)
#
#         alpha = core.CONSTS['PV_ratio']*gamma/V * CV/KT
#         KS = KT*(1+alpha*gamma*T)
#
#         dVdP_S = -V/KS
#         # dTdP_S = 1/(alpha*KS)
#         dTdP_S = T/KS*gamma
#
#         return [np.squeeze(dVdP_S), np.squeeze(dTdP_S)]
#
#     def _calc_thermal_press_S(self, V_a, T_a):
#         V_a, T_a = core.fill_array(V_a, T_a)
#
#         thermal_calc = self._calculators['thermal']
#         gamma_calc = self.calculators['gamma']
#         T0 = self.refstate.ref_temp()
#
#         PV_ratio, = core.get_consts(['PV_ratio'])
#         mexp = self.get_param_values(param_names='mexp')
#
#         b_V = self.calc_RTcoefs(V_a)
#         b_deriv_V = self.calc_RTcoefs_deriv(V_a)
#
#         Tref_adiabat = self.ref_temp_adiabat(V_a)
#
#         heat_capacity_0S_a = thermal_calc._calc_heat_capacity(
#             Tref_adiabat, opt_args={'bcoef':b_V})
#
#         dtherm_dev_deriv_T = (
#             thermal_calc._calc_therm_dev_deriv(T_a)
#             - thermal_calc._calc_therm_dev_deriv(Tref_adiabat) )
#         dtherm_dev_deriv_T0 = (
#             thermal_calc._calc_therm_dev_deriv(T0)
#             - thermal_calc._calc_therm_dev_deriv(Tref_adiabat) )
#
#         gamma_a = gamma_calc._calc_gamma(V_a)
#
#         P_therm_S = +PV_ratio*(
#             b_deriv_V/(mexp-1)*(T_a*dtherm_dev_deriv_T
#                                 -T0*dtherm_dev_deriv_T0) +
#             gamma_a/V_a*(T_a-T0)*heat_capacity_0S_a
#             )
#
#         return P_therm_S
#
#     def _calc_thermal_press_E(self, V_a, T_a):
#         V_a, T_a = core.fill_array(V_a, T_a)
#
#         thermal_calc = self._calculators['thermal']
#         T0 = self.refstate.ref_temp()
#
#         PV_ratio, = core.get_consts(['PV_ratio'])
#         mexp = self.get_param_values(param_names='mexp')
#
#         b_deriv_V = self.calc_RTcoefs_deriv(V_a)
#
#         dtherm_dev = (
#             thermal_calc._calc_therm_dev(T_a)
#             - thermal_calc._calc_therm_dev(T0) )
#
#         P_therm_E = -PV_ratio*b_deriv_V*dtherm_dev
#
#         return P_therm_E
#
#     def thermal_energy(self, V_a, T_a):
#         """
#         Internal Energy difference from T=T0
#         """
#
#         V_a, T_a = core.fill_array(V_a, T_a)
#         T0 = self.refstate.ref_temp()
#
#         b_V = self.calc_RTcoefs(V_a)
#         thermal_calc = self.calculators['thermal']
#         electronic_calc = self.calculators['electronic']
#         # T0, = self.get_param_values(param_names=['T0',])
#
#         opt_args = {'bcoef':b_V, 'Tref':T0}
#         thermal_energy_a = (
#             thermal_calc._calc_energy(T_a, opt_args=opt_args) +
#             electronic_calc._calc_energy(V_a, T_a) )
#
#         return  thermal_energy_a
#
#     def thermal_entropy(self, V_a, T_a):
#         V_a, T_a = core.fill_array(V_a, T_a)
#
#         Tref_adiabat = self.ref_temp_adiabat(V_a)
#         thermal_calc = self._calculators['thermal']
#         b_V = self.calc_RTcoefs(V_a)
#
#         opt_args = {'bcoef':b_V, 'Tref':Tref_adiabat}
#         thermal_entropy_a = (
#             thermal_calc._calc_entropy(T_a, opt_args=opt_args) )
#
#         return  thermal_entropy_a
#
#     def internal_energy(self, V_a, T_a):
#         V_a, T_a = core.fill_array(V_a, T_a)
#         compress_calc = self.calculators['compress']
#         # electronic_calc = self.calculators['electronic']
#
#         compress_path_const = compress_calc.path_const
#         assert compress_path_const=='T', (
#             'Only isothermal compress models supported now.')
#
#         T0 = self.refstate.ref_temp()
#         S0 = self.refstate.ref_entropy()
#         E0 = self.refstate.ref_internal_energy()
#
#         # S0, T0 = self.get_param_values(param_names=['S0','T0'])
#
#         # if   compress_path_const=='T':
#         #     F0, T0, S0 = self.get_param_values(param_names=['F0','T0','S0'])
#         #     E0 = F0 + T0*S0
#
#         # elif (compress_path_const=='S')|(compress_path_const=='0K'):
#         #     E0, = self.get_param_values(param_names=['E0'])
#
#         # else:
#         #     raise NotImplementedError(
#         #       'path_const '+path_const+' is not valid for CompressEos.')
#
#         F_compress = self.compress_energy(V_a)
#         # Sref = S0 + self.thermal_entropy(V_a, T0)
#         Sref = self.entropy(V_a, T0)
#         E_compress = F_compress + T0*Sref
#
#         # E_electronic_a = electronic_calc._calc_energy(V_a, T_a)
#
#         thermal_energy_a = self.thermal_energy(V_a, T_a)
#
#         # internal_energy_a = E0 + E_compress + thermal_energy_a + E_electronic_a
#         internal_energy_a = E0 + E_compress + thermal_energy_a
#         return internal_energy_a
#
#     def entropy(self, V_a, T_a):
#         V_a, T_a = core.fill_array(V_a, T_a)
#         S0 = self.refstate.ref_entropy()
#
#         thermal_entropy_a = self.thermal_entropy(V_a, T_a)
#
#         entropy_a = (S0 + thermal_entropy_a)
#         return entropy_a
#
#
#         pass
#
#     def _calc_energy(self, T_a, opt_args=None):
#         """Returns heat capacity as a function of temperature."""
#         pass
#
#     def _calc_entropy(self, T_a, opt_args=None):
#         """Returns heat capacity as a function of temperature."""
#         pass
#
#     def _calc_entropy_pot(self, T_a, opt_args=None):
#         """Returns heat capacity as a function of temperature."""
#         pass
#
#     def _calc_dEdV_T(self, V_a, T_a, theta_a, gamma_a):
#         pass
#
#     def _calc_dEdV_S(self, V_a, T_a, theta_a, gamma_a):
#         pass
# #====================================================================
class _GeneralPolyCalc(with_metaclass(ABCMeta, core.Calculator)):
    _kind_opts = ['V','logV']

    def __init__(self, eos_mod, order=6, kind='V', coef_basename='bcoef'):

        if kind not in self._kind_opts:
            raise NotImplementedError(
                'kind '+kind+' is not valid for GeneralPolyCalc.')

        if ((not np.isscalar(order)) | (order < 0) | (np.mod(order,0) !=0)):
            raise ValueError(
                'order ' + str(order) +' is not valid for GeneralPolyCalc. '+
                'It must be a positive integer.')

        self._eos_mod = eos_mod
        self._coef_basename = coef_basename
        self._kind = kind

        self._init_params(order)
        self._required_calculators = None

    def _get_polyval_coef(self):
        coef_basename = self._coef_basename

        param_names = self.eos_mod.get_array_param_names(coef_basename)
        param_values = self.eos_mod.get_param_values(param_names=param_names)

        coef_index = core.get_array_param_index(param_names)
        order = np.max(coef_index)+1
        param_full = np.zeros(order)
        param_full[coef_index] = param_values

        coefs_a = param_full[::-1]  # Reverse array for np.polyval
        return coefs_a

    def _calc_vol_dev(self, V_a):
        kind = self._kind
        V0 = self.eos_mod.get_param_values(param_names='V0')

        if kind=='V':
            vol_dev = V_a/V0 - 1
        elif kind=='logV':
            vol_dev = np.log(V_a/V0)
        elif kind=='rho':
            vol_dev = V0/V_a - 1

        return vol_dev

    def _calc_vol_dev_deriv(self, V_a):
        kind = self._kind
        V0 = self.eos_mod.get_param_values(param_names='V0')

        if kind=='V':
            vol_dev_deriv = 1/V0*np.ones(V_a.shape)
        elif kind=='logV':
            vol_dev_deriv = +1/V_a
        elif kind=='rho':
            vol_dev_deriv = -V0/V_a**2

        return vol_dev_deriv

    def calc_coef(self, V_a):
        vol_dev = self._calc_vol_dev(V_a)
        coefs_a = self._get_polyval_coef()
        coef_V = np.polyval(coefs_a, vol_dev)
        return coef_V

    def calc_coef_deriv(self, V_a):
        vol_dev = self._calc_vol_dev(V_a)
        vol_dev_deriv = self._calc_vol_dev_deriv(V_a)
        coefs_a = self._get_polyval_coef()
        order = coefs_a.size-1
        coefs_deriv_a = np.polyder(coefs_a)
        coef_deriv_V = vol_dev_deriv * np.polyval(coefs_deriv_a, vol_dev)
        return coef_deriv_V
#====================================================================
class _RTPolyCalc(with_metaclass(ABCMeta, _GeneralPolyCalc)):

    def __init__(self, eos_mod, order=6, kind='V', coef_basename='bcoef',
                 RTpress=False):
        self.RTpress=RTpress
        super(_RTPolyCalc, self).__init__(eos_mod, order=order, kind=kind,
                                          coef_basename=coef_basename)
        pass

    def _init_params(self, order):
        RTpress = self.RTpress
        kind = self._kind
        coef_basename = self._coef_basename

        if kind=='V':
            # Defaults from Spera2011
            # NOTE switch units cc/g -> ang3,  kJ/g -> eV

            V0 = 0.408031

            if coef_basename == 'bcoef':

                if RTpress:
                    coefs = np.array([+0.9821, +0.615, +1.31,
                                      -3.0, -4.1,0])
                else:
                    shifted_coefs = np.array([-.371466, 7.09542, -45.7362,
                                              139.020, -201.487, 112.513])
                    coefs = core.shift_poly(shifted_coefs, xscale=V0)

            elif coef_basename == 'acoef':
                shifted_coefs = np.array([127.116, -3503.98, 20724.4, -60212.0,
                                          86060.5, -48520.4])

                coefs = core.shift_poly(shifted_coefs, xscale=V0)

            else:
                raise NotImplemented('This is not a valid RTcoef type')


        elif kind=='logV':
            # Defaults from Spera2011
            # NOTE switch units cc/g -> ang3,  kJ/g -> eV
            V0 = 0.408031

            if coef_basename == 'bcoef':
                coefs = np.array([ 0.04070134,  0.02020084, -0.07904852,
                                  -0.45542896, -0.55941513, -0.20257299])

            elif coef_basename == 'acoef':
                coefs = np.array([-105.88653606, -1.56279233, 16.34275157,
                                  87.28979726, 121.16123888,   40.31492443])

            else:
                raise NotImplemented('This is not a valid RTcoef type')

        param_names = core.make_array_param_names(
            coef_basename, order, skipzero=False)
        param_defaults = [0 for ind in range(0,order+1)]
        if order>5:
            param_defaults[0:6] = coefs
        else:
            param_defaults[0:order+1] = coefs[0:order+1]

        param_scales = [1 for ind in range(0,order+1)]
        param_units = core.make_array_param_units(
            param_names, base_unit='kJ/g', deriv_unit='(cc/g)')

        param_names.append('V0')
        param_defaults.append(V0)
        param_scales.append(V0/10)
        param_units.append('cc/g')

        self._set_params(param_names, param_units,
                         param_defaults, param_scales, order=order)
#====================================================================
class _logPThermPolyCalc(with_metaclass(ABCMeta, _GeneralPolyCalc)):
    def __init__(self, eos_mod, order=5, kind='V'):
        coef_basename='logPth_coef'
        super(_logPThermPolyCalc, self).__init__(
            eos_mod, order=order, kind=kind, coef_basename=coef_basename)
        pass

    def _init_params(self, order):
        kind = self._kind
        coef_basename = self._coef_basename

        Vref = 38.88 # cm^3/mol
        Natom = 5
        Vconv = 1./Natom*core.CONSTS['ang3percc']/core.CONSTS['Nmol']
        V0 = Vref*Vconv
        # units on Pth = [GPa/K]
        base_unit='GPa/K'

        if kind=='V':
            # Defaults from deKoker2009
            coefs = np.array([-5.96440829, -1.68059219,  1.12415712,
                              -0.6361655 ,  1.58385018, -0.34339286])
            deriv_unit='(ang^3)'

        elif kind=='logV':
            # Defaults from deKoker2009
            coefs = np.array([-5.96467796, -1.66870847,  0.32322974,
                              -0.04433057, -0.07772874, -0.00781603])
            deriv_unit='(1)'

        else:
            raise NotImplemented('This is not a valid Pth_coef kind')

        param_names = core.make_array_param_names(
            coef_basename, order, skipzero=False)
        param_defaults = [0 for ind in range(0,order+1)]
        if order>5:
            param_defaults[0:6] = coefs
        else:
            param_defaults[0:order+1] = coefs[0:order+1]

        param_scales = [1 for ind in range(0,order+1)]
        param_units = core.make_array_param_units(
            param_names, base_unit=base_unit, deriv_unit=deriv_unit)

        # Add V0 to array param lists
        param_names.append('V0')
        param_defaults.append(V0)
        param_scales.append(V0/10)
        param_units.append('ang^3')

        self._set_params(param_names, param_units, param_defaults,
                         param_scales, order=order)
#====================================================================
