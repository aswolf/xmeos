# -*- coding: utf-8 -*-
# from __future__ import absolute_import, print_function, division
from future.utils import with_metaclass

from builtins import str
from builtins import range
import numpy as np
import scipy as sp
from abc import ABCMeta, abstractmethod
from scipy import integrate
import scipy.interpolate as interpolate

from . import core
from . import refstate
from . import _debye

__all__ = ['ThermalEos','ThermalCalc']
# 'CompressedThermalEos','CompressedThermalCalc']

#====================================================================
# Base Classes
#====================================================================
def set_calculator(eos_mod, kind, kind_opts, external_bcoef=False):
    assert kind in kind_opts, (
        kind + ' is not a valid thermal calculator. '+
        'You must select one of: ' +  str(kind_opts))

    if   kind=='Debye':
        calc = _Debye(eos_mod)
    elif kind=='Einstein':
        calc = _Einstein(eos_mod)
    elif kind=='PTherm':
        calc = _PTherm(eos_mod)
    elif kind=='GenRosenfeldTarazona':
        calc = _GenRosenfeldTarazona(eos_mod, external_bcoef=external_bcoef)
    elif kind=='ConstHeatCap':
        calc = _ConstHeatCap(eos_mod)
    elif kind=='Cp-Berman':
        calc = _Cp_Berman(eos_mod)
    elif kind=='Cp-Fei':
        calc = _Cp_Fei(eos_mod)
    elif kind=='Cp-Maier-Kelley':
        calc = _Cp_Maier_Kelley(eos_mod)
    else:
        raise NotImplementedError(kind+' is not a valid '+\
                                  'Thermal Calculator.')

    eos_mod._add_calculator(calc, calc_type='thermal')
    pass
#====================================================================
class ThermalEos(with_metaclass(ABCMeta, core.Eos)):
    """
    EOS model for thermal energy heating path.

    Parameters
    ----------
    Path can either be isochoric (V=const) or isobaric (P=const)

    For this restricted path, thermodyn properties depend only on temperature.

    """

    _path_opts = ['V','P']
    _kind_opts = ['Debye','Einstein','GenRosenfeldTarazona','ConstHeatCap',
                  'Cp-Berman','Cp-Fei','Cp-Maier-Kelley']

    def __init__(self, kind='Debye', natom=1, model_state={}):

        ref_compress_state='P0'
        ref_thermal_state='T0'
        ref_energy_type='E0'
        self._pre_init(natom=natom)

        set_calculator(self, kind, self._kind_opts)
        # self._set_ref_state()


        refstate.set_calculator(self, ref_compress_state=ref_compress_state,
                                ref_thermal_state=ref_thermal_state,
                                ref_energy_type=ref_energy_type)

        self._post_init(model_state=model_state)
        pass

    def __repr__(self):
        calc = self.calculators['thermal']
        return ("ThermalEos(kind={kind}, natom={natom}, "
                "model_state={model_state}, "
                ")"
                .format(kind=repr(calc.name),
                        natom=repr(self.natom),
                        model_state=self.model_state
                        )
                )

    def _set_ref_state(self):
        calc = self.calculators['thermal']
        path_const = calc.path_const

        # Add needed extra parameters (depending on path_const)
        if path_const=='V':
            param_ref_names = ['V0']
            param_ref_units = ['ang^3']
            param_ref_defaults = [100]
            param_ref_scales = [100]

        elif path_const=='P':
            P0 = 0
            param_ref_names = ['P0']
            param_ref_units = ['GPa']
            param_ref_defaults = [0.0]
            param_ref_scales = [100]

        else:
            raise NotImplementedError(
                'path_const '+path_const+' is not valid for ThermalEos.')


        self._path_const = path_const
        self._param_ref_names = param_ref_names
        self._param_ref_units = param_ref_units
        self._param_ref_defaults = param_ref_defaults
        self._param_ref_scales = param_ref_scales
        pass

    @property
    def path_opts(self):
        return self._path_opts

    @property
    def path_const(self):
        return self._path_const

    def energy(self, T_a):
        calculator = self.calculators['thermal']
        energy_a =  calculator._calc_energy(T_a)
        return energy_a

    def heat_capacity(self, T_a):
        calculator = self.calculators['thermal']
        heat_capacity_a =  calculator._calc_heat_capacity(T_a)
        return heat_capacity_a

    def entropy(self, T_a):
        calculator = self.calculators['thermal']
        entropy_a =  calculator._calc_entropy(T_a)
        return entropy_a

    def dEdV_T(self, T_a):
        pass

    def dEdV_S(self, T_a):
        pass
#====================================================================

#====================================================================
# Calculators
#====================================================================
class ThermalCalc(with_metaclass(ABCMeta, core.Calculator)):
    """
    Abstract Equation of State class for a reference Thermal Energy Path

    Path can either be isothermal (T=const) or adiabatic (S=const)

    For this restricted path, thermodyn properties depend only on volume

    """

    _path_opts = ['V','P']

    def __init__(self, eos_mod, path_const=None):
        # assert path_const in self.path_opts, path_const + ' is not a valid ' + \
        #     'path const. You must select one of: ' + path_opts

        self._eos_mod = eos_mod
        self._init_params()
        self._required_calculators = None

        self._path_const = path_const
        pass

    @property
    def path_opts(self):
        return self._path_opts

    @property
    def path_const(self):
        return self._path_const

    @property
    def ndof(self):
        return self._ndof

    ####################
    # Required Methods #
    ####################
    @abstractmethod
    def _init_params(self):
        """Initialize list of calculator parameter names."""
        pass

    # @abstractmethod
    # def _calc_heat_capacity(self, T_a):
    #     """Returns heat capacity as a function of temperature."""
    #     pass

    def _get_Cv_limit(self):
        Cvlimfac, = self.eos_mod.get_param_values(param_names=['Cvlimfac'])
        ndof = self.ndof
        natom = self.eos_mod.natom
       #  print('ndof = ',ndof)
       #  print('natom = ',natom)
       #  print('Cvlimfac = ',Cvlimfac)
        Cvlim = Cvlimfac*ndof/2*natom*core.CONSTS['kboltz']
        return Cvlim

    # @abstractmethod
    # def _calc_energy(self, T_a):
    #     """Returns thermal energy as a function of temperature."""
    #     pass

    @abstractmethod
    def _calc_entropy(self, T_a):
        pass

    @abstractmethod
    def _calc_dEdV_T(self, T_a):
        pass

    @abstractmethod
    def _calc_dEdV_S(self, T_a):
        pass

    ####################
    # Optional Methods #
    ####################

    def _calc_param_deriv(self, fname, paramname, V_a, dxfrac=1e-6):
        scale_a, paramkey_a = self.get_param_scale(apply_expand_adj=True )
        scale = scale_a[paramkey_a==paramname][0]
        # print 'scale: ' + np.str(scale)

        #if (paramname is 'E0') and (fname is 'energy'):
        #    return np.ones(V_a.shape)
        try:
            fun = getattr(self, fname)
            # Note that self is implicitly included
            val0_a = fun(V_a)
        except:
            assert False, 'That is not a valid function name ' + \
                '(e.g. it should be press or energy)'

        try:
            param = core.get_params([paramname])[0]
            dparam = scale*dxfrac
            # print 'param: ' + np.str(param)
            # print 'dparam: ' + np.str(dparam)
        except:
            assert False, 'This is not a valid parameter name'

        # set param value in eos_d dict
        core.set_params([paramname,], [param+dparam,])

        # Note that self is implicitly included
        dval_a = fun(V_a) - val0_a

        # reset param to original value
        core.set_params([paramname], [param])

        deriv_a = dval_a/dxfrac
        return deriv_a

    def _calc_energy_perturb(self, V_a):
        """Returns Energy pertubation basis functions resulting from fractional changes to EOS params."""

        fname = 'energy'
        scale_a, paramkey_a = self.get_param_scale(
            apply_expand_adj=self.expand_adj)
        Eperturb_a = []
        for paramname in paramkey_a:
            iEperturb_a = self._calc_param_deriv(fname, paramname, V_a)
            Eperturb_a.append(iEperturb_a)

        Eperturb_a = np.array(Eperturb_a)

        return Eperturb_a, scale_a, paramkey_a
#====================================================================

#====================================================================
# Implementations
#====================================================================
class _Debye(ThermalCalc):
    """
    Implimentation copied from Burnman.

    """

    _path_opts=['V']
    _ndof = 6

    def __init__(self, eos_mod):
        super(_Debye, self).__init__(eos_mod, path_const='V')
        pass

    def _init_params(self):
        """Initialize list of calculator parameter names."""

        T0 = 0
        T0_scale = 300
        theta0 = 1000
        Cvlimfac = 1

        param_names = ['theta0', 'Cvlimfac']
        param_units = ['K', '1']
        param_defaults = [theta0, Cvlimfac]
        param_scales = [theta0, Cvlimfac]

        self._set_params(param_names, param_units,
                         param_defaults, param_scales)

        pass

    def _calc_heat_capacity(self, T_a, theta=None):
        """Returns heat capacity as a function of temperature."""

        T_a = core.fill_array(T_a)
        Cvlim = self._get_Cv_limit()

        if theta is None:
            theta, = self.eos_mod.get_param_values(param_names=['theta0'])

        x = theta/T_a
        Cv_values = Cvlim*_debye.debye_heat_capacity_fun(x)
        return Cv_values

    def _calc_energy(self, T_a, theta=None, T0=None):
        """Returns heat capacity as a function of temperature."""

        T_a = core.fill_array(T_a)
        Cvlim = self._get_Cv_limit()

        if theta is None:
            theta, = self.eos_mod.get_param_values(param_names=['theta0'])
        if T0 is None:
            T0 = self.eos_mod.refstate.ref_temp()

        x = core.fill_array(theta/T_a)
        xref = core.fill_array(theta/T0)

        energy = Cvlim*(T_a*_debye.debye3_fun(x)
                        -T0*_debye.debye3_fun(xref))

        return energy

    def _calc_entropy(self, T_a, theta=None, T0=None, theta0=None):
        """Returns heat capacity as a function of temperature."""

        T_a = core.fill_array(T_a)
        Cvlim = self._get_Cv_limit()

        if T0 is None:
            T0 = self.eos_mod.refstate.ref_temp()
        if theta is None:
            theta, = self.eos_mod.get_param_values(param_names=['theta0'])
        if theta0 is None:
            theta0, = self.eos_mod.get_param_values(param_names=['theta0'])

        x = core.fill_array(theta/T_a)
        xref = core.fill_array(theta0/T0)

        entropy = Cvlim*(+_debye.debye_entropy_fun(x)
                         -_debye.debye_entropy_fun(xref))

        return entropy

    def _calc_dEdV_T(self, V_a, T_a, theta_a, gamma_a):
        Cvlim = self._get_Cv_limit()

        x = theta_a/np.array(T_a)
        dEdV_T = -Cvlim*gamma_a/V_a*theta_a*_debye.debye3_deriv_fun(x)
        return dEdV_T

    def _calc_dEdV_S(self, V_a, T_a, theta_a, gamma_a):
        x = theta_a/np.array(T_a)
        dEdV_S = 1/x*self._calc_dEdV_T(V_a, T_a, theta_a, gamma_a)
        return dEdV_S
#====================================================================
class _Einstein(ThermalCalc):

    _ndof = 6
    _EPS = np.finfo(np.float).eps
    _path_opts=['V']

    def __init__(self, eos_mod):
        super(_Einstein, self).__init__(eos_mod, path_const='V')
        pass

    def _init_params(self):
        """Initialize list of calculator parameter names."""

        natom = self.eos_mod.natom

        T0 = 0
        T0_scale = 300
        theta0 = 1000
        Cvlimfac = 1

        param_names = ['theta0', 'Cvlimfac']
        param_units = ['K', '1']
        param_defaults = [theta0, Cvlimfac]
        param_scales = [theta0, Cvlimfac]

        self._set_params(param_names, param_units,
                         param_defaults, param_scales)

        pass

    def _calc_energy_factor(self, x):
        fac = 1/(np.exp(x)-1)
        try:
            fac[1/x < self._EPS] = 0
        except TypeError:
            if 1/x < self._EPS:
                fac = 0

        return fac

    def _calc_flogf(self, x, Nosc):
        f = Nosc*self._calc_energy_factor(x)
        flogf = f*np.log(f)
        try:
            flogf[f==0] = 0.0
        except TypeError:
            if f==0:
                flogf = 0

        return flogf

    def _calc_heat_capacity(self, T_a, theta=None):
        """Returns heat capacity as a function of temperature."""

        theta0, = self.eos_mod.get_param_values(param_names=['theta0'])
        Cvlim = self._get_Cv_limit()

        if theta is None:
            theta = theta0

        T_a = np.array(T_a)

        x = theta/T_a
        Cv_a = Cvlim*x**2*np.exp(x)/(np.exp(x)-1)**2
        Cv_a[1/x < self._EPS] = 0

        return Cv_a

    def _calc_energy(self, T_a, theta=None, T0=None):
        """Returns heat capacity as a function of temperature."""

        T_a = core.fill_array(T_a)
        Cvlim = self._get_Cv_limit()

        if theta is None:
            theta, = self.eos_mod.get_param_values(param_names=['theta0'])
        if T0 is None:
            T0 = self.eos_mod.refstate.ref_temp()

        x = core.fill_array(theta/T_a)
        xref = core.fill_array(theta/T0)

        # NOTE: Cannot include zero-pt energy since we are using energy diff
        energy = Cvlim*theta*(
            self._calc_energy_factor(x)-self._calc_energy_factor(xref))
        return energy

    def _calc_entropy(self, T_a, theta=None, T0=None, theta0=None):
        """Returns heat capacity as a function of temperature."""

        T_a = core.fill_array(T_a)
        Cvlim = self._get_Cv_limit()

        if T0 is None:
            T0 = self.eos_mod.refstate.ref_temp()
        if theta is None:
            theta, = self.eos_mod.get_param_values(param_names=['theta0'])
        if theta0 is None:
            theta0, = self.eos_mod.get_param_values(param_names=['theta0'])

        x = core.fill_array(theta/T_a)
        xref = core.fill_array(theta0/T0)

        Nosc = Cvlim/core.CONSTS['kboltz']

        Equanta = Nosc*self._calc_energy_factor(x)
        Squanta = self._calc_flogf(x, Nosc)

        Equanta0 = Nosc*self._calc_energy_factor(xref)
        Squanta0 = self._calc_flogf(xref, Nosc)

        entropy = core.CONSTS['kboltz']*(
            (Nosc+Equanta)*np.log(Nosc+Equanta)
            - (Nosc+Equanta0)*np.log(Nosc+Equanta0)
            - (Squanta-Squanta0))

        return entropy

    def _einstein_fun(self, x):
        energy_fac = 1/2 + 1/(np.exp(x)-1)
        return energy_fac

    def _einstein_deriv_fun(self, x):
        deriv_fac = -np.exp(x)/(np.exp(x)-1)**2
        return deriv_fac

    # FIX THESE!!!!
    def _calc_dEdV_T(self, V_a, T_a, theta_a, gamma_a, Cvmax=None):
        Cvlim = self._get_Cv_limit()
        # Cvmax, = self.eos_mod.get_param_values(
        #     param_names=['Cvmax'], overrides=[Cvmax])

        x = theta_a/np.array(T_a)
        dEdV_S = self._calc_dEdV_S(V_a, T_a, theta_a, gamma_a, Cvmax=Cvlim)
        dEdV_T = dEdV_S - Cvlim*theta_a*gamma_a/V_a*x*self._einstein_deriv_fun(x)
        return dEdV_T

    def _calc_dEdV_S(self, V_a, T_a, theta_a, gamma_a, Cvmax=None):
        Cvlim = self._get_Cv_limit()
        # Cvmax, = self.eos_mod.get_param_values(
        #     param_names=['Cvmax'], overrides=[Cvmax])

        x = theta_a/np.array(T_a)
        dEdV_S = -Cvlim*theta_a*gamma_a/V_a*self._einstein_fun(x)
        return dEdV_S
#====================================================================
class _GenRosenfeldTarazona(ThermalCalc):

    _ndof = 3
    _EPS = np.finfo(np.float).eps
    _path_opts=['V']

    def __init__(self, eos_mod, external_bcoef=False):
        self._external_bcoef = external_bcoef
        super(_GenRosenfeldTarazona, self).__init__(eos_mod, path_const='V')
        pass

    def _init_params(self):
        """Initialize list of calculator parameter names."""

        # T0 = 3000
        # T0_scl = T0*0.1
        mexp = 3/5
        bcoef = -5
        Cvlimfac = 1
        Cvlimfac_scl = 0.03
        coef_scl = np.abs(bcoef)

        param_names = ['mexp', 'Cvlimfac']
        param_units = ['1', '1']
        param_defaults = [mexp, Cvlimfac]
        param_scales = [mexp, Cvlimfac_scl]
        # acoef = -20
        if not self._external_bcoef:
            param_names.append('bcoef')
            param_units.append('eV')
            param_defaults.append(bcoef)
            param_scales.append(coef_scl)

        self._set_params(param_names, param_units,
                         param_defaults, param_scales)

    def _calc_heat_capacity(self, T_a, bcoef=None):
        """Returns heat capacity as a function of temperature."""

        T_a = core.fill_array(T_a)
        Cvlim = self._get_Cv_limit()

        if bcoef is None:
            bcoef, = self.eos_mod.get_param_values(param_names=['bcoef'])

        dtherm_dev = self._calc_therm_dev_deriv(T_a)
        Cv_pot = bcoef*dtherm_dev
        Cv_kin = Cvlim
        Cv_values = Cv_pot + Cv_kin

        return Cv_values

    def _calc_therm_dev(self, T_a):
        T_a = core.fill_array(T_a)

        T0 = self.eos_mod.refstate.ref_temp()
        # T0, mexp = self.eos_mod.get_param_values(param_names=['T0','mexp'])
        mexp = self.eos_mod.get_param_values(param_names='mexp')
        therm_dev_a = (T_a/T0)**mexp - 1
        return therm_dev_a

    def _calc_therm_dev_deriv(self, T_a):
        T_a = core.fill_array(T_a)
        T0 = self.eos_mod.refstate.ref_temp()
        # T0, mexp = self.eos_mod.get_param_values(param_names=['T0','mexp'])
        mexp = self.eos_mod.get_param_values(param_names='mexp')
        dtherm_dev_a = (mexp/T0)*(T_a/T0)**(mexp-1)
        return dtherm_dev_a

    def _calc_energy(self, T_a, bcoef=None, Tref=None):
        """Returns heat capacity as a function of temperature."""

        T_a = core.fill_array(T_a)
        Cvlim = self._get_Cv_limit()

        if bcoef is None:
            bcoef, = self.eos_mod.get_param_values(param_names=['bcoef'])
        if Tref is None:
            Tref = self.eos_mod.refstate.ref_temp()
            # Tref, = self.eos_mod.get_param_values(param_names=['T0'])

        therm_dev = self._calc_therm_dev(T_a)
        energy_pot = bcoef*therm_dev
        energy_kin = Cvlim*(T_a-Tref)
        energy = energy_pot + energy_kin

        return energy

    def _calc_entropy(self, T_a, bcoef=None, Tref=None):
        """Returns heat capacity as a function of temperature."""

        Cvlim = self._get_Cv_limit()

        if bcoef is None:
            bcoef, = self.eos_mod.get_param_values(param_names=['bcoef'])
        if Tref is None:
            Tref = self.eos_mod.refstate.ref_temp()
            # Tref, = self.eos_mod.get_param_values(param_names=['T0'])

        T_a, Tref = core.fill_array(T_a, Tref)

        mexp, = self.eos_mod.get_param_values(param_names=['mexp'])
        entropy_pot = bcoef/(mexp-1)*(self._calc_therm_dev_deriv(T_a)
                                      - self._calc_therm_dev_deriv(Tref))
        entropy_kin = Cvlim*np.log(T_a/Tref)
        entropy = entropy_pot + entropy_kin

        return entropy

    def _calc_entropy_pot(self, T_a, bcoef=None, Tref=None):
        """Returns heat capacity as a function of temperature."""

        Cvlim = self._get_Cv_limit()

        if bcoef is None:
            bcoef, = self.eos_mod.get_param_values(param_names=['bcoef'])
        if Tref is None:
            Tref = self.eos_mod.refstate.ref_temp()
            # Tref, = self.eos_mod.get_param_values(param_names=['T0'])

        T_a, Tref = core.fill_array(T_a, Tref)

        mexp, = self.eos_mod.get_param_values(param_names=['mexp'])
        entropy_pot = bcoef/(mexp-1)*(self._calc_therm_dev_deriv(T_a)
                                      - self._calc_therm_dev_deriv(Tref))

        return entropy_pot

    def _calc_dEdV_T(self, V_a, T_a, theta_a, gamma_a):
        return np.nan

    def _calc_dEdV_S(self, V_a, T_a, theta_a, gamma_a):
        return np.nan
#====================================================================
class _PTherm(ThermalCalc):
    """

    """

    _path_opts=['V']
    _ndof = 6

    def __init__(self, eos_mod):
        super(_PTherm, self).__init__(eos_mod, path_const='V')
        pass

    def _init_params(self):
        """Initialize list of calculator parameter names."""

        T0 = 300
        T0_scale = 300
        Pth0 = 3e-3 # GPa/K
        gamma0 = 1 # GPa/K

        param_names = ['Pth0', 'gamma0']
        param_units = ['GPa/K', '1']
        param_defaults = [Pth0, gamma0]
        param_scales = [1e-3, 1]

        self._set_params(param_names, param_units,
                         param_defaults, param_scales)

        pass

    def _calc_press(self, T_a, Pth=None, T0=None):
        T_a = core.fill_array(T_a)

        if T0 is None:
            T0 = self.eos_mod.refstate.ref_temp()
        if Pth is None:
            Pth, = self.eos_mod.get_param_values(param_names=['Pth0'])

        dPtherm = (T_a-T0)*Pth
        return dPtherm

    def _calc_energy(self, T_a, gamma=None, Pth=None, T0=None):
        T_a = core.fill_array(T_a)

        if gamma is None:
            gamma, = self.eos_mod.get_param_values(param_names=['gamma0'])

        dPtherm = self._calc_press(T_a, Pth=Pth, T0=T0)
        dEtherm = dPtherm/(core.CONSTS['PV_ratio']*gamma/V)
        return dEtherm

    def _calc_heat_capacity(self, T_a, gamma=None, V=None, Pth=None, T0=None):
        """Returns heat capacity as a function of temperature."""
        T_a = core.fill_array(T_a)

        if gamma is None:
            gamma, = self.eos_mod.get_param_values(param_names=['gamma0'])
        if T0 is None:
            T0 = self.eos_mod.refstate.ref_temp()
        if V is None:
            V = self.eos_mod.refstate.ref_volume()
        if Pth is None:
            Pth, = self.eos_mod.get_param_values(param_names=['Pth0'])

        V_a, T_a = core.fill_array(V, T_a)
        Cv = Pth/(core.CONSTS['PV_ratio']*gamma/V_a)

        return Cv

    def _calc_entropy(self, T_a, gamma=None, V=None, Pth=None, T0=None):
        """Returns heat capacity as a function of temperature."""

        T_a = core.fill_array(T_a)

        Cv_const = self._calc_heat_capacity(T_a, gamma=gamma, V=V,
                                            Pth=Pth, T0=T0)
        entropy = Cv_const*np.log(T_a/T0)
        return entropy

    def _calc_dEdV_T(self, T_a):
        return None

    def _calc_dEdV_S(self, T_a):
        return None
#====================================================================
class _ConstHeatCap(ThermalCalc):

    _EPS = np.finfo(np.float).eps
    _path_opts=['V']

    def __init__(self, eos_mod, ndof=3):
        """
        default ndof is 3 relevant for liquids
        """
        super(_ConstHeatCap, self).__init__(eos_mod, path_const='V')
        self._ndof = ndof
        pass

    def _init_params(self):
        """Initialize list of calculator parameter names."""

        natom = self.eos_mod.natom

        T0 = 1000
        T0_scale = 300
        Cvlimfac = 1
        theta0 = np.nan

        param_names = ['theta0','Cvlimfac']
        param_units = ['K','1']
        param_defaults = [theta0, Cvlimfac]
        param_scales = [theta0, 0.03]

        # param_names = ['Cvlimfac']
        # param_units = ['1']
        # param_defaults = [Cvlimfac]
        # param_scales = [1]

        self._set_params(param_names, param_units,
                         param_defaults, param_scales)

        pass

    def _calc_heat_capacity(self, T_a, theta=None, T0=None):
        """
        Returns heat capacity as a function of temperature.

        T0, theta included for compatibility with MieGruneisenEos.
        """

        T_a = core.fill_array(T_a)

        Cvlimfac, = self.eos_mod.get_param_values(param_names=['Cvlimfac'])
        Cvlim = self._get_Cv_limit()
        Cv = Cvlimfac*Cvlim

        Cv_a, T_a = core.fill_array(Cv, T_a)
        return Cv_a

    def _calc_energy(self, T_a, theta=None, T0=None):
        """
        Returns heat capacity as a function of temperature.

        theta included for compatibility with MieGruneisenEos.
        """

        T_a = core.fill_array(T_a)
        if T0 is None:
            T0 = self.eos_mod.refstate.ref_temp()

        # print(T0)
        # print(theta)

        Cv_a = self._calc_heat_capacity(T_a, T0=T0)
        energy = Cv_a*(T_a-T0)

        return energy

    def _calc_entropy(self, T_a, T0=None, theta=None, theta0=None):
        """
        Returns heat capacity as a function of temperature.

        theta & theta0 included for compatibility with MieGruneisenEos.
        """

        T_a = core.fill_array(T_a)
        if T0 is None:
            T0 = self.eos_mod.refstate.ref_temp()

        Cv_a = self._calc_heat_capacity(T_a, T0=T0)
        S_a = Cv_a*np.log(T_a/T0)

        return S_a

    # FIX THESE!!!!
    def _calc_dEdV_T(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)
        return 0*V_a


    def _calc_dEdV_S(self, V_a, T_a, theta_a, gamma_a, Cvmax=None):
        V_a, T_a = core.fill_array(V_a, T_a)
        return 0*V_a
#====================================================================
# class _PowLawHeatCap(ThermalCalc):
#     _EPS = np.finfo(np.float).eps
#     _path_opts=['V']
#
#     def __init__(self, eos_mod, ndof=3):
#         """
#         default ndof is 3 relevant for liquids
#         """
#         super(_PowLawHeatCap, self).__init__(eos_mod, path_const='V')
#         self._ndof = ndof
#         pass
#
#     def _init_params(self):
#         """Initialize list of calculator parameter names."""
#
#         natom = self.eos_mod.natom
#
#         T0 = 1000
#         T0_scale = 300
#         Cvlimfac = 1
#         theta0 = np.nan
#         Cvexp = 1
#
#         param_names = ['theta0','Cvlimfac','Cvexp']
#         param_units = ['K','1','Cvexp']
#         param_defaults = [theta0, Cvlimfac, Cvexp]
#         param_scales = [theta0, 0.1, .1]
#
#         self._set_params(param_names, param_units,
#                          param_defaults, param_scales)
#
#         pass
#
#     def _calc_heat_capacity(self, T_a, theta=None, T0=None):
#         """
#         Returns heat capacity as a function of temperature.
#
#         T0, theta included for compatibility with MieGruneisenEos.
#         """
#
#         T_a = core.fill_array(T_a)
#
#         Cvlimfac, Cvexp = self.eos_mod.get_param_values(param_names=['Cvlimfac','Cvexp'])
#         Cvlim = self._get_Cv_limit()
#         Cv = Cvlimfac*Cvlim*
#
#         Cv_a, T_a = core.fill_array(Cv, T_a)
#         return Cv_a
#
#     def _calc_energy(self, T_a, theta=None, T0=None):
#         """
#         Returns heat capacity as a function of temperature.
#
#         theta included for compatibility with MieGruneisenEos.
#         """
#
#         T_a = core.fill_array(T_a)
#         if T0 is None:
#             T0 = self.eos_mod.refstate.ref_temp()
#
#         # print(T0)
#         # print(theta)
#
#         Cv_a = self._calc_heat_capacity(T_a, T0=T0)
#         energy = Cv_a*(T_a-T0)
#
#         return energy
#
#     def _calc_entropy(self, T_a, T0=None, theta=None, theta0=None):
#         """
#         Returns heat capacity as a function of temperature.
#
#         theta & theta0 included for compatibility with MieGruneisenEos.
#         """
#
#         T_a = core.fill_array(T_a)
#         if T0 is None:
#             T0 = self.eos_mod.refstate.ref_temp()
#
#         Cv_a = self._calc_heat_capacity(T_a, T0=T0)
#         S_a = Cv_a*np.log(T_a/T0)
#
#         return S_a
#
#     # FIX THESE!!!!
#     def _calc_dEdV_T(self, V_a, T_a):
#         V_a, T_a = core.fill_array(V_a, T_a)
#         return 0*V_a
#
#
#     def _calc_dEdV_S(self, V_a, T_a, theta_a, gamma_a, Cvmax=None):
#         V_a, T_a = core.fill_array(V_a, T_a)
#         return 0*V_a
# #====================================================================
