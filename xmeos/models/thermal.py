# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from future.utils import with_metaclass

from builtins import str
from builtins import range
import numpy as np
import scipy as sp
from abc import ABCMeta, abstractmethod
from scipy import integrate
import scipy.interpolate as interpolate

from . import core
from . import _debye

__all__ = ['ThermalEos','ThermalCalc']
# 'CompressedThermalEos','CompressedThermalCalc']

#====================================================================
# Base Classes
#====================================================================
def set_calculator(eos_mod, kind, kind_opts):
    assert kind in kind_opts, (
        kind + ' is not a valid thermal calculator. '+
        'You must select one of: ' +  kind_opts)

    if   kind=='Debye':
        calc = _Debye(eos_mod)
    elif kind=='Einstein':
        calc = _Einstein(eos_mod)
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
    _kind_opts = ['Debye','Einstein','Cp-Berman','Cp-Fei','Cp-Maier-Kelley']

    def __init__(self, kind='Debye', natom=1, model_state={}):

        self._pre_init(natom=natom)

        set_calculator(self, kind, self._kind_opts)
        self._set_ref_state()

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

    def __init__(self, eos_mod):
        super(_Debye, self).__init__(eos_mod, path_const='V')
        pass

    def _init_params(self, theta_param=None):
        """Initialize list of calculator parameter names."""
        natom = self.eos_mod.natom
        T0 = 0
        T0_scale = 300
        theta0 = 1000
        Cvmax = 3*natom*core.CONSTS['kboltz']

        param_names = ['theta0', 'Cvmax', 'T0']
        param_units = ['K', 'eV/K', 'K']
        param_defaults = [theta0, Cvmax, T0]
        param_scales = [theta0, Cvmax, T0_scale]

        self._set_params(param_names, param_units,
                         param_defaults, param_scales)

        pass

    def _calc_heat_capacity(self, T_a, theta=None):
        """Returns heat capacity as a function of temperature."""

        Cvmax, = self.eos_mod.get_param_values(param_names=['Cvmax'])

        if theta is None:
            theta, = self.eos_mod.get_param_values(param_names=['theta0'])

        x = theta/np.array(T_a)
        Cv_values = Cvmax*_debye.debye_heat_capacity_fun(x)
        return Cv_values

    def _calc_energy(self, T_a, theta=None, T0=None):
        """Returns heat capacity as a function of temperature."""

        Cvmax, = self.eos_mod.get_param_values(param_names=['Cvmax'])
        T0, = self.eos_mod.get_param_values(param_names=['T0'], overrides=[T0])

        if theta is None:
            theta, = self.eos_mod.get_param_values(param_names=['theta0'])

        T_a = np.array(T_a)
        x = core.fill_array(theta/T_a)
        x0 = core.fill_array(theta/T0)

        energy = Cvmax*(T_a*_debye.debye3_fun(x)
                        -T0*_debye.debye3_fun(x0))

        # energy = Cvmax*T_a*_debye.debye3_fun(x)

        return energy

    def _calc_entropy(self, T_a, theta=None, T0=None):
        """Returns heat capacity as a function of temperature."""

        Cvmax, = self.eos_mod.get_param_values(param_names=['Cvmax'])
        T0, = self.eos_mod.get_param_values(param_names=['T0'], overrides=[T0])

        if theta is None:
            theta, = self.eos_mod.get_param_values(param_names=['theta0'])

        T_a = np.array(T_a)
        x = core.fill_array(theta/T_a)
        x0 = core.fill_array(theta/T0)

        entropy = Cvmax*(_debye.debye_entropy_fun(x)
                         -_debye.debye_entropy_fun(x0))

        return entropy

    def _calc_dEdV_T(self, V_a, T_a, theta_a, gamma_a):
        Cvmax, = self.eos_mod.get_param_values(param_names=['Cvmax'])

        x = theta_a/np.array(T_a)
        dEdV_T = -Cvmax*gamma_a/V_a*theta_a*_debye.debye3_deriv_fun(x)
        return dEdV_T

    def _calc_dEdV_S(self, V_a, T_a, theta_a, gamma_a):
        Cvmax, = self.eos_mod.get_param_values(param_names=['Cvmax'])

        x = theta_a/np.array(T_a)
        dEdV_S = 1/x*self._calc_dEdV_T(V_a, T_a, theta_a, gamma_a)
        return dEdV_S
#====================================================================
class _Einstein(ThermalCalc):

    _EPS = np.finfo(np.float).eps
    _path_opts=['V']

    def __init__(self, eos_mod):
        super(_Einstein, self).__init__(eos_mod, path_const='V')
        pass

    def _init_params(self, theta_param=None):
        """Initialize list of calculator parameter names."""

        natom = self.eos_mod.natom

        T0 = 0
        T0_scale = 300
        theta0 = 1000
        Cvmax = 3*natom*core.CONSTS['kboltz']

        self._param_names = ['theta0', 'Cvmax', 'T0']
        self._param_units = ['K', 'eV/K', 'K']
        self._param_defaults = [theta0, Cvmax, T0]
        self._param_scales = [theta0, Cvmax, T0_scale]
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

        theta0, Cvmax = self.eos_mod.get_param_values(
            param_names=['theta0','Cvmax'])

        if theta is None:
            theta = theta0

        T_a = np.array(T_a)

        x = theta/T_a
        Cv_a = Cvmax*x**2*np.exp(x)/(np.exp(x)-1)**2
        Cv_a[1/x < self._EPS] = 0

        return Cv_a

    def _calc_energy(self, T_a, theta=None):
        """Returns heat capacity as a function of temperature."""

        theta0, Cvmax, T0 = self.eos_mod.get_param_values(
            param_names=['theta0', 'Cvmax', 'T0'])

        if theta is None:
            theta = theta0

        T_a = np.array(T_a)
        x = theta/T_a
        x0 = theta0/T0

        energy = Cvmax*theta0*(
            1/2 + self._calc_energy_factor(x)-self._calc_energy_factor(x0))
        return energy

    def _calc_entropy(self, T_a, theta=None):
        """Returns heat capacity as a function of temperature."""

        theta0, Cvmax, T0 = self.eos_mod.get_param_values(
            param_names=['theta0', 'Cvmax', 'T0'])

        if theta is None:
            theta = theta0

        T_a = np.array(T_a)
        x = theta/T_a
        x0 = theta0/T0
        Nosc = Cvmax/core.CONSTS['kboltz']

        Equanta = Nosc*self._calc_energy_factor(x)
        Squanta = self._calc_flogf(x, Nosc)

        Equanta0 = Nosc*self._calc_energy_factor(x0)
        Squanta0 = self._calc_flogf(x0, Nosc)

        entropy = core.CONSTS['kboltz']*(
            (Nosc+Equanta)*np.log(Nosc+Equanta)
            - (Nosc+Equanta0)*np.log(Nosc+Equanta0)
            - (Squanta-Squanta0))

        # NOTE that Nosc*log(Nosc) has been removed due to difference
        # - Nosc*np.log(Nosc)
        return entropy

    def _einstein_fun(self, x):
        energy_fac = 1/2 + 1/(np.exp(x)-1)
        return energy_fac

    def _einstein_deriv_fun(self, x):
        deriv_fac = -np.exp(x)/(np.exp(x)-1)**2
        return deriv_fac

    # FIX THESE!!!!
    def _calc_dEdV_T(self, V_a, T_a, theta_a, gamma_a, Cvmax=None):
        Cvmax, = self.eos_mod.get_param_values(
            param_names=['Cvmax'], overrides=[Cvmax])

        x = theta_a/np.array(T_a)
        dEdV_S = self._calc_dEdV_S(V_a, T_a, theta_a, gamma_a, Cvmax=Cvmax)
        dEdV_T = dEdV_S - Cvmax*theta_a*gamma_a/V_a*x*self._einstein_deriv_fun(x)
        return dEdV_T

    def _calc_dEdV_S(self, V_a, T_a, theta_a, gamma_a, Cvmax=None):
        Cvmax, = self.eos_mod.get_param_values(
            param_names=['Cvmax'], overrides=[Cvmax])

        x = theta_a/np.array(T_a)
        dEdV_S = -Cvmax*theta_a*gamma_a/V_a*self._einstein_fun(x)
        return dEdV_S
#====================================================================
