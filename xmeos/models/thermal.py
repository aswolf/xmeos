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

__all__ = ['ThermalEnergyEos','ThermalEnergyCalc']

#====================================================================
# Base Classes
#====================================================================
class ThermalEnergyEos(with_metaclass(ABCMeta, core.Eos)):
    """
    EOS model for thermal energy heating path.

    Parameters
    ----------
    Path can either be isochoric (V=const) or isobaric (P=const)

    For this restricted path, thermodyn properties depend only on temperature.

    """

    _path_opts = ['V','P']
    _kind_opts = ['Debye','Einstein','Cp-Berman','Cp-Fei','Cp-Maier-Kelley']

    def __init__(self, kind='Debye', natom=1, level_const=100,
                 model_state={}):
        self._pre_init(natom=natom)

        self._init_calculator(kind, level_const)

        self._post_init(model_state=model_state)

        pass


    def __repr__(self):
        return ("ThermalEnergyEos(kind={kind}, natom={natom}, "
                "level_const={level_const}, "
                "model_state={model_state}, "
                ")"
                .format(kind=self._kind,
                        natom=repr(self.natom),
                        level_const=repr(self.level_const),
                        model_state=self.model_state
                        )
                )

    def _init_calculator(self, kind, level_const):
        assert kind in self._kind_opts, kind + ' is not a valid ' + \
            'ThermalEnergyEos Calculator. You must select one of: ' + self._kind_opts

        self._kind = kind
        self._level_const = level_const

        _kind_opts = ['Debye','Einstein','Cp-Berman','Cp-Fei','Cp-Maier-Kelley']
        if   kind=='Debye':
            calc = _Debye(self, level_const=level_const)
        elif kind=='Einstein':
            calc = _Einstein(self, level_const=level_const)
        elif kind=='Cp-Berman':
            calc = _Cp_Berman(self, level_const=level_const)
        elif kind=='Cp-Fei':
            calc = _Cp_Fei(self, level_const=level_const)
        elif kind=='Cp-Maier-Kelley':
            calc = _Cp_Maier_Kelley(self, level_const=level_const)
        else:
            raise NotImplementedError(kind+' is not a valid '+\
                                      'ThermalEnergyEos Calculator.')

        path_const = calc.path_const
        self._add_calculator( calc, kind='thermal_energy' )
        self._path_const = path_const

        pass

    @property
    def path_opts(self):
        return self._path_opts

    @property
    def path_const(self):
        return self._path_const

    @property
    def level_const(self):
        return self._level_const

    def energy(self, T_a):
        calculator = self.calculators['thermal_energy']
        energy_a =  calculator._calc_energy(T_a)
        return energy_a

    def heat_capacity(self, T_a):
        calculator = self.calculators['thermal_energy']
        heat_capacity_a =  calculator._calc_heat_capacity(T_a)
        return heat_capacity_a
#====================================================================
# class CompressedThermalEnergyEos(with_metaclass(ABCMeta, core.Eos)):
#====================================================================
# class MieGruneisenEos(with_metaclass(ABCMeta, core.Eos)):
#====================================================================


#====================================================================
# Calculators
#====================================================================
class ThermalEnergyCalc(with_metaclass(ABCMeta, core.Calculator)):
    """
    Abstract Equation of State class for a reference Thermal Energy Path

    Path can either be isothermal (T=const) or adiabatic (S=const)

    For this restricted path, thermodyn properties depend only on volume

    """

    _path_opts = ['V','P']

    def __init__(self, eos_mod, path_const='P', level_const=0.0):
        assert path_const in self.path_opts, path_const + ' is not a valid ' + \
            'path const. You must select one of: ' + path_opts

        self._eos_mod = eos_mod
        self._init_params()
        self._required_calculators = None

        self.path_const = path_const
        self.level_const = level_const
        pass

    @property
    def path_opts(self):
        return self._path_opts

    def get_path_const(self):
        return self.path_const

    def get_level_const(self):
        return self.level_const

    ####################
    # Required Methods #
    ####################
    @abstractmethod
    def _init_params(self):
        """Initialize list of calculator parameter names."""
        pass

    @abstractmethod
    def _init_required_calculators(self):
        """Initialize list of other required calculators."""
        pass

    @abstractmethod
    def _calc_heat_capacity(self, T_a):
        """Returns heat capacity as a function of temperature."""
        pass

    @abstractmethod
    def _calc_energy(self, T_a):
        """Returns thermal energy as a function of temperature."""
        pass

    ####################
    # Optional Methods #
    ####################
    def _calc_entropy(self, T_a):
        raise NotImplemented('Entropy function not implemented for this calculator.')
        pass

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
class CompressedThermalEnergyCalc(with_metaclass(ABCMeta, core.Calculator)):
    """
    Abstract Equation of State class for a reference Thermal Energy Path

    Path can either be isothermal (T=const) or adiabatic (S=const)

    For this restricted path, thermodyn properties depend only on volume

    """

    def __init__(self, eos_mod):
        self._eos_mod = eos_mod
        self._init_params()
        self._required_calculators = None
        pass

    ####################
    # Required Methods #
    ####################
    @abstractmethod
    def _init_params(self):
        """Initialize list of calculator parameter names."""
        pass

    @abstractmethod
    def _init_required_calculators(self):
        """Initialize list of other required calculators."""
        pass

    @abstractmethod
    def _calc_heat_capacity(self, V_a, T_a):
        """Returns heat capacity as a function of temperature."""
        pass

    @abstractmethod
    def _calc_energy(self, V_a, T_a):
        """Returns thermal energy as a function of temperature."""
        pass
#====================================================================

#====================================================================
# Implementations
#====================================================================
class _Debye(ThermalEnergyCalc):
    """
    Implimentation copied from Burnman.

    """

    _path_opts=['V']

    def __init__(self, eos_mod, level_const=100):
        super(_Debye, self).__init__(eos_mod, path_const='V',
                                     level_const=level_const)

    def _init_required_calculators(self):
        """Initialize list of other required calculators."""

        self._required_calculators = None
        pass

    def _init_params(self, theta_param=None):
        """Initialize list of calculator parameter names."""

        natom = self.eos_mod.natom

        theta0 = 1000
        Cvmax = 3*natom*core.CONSTS['kboltz']

        self._param_names = ['theta0', 'Cvmax']
        self._param_units = ['K', 'eV/K']
        self._param_defaults = [theta0, Cvmax]
        self._param_scales = [theta0, Cvmax]
        pass

    def _calc_heat_capacity(self, T_a, theta0=None, Cvmax=None):
        """Returns heat capacity as a function of temperature."""

        theta0, Cvmax = self.eos_mod.get_param_values(
            param_names=['theta0','Cvmax'], overrides=[theta0, Cvmax])

        x_values = theta0/np.array(T_a)
        Cv_values = Cvmax*_debye.debye_heat_capacity_fun(x_values)
        return Cv_values

    def _calc_energy(self, T_a, theta0=None, Cvmax=None):
        """Returns heat capacity as a function of temperature."""

        theta0, Cvmax = self.eos_mod.get_param_values(
            param_names=['theta0','Cvmax'], overrides=[theta0, Cvmax])

        T_a = np.array(T_a)
        x_values = theta0/T_a
        energy = Cvmax*T_a*_debye.debye_energy_fun(x_values)
        return energy
#====================================================================
# class _CompressedDebye(CompressedThermalEnergyCalc):
#     """
#     Implimentation copied from Burnman.
#
#     """
#
#     def __init__(self, eos_mod, path_const='P', level_const=0.0):
#
#     def _init_required_calculators(self):
#         """Initialize list of other required calculators."""
#
#         self._eos_mod.
#
#         self._required_calculators = None
#
#         pass
#     def _init_params(self, theta_param=None):
#         """Initialize list of calculator parameter names."""
#
#         theta0, Cvmax = 100, 150
#         E0_scale = np.round(V0*KP0/core.CONSTS['PV_ratio'],decimals=2)
#         self._param_names = ['V0','K0','KP0','E0']
#         self._param_units = ['ang^3','GPa','1','eV']
#         self._param_defaults = [V0,K0,KP0,0]
#         self._param_scales = [V0,K0,KP0,E0_scale]
#
#         pass
#
#     def _calc_heat_capacity(self, T_a, Theta=None, Cvmax=None):
#         """Returns heat capacity as a function of temperature."""
#
#         if Theta is None:
#             Theta, = self.eos_mod.get_param_values(param_names=['Theta'])
#
#         if Cvmax is None:
#             Cvmax, = self.eos_mod.get_param_values(param_names=['Cvmax'])
#
#         x_values = Theta/np.array(T_a)
#         Cv_values = Cvmax*_debye.debye_heat_capacity_fun(x_values)
#         return Cv_values
#
#     def _calc_energy(self, T_a, Theta=None, Cvmax=None):
#         """Returns heat capacity as a function of temperature."""
#
#         if Theta is None:
#             Theta, = self.eos_mod.get_param_values(param_names=['Theta'])
#
#         if Cvmax is None:
#             Cvmax, = self.eos_mod.get_param_values(param_names=['Cvmax'])
#
#         x_values = Theta/np.array(T_a)
#         Cv_values = Cvmax*_debye.debye_heat_capacity_fun(x_values)
#         return Cv_values
#====================================================================
