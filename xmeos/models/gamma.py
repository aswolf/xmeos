# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from future.utils import with_metaclass
import numpy as np
import scipy as sp
from abc import ABCMeta, abstractmethod
from scipy import integrate
import scipy.interpolate as interpolate

from . import core
from . import refstate

__all__ = ['GammaEos','GammaCalc']


#====================================================================
# Base Class
#====================================================================
def set_calculator(eos_mod, kind, kind_opts):
    assert kind in kind_opts, (
        kind + ' is not a valid thermal calculator. '+
        'You must select one of: ' +  str(kind_opts))

    eos_mod._kind = kind

    if   kind=='GammaPowLaw':
        calc = _GammaPowLaw(eos_mod)
    elif kind=='GammaShiftPowLaw':
        calc = _GammaShiftPowLaw(eos_mod)
    elif kind=='GammaFiniteStrain':
        calc = _GammaFiniteStrain(eos_mod)
    else:
        raise NotImplementedError(kind+' is not a valid '+
                                  'GammaEos Calculator.')

    eos_mod._add_calculator(calc, calc_type='gamma')
    pass
#====================================================================
class GammaEos(with_metaclass(ABCMeta, core.Eos)):
    """
    EOS model for compression dependence of Grüneisen parameter.

    Parameters
    ----------
    Thermodyn properties depend only on volume

    """

    _kind_opts = ['GammaPowLaw','GammaShiftPowLaw','GammaFiniteStrain']

    def __init__(self, kind='GammaPowLaw', natom=1, model_state={}):
        self._pre_init(natom=natom)

        set_calculator(self, kind, self._kind_opts)

        ref_compress_state='P0'
        ref_thermal_state='T0'
        ref_energy_type = 'E0'
        refstate.set_calculator(self, ref_compress_state=ref_compress_state,
                                ref_thermal_state=ref_thermal_state,
                                ref_energy_type=ref_energy_type)
        # self._set_ref_state()
        self._post_init(model_state=model_state)

        pass

    def __repr__(self):
        calc = self.calculators['gamma']
        return ("GammaEos(kind={kind}, natom={natom}, "
                "model_state={model_state}, "
                ")"
                .format(kind=repr(calc.name),
                        natom=repr(self.natom),
                        model_state=self.model_state
                        )
                )

    def _set_ref_state(self):
        calc = self.calculators['gamma']
        path_const = calc.path_const

        if path_const=='S':
            param_ref_names = []
            param_ref_units = []
            param_ref_defaults = []
            param_ref_scales = []
        else:
            raise NotImplementedError(
                'path_const '+path_const+' is not valid for ThermalEos.')

        self._path_const = calc.path_const
        self._param_ref_names = param_ref_names
        self._param_ref_units = param_ref_units
        self._param_ref_defaults = param_ref_defaults
        self._param_ref_scales = param_ref_scales
        pass

    def gamma(self, V_a):
        gamma_a = self.calculators['gamma']._calc_gamma(V_a)
        return gamma_a

    def gamma_deriv(self, V_a):
        gamma_deriv_a = self.calculators['gamma']._calc_gamma_deriv(V_a)
        return gamma_deriv_a

    def temp(self, V_a, T0=None):
        temp_a = self.calculators['gamma']._calc_temp(V_a, T0=T0)
        return temp_a
#====================================================================
class GammaCalc(with_metaclass(ABCMeta, core.Calculator)):
    """
    Abstract Equation of State class for a reference Compression Path

    Path can either be isothermal (T=const) or adiabatic (S=const)

    For this restricted path, thermodyn properties depend only on volume

    """

    def __init__(self, eos_mod):
        self._eos_mod = eos_mod
        self._init_params()

        self._path_const = 'S'
        pass

    @property
    def path_const( self ):
        return self._path_const

    ####################
    # Required Methods #
    ####################
    @abstractmethod
    def _init_params( self ):
        """Initialize list of calculator parameter names."""
        pass

    @abstractmethod
    def _calc_gamma(self, V_a):
        pass

    @abstractmethod
    def _calc_gamma_deriv(self, V_a):
        pass

    @abstractmethod
    def _calc_temp(self, V_a, T0=None):
        pass

    def _calc_theta(self, V_a):
        theta0 = self.eos_mod.get_param_values(param_names=['theta0'])
        theta = self._calc_temp(V_a, T0=theta0)
        return theta

    ####################
    # Optional Methods #
    ####################
    # EOS property functions
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



# Implementations
#====================================================================
class _GammaPowLaw(GammaCalc):
    _path_opts=['S']

    def __init__(self, eos_mod):
        super(_GammaPowLaw, self).__init__(eos_mod)
        pass

    def _init_params(self):
        """Initialize list of calculator parameter names."""

        V0 = 100
        gamma0 = 1.0
        q = 1.0

        self._param_names = ['V0', 'gamma0', 'q']
        self._param_units = ['ang^3', '1', '1']
        self._param_defaults = [V0, gamma0, q]
        self._param_scales = [V0, gamma0, q]
        pass

    def _calc_gamma(self, V_a):
        V0, gamma0, q = self.eos_mod.get_param_values(
            param_names=['V0','gamma0','q'])

        gamma_a = gamma0 *(V_a/V0)**q

        return gamma_a

    def _calc_gamma_deriv(self, V_a):
        q, = self.eos_mod.get_param_values(param_names=['q'])

        gamma_a = self._calc_gamma(V_a)
        gamma_deriv_a = q*gamma_a/V_a
        return gamma_deriv_a

    def _calc_temp(self, V_a, T0=None):
        if T0 is None:
            T0 = self.eos_mod.refstate.ref_temp()
        # T0, = self.eos_mod.get_param_values(param_names=['T0'], overrides=[T0])
        gamma0, q = self.eos_mod.get_param_values(
            param_names=['gamma0','q'])

        gamma_a = self._calc_gamma(V_a)
        T_a = T0*np.exp(-(gamma_a - gamma0)/q)

        return T_a
#====================================================================
class _GammaShiftPowLaw(GammaCalc):
    """
    Shifted Power Law description of Grüneisen Parameter (Al’tshuler, 1987)

    """
    _path_opts=['S']

    def __init__(self, eos_mod):
        super(_GammaShiftPowLaw, self).__init__(eos_mod)
        pass

    def _init_params(self):
        """Initialize list of calculator parameter names."""

        V0 = 100
        gamma0 = 1.5
        gamma_inf = 2/3
        beta = 1.4
        T0 = 300

        self._param_names = ['V0', 'gamma0', 'gamma_inf', 'beta', 'T0']
        self._param_units = ['ang^3', '1', '1', '1', 'K']
        self._param_defaults = [V0, gamma0, gamma_inf, beta, T0]
        self._param_scales = [V0, gamma0, gamma_inf, beta, T0]
        pass

    def _calc_gamma(self, V_a):
        V0, gamma0, gamma_inf, beta = self.eos_mod.get_param_values(
            param_names=['V0','gamma0','gamma_inf','beta'])

        gamma_a = gamma_inf + (gamma0-gamma_inf)*(V_a/V0)**beta
        return gamma_a

    def _calc_gamma_deriv(self, V_a):
        gamma_inf, beta = self.eos_mod.get_param_values(
            param_names=['gamma_inf','beta'])

        gamma_a = self._calc_gamma(V_a)
        gamma_deriv_a = beta/V_a*(gamma_a-gamma_inf)
        return gamma_deriv_a

    def _calc_temp(self, V_a, T0=None):
        T0, = self.eos_mod.get_param_values(param_names=['T0'], overrides=[T0])
        V0, gamma0, gamma_inf, beta = self.eos_mod.get_param_values(
            param_names=['V0','gamma0','gamma_inf','beta'])

        gamma_a = self._calc_gamma(V_a)
        x = V_a/V0
        T_a = T0*x**(-gamma_inf)*np.exp((gamma0-gamma_inf)/beta*(1-x**beta))

        return T_a
#====================================================================
class _GammaFiniteStrain(GammaCalc):
    _path_opts=['S']

    def __init__(self, eos_mod):
        super(_GammaFiniteStrain, self).__init__(eos_mod)
        pass

    def _init_params(self):
        """Initialize list of calculator parameter names."""

        V0 = 100
        gamma0 = 0.5
        gammap0 = -2

        self._param_names = ['V0', 'gamma0', 'gammap0']
        self._param_units = ['ang^3', '1', '1']
        self._param_defaults = [V0, gamma0, gammap0]
        self._param_scales = [V0, gamma0, gammap0]
        pass

    def _calc_strain_coefs(self):
        V0, gamma0, gammap0 = self.eos_mod.get_param_values(
            param_names=['V0','gamma0','gammap0'])

        a1 = 6*gamma0
        a2 = -12*gamma0 +36*gamma0**2 -18*gammap0
        return a1, a2

    def _calc_fstrain(self, V_a, deriv=False):
        V0, = self.eos_mod.get_param_values(param_names=['V0'])

        x = V_a/V0
        if deriv:
            return -1/(3*V0)*x**(-5/3)
        else:
            return 1/2*(x**(-2/3)-1)

        pass

    def _calc_gamma(self, V_a):
        a1, a2 = self._calc_strain_coefs()
        fstr_a = self._calc_fstrain(V_a)

        gamma_a = (2*fstr_a+1)*(a1+a2*fstr_a)/(6*(1+a1*fstr_a+0.5*a2*fstr_a**2))

        return gamma_a

    def _calc_gamma_deriv(self, V_a):
        a1, a2 = self._calc_strain_coefs()
        fstr_a = self._calc_fstrain(V_a)
        fstr_deriv = self._calc_fstrain(V_a, deriv=True)

        gamma_a = self._calc_gamma(V_a)
        gamma_deriv_a = gamma_a*fstr_deriv*(
            2/(2*fstr_a+1)+a2/(a1+a2*fstr_a)
            -(a1+a2*fstr_a)/(1+a1*fstr_a+.5*a2*fstr_a**2))

        return gamma_deriv_a

    def _calc_temp(self, V_a, T0=None):
        if T0 is None:
            T0 = self.eos_mod.refstate.ref_temp()

        a1, a2 = self._calc_strain_coefs()
        fstr_a = self._calc_fstrain(V_a)
        T_a = T0*np.sqrt(1 + a1*fstr_a + 0.5*a2*fstr_a**2)
        return T_a
#====================================================================
