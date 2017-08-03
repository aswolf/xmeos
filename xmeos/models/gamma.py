# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from future.utils import with_metaclass
import numpy as np
import scipy as sp
from abc import ABCMeta, abstractmethod
from scipy import integrate
import scipy.interpolate as interpolate

from . import core

__all__ = ['GammaEos','GammaCalc']

#====================================================================
# Base Class
#====================================================================
class GammaEos(with_metaclass(ABCMeta, core.Eos)):
    """
    EOS model for compression dependence of Grüneisen parameter.

    Parameters
    ----------
    Thermodyn properties depend only on volume

    """

    _kind_opts = ['GammaPowLaw','GammaShiftPowLaw','GammaFiniteStrain']

    def __init__(self, kind='GammaPowLaw', natom=1, level_const=0,
                 model_state={}):
        self._pre_init(natom=natom)

        self._init_calculator(kind, level_const)

        self._post_init(model_state=model_state)

        pass

    def __repr__(self):
        return ("GammaEos(kind={kind}, natom={natom}, "
                "model_state={model_state}, "
                ")"
                .format(kind=self._kind,
                        natom=repr(self.natom),
                        model_state=self.model_state
                        )
                )

    def _init_calculator(self, kind, level_const):
        assert kind in self._kind_opts, kind + ' is not a valid ' + \
            'CompressEos Calculator. You must select one of: ' + self._kind_opts

        self._kind = kind

        if   kind=='GammaPowLaw':
            calc = _GammaPowLaw(self)
        elif kind=='GammaShiftPowLaw':
            calc = _GammaShiftPowLaw(self)
        elif kind=='GammaFiniteStrain':
            calc = _GammaFiniteStrain(self)
        else:
            raise NotImplementedError(kind+' is not a valid '+
                                      'GammaEos Calculator.')

        self._add_calculator(calc, kind='gamma')
        self._level_const = level_const
        pass

    @property
    def level_const(self):
        return self._level_const

    def gamma(self, V_a):
        gamma_a = self.calculators['gamma']._calc_gamma(V_a)
        return gamma_a

    def gamma_deriv(self, V_a):
        gamma_deriv_a = self.calculators['gamma']._calc_gamma_deriv(V_a)
        return gamma_deriv_a

    def temp(self, V_a, V0=None, T0=None):
        temp_a = self.calculators['gamma']._calc_temp(V_a, V0=V0, T0=T0)
        return temp_a
#====================================================================
class GammaCalc(with_metaclass(ABCMeta, core.Calculator)):
    """
    Abstract Equation of State class for a reference Compression Path

    Path can either be isothermal (T=const) or adiabatic (S=const)

    For this restricted path, thermodyn properties depend only on volume

    """

    def __init__(self, eos_mod, level_const=None):
        self._eos_mod = eos_mod
        self._init_params()
        self._required_calculators = None

        self._path_const = 'S'
        self._level_const = level_const
        pass

    @property
    def path_const( self ):
        return self._path_const

    @property
    def level_const( self ):
        return self._level_const

    ####################
    # Required Methods #
    ####################
    @abstractmethod
    def _init_params( self ):
        """Initialize list of calculator parameter names."""
        pass

    @abstractmethod
    def _init_required_calculators( self ):
        """Initialize list of other required calculators."""
        pass

    @abstractmethod
    def _calc_gamma(self, V_a):
        pass

    @abstractmethod
    def _calc_gamma_deriv(self, V_a):
        pass

    @abstractmethod
    def _calc_temp(self, V_a, V0=None, T0=None):
        pass

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

    def __init__(self, eos_mod, level_const=0):
        super(_GammaPowLaw, self).__init__(eos_mod, level_const=level_const)
        pass

    def _init_required_calculators(self):
        """Initialize list of other required calculators."""

        self._required_calculators = None
        pass

    def _init_params(self):
        """Initialize list of calculator parameter names."""

        V0 = 100
        gamma0 = 1.0
        q = 1.0
        T0 = 300

        self._param_names = ['V0', 'gamma0', 'q', 'T0']
        self._param_units = ['ang^3', '1', '1', 'K']
        self._param_defaults = [V0, gamma0, q, T0]
        self._param_scales = [V0, gamma0, q, T0]
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

    def _calc_temp(self, V_a, V0=None, T0=None):
        V0, T0 = self.eos_mod.get_param_values(
            param_names=['V0','T0'], overrides=[V0,T0])
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

    def __init__(self, eos_mod, level_const=0):
        super(_GammaShiftPowLaw, self).__init__(eos_mod, level_const=level_const)
        pass

    def _init_required_calculators(self):
        """Initialize list of other required calculators."""

        self._required_calculators = None
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

    def _calc_temp(self, V_a, V0=None, T0=None):
        V0, T0 = self.eos_mod.get_param_values(
            param_names=['V0','T0'], overrides=[V0,T0])
        gamma0, gamma_inf, beta = self.eos_mod.get_param_values(
            param_names=['gamma0','gamma_inf','beta'])

        gamma_a = self._calc_gamma(V_a)
        x = V_a/V0
        T_a = T0*x**(-gamma_inf)*np.exp((gamma0-gamma_inf)/beta*(1-x**beta))

        return T_a
#====================================================================
class _GammaFiniteStrain(GammaCalc):
    _path_opts=['S']

    def __init__(self, eos_mod, level_const=0):
        super(_GammaFiniteStrain, self).__init__(eos_mod, level_const=level_const)
        pass

    def _init_required_calculators(self):
        """Initialize list of other required calculators."""

        self._required_calculators = None
        pass

    def _init_params(self):
        """Initialize list of calculator parameter names."""

        V0 = 100
        gamma0 = 1.0
        gammap0 = 1.0
        T0 = 300

        self._param_names = ['V0', 'gamma0', 'gammap0', 'T0']
        self._param_units = ['ang^3', '1', '1', 'K']
        self._param_defaults = [V0, gamma0, gammap0, T0]
        self._param_scales = [V0, gamma0, gammap0, T0]
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

    def _calc_temp(self, V_a, V0=None, T0=None):
        V0, T0 = self.eos_mod.get_param_values(
            param_names=['V0','T0'], overrides=[V0,T0])

        a1, a2 = self._calc_strain_coefs()
        fstr_a = self._calc_fstrain(V_a)
        T_a = T0*np.sqrt(1 + a1*fstr_a + 0.5*a2*fstr_a**2)

        return T_a
#====================================================================

#
#        TOL = 1e-8
#        Nsamp = 81
#        # Nsamp = 281
#        # Nsamp = 581
#
#        if self.V0ref:
#            VR, = core.get_params( ['V0'], eos_d )
#        else:
#            VR, = core.get_params( ['VR'], eos_d )
#
#
#        Vmin = np.min(V_a)
#        Vmax = np.max(V_a)
#
#        dVmax = np.log(Vmax/VR)
#        dVmin = np.log(Vmin/VR)
#
#        T_a = TR*np.ones(V_a.size)
#
#        if np.abs(dVmax) < TOL:
#            dVmax = 0.0
#        if np.abs(dVmin) < TOL:
#            dVmin = 0.0
#
#
#        if dVmax > TOL:
#            indhi_a = np.where(np.log(V_a/VR) > TOL)[0]
#            # indhi_a = np.where(V_a > VR)[0]
#
#            # ensure numerical stability by shifting
#            # if (Vmax-VR)<=TOL:
#            #     T_a[indhi_a] = TR
#            # else:
#            Vhi_a = np.linspace(VR,Vmax,Nsamp)
#            gammahi_a = self.gamma( Vhi_a, eos_d )
#            logThi_a = integrate.cumtrapz(-gammahi_a/Vhi_a,x=Vhi_a)
#            logThi_a = np.append([0],logThi_a)
#            logtemphi_f = interpolate.interp1d(Vhi_a,logThi_a,kind='cubic')
#            T_a[indhi_a] = TR*np.exp(logtemphi_f(V_a[indhi_a]))
#
#        if dVmin < -TOL:
#            indlo_a = np.where(np.log(V_a/VR) < -TOL)[0]
#            # indlo_a = np.where(V_a <= VR)[0]
#
#            # # ensure numerical stability by shifting
#            # if (VR-Vmin)<TOL:
#            #     T_a[indlo_a] = TR
#            # else:
#            Vlo_a = np.linspace(VR,Vmin,Nsamp)
#            gammalo_a = self.gamma( Vlo_a, eos_d )
#            logTlo_a = integrate.cumtrapz(-gammalo_a/Vlo_a,x=Vlo_a)
#            logTlo_a = np.append([0],logTlo_a)
#            logtemplo_f = interpolate.interp1d(Vlo_a,logTlo_a,kind='cubic')
#            T_a[indlo_a] = TR*np.exp(logtemplo_f(V_a[indlo_a]))
#
#        return T_a
