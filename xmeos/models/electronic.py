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

__all__ = ['ElectronicEos','ElectronicCalc']
# 'CompressedThermalEos','CompressedThermalCalc']

#====================================================================
# Base Classes
#====================================================================
def set_calculator(eos_mod, kind, kind_opts, apply_correction=False):
    assert kind in kind_opts, (
        kind + ' is not a valid electronic calculator. '+
        'You must select one of: ' + str(kind_opts))

    if   kind=='None':
        calc = _None(eos_mod, apply_correction=apply_correction)
    elif kind=='CvPowLaw':
        calc = _CvPowLaw(eos_mod, apply_correction=apply_correction)
    else:
        raise NotImplementedError(kind+' is not a valid '+\
                                  'Electronic Calculator.')

    eos_mod._add_calculator(calc, calc_type='electronic')
    pass
#====================================================================
class ElectronicEos(with_metaclass(ABCMeta, core.Eos)):
    """
    EOS model for thermal-electronic contributions.

    Parameters
    ----------

    """

    _kind_opts = ['None','CvPowLaw']

    def __init__(self, kind='None', natom=1, apply_correction=True, model_state={}):
        self._pre_init(natom=natom)
        set_calculator(self, kind, self._kind_opts,
                       apply_correction=apply_correction)
        self._post_init(model_state=model_state)
        pass

    @property
    def apply_correction(self):
        calc = self.calculators['electronic']
        return calc.apply_correction

    @apply_correction.setter
    def apply_correction(self, apply_correction):
        calc = self.calculators['electronic']
        calc.apply_correction = apply_correction

    def __repr__(self):
        calc = self.calculators['electronic']
        return ("ElectronicEos(kind={kind}, natom={natom}, "
                "apply_correction={apply_correction}, "
                "model_state={model_state}, )".format(
                    kind=repr(calc.name),
                    natom=repr(self.natom),
                    apply_correction=repr(calc.apply_correction),
                    model_state=self.model_state
                    )
                )

    def heat_capacity(self, V_a, T_a):
        calculator = self.calculators['electronic']
        heat_capacity_a =  calculator._calc_heat_capacity(V_a, T_a)
        return heat_capacity_a

    def entropy(self, V_a, T_a):
        calculator = self.calculators['electronic']
        entropy_a =  calculator._calc_entropy(V_a, T_a)
        return entropy_a

    def energy(self, V_a, T_a):
        calculator = self.calculators['electronic']
        energy_a =  calculator._calc_energy(V_a, T_a)
        return energy_a

    def helmholtz_energy(self, V_a, T_a):
        calculator = self.calculators['electronic']
        helmholtz_energy_a =  calculator._calc_helmholtz_energy(V_a, T_a)
        return helmholtz_energy_a

    def press(self, V_a, T_a):
        calculator = self.calculators['electronic']
        energy_a =  calculator._calc_press(V_a, T_a)
        return energy_a
#====================================================================

#====================================================================
# Calculators
#====================================================================
class ElectronicCalc(with_metaclass(ABCMeta, core.Calculator)):
    """
    Abstract Equation of State class for a reference Electronic EOS

    """


    def __init__(self, eos_mod, apply_correction=True):
        # assert path_const in self.path_opts, path_const + ' is not a valid ' + \
        #     'path const. You must select one of: ' + path_opts

        self._eos_mod = eos_mod
        self._init_params()
        self._required_calculators = None
        self.apply_correction = apply_correction
        pass


    @property
    def ndof(self):
        return self._ndof

    @property
    def apply_correction(self):
        return self._apply_correction

    @apply_correction.setter
    def apply_correction(self, apply_correction):
        assert apply_correction in [True, False], 'apply_correction must be a boolean'
        self._apply_correction = apply_correction

    ####################
    # Required Methods #
    ####################
    @abstractmethod
    def _init_params(self):
        """Initialize list of calculator parameter names."""
        pass

    def _get_Cv_scale(self):
        # Cvlimfac, = self.eos_mod.get_param_values(param_names=['Cvlimfac'])
        natom = self.eos_mod.natom
        Cv_scale = natom*core.CONSTS['kboltz']
        return Cv_scale

    @abstractmethod
    def _do_calc_heat_capacity(self, V_a, T_a):
        pass

    @abstractmethod
    def _do_calc_entropy(self, V_a, T_a):
        pass

    @abstractmethod
    def _do_calc_dSdV_T(self, V_a, T_a):
        pass

    @abstractmethod
    def _do_calc_energy(self, V_a, T_a):
        pass

    @abstractmethod
    def _do_calc_helmholtz_energy(self, V_a, T_a):
        pass

    @abstractmethod
    def _do_calc_press(self, V_a, T_a):
        pass

    def _calc_heat_capacity(self, V_a, T_a):
        if self.apply_correction:
            return self._do_calc_heat_capacity(V_a, T_a)
        else:
            return 0

    def _calc_entropy(self, V_a, T_a):
        if self.apply_correction:
            return self._do_calc_entropy(V_a, T_a)
        else:
            return 0
        pass

    def _calc_dSdV_T(self, V_a, T_a):
        if self.apply_correction:
            return self._do_calc_dSdV_T(V_a, T_a)
        else:
            return 0
        pass

    def _calc_energy(self, V_a, T_a):
        if self.apply_correction:
            return self._do_calc_energy(V_a, T_a)
        else:
            return 0

    def _calc_helmholtz_energy(self, V_a, T_a):
        if self.apply_correction:
            return self._do_calc_helmholtz_energy(V_a, T_a)
        else:
            return 0

    def _calc_press(self, V_a, T_a):
        if self.apply_correction:
            return self._do_calc_press(V_a, T_a)
        else:
            return 0

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
class _CvPowLaw(ElectronicCalc):
    """
    Implimentation copied from Burnman.

    """

    def __init__(self, eos_mod, apply_correction=True):
        super(_CvPowLaw, self).__init__(
            eos_mod, apply_correction=apply_correction)
        pass

    def _init_params(self):
        """Initialize list of calculator parameter names."""

        V0 = 100.0
        CvelFac0, CvelFacExp = 3.0e-4, +0.6
        Tel0, TelExp = 3000.0, -0.3

        param_names = ['V0', 'CvelFac0', 'CvelFacExp', 'Tel0', 'TelExp']
        param_units = ['ang^3', '1', '1', 'K', '1']
        param_defaults = [V0, CvelFac0, CvelFacExp, Tel0, TelExp]
        param_scales = [V0, 1.0e-4, 1.0, 3000.0, 1.0]

        self._set_params(param_names, param_units,
                         param_defaults, param_scales)
        pass

    def _calc_CvFac(self, V_a, deriv=0):
        V_a = core.fill_array(V_a)
        V0, CvelFac0, CvelFacExp = self.eos_mod.get_param_values(
            param_names=['V0', 'CvelFac0', 'CvelFacExp'])

        vratio_a = V_a/V0
        Cv_scl = self._get_Cv_scale()
        CvFac = Cv_scl*CvelFac0*vratio_a**CvelFacExp

        if deriv==0:
            return CvFac

        elif deriv==1:
            CvFac_deriv = CvelFacExp*CvFac/V_a
            return CvFac_deriv

        else:
            assert False, 'That is not a valid deriv option'

        pass

    def _calc_Tel(self, V_a, deriv=0):
        V_a = core.fill_array(V_a)
        V0, Tel0, TelExp = self.eos_mod.get_param_values(
            param_names=['V0', 'Tel0', 'TelExp'])

        vratio_a = V_a/V0
        Tel = Tel0*vratio_a**TelExp

        if deriv==0:
            return Tel

        elif deriv==1:
            Tel_deriv = TelExp*Tel/V_a
            return Tel_deriv

        else:
            assert False, 'That is not a valid deriv option'

        pass

    def _apply_electron_threshold(self, V_a, T_a, vals):
        Tel = self._calc_Tel(V_a)
        vals[T_a <= Tel] = 0
        return vals

    def _do_calc_heat_capacity(self, V_a, T_a):
        """Returns electronic heat capacity."""

        V_a, T_a = core.fill_array(V_a, T_a)

        CvFac = self._calc_CvFac(V_a)
        Tel = self._calc_Tel(V_a)

        Cv_values = CvFac*(T_a-Tel)
        Cv_values = self._apply_electron_threshold(V_a, T_a, Cv_values)
        return Cv_values

    def _do_calc_energy(self, V_a, T_a):
        """Returns electronic energy."""

        V_a, T_a = core.fill_array(V_a, T_a)

        CvFac = self._calc_CvFac(V_a)
        Tel = self._calc_Tel(V_a)

        energy = 0.5*CvFac*(T_a-Tel)**2
        energy = self._apply_electron_threshold(V_a, T_a, energy)
        return energy

    def _do_calc_helmholtz_energy(self, V_a, T_a):
        """Returns electronic entropy."""

        V_a, T_a = core.fill_array(V_a, T_a)

        CvFac = self._calc_CvFac(V_a)
        Tel = self._calc_Tel(V_a)

        F1 = -CvFac*(.5*(T_a**2 - Tel**2))
        F2 = +CvFac*T_a*Tel*np.log(T_a/Tel)
        helmholtz_energy = F1 + F2

        helmholtz_energy = self._apply_electron_threshold( V_a, T_a, helmholtz_energy)

        return helmholtz_energy

    def _do_calc_entropy(self, V_a, T_a):
        """Returns electronic entropy."""

        V_a, T_a = core.fill_array(V_a, T_a)

        CvFac = self._calc_CvFac(V_a)
        Tel = self._calc_Tel(V_a)

        entropy = CvFac*(T_a - Tel - Tel*np.log(T_a/Tel))
        entropy = self._apply_electron_threshold(V_a, T_a, entropy)
        return entropy

    def _do_calc_dSdV_T(self, V_a, T_a):
        """Returns electronic entropy."""

        V_a, T_a = core.fill_array(V_a, T_a)

        CvFac = self._calc_CvFac(V_a)
        CvFac_deriv = self._calc_CvFac(V_a, deriv=1)

        Tel = self._calc_Tel(V_a)
        Tel_deriv = self._calc_Tel(V_a, deriv=1)

        dSdV_T = (CvFac_deriv*(T_a - Tel - Tel*np.log(T_a/Tel))
                  +CvFac*(0-Tel_deriv-Tel_deriv*np.log(T_a/Tel) +Tel_deriv))

        # -Tel*np.log(T_a/Tel) = -Tel*np.log(T_a)+Tel*np.log(Tel)
        # -Tel_deriv*np.log(T_a)+Tel_deriv*np.log(Tel)+Tel/Tel*Tel_deriv

        dSdV_T = self._apply_electron_threshold(V_a, T_a, dSdV_T)
        return dSdV_T

    def _do_calc_press(self, V_a, T_a):
        """Returns electronic press."""

        PV_ratio, = core.get_consts(['PV_ratio'])
        V_a, T_a = core.fill_array(V_a, T_a)

        CvFac = self._calc_CvFac(V_a)
        CvFac_deriv = self._calc_CvFac(V_a, deriv=1)

        Tel = self._calc_Tel(V_a)
        Tel_deriv = self._calc_Tel(V_a, deriv=1)

        # press = -PV_ratio*(
        #     + CvFac_deriv*(0.5*(T_a**2-Tel**2) - T_a*Tel*np.log(T_a/Tel))
        #     + CvFac*Tel_deriv*(T_a -Tel -T_a*np.log(T_a/Tel))
        #     )

        # P1 = -PV_ratio*( -CvFac_deriv*(.5*(T_a**2 - Tel**2)) + CvFac*Tel*Tel_deriv)
        # P2 = -PV_ratio*( CvFac_deriv*T_a*Tel*np.log(T_a/Tel) + CvFac*T_a*Tel_deriv*np.log(T_a/Tel) - CvFac*T_a*Tel_deriv)

        P1 = -PV_ratio*(
            -CvFac_deriv*(.5*(T_a**2 - Tel**2))
            +CvFac*Tel*Tel_deriv
            )
        P2 = -PV_ratio*(
            +CvFac_deriv*T_a*Tel*np.log(T_a/Tel)
            +CvFac*T_a*Tel_deriv*np.log(T_a/Tel)
            -CvFac*T_a*Tel_deriv
            )

        press = P1 + P2

        press = self._apply_electron_threshold(V_a, T_a, press)
        return press
#====================================================================
class _None(ElectronicCalc):
    """
    Implimentation copied from Burnman.

    """

    def __init__(self, eos_mod, apply_correction=True):
        super(_None, self).__init__(
            eos_mod, apply_correction=apply_correction)
        pass

    def _init_params(self):
        """Initialize list of calculator parameter names."""

        V0 = 100.0
        CvelFac0, CvelFacExp = 3.0e-4, +0.6
        Tel0, TelExp = 3000.0, -0.3

        param_names = []
        param_units = []
        param_defaults = []
        param_scales = []

        self._set_params(param_names, param_units,
                         param_defaults, param_scales)
        pass

    def _do_calc_heat_capacity(self, V_a, T_a):
        """Returns electronic heat capacity."""

        V_a, T_a = core.fill_array(V_a, T_a)
        return np.zeros(V_a.shape)

    def _do_calc_energy(self, V_a, T_a):
        """Returns electronic energy."""

        V_a, T_a = core.fill_array(V_a, T_a)
        return np.zeros(V_a.shape)

    def _do_calc_helmholtz_energy(self, V_a, T_a):
        """Returns electronic helmholtz energy."""

        V_a, T_a = core.fill_array(V_a, T_a)
        return np.zeros(V_a.shape)

    def _do_calc_entropy(self, V_a, T_a):
        """Returns electronic entropy."""

        V_a, T_a = core.fill_array(V_a, T_a)
        return np.zeros(V_a.shape)

    def _do_calc_dSdV_T(self, V_a, T_a):
        """Returns electronic entropy derivative dSdV_T."""

        V_a, T_a = core.fill_array(V_a, T_a)
        return np.zeros(V_a.shape)

    def _do_calc_press(self, V_a, T_a):
        """Returns electronic press."""

        V_a, T_a = core.fill_array(V_a, T_a)
        return np.zeros(V_a.shape)
#====================================================================
