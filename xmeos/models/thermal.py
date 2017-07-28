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

    def __init__(self, kind='Debye', level_const=None):
        self._pre_init()

        self._init_calculator(kind, level_const)

        self._post_init()
        pass

    def _init_calculator(self, kind, level_const):
        assert kind in self._kind_opts, kind + ' is not a valid ' + \
            'ThermalEnergyEos Calculator. You must select one of: ' + self._kind_opts


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
        self._add_calculator( calc, kind='compress' )

        self._kind = kind
        self._path_const = path_const
        self._level_const = level_const
    _path_opts = ['V','P']


        pass

    @property
    def path_const(self):
        return self._path_const

    @property
    def level_const(self):
        return self._level_const

    def press( self, V_a, apply_expand_adj=True):
        press_a = self.compress_calculator._calc_press(V_a)
        # if self.expand_adj and apply_expand_adj:
        #     ind_exp = self.get_ind_expand(V_a, eos_d)
        #     if (ind_exp.size>0):
        #         press_a[ind_exp] = self.expand_adj_mod._calc_press( V_a[ind_exp], eos_d )

        return press_a

    def energy( self, V_a, apply_expand_adj=True ):
        energy_a =  self.compress_calculator._calc_energy(V_a)
        # if self.expand_adj and apply_expand_adj:
        #     ind_exp = self.get_ind_expand(V_a, eos_d)
        #     if apply_expand_adj and (ind_exp.size>0):
        #         energy_a[ind_exp] = self.expand_adj_mod._calc_energy( V_a[ind_exp], eos_d )

        return energy_a

    def bulk_mod( self, V_a, apply_expand_adj=True ):
        bulk_mod_a =  self.compress_calculator._calc_bulk_mod(V_a)
        if self.expand_adj and apply_expand_adj:
            ind_exp = self.get_ind_expand(V_a)
            if apply_expand_adj and (ind_exp.size>0):
                bulk_mod_a[ind_exp] = self.expand_adj_mod._calc_bulk_mod(V_a[ind_exp])

        return bulk_mod_a

    def bulk_mod_deriv(  self,V_a, apply_expand_adj=True ):
        bulk_mod_deriv_a =  self.compress_calculator._calc_bulk_mod_deriv(V_a)
        if self.expand_adj and apply_expand_adj:
            ind_exp = self.get_ind_expand(V_a)
            if apply_expand_adj and (ind_exp.size>0):
                bulk_mod_deriv_a[ind_exp] = self.expand_adj_mod_deriv._calc_bulk_mod_deriv(V_a[ind_exp])

        return bulk_mod_deriv_a

    def energy_perturb( self, V_a, apply_expand_adj=True ):
        # Eval positive press values
        Eperturb_pos_a, scale_a, paramkey_a  = self.compress_calculator._calc_energy_perturb(V_a)

        if (self.expand_adj==False) or (apply_expand_adj==False):
            return Eperturb_pos_a, scale_a, paramkey_a
        else:
            Nparam_pos = Eperturb_pos_a.shape[0]

            scale_a, paramkey_a, ind_pos = \
                self.get_param_scale(apply_expand_adj=True,
                                     output_ind=True)

            Eperturb_a = np.zeros((paramkey_a.size, V_a.size))
            Eperturb_a[ind_pos,:] = Eperturb_pos_a

            # Overwrite negative pressure Expansion regions
            ind_exp = self.get_ind_expand(V_a)
            if ind_exp.size>0:
                Eperturb_adj_a = \
                    self.expand_adj_mod._calc_energy_perturb(V_a[ind_exp])[0]
                Eperturb_a[:,ind_exp] = Eperturb_adj_a

            return Eperturb_a, scale_a, paramkey_a

    #   Standard methods must be overridden (as needed) by implimentation model

    def get_param_scale_sub(self):
        raise NotImplementedError("'get_param_scale_sub' function not implimented for this model")

    ####################
    # Required Methods #
    ####################


    ####################
    # Optional Methods #
    ####################
    def _calc_energy_perturb(self, V_a):
        """Returns Energy pertubation basis functions resulting from fractional changes to EOS params."""

        fname = 'energy'

        scale_a, paramkey_a = self.get_param_scale(
            apply_expand_adj=self.expand_adj)
        Eperturb_a = []
        for paramname in paramkey_a:
            iEperturb_a = self.param_deriv(fname, paramname, V_a)
            Eperturb_a.append(iEperturb_a)

        Eperturb_a = np.array(Eperturb_a)

        return Eperturb_a, scale_a, paramkey_a
#====================================================================
# class MieGruneisenEos(with_metaclass(ABCMeta, core.Eos)):

# class CompressedThermalEnergyEos(with_metaclass(ABCMeta, core.Eos)):



#====================================================================
# Calculators
#====================================================================
class CompressCalc(with_metaclass(ABCMeta, core.Calculator)):
    """
    Abstract Equation of State class for a reference Compression Path

    Path can either be isothermal (T=const) or adiabatic (S=const)

    For this restricted path, thermodyn properties depend only on volume

    """

    path_opts = ['T','S']
    supress_energy = False
    supress_press = False

    def __init__( self, eos_mod, path_const='T', level_const=300,
                 expand_adj_mod=None, expand_adj=None,
                 supress_energy=False, supress_press=False ):
        assert path_const in self.path_opts, path_const + ' is not a valid ' + \
            'path const. You must select one of: ' + path_opts

        self._eos_mod = eos_mod
        self._init_params()
        self._required_calculators = None

        self.path_const = path_const
        self.level_const = level_const
        self.supress_energy = supress_energy
        self.supress_press = supress_press

        # Use Expansion Adjustment for negative pressure region?
        if expand_adj is None:
            self.expand_adj = False
        else:
            self.expand_adj = expand_adj

        if expand_adj_mod is None:
            self.expand_adj = False
            self.expand_adj_mod = None
        else:
            self.expand_adj = True
            self.expand_adj_mod = expand_adj_mod
        pass

    def get_ind_expand(self, V_a):
        V0 = core.get_params(['V0'])
        ind_exp = np.where( V_a > V0 )[0]
        return ind_exp

    def get_path_const( self ):
        return self.path_const

    def get_level_const( self ):
        return self.level_const


    # NEED to write infer volume function
    #   Standard methods must be overridden (as needed) by implimentation model

    def get_param_scale_sub(self):
        raise NotImplementedError("'get_param_scale_sub' function not implimented for this model")

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
    def _calc_press(self, V_a):
        """Returns Press variation along compression curve."""
        pass

    @abstractmethod
    def _calc_energy(self, V_a):
        """Returns Energy along compression curve."""
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

    def _calc_bulk_mod(self, V_a):
        """Returns Bulk Modulus variation along compression curve."""
        raise NotImplementedError("'bulk_mod' function not implimented for this model")

    def _calc_bulk_mod_deriv(self, V_a):
        """Returns Bulk Modulus Deriv (K') variation along compression curve."""
        raise NotImplementedError("'bulk_mod_deriv' function not implimented for this model")
#====================================================================







