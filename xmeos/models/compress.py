# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from future.utils import with_metaclass
import numpy as np
import scipy as sp
from abc import ABCMeta, abstractmethod
from scipy import integrate
import scipy.interpolate as interpolate

from . import core

__all__ = ['CompressEos','CompressCalc']

#====================================================================
# Models
#====================================================================
#====================================================================
class CompressEos(with_metaclass(ABCMeta, core.Eos)):
    """
    EOS model for reference compression path.

    Parameters
    ----------
    Path can either be isothermal (T=const) or adiabatic (S=const)

    For this restricted path, thermodyn properties depend only on volume

    """

    _path_opts = ['T','S']
    _kind_opts = ['Vinet','BirchMurn3','BirchMurn4','GenFiniteStrain','Tait']

    def __init__(self, kind='Vinet', natom=1, path_const='T', level_const=300,
                 model_state={}):
        self._pre_init(natom=natom)

        self._init_calculator(kind, path_const, level_const)

        self._post_init(model_state=model_state)

        pass

    @property
    def path_opts(self):
        return self._path_opts

    def __repr__(self):
        return ("CompressEos(kind={kind}, natom={natom}, "
                "path_const={path_const}, level_const={level_const}, "
                "model_state={model_state}, "
                ")"
                .format(kind=self._kind,
                        natom=repr(self.natom),
                        path_const=repr(self.path_const),
                        level_const=repr(self.level_const),
                        model_state=self.model_state
                        )
                )

    def _init_calculator(self, kind, path_const, level_const):
        assert kind in self._kind_opts, kind + ' is not a valid ' + \
            'CompressEos Calculator. You must select one of: ' + self._kind_opts

        assert path_const in self._path_opts, path_const + ' is not a valid ' + \
            'path const. You must select one of: ' + self._path_opts

        self._kind = kind
        self._path_const = path_const
        self._level_const = level_const

        if   kind=='Vinet':
            calc = _Vinet(self, path_const=path_const,
                          level_const=level_const)
        elif kind=='BirchMurn3':
            calc = _BirchMurn3(self, path_const=path_const,
                               level_const=level_const)
        elif kind=='BirchMurn4':
            calc = _BirchMurn4(self, path_const=path_const,
                               level_const=level_const)
        elif kind=='GenFiniteStrain':
            calc = _GenFiniteStrain(self, path_const=path_const,
                                    level_const=level_const)
        elif kind=='Tait':
            calc = _Tait(self, path_const=path_const,
                         level_const=level_const)
        else:
            raise NotImplementedError(kind+' is not a valid '+\
                                      'CompressEos Calculator.')

        self._add_calculator( calc, kind='compress' )
        pass

    @property
    def path_const(self):
        return self._path_const

    @property
    def level_const(self):
        return self._level_const

    def press(self, V_a, apply_expand_adj=True):
        press_a = self.calculators['compress']._calc_press(V_a)
        return press_a

    def energy( self, V_a, apply_expand_adj=True ):
        energy_a =  self.calculators['compress']._calc_energy(V_a)
        # if self.expand_adj and apply_expand_adj:
        #     ind_exp = self.get_ind_expand(V_a, eos_d)
        #     if apply_expand_adj and (ind_exp.size>0):
        #         energy_a[ind_exp] = self.expand_adj_mod._calc_energy( V_a[ind_exp], eos_d )

        return energy_a

    def bulk_mod( self, V_a, apply_expand_adj=True ):
        bulk_mod_a =  self.calculators['compress']._calc_bulk_mod(V_a)
        if self.expand_adj and apply_expand_adj:
            ind_exp = self.get_ind_expand(V_a)
            if apply_expand_adj and (ind_exp.size>0):
                bulk_mod_a[ind_exp] = self.expand_adj_mod._calc_bulk_mod(V_a[ind_exp])

        return bulk_mod_a

    def bulk_mod_deriv(  self,V_a, apply_expand_adj=True ):
        bulk_mod_deriv_a =  self.calculators['compress']._calc_bulk_mod_deriv(V_a)
        if self.expand_adj and apply_expand_adj:
            ind_exp = self.get_ind_expand(V_a)
            if apply_expand_adj and (ind_exp.size>0):
                bulk_mod_deriv_a[ind_exp] = self.expand_adj_mod_deriv._calc_bulk_mod_deriv(V_a[ind_exp])

        return bulk_mod_deriv_a

    def energy_perturb( self, V_a, apply_expand_adj=True ):
        # Eval positive press values
        Eperturb_pos_a, scale_a, paramkey_a  = self.calculators['compress']._calc_energy_perturb(V_a)

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
# class HeatedCompressEos(with_metaclass(ABCMeta, core.Eos)):
#====================================================================


#====================================================================
# Calculators
#====================================================================
class CompressCalc(with_metaclass(ABCMeta, core.Calculator)):
    """
    Abstract Equation of State class for a reference Compression Path

    Path can either be isothermal (T=const) or adiabatic (S=const)

    For this restricted path, thermodyn properties depend only on volume

    """

    _path_opts = ['T','S']
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

        self._path_const = path_const
        self._level_const = level_const
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

    @property
    def path_opts(self):
        return self._path_opts

    def get_ind_expand(self, V_a):
        V0 = core.get_params(['V0'])
        ind_exp = np.where( V_a > V0 )[0]
        return ind_exp

    @property
    def path_const(self):
        return self._path_const

    @property
    def level_const(self):
        return self._level_const


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

#====================================================================
# Implementations
#====================================================================
class _Vinet(CompressCalc):
    def get_param_scale_sub(self):
        """Return scale values for each parameter"""
        V0, K0, KP0 = core.get_params(['V0','K0','KP0'])
        PV_ratio, = core.get_consts(['PV_ratio'])

        paramkey_a = np.array(['V0','K0','KP0','E0'])
        scale_a = np.array([V0,K0,KP0,K0*V0/PV_ratio])

        return scale_a, paramkey_a

    def _calc_press(self, V_a):
        V0, K0, KP0 = self.eos_mod.get_param_values(
            param_names=['V0','K0','KP0'])

        eta = 3/2*(KP0-1)
        vratio_a = V_a/V0
        x_a = vratio_a**(1/3)

        press_a = 3*K0*(1-x_a)*x_a**(-2)*np.exp(eta*(1-x_a))

        return press_a

    def _calc_energy(self, V_a):
        V0, K0, KP0, E0 = self.eos_mod.get_param_values(
            param_names=['V0','K0','KP0','E0'])
        PV_ratio, = core.get_consts(['PV_ratio'])

        eta = 3/2*(KP0-1)
        vratio_a = V_a/V0
        x_a = vratio_a**(1/3)


        energy_a = E0 + 9*K0*V0/PV_ratio/eta**2*\
            (1 + (eta*(1-x_a)-1)*np.exp(eta*(1-x_a)))

        return energy_a

    def _calc_energy_perturb(self, V_a):
        """Returns Energy pertubation basis functions resulting from fractional changes to EOS params."""

        V0, K0, KP0, E0 = core.get_params(['V0','K0','KP0','E0'])
        PV_ratio, = core.get_consts(['PV_ratio'])

        eta = 3/2*(KP0-1)
        vratio_a = V_a/V0
        x = vratio_a**(1/3)

        scale_a, paramkey_a = self.get_param_scale_sub()

        # NOTE: CHECK UNITS (PV_RATIO) here
        dEdp_a = 1/PV_ratio*np.vstack\
            ([-3*K0*(eta**2*x*(x-1) + 3*eta*(x-1) - 3*np.exp(eta*(x-1)) + 3)\
              *np.exp(-eta*(x-1))/eta**2,
              -9*V0*(eta*(x-1) - np.exp(eta*(x-1)) + 1)*np.exp(-eta*(x-1))/eta**2,
              27*K0*V0*(2*eta*(x-1) + eta*(-x + (x-1)*(eta*(x-1) + 1) + 1)
                        -2*np.exp(eta*(x-1)) + 2)*np.exp(-eta*(x-1))/(2*eta**3),
              PV_ratio*np.ones(V_a.shape)])

        Eperturb_a = np.expand_dims(scale_a,1)*dEdp_a
        #Eperturb_a = np.expand_dims(scale_a)*dEdp_a

        return Eperturb_a, scale_a, paramkey_a

    def _init_params(self):
        """Initialize list of calculator parameter names."""

        V0, K0, KP0 = 100, 150, 4
        E0_scale = np.round(V0*KP0/core.CONSTS['PV_ratio'],decimals=2)
        self._param_names = ['V0','K0','KP0','E0']
        self._param_units = ['ang^3','GPa','1','eV']
        self._param_defaults = [V0,K0,KP0,0]
        self._param_scales = [V0,K0,KP0,E0_scale]

        pass

    def _init_required_calculators(self):
        """Initialize list of other required calculators."""
        self._required_calculators = None
        pass
#====================================================================
class _BirchMurn3(CompressCalc):
    def _calc_press(self, V_a):
        V0, K0, KP0 = self.eos_mod.get_param_values(param_names=['V0','K0','KP0'])

        vratio_a = V_a/V0

        press_a = 3/2*K0 * (vratio_a**(-7/3) - vratio_a**(-5/3)) * \
            (1 + 3/4*(KP0-4)*(vratio_a**(-2/3)-1))

        return press_a

    def _calc_energy(self, V_a):
        V0, K0, KP0, E0 = self.eos_mod.get_param_values(
            param_names=['V0','K0','KP0','E0'])
        PV_ratio = core.CONSTS['PV_ratio']

        vratio_a = V_a/V0

        fstrain_a = 1/2*(vratio_a**(-2/3) - 1)

        energy_a = E0 + 9/2*(V0*K0/PV_ratio)*\
            ( KP0*fstrain_a**3 + fstrain_a**2*(1-4*fstrain_a) )

        return energy_a

    def _init_params(self):
        """Initialize list of calculator parameter names."""

        V0, K0, KP0 = 100, 150, 4
        E0_scale = V0*KP0/core.CONSTS['PV_ratio']
        self._param_names = ['V0','K0','KP0','E0']
        self._param_units = ['ang^3','GPa','1','eV']
        self._param_defaults = [V0,K0,KP0,0]
        self._param_scales = [V0,K0,KP0,E0_scale]
        pass

    def _init_required_calculators(self):
        """Initialize list of other required calculators."""
        self._required_calculators = None
        pass
#====================================================================
class _BirchMurn4(CompressCalc):
    def get_param_scale_sub(self):
        """Return scale values for each parameter"""
        V0, K0, KP0, KP20 = core.get_params(['V0','K0','KP0','KP20'])
        PV_ratio = core.CONSTS['PV_ratio']

        paramkey_a = np.array(['V0','K0','KP0','KP20','E0'])
        scale_a = np.array([V0,K0,KP0,KP0/K0,K0*V0/PV_ratio])

        return scale_a, paramkey_a

    def _calc_strain_energy_coeffs(self, nexp, K0, KP0, KP20):
        a1 = 3/2*(KP0-nexp-2)
        a2 = 3/2*(K0*KP20 + KP0*(KP0-2*nexp-3)+3+4*nexp+11/9*nexp**2)
        return a1,a2

    def _calc_press(self, V_a):
        V0, K0, KP0, KP20 = self.eos_mod.get_param_values(
            param_names=['V0','K0','KP0','KP20'])
        nexp = +2

        vratio_a = V_a/V0
        fstrain_a = 1/nexp*(vratio_a**(-nexp/3) - 1)

        a1,a2 = self._calc_strain_energy_coeffs(nexp,K0,KP0,KP20)

        press_a = 3*K0*(1+a1*fstrain_a + a2*fstrain_a**2)*\
            fstrain_a*(nexp*fstrain_a+1)**((nexp+3)/nexp)
        return press_a

    def _calc_energy(self, V_a):
        V0, K0, KP0, KP20, E0 = self.eos_mod.get_param_values(
            param_names=['V0','K0','KP0','KP20','E0'])
        nexp = +2

        PV_ratio = core.CONSTS['PV_ratio']

        vratio_a = V_a/V0
        fstrain_a = 1/nexp*(vratio_a**(-nexp/3) - 1)

        a1,a2 = self._calc_strain_energy_coeffs(nexp,K0,KP0,KP20)


        energy_a = E0 + 9*(V0*K0/PV_ratio)*\
            ( 1/2*fstrain_a**2 + a1/3*fstrain_a**3 + a2/4*fstrain_a**4)

        return energy_a

    def _init_params(self):
        """Initialize list of calculator parameter names."""

        V0, K0, KP0 = 100, 150, 4
        KP20 = -KP0/KP0
        KP20_scale = np.abs(KP20)
        E0_scale = V0*KP0/core.CONSTS['PV_ratio']
        self._param_names = ['V0','K0','KP0','KP20','E0']
        self._param_units = ['ang^3','GPa','1','GPa^-1','eV']
        self._param_defaults = [V0,K0,KP0,KP20,0]
        self._param_scales = [V0,K0,KP0,KP20_scale,E0_scale]
        pass

    def _init_required_calculators(self):
        """Initialize list of other required calculators."""
        self._required_calculators = None
        pass
#====================================================================
class _GenFiniteStrain(CompressCalc):
    """
    Generalized Finite Strain EOS from Jeanloz1989b

    Note: nexp=2 yields Birch Murnaghan (eulerian strain) EOS
          nexp=-2 yields lagragian strain EOS
    """

    def _calc_strain_energy_coeffs(self, nexp, K0, KP0, KP20=None, KP30=None):
        a1 = 3/2*(KP0-nexp-2)
        if KP20 is None:
            return a1
        else:
            a2 = 3/2*(K0*KP20 + KP0*(KP0-2*nexp-3)+3+4*nexp+11/9*nexp**2)
            if KP30 is None:
                return a1,a2
            else:
                a3 = 1/8*(9*K0**2*KP30 + 6*(6*KP0-5*nexp-6)*K0*KP20
                           +((3*KP0-5*nexp-6)**2 +10*nexp**2 + 30*nexp + 18)*KP0
                           -(50/3*nexp**3 + 70*nexp**2 + 90*nexp + 36))
                return a1,a2,a3

    def _calc_press(self, V_a):
        V0, K0, KP0, KP20, nexp = self.eos_mod.get_param_values(
            param_names=['V0','K0','KP0','KP20','nexp'])

        vratio_a = V_a/V0
        fstrain_a = 1/nexp*(vratio_a**(-nexp/3) - 1)

        a1,a2 = self._calc_strain_energy_coeffs(nexp,K0,KP0,KP20=KP20)

        press_a = 3*K0*(1+a1*fstrain_a + a2*fstrain_a**2)*\
            fstrain_a*(nexp*fstrain_a+1)**((nexp+3)/nexp)
        return press_a

    def _calc_energy(self, V_a):
        V0, K0, KP0, KP20, E0, nexp = self.eos_mod.get_param_values(
            param_names=['V0','K0','KP0','KP20','E0','nexp'])

        PV_ratio = core.CONSTS['PV_ratio']

        vratio_a = V_a/V0
        fstrain_a = 1/nexp*(vratio_a**(-nexp/3) - 1)

        a1,a2 = self._calc_strain_energy_coeffs(nexp,K0,KP0,KP20=KP20)


        energy_a = E0 + 9*(V0*K0/PV_ratio)*\
            ( 1/2*fstrain_a**2 + a1/3*fstrain_a**3 + a2/4*fstrain_a**4)

        return energy_a

    def _init_params(self):
        """Initialize list of calculator parameter names."""

        V0, K0, KP0, nexp = 100, 150, 4, 2
        KP20 = -KP0/KP0
        KP20_scale = np.abs(KP20)
        E0_scale = V0*KP0/core.CONSTS['PV_ratio']
        self._param_names = ['V0','K0','KP0','KP20','E0','nexp']
        self._param_units = ['ang^3','GPa','1','GPa^-1','eV','1']
        self._param_defaults = [V0,K0,KP0,KP20,0,nexp]
        self._param_scales = [V0,K0,KP0,KP20_scale,E0_scale,nexp]
        pass

    def _init_required_calculators(self):
        """Initialize list of other required calculators."""
        self._required_calculators = None
        pass
#====================================================================
class _Tait(CompressCalc):
    # def __init__( self, setlogPmin=False,
    #              path_const='T', level_const=300, expand_adj_mod=None,
    #              expand_adj=None, supress_energy=False, supress_press=False ):
    #     super(Tait, self).__init__( expand_adj=None )
    #     self.setlogPmin = setlogPmin
    #     pass
    # def __init__( self, setlogPmin=False, expand_adj=False ):
    #     self.setlogPmin = setlogPmin
    #     self.expand_adj = expand_adj
    #     pass

    def _get_eos_params(self):
        V0, K0, KP0 = self.eos_mod.get_param_values(
            param_names=['V0','K0','KP0'])
        if self.setlogPmin:
            logPmin, = self.eos_mod.get_param_values(
                param_names=['logPmin'])
            Pmin = np.exp(logPmin)
            # assert Pmin>0, 'Pmin must be positive.'
            KP20 = (KP0+1)*(KP0/K0 - 1/Pmin)
        else:
            KP20, = self.eos_mod.get_param_values(
                param_names=['KP20'])

        return V0,K0,KP0,KP20

    def get_param_scale_sub(self):
        """Return scale values for each parameter"""

        V0, K0, KP0, KP20 = self.eos_mod.get_param_values(
            param_names=['V0','K0','KP0','KP20'])

        PV_ratio = core.CONSTS['PV_ratio']

        if self.setlogPmin:
            # [V0,K0,KP0,E0]
            paramkey_a = np.array(['V0','K0','KP0','E0'])
            scale_a = np.array([V0,K0,KP0,K0*V0/PV_ratio])
        else:
            # [V0,K0,KP0,KP20,E0]
            paramkey_a = np.array(['V0','K0','KP0','KP20','E0'])
            scale_a = np.array([V0,K0,KP0,KP0/K0,K0*V0/PV_ratio])


        return scale_a, paramkey_a

    def _eos_to_abc_params(self, K0, KP0, KP20):
        a = (KP0 + 1)/(K0*KP20 + KP0 + 1)
        b = -KP20/(KP0+1) + KP0/K0
        c = (K0*KP20 + KP0 + 1)/(-K0*KP20 + KP0**2 + KP0)

        return a,b,c

    def _calc_press(self, V_a):
        V0, K0, KP0, KP20 = self.eos_mod.get_param_values(
            param_names=['V0','K0','KP0','KP20'])
        a,b,c = self._eos_to_abc_params(K0,KP0,KP20)
        vratio_a = V_a/V0

        press_a = 1/b*(((vratio_a + a - 1)/a)**(-1/c) - 1)

        return press_a

    def _calc_energy(self, V_a):
        V0, K0, KP0, KP20, E0 = self.eos_mod.get_param_values(
            param_names=['V0','K0','KP0','KP20','E0'])
        a,b,c = self._eos_to_abc_params(K0,KP0,KP20)

        PV_ratio = core.CONSTS['PV_ratio']

        vratio_a = V_a/V0

        press_a = self._calc_press(V_a)
        eta_a = b*press_a + 1
        eta_pow_a = eta_a**(-c)
        #  NOTE: Need to simplify energy expression here
        energy_a = E0 + (V0/b)/PV_ratio*(a*c/(c-1)-1)\
            - (V0/b)/PV_ratio*( a*c/(c-1)*eta_a*eta_pow_a - a*eta_pow_a + a - 1)

        return energy_a

    def _calc_energy_perturb_deprecate(self, V_a):
        """
        Returns Energy pertubation basis functions resulting from
        fractional changes to EOS params.

        """
        V0, K0, KP0, KP20 = self._get_eos_params()
        E0, = core.get_params(['E0'])

        a,b,c = self._eos_to_abc_params(K0,KP0,KP20)

        PV_ratio = core.CONSTS['PV_ratio']

        vratio_a = V_a/V0

        press_a = self._calc_press(V_a)
        eta_a = b*press_a + 1
        eta_pow_a = eta_a**(-c)

        scale_a, paramkey_a = self.get_param_scale_sub()

        # [V0,K0,KP0,KP20,E0]
        dEdp_a = np.ones((4, V_a.size))
        # dEdp_a[0,:] = 1/(PV_ratio*b*(c-1))*eta_a*(-a*eta_pow_a -1 + (1-a)*(a+c))
        dEdp_a[0,:] = 1/(PV_ratio*b*(c-1))*eta_a*(-a*eta_pow_a +a -1 -a*c+c) \
            + 1/(PV_ratio*b)*(a*c/(c-1)-1)
        dEdp_a[-1,:] = 1

        # from IPython import embed; embed(); import ipdb; ipdb.set_trace()
        # 1x3
        dEdabc_a = np.vstack\
            ([V0*eta_a/(a*b*(c-1))*(-a*eta_pow_a + a*(1-c))+c*V0/(b*(c-1)),
              V0/(b**2*(c-1))*((-a*eta_pow_a+a-1)*(c-1) + c*a*eta_a*eta_pow_a) \
              - V0/b**2*(a*c/(c-1) - 1),
              -a*V0/(b*(c-1)**2)*eta_a*eta_pow_a*(-c+(c-1)*(1-np.log(eta_a)))\
              +a*V0/(b*(c-1))*(1-c/(c-1))])
        # 3x3
        abc_jac = np.array([[-KP20*(KP0+1)/(K0*KP20+KP0+1)**2,
                             K0*KP20/(K0*KP20+KP0+1)**2,
                             -K0*(KP0+1)/(K0*KP20+KP0+1)**2],
                            [-KP0/K0**2, KP20/(KP0+1)**2 + 1/K0, -1/(KP0+1)],
                            [KP20*(KP0**2+2*KP0+1)/(-K0*KP20+KP0**2+KP0)**2,
                             (-K0*KP20+KP0**2+KP0-(2*KP0+1)*(K0*KP20+KP0+1))/\
                             (-K0*KP20+KP0**2+KP0)**2,
                             K0*(KP0**2+2*KP0+1)/(-K0*KP20+KP0**2+KP0)**2]])

        dEdp_a[1:4,:] = 1/PV_ratio*np.dot(abc_jac.T,dEdabc_a)

        print(dEdp_a.shape)

        if self.setlogPmin:
            # [V0,K0,KP0,E0]
            print(dEdp_a.shape)
            dEdp_a = dEdp_a[[0,1,2,4],:]


        Eperturb_a = np.expand_dims(scale_a,1)*dEdp_a


        #Eperturb_a = np.expand_dims(scale_a)*dEdp_a

        return Eperturb_a, scale_a, paramkey_a

    def _init_params(self):
        """Initialize list of calculator parameter names."""

        V0, K0, KP0 = 100, 150, 4
        KP20 = -KP0/K0
        E0_scale = V0*KP0/core.CONSTS['PV_ratio']
        KP20_scale = np.abs(KP20)
        self._param_names = ['V0','K0','KP0','KP20','E0']
        self._param_units = ['ang^3','GPa','1','GPa^-1','eV']
        self._param_defaults = [V0,K0,KP0,KP20,0]
        self._param_scales = [V0,K0,KP0,KP20_scale,E0_scale]
        pass

    def _init_required_calculators(self):
        """Initialize list of other required calculators."""
        self._required_calculators = None
        pass
#====================================================================
