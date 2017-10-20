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
def set_calculator(eos_mod, kind, kind_opts, path_const, order=3):
    assert kind in kind_opts, (
        kind + ' is not a valid thermal calculator. '+
        'You must select one of: ' +  kind_opts)

    if   kind=='Vinet':
        calc = _Vinet(eos_mod, path_const=path_const, order=order)
    elif kind=='BirchMurn3':
        calc = _BirchMurn3(eos_mod, path_const=path_const, order=order)
    elif kind=='BirchMurn4':
        calc = _BirchMurn4(eos_mod, path_const=path_const, order=order)
    elif kind=='GenFiniteStrain':
        calc = _GenFiniteStrain(eos_mod, path_const=path_const, order=order)
    elif kind=='Tait':
        calc = _Tait(eos_mod, path_const=path_const, order=order)
    elif kind=='PolyRho':
        calc = _PolyRho(eos_mod, path_const=path_const, order=order)
    else:
        raise NotImplementedError(kind+' is not a valid '+\
                                  'CompressEos Calculator.')

    eos_mod._add_calculator(calc, calc_type='compress')
    pass
#====================================================================
class CompressEos(with_metaclass(ABCMeta, core.Eos)):
    """
    EOS model for reference compression path.

    Parameters
    ----------
    Path can either be isothermal (T=const) or adiabatic (S=const)

    For this restricted path, thermodyn properties depend only on volume

    """

    _path_opts = ['T','S','0K']
    _kind_opts = ['Vinet','BirchMurn3','BirchMurn4','GenFiniteStrain',
                  'Tait','PolyRho']

    def __init__(self, kind='Vinet', natom=1, molar_mass=100, path_const='T',
                 order=3, model_state={}):

        self._pre_init(natom=natom, molar_mass=molar_mass)

        set_calculator(self, kind, self._kind_opts, path_const, order=order)
        self._set_ref_state()

        self._post_init(model_state=model_state)

        pass

    def __repr__(self):
        calc = self.calculators['compress']
        return ("CompressEos(kind={kind}, natom={natom}, "
                "molar_mass={molar_mass},"
                "path_const={path_const}, order={order}, "
                "model_state={model_state}, "
                ")"
                .format(kind=repr(calc.name),
                        natom=repr(self.natom),
                        molar_mass=repr(self.molar_mass),
                        path_const=repr(self.path_const),
                        order=repr(calc.order),
                        model_state=self.model_state
                        )
                )

    def _set_ref_state(self):
        calc = self.calculators['compress']
        path_const = calc.path_const

        energy_scale = calc.get_energy_scale()
        T0 = 300

        # Add needed extra parameters (depending on path_const)
        if path_const=='T':
            param_ref_names = ['T0','F0']
            param_ref_units = ['K','eV']
            param_ref_defaults = [T0, 0.0]
            param_ref_scales = [T0, energy_scale]

        elif path_const=='S':
            param_ref_names = ['T0','E0']
            param_ref_units = ['K','eV']
            param_ref_defaults = [T0, 0.0]
            param_ref_scales = [T0, energy_scale]

        elif path_const=='0K':
            param_ref_names = []
            param_ref_units = []
            param_ref_defaults = []
            param_ref_scales = []
            pass

        else:
            raise NotImplementedError(
                'path_const '+path_const+' is not valid for CompressEos.')

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

    def press(self, V_a, apply_expand_adj=True):
        press_a = self.calculators['compress']._calc_press(V_a)
        return press_a

    def energy( self, V_a, apply_expand_adj=True ):
        energy0 = 0.0
        try:
            energy0 = self.get_param_values(param_names=['F0'])
        except:
            pass
        try:
            energy0 = self.get_param_values(param_names=['E0'])
        except:
            pass

        energy_a =  energy0 + self.calculators['compress']._calc_energy(V_a)
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

#====================================================================
# Calculators
#====================================================================
class CompressCalc(with_metaclass(ABCMeta, core.Calculator)):
    """
    Abstract Equation of State class for a reference Compression Path

    Path can either be isothermal (T=const) or adiabatic (S=const)

    For this restricted path, thermodyn properties depend only on volume

    """

    _path_opts = ['T','S','0K']
    supress_energy = False
    supress_press = False

    def __init__( self, eos_mod, path_const='T', order=None,
                 expand_adj_mod=None, expand_adj=None,
                 supress_energy=False, supress_press=False ):
        assert path_const in self.path_opts, path_const + ' is not a valid ' + \
            'path const. You must select one of: ' + path_opts

        assert (np.isscalar(order))&(order>0)&(np.mod(order,1)==0), (
            'order must be a positive integer.')

        self._eos_mod = eos_mod
        self._init_params(order)
        self._required_calculators = None

        self._path_const = path_const
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
    def order(self):
        return self._order

    # NEED to write infer volume function
    #   Standard methods must be overridden (as needed) by implimentation model

    def get_energy_scale(self):
        V0, K0 = self.get_param_defaults(['V0','K0'])
        energy_scale = np.round(V0*K0/core.CONSTS['PV_ratio'],decimals=2)
        return energy_scale

    def get_param_scale_sub(self):
        raise NotImplementedError("'get_param_scale_sub' function not implimented for this model")

    ####################
    # Required Methods #
    ####################
    @abstractmethod
    def _init_params(self, order):
        """Initialize list of calculator parameter names."""
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
    _name='Vinet'

    def _init_params(self, order):
        """Initialize list of calculator parameter names."""

        order = 3 # ignore order input

        V0, K0, KP0 = 100, 150, 4
        param_names = ['V0','K0','KP0']
        param_units = ['ang^3','GPa','1']
        param_defaults = [V0,K0,KP0]
        param_scales = [V0,K0,KP0]

        self._set_params(param_names, param_units,
                         param_defaults, param_scales, order=order)

        pass

    def _calc_press(self, V_a):
        V0, K0, KP0 = self.eos_mod.get_param_values(
            param_names=['V0','K0','KP0'])

        eta = 3/2*(KP0-1)
        vratio_a = V_a/V0
        x_a = vratio_a**(1/3)

        press_a = 3*K0*(1-x_a)*x_a**(-2)*np.exp(eta*(1-x_a))

        return press_a

    def _calc_energy(self, V_a):
        V0, K0, KP0 = self.eos_mod.get_param_values(
            param_names=['V0','K0','KP0'])
        PV_ratio, = core.get_consts(['PV_ratio'])

        eta = 3/2*(KP0-1)
        vratio_a = V_a/V0
        x_a = vratio_a**(1/3)


        energy_a = 9*K0*V0/PV_ratio/eta**2*\
            (1 + (eta*(1-x_a)-1)*np.exp(eta*(1-x_a)))

        return energy_a

    # def get_param_scale_sub(self):
    #     """Return scale values for each parameter"""
    #     V0, K0, KP0 = core.get_params(['V0','K0','KP0'])
    #     PV_ratio, = core.get_consts(['PV_ratio'])

    #     paramkey_a = np.array(['V0','K0','KP0','E0'])
    #     scale_a = np.array([V0,K0,KP0,K0*V0/PV_ratio])

    #     return scale_a, paramkey_a

    # def _calc_energy_perturb(self, V_a):
    #     """Returns Energy pertubation basis functions resulting from fractional changes to EOS params."""

    #     V0, K0, KP0, E0 = core.get_params(['V0','K0','KP0','E0'])
    #     PV_ratio, = core.get_consts(['PV_ratio'])

    #     eta = 3/2*(KP0-1)
    #     vratio_a = V_a/V0
    #     x = vratio_a**(1/3)

    #     scale_a, paramkey_a = self.get_param_scale_sub()

    #     # NOTE: CHECK UNITS (PV_RATIO) here
    #     dEdp_a = 1/PV_ratio*np.vstack\
    #         ([-3*K0*(eta**2*x*(x-1) + 3*eta*(x-1) - 3*np.exp(eta*(x-1)) + 3)\
    #           *np.exp(-eta*(x-1))/eta**2,
    #           -9*V0*(eta*(x-1) - np.exp(eta*(x-1)) + 1)*np.exp(-eta*(x-1))/eta**2,
    #           27*K0*V0*(2*eta*(x-1) + eta*(-x + (x-1)*(eta*(x-1) + 1) + 1)
    #                     -2*np.exp(eta*(x-1)) + 2)*np.exp(-eta*(x-1))/(2*eta**3),
    #           PV_ratio*np.ones(V_a.shape)])

    #     Eperturb_a = np.expand_dims(scale_a,1)*dEdp_a
    #     #Eperturb_a = np.expand_dims(scale_a)*dEdp_a

    #     return Eperturb_a, scale_a, paramkey_a
#====================================================================
class _BirchMurn3(CompressCalc):
    def _calc_press(self, V_a):
        V0, K0, KP0 = self.eos_mod.get_param_values(param_names=['V0','K0','KP0'])

        vratio_a = V_a/V0

        press_a = 3/2*K0 * (vratio_a**(-7/3) - vratio_a**(-5/3)) * \
            (1 + 3/4*(KP0-4)*(vratio_a**(-2/3)-1))

        return press_a

    def _calc_energy(self, V_a):
        V0, K0, KP0 = self.eos_mod.get_param_values(
            param_names=['V0','K0','KP0'])
        PV_ratio = core.CONSTS['PV_ratio']

        vratio_a = V_a/V0

        fstrain_a = 1/2*(vratio_a**(-2/3) - 1)

        energy_a = 9/2*(V0*K0/PV_ratio)*\
            ( KP0*fstrain_a**3 + fstrain_a**2*(1-4*fstrain_a) )

        return energy_a

    def _init_params(self, order):
        """Initialize list of calculator parameter names."""

        order = 3 # ignore order input

        V0, K0, KP0 = 100, 150, 4
        param_names = ['V0','K0','KP0']
        param_units = ['ang^3','GPa','1']
        param_defaults = [V0,K0,KP0]
        param_scales = [V0,K0,KP0]

        self._set_params(param_names, param_units,
                         param_defaults, param_scales, order=order)

        pass
#====================================================================
class _BirchMurn4(CompressCalc):
    # def get_param_scale_sub(self):
    #     """Return scale values for each parameter"""
    #     V0, K0, KP0, KP20 = core.get_params(['V0','K0','KP0','KP20'])
    #     PV_ratio = core.CONSTS['PV_ratio']

    #     paramkey_a = np.array(['V0','K0','KP0','KP20','E0'])
    #     scale_a = np.array([V0,K0,KP0,KP0/K0,K0*V0/PV_ratio])

    #     return scale_a, paramkey_a

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
        V0, K0, KP0, KP20 = self.eos_mod.get_param_values(
            param_names=['V0','K0','KP0','KP20'])
        nexp = +2

        PV_ratio = core.CONSTS['PV_ratio']

        vratio_a = V_a/V0
        fstrain_a = 1/nexp*(vratio_a**(-nexp/3) - 1)

        a1,a2 = self._calc_strain_energy_coeffs(nexp,K0,KP0,KP20)


        energy_a = 9*(V0*K0/PV_ratio)*\
            ( 1/2*fstrain_a**2 + a1/3*fstrain_a**3 + a2/4*fstrain_a**4)

        return energy_a

    def _init_params(self, order):
        """Initialize list of calculator parameter names."""

        order = 4 # ignore order input

        V0, K0, KP0 = 100, 150, 4
        KP20 = -KP0/K0
        KP20_scale = np.abs(KP20)
        param_names = ['V0','K0','KP0','KP20']
        param_units = ['ang^3','GPa','1','GPa^-1']
        param_defaults = [V0,K0,KP0,KP20]
        param_scales = [V0,K0,KP0,KP20_scale]

        self._set_params(param_names, param_units,
                         param_defaults, param_scales, order=order)

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
        V0, K0, KP0, KP20, nexp = self.eos_mod.get_param_values(
            param_names=['V0','K0','KP0','KP20','nexp'])

        PV_ratio = core.CONSTS['PV_ratio']

        vratio_a = V_a/V0
        fstrain_a = 1/nexp*(vratio_a**(-nexp/3) - 1)

        a1,a2 = self._calc_strain_energy_coeffs(nexp,K0,KP0,KP20=KP20)


        energy_a = 9*(V0*K0/PV_ratio)*\
            ( 1/2*fstrain_a**2 + a1/3*fstrain_a**3 + a2/4*fstrain_a**4)

        return energy_a

    def _init_params(self, order):
        """Initialize list of calculator parameter names."""

        order = 4 #ignore input order
        V0, K0, KP0, nexp = 100, 150, 4, 2
        nexp_scale = 1
        KP20 = -KP0/K0
        KP20_scale = np.abs(KP20)
        param_names = ['V0','K0','KP0','KP20','nexp']
        param_units = ['ang^3','GPa','1','GPa^-1','1']
        param_defaults = [V0,K0,KP0,KP20,nexp]
        param_scales = [V0,K0,KP0,KP20_scale,nexp_scale]

        self._set_params(param_names, param_units,
                         param_defaults, param_scales, order=order)
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

    # def get_param_scale_sub(self):
    #     """Return scale values for each parameter"""

    #     V0, K0, KP0, KP20 = self.eos_mod.get_param_values(
    #         param_names=['V0','K0','KP0','KP20'])

    #     PV_ratio = core.CONSTS['PV_ratio']

    #     if self.setlogPmin:
    #         # [V0,K0,KP0,E0]
    #         paramkey_a = np.array(['V0','K0','KP0','E0'])
    #         scale_a = np.array([V0,K0,KP0,K0*V0/PV_ratio])
    #     else:
    #         # [V0,K0,KP0,KP20,E0]
    #         paramkey_a = np.array(['V0','K0','KP0','KP20','E0'])
    #         scale_a = np.array([V0,K0,KP0,KP0/K0,K0*V0/PV_ratio])


    #     return scale_a, paramkey_a

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

        # from IPython import embed; import pdb; embed(); pdb.set_trace()

        return press_a

    def _calc_energy(self, V_a):
        V0, K0, KP0, KP20 = self.eos_mod.get_param_values(
            param_names=['V0','K0','KP0','KP20'])
        a,b,c = self._eos_to_abc_params(K0,KP0,KP20)

        PV_ratio = core.CONSTS['PV_ratio']

        vratio_a = V_a/V0

        press_a = self._calc_press(V_a)
        eta_a = b*press_a + 1
        eta_pow_a = eta_a**(-c)
        #  NOTE: Need to simplify energy expression here
        energy_a = (V0/b)/PV_ratio*(a*c/(c-1)-1)\
            - (V0/b)/PV_ratio*( a*c/(c-1)*eta_a*eta_pow_a - a*eta_pow_a + a - 1)

        return energy_a

    # def _calc_energy_perturb_deprecate(self, V_a):
    #     """
    #     Returns Energy pertubation basis functions resulting from
    #     fractional changes to EOS params.

    #     """
    #     V0, K0, KP0, KP20 = self._get_eos_params()
    #     E0, = core.get_params(['E0'])

    #     a,b,c = self._eos_to_abc_params(K0,KP0,KP20)

    #     PV_ratio = core.CONSTS['PV_ratio']

    #     vratio_a = V_a/V0

    #     press_a = self._calc_press(V_a)
    #     eta_a = b*press_a + 1
    #     eta_pow_a = eta_a**(-c)

    #     scale_a, paramkey_a = self.get_param_scale_sub()

    #     # [V0,K0,KP0,KP20,E0]
    #     dEdp_a = np.ones((4, V_a.size))
    #     # dEdp_a[0,:] = 1/(PV_ratio*b*(c-1))*eta_a*(-a*eta_pow_a -1 + (1-a)*(a+c))
    #     dEdp_a[0,:] = 1/(PV_ratio*b*(c-1))*eta_a*(-a*eta_pow_a +a -1 -a*c+c) \
    #         + 1/(PV_ratio*b)*(a*c/(c-1)-1)
    #     dEdp_a[-1,:] = 1

    #     # from IPython import embed; embed(); import ipdb; ipdb.set_trace()
    #     # 1x3
    #     dEdabc_a = np.vstack\
    #         ([V0*eta_a/(a*b*(c-1))*(-a*eta_pow_a + a*(1-c))+c*V0/(b*(c-1)),
    #           V0/(b**2*(c-1))*((-a*eta_pow_a+a-1)*(c-1) + c*a*eta_a*eta_pow_a) \
    #           - V0/b**2*(a*c/(c-1) - 1),
    #           -a*V0/(b*(c-1)**2)*eta_a*eta_pow_a*(-c+(c-1)*(1-np.log(eta_a)))\
    #           +a*V0/(b*(c-1))*(1-c/(c-1))])
    #     # 3x3
    #     abc_jac = np.array([[-KP20*(KP0+1)/(K0*KP20+KP0+1)**2,
    #                          K0*KP20/(K0*KP20+KP0+1)**2,
    #                          -K0*(KP0+1)/(K0*KP20+KP0+1)**2],
    #                         [-KP0/K0**2, KP20/(KP0+1)**2 + 1/K0, -1/(KP0+1)],
    #                         [KP20*(KP0**2+2*KP0+1)/(-K0*KP20+KP0**2+KP0)**2,
    #                          (-K0*KP20+KP0**2+KP0-(2*KP0+1)*(K0*KP20+KP0+1))/\
    #                          (-K0*KP20+KP0**2+KP0)**2,
    #                          K0*(KP0**2+2*KP0+1)/(-K0*KP20+KP0**2+KP0)**2]])

    #     dEdp_a[1:4,:] = 1/PV_ratio*np.dot(abc_jac.T,dEdabc_a)

    #     print(dEdp_a.shape)

    #     if self.setlogPmin:
    #         # [V0,K0,KP0,E0]
    #         print(dEdp_a.shape)
    #         dEdp_a = dEdp_a[[0,1,2,4],:]


    #     Eperturb_a = np.expand_dims(scale_a,1)*dEdp_a


    #     #Eperturb_a = np.expand_dims(scale_a)*dEdp_a

    #     return Eperturb_a, scale_a, paramkey_a

    def _init_params(self, order):
        """Initialize list of calculator parameter names."""

        order = 4 # ignore input order
        V0, K0, KP0 = 100, 150, 4
        KP20 = -KP0/K0
        KP20_scale = np.abs(KP20)
        param_names = ['V0','K0','KP0','KP20']
        param_units = ['ang^3','GPa','1','GPa^-1']
        param_defaults = [V0,K0,KP0,KP20]
        param_scales = [V0,K0,KP0,KP20_scale]

        self._set_params(param_names, param_units,
                         param_defaults, param_scales, order=order)

        pass
#====================================================================
class _PolyRho(CompressCalc):
    """
    Needed for Spera 2011
    """

    # def __init__(self, eos_mod, path_const='T', order=5, mass=100 ):
    # def _get_coef_array(self):
    #     basename = 'Pcoef'
    #     param_names = core.make_array_param_defaults(basename, self.order)
    #     param_values = np.array(self.eos_mod.get_param_values(
    #         param_names=param_names))

    #     coef_index = core.get_array_param_index(param_names)
    #     order = np.max(coef_index)
    #     param_full = np.zeros(order)
    #     param_full[coef_index] = param_values

    def _vol_to_rho(self, V):
        rho = (self.eos_mod.molar_mass/V)*(core.CONSTS['ang3percc']/core.CONSTS['Nmol'])
        return rho

    def _rho_to_vol(self, rho):
        V = (self.eos_mod.molar_mass/rho)*(core.CONSTS['ang3percc']/core.CONSTS['Nmol'])
        return V

    def _get_poly_coef(self):
        param_names = self.eos_mod.get_array_param_names('Pcoef')
        param_values = self.eos_mod.get_param_values(param_names=param_names)
        V0, = self.eos_mod.get_param_values(param_names=['V0'])
        rho0 = self._vol_to_rho(V0)

        coef_index = core.get_array_param_index(param_names)
        order = np.max(coef_index)+1
        param_full = np.zeros(order)
        param_full[coef_index] = param_values

        coef_a = np.flipud(param_full)

        return coef_a, rho0

    def _get_unshifted_poly_coef(self):
        coef_a, rho0 = self._get_poly_coef()
        order = coef_a.size
        Pcoef_a = coef_a*rho0**np.flipud(np.arange(order))
        core.simplify_poly(coef_a)

    def _calc_press(self, V_a):
        V_a = core.fill_array(V_a)
        coef_a, rho0 = self._get_poly_coef()
        rho_a = self._vol_to_rho(V_a)

        order = coef_a.size
        Pcoef_a = coef_a*rho0**np.flipud(np.arange(order))
        x = rho_a/rho0
        press_a = np.polyval(Pcoef_a, x-1)
        return press_a

    def _calc_energy(self, V_a):
        V_a = core.fill_array(V_a)
        PV_ratio = core.CONSTS['PV_ratio']

        coef_a, rho0 = self._get_poly_coef()
        rho_a = self._vol_to_rho(V_a)

        order = coef_a.size
        Pcoef_a = coef_a*rho0**np.flipud(np.arange(order))
        x = rho_a/rho0

        press_a = np.polyval(Pcoef_a, x-1)


        core.simplify_poly(Pcoef_a)


        V0, = self.eos_mod.get_param_values(param_names=['V0'])
        coef_a, rho0 = self._get_poly_coef()

        coef_rev_a = np.flipud(coef_a)
        order = coef_a.size
        coef_exp_a = np.flipud(np.arange(0,order))

        energy_a = np.zeros(V_a.shape)
        energy_a += coef_rev_a[0]*(V_a-V0)*PV_ratio
        energy_a += coef_rev_a[1]*np.log(V_a/V0)*PV_ratio

        for deg in range(2,order):
            energy_a += coef_rev_a[deg]*()

        return energy_a

    def get_energy_scale(self):
        V0, dPdrho = self.get_param_defaults(['V0','_Pcoef_1'])
        rho0 = self._vol_to_rho(V0)
        K0 = rho0*dPdrho
        energy_scale = np.round(V0*K0/core.CONSTS['PV_ratio'],decimals=2)
        return energy_scale

    def _init_params(self, order):
        """Initialize list of calculator parameter names."""

        rho0 = 2.58516
        coef_basename = 'Pcoef'
        param_names = core.make_array_param_names(coef_basename, order,
                                                  skipzero=True)

        param_values_sio2 = [8.78411, 12.08481, -5.5986, 4.92863, -0.90499]
        if order>6:
            param_defaults = [0 for ind in range(1,order)]
            param_defaults[0:5] = param_values_sio2
        else:
            param_defaults = param_values_sio2[0:order-1]

        param_scales = [1 for ind in range(1,order)]
        param_units = core.make_array_param_units(param_names, base_unit='GPa',
                                                  deriv_unit='(g/cc)')

        V0 = self._rho_to_vol(rho0)

        param_names.append('V0')
        param_scales.append(V0)
        param_units.append('ang^3')
        param_defaults.append(V0)

        self._set_params(param_names, param_units,
                         param_defaults, param_scales, order=order)

        pass
#====================================================================
