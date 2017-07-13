import numpy as np
import scipy as sp
from abc import ABCMeta, abstractmethod
from scipy import integrate
import scipy.interpolate as interpolate

from core import EosMod
import core

#====================================================================
# Base Classes
#====================================================================
class CompressMod(EosMod):
    """
    Abstract Equation of State class to describe Compression Behavior

    generally depends on both vol and temp
    """
    __metaclass__ = ABCMeta

    def press( self, V_a, T_a, eos_d ):
        """Returns Press behavior due to compression."""
        return self.calc_press(V_a, T_a, eos_d)

    def vol( self, P_a, T_a, eos_d ):
        """Returns Vol behavior due to compression."""
        return self.calc_vol(V_a, T_a, eos_d)

    def energy( self, V_a, T_a, eos_d ):
        """Returns Energy behavior due to compression."""
        return self.calc_energy(V_a, T_a, eos_d)

    def energy_perturb( self, V_a, T_a, eos_d ):
        """Returns Energy pertubation basis functions resulting from fractional changes to EOS params."""
        return self.calc_energy_perturb(V_a, T_a, eos_d)

    def bulk_mod( self, V_a, T_a, eos_d ):
        """Returns Bulk Modulus behavior due to compression."""
        return self.calc_bulk_mod(V_a, T_a, eos_d)

    def bulk_mod_deriv( self, V_a, T_a, eos_d ):
        """Returns Bulk Modulus Deriv (K') behavior due to compression."""
        return self.calc_bulk_mod_deriv(V_a, T_a, eos_d)

    # Standard methods must be overridden (as needed) by implimentation model
    def calc_press( self, V_a, T_a, eos_d ):
        """Calculates Press behavior due to compression."""
        raise NotImplementedError("'calc_press' function not implimented for this model")

    def calc_vol( self, P_a, T_a, eos_d ):
        """Calculates Vol behavior due to compression."""
        raise NotImplementedError("'calc_vol' function not implimented for this model")

    def calc_energy( self, V_a, T_a, eos_d ):
        """Calculates Energy behavior due to compression."""
        raise NotImplementedError("'calc_energy' function not implimented for this model")

    def calc_energy_perturb( self, V_a, T_a, eos_d ):
        """Calculates Energy pertubation basis functions resulting from fractional changes to EOS params."""
        raise NotImplementedError("'calc_energy_perturb' function not implimented for this model")

    def calc_bulk_mod( self, V_a, T_a, eos_d ):
        """Calculates Bulk Modulus behavior due to compression."""
        raise NotImplementedError("'calc_bulk_mod' function not implimented for this model")

    def calc_bulk_mod_deriv( self, V_a, T_a, eos_d ):
        """Calculates Bulk Modulus Deriv (K') behavior due to compression."""
        raise NotImplementedError("'calc_bulk_mod_deriv' function not implimented for this model")
#====================================================================
class CompressPathMod(CompressMod):
    """
    Abstract Equation of State class for a reference Compression Path

    Path can either be isothermal (T=const) or adiabatic (S=const)

    For this restricted path, thermodyn properties depend only on volume
    """
    __metaclass__ = ABCMeta

    path_opts = ['T','S']
    supress_energy = False
    supress_press = False

    def __init__( self, path_const='T', level_const=300, expand_adj_mod=None,
                 expand_adj=None, supress_energy=False, supress_press=False ):
        assert path_const in self.path_opts, path_const + ' is not a valid ' + \
            'path const. You must select one of: ' + path_opts
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

    def validate_shared_param_scale( self, scale_pos_a, paramkey_pos_a,
                                    scale_neg_a, paramkey_neg_a ):
        TOL = 1e-4
        assert np.all(np.in1d(paramkey_pos_a,paramkey_neg_a)),\
            'paramkey_neg_a must be a superset of paramkey_pos_a'
        assert len(paramkey_neg_a) <= len(paramkey_pos_a)+1,\
            'paramkey_neg_a must have at most one more parameter than paramkey_neg_a'

        # shared_mask = np.in1d(paramkey_neg_a,paramkey_pos_a)
        # paramkey_shared_a = paramkey_neg_a[shared_mask]
        # scale_shared_a = scale_neg_a[shared_mask]

        ind_pos_a = np.array([np.where(paramkey_neg_a==paramkey)[0][0] \
                              for paramkey in paramkey_pos_a])
        # scale_a[ind_pos_a] = scale_pos_a

        assert np.all(np.log(scale_neg_a[ind_pos_a]/scale_pos_a)<TOL),\
            'Shared param scales must match to within TOL.'

        return ind_pos_a

    def get_param_scale( self, eos_d, apply_expand_adj=False , output_ind=False):
        if not self.expand_adj :
            return self.get_param_scale_sub( eos_d )
        else:
            scale_a, paramkey_a = self.get_param_scale_sub( eos_d )
            scale_a = np.append(scale_a,0.01)
            paramkey_a = np.append(paramkey_a,'logPmin')

            return scale_a, paramkey_a
            # paramkey_pos_a = np.append(paramkey_pos_a,1.0)

            # scale_neg_a, paramkey_neg_a = self.expand_adj_mod.get_param_scale_sub( eos_d )

            # ind_pos_a = self.validate_shared_param_scale(scale_pos_a,paramkey_pos_a,
            #                                              scale_neg_a,paramkey_neg_a)

            # # Since negative expansion EOS model params are a superset of those
            # # required for the positive compression model, we can simply return the
            # # scale and paramkey values from the negative expansion model
            # scale_a = scale_neg_a
            # paramkey_a = paramkey_neg_a

            # if output_ind:
            #     return scale_a, paramkey_a, ind_pos_a
            # else:
            #     return scale_a, paramkey_a

    def get_ind_exp( self, V_a, eos_d ):
        V0 = core.get_params( ['V0'], eos_d )
        ind_exp = np.where( V_a > V0 )[0]
        return ind_exp

    def get_path_const( self ):
        return self.path_const

    def get_level_const( self ):
        return self.level_const

    # EOS property functions
    def param_deriv( self, fname, paramname, V_a, eos_d, dxfrac=1e-6):
        scale_a, paramkey_a = self.get_param_scale( eos_d, apply_expand_adj=True )
        scale = scale_a[paramkey_a==paramname][0]
        # print 'scale: ' + np.str(scale)

        #if (paramname is 'E0') and (fname is 'energy'):
        #    return np.ones(V_a.shape)
        try:
            fun = getattr(self, fname)
            # Note that self is implicitly included
            val0_a = fun( V_a, eos_d)
        except:
            assert False, 'That is not a valid function name ' + \
                '(e.g. it should be press or energy)'

        try:
            param = core.get_params( [paramname], eos_d )[0]
            dparam = scale*dxfrac
            # print 'param: ' + np.str(param)
            # print 'dparam: ' + np.str(dparam)
        except:
            assert False, 'This is not a valid parameter name'

        # set param value in eos_d dict
        core.set_params( [paramname,], [param+dparam,], eos_d )

        # Note that self is implicitly included
        dval_a = fun(V_a, eos_d) - val0_a

        # reset param to original value
        core.set_params( [paramname], [param], eos_d )

        deriv_a = dval_a/dxfrac
        return deriv_a

    def press( self, V_a, eos_d, apply_expand_adj=True):
        if self.supress_press:
            zero_a = 0.0*V_a
            return zero_a

        else:
            press_a = self.calc_press(V_a, eos_d)
            if self.expand_adj and apply_expand_adj:
                ind_exp = self.get_ind_exp(V_a, eos_d)
                if (ind_exp.size>0):
                    press_a[ind_exp] = self.expand_adj_mod.calc_press( V_a[ind_exp], eos_d )

            return press_a
        pass

    def energy( self, V_a, eos_d, apply_expand_adj=True ):
        if self.supress_energy:
            zero_a = 0.0*V_a
            return zero_a

        else:
            energy_a =  self.calc_energy(V_a, eos_d)
            if self.expand_adj and apply_expand_adj:
                ind_exp = self.get_ind_exp(V_a, eos_d)
                if apply_expand_adj and (ind_exp.size>0):
                    energy_a[ind_exp] = self.expand_adj_mod.calc_energy( V_a[ind_exp], eos_d )

            return energy_a

    def bulk_mod( self, V_a, eos_d, apply_expand_adj=True ):
        bulk_mod_a =  self.calc_bulk_mod(V_a, eos_d)
        if self.expand_adj and apply_expand_adj:
            ind_exp = self.get_ind_exp(V_a, eos_d)
            if apply_expand_adj and (ind_exp.size>0):
                bulk_mod_a[ind_exp] = self.expand_adj_mod.calc_bulk_mod( V_a[ind_exp], eos_d )

        return bulk_mod_a

    def bulk_mod_deriv(  self,V_a, eos_d, apply_expand_adj=True ):
        bulk_mod_deriv_a =  self.calc_bulk_mod_deriv(V_a, eos_d)
        if self.expand_adj and apply_expand_adj:
            ind_exp = self.get_ind_exp(V_a, eos_d)
            if apply_expand_adj and (ind_exp.size>0):
                bulk_mod_deriv_a[ind_exp] = self.expand_adj_mod_deriv.calc_bulk_mod_deriv( V_a[ind_exp], eos_d )

        return bulk_mod_deriv_a

    def energy_perturb( self, V_a, eos_d, apply_expand_adj=True ):
        # Eval positive press values
        Eperturb_pos_a, scale_a, paramkey_a  = self.calc_energy_perturb( V_a, eos_d )

        if (self.expand_adj==False) or (apply_expand_adj==False):
            return Eperturb_pos_a, scale_a, paramkey_a
        else:
            Nparam_pos = Eperturb_pos_a.shape[0]

            scale_a, paramkey_a, ind_pos = \
                self.get_param_scale( eos_d, apply_expand_adj=True,
                                     output_ind=True )

            Eperturb_a = np.zeros((paramkey_a.size, V_a.size))
            Eperturb_a[ind_pos,:] = Eperturb_pos_a

            # Overwrite negative pressure Expansion regions
            ind_exp = self.get_ind_exp(V_a, eos_d)
            if ind_exp.size>0:
                Eperturb_adj_a = \
                    self.expand_adj_mod.calc_energy_perturb( V_a[ind_exp],
                                                            eos_d )[0]
                Eperturb_a[:,ind_exp] = Eperturb_adj_a

            return Eperturb_a, scale_a, paramkey_a

    #   Standard methods must be overridden (as needed) by implimentation model

    def get_param_scale_sub( self, eos_d):
        raise NotImplementedError("'get_param_scale_sub' function not implimented for this model")

    def calc_press( self, V_a, eos_d ):
        """Returns Press variation along compression curve."""
        raise NotImplementedError("'press' function not implimented for this model")

    def calc_energy( self, V_a, eos_d ):
        """Returns Energy along compression curve."""
        raise NotImplementedError("'energy' function not implimented for this model")

    def calc_energy_perturb( self, V_a, eos_d ):
        """Returns Energy pertubation basis functions resulting from fractional changes to EOS params."""

        fname = 'energy'
        scale_a, paramkey_a = self.get_param_scale\
            ( eos_d, apply_expand_adj=self.expand_adj )
        Eperturb_a = []
        for paramname in paramkey_a:
            iEperturb_a = self.param_deriv( fname, paramname, V_a, eos_d)
            Eperturb_a.append(iEperturb_a)

        Eperturb_a = np.array(Eperturb_a)

        return Eperturb_a, scale_a, paramkey_a

    def calc_bulk_mod( self, V_a, eos_d ):
        """Returns Bulk Modulus variation along compression curve."""
        raise NotImplementedError("'bulk_mod' function not implimented for this model")

    def calc_bulk_mod_deriv( self, V_a, eos_d ):
        """Returns Bulk Modulus Deriv (K') variation along compression curve."""
        raise NotImplementedError("'bulk_mod_deriv' function not implimented for this model")
#====================================================================

#====================================================================
# Implementations
#====================================================================
class BirchMurn3(CompressPathMod):
    def calc_press( self, V_a, eos_d ):
        V0, K0, KP0 = core.get_params( ['V0','K0','KP0'], eos_d )

        vratio_a = 1.0*V_a/V0

        press_a = 3.0/2*K0 * (vratio_a**(-7.0/3) - vratio_a**(-5.0/3)) * \
            (1 + 3.0/4*(KP0-4)*(vratio_a**(-2.0/3)-1))

        return press_a

    def calc_energy( self, V_a, eos_d ):
        V0, K0, KP0, E0 = core.get_params( ['V0','K0','KP0','E0'], eos_d )
        PV_ratio, = core.get_consts( ['PV_ratio'], eos_d )

        vratio_a = 1.0*V_a/V0

        fstrain_a = 0.5*(vratio_a**(-2.0/3) - 1)

        energy_a = E0 + 9.0/2*(V0*K0/PV_ratio)*\
            ( KP0*fstrain_a**3 + fstrain_a**2*(1-4*fstrain_a) )

        return energy_a
#====================================================================
class BirchMurn4(CompressPathMod):
    def get_param_scale_sub( self, eos_d):
        """Return scale values for each parameter"""
        V0, K0, KP0, KP20 = core.get_params( ['V0','K0','KP0','KP20'], eos_d )
        PV_ratio, = core.get_consts( ['PV_ratio'], eos_d )

        paramkey_a = np.array(['V0','K0','KP0','KP20','E0'])
        scale_a = np.array([V0,K0,KP0,KP0/K0,K0*V0/PV_ratio])

        return scale_a, paramkey_a

    def calc_strain_energy_coeffs(self, nexp, K0, KP0, KP20 ):
        a1 = 3./2*(KP0-nexp-2)
        a2 = 3./2*(K0*KP20 + KP0*(KP0-2*nexp-3)+3+4*nexp+11./9*nexp**2)
        return a1,a2

    def calc_press( self, V_a, eos_d ):
        # globals()['set_param']( ['nexp'], [self.nexp], eos_d )
        # press_a = self.gen_finite_strain_mod.press( V_a, eos_d )
        V0, K0, KP0, KP20 = core.get_params( ['V0','K0','KP0','KP20'], eos_d )
        nexp = +2.0

        vratio_a = 1.0*V_a/V0
        fstrain_a = 1./nexp*(vratio_a**(-nexp/3) - 1)

        a1,a2 = self.calc_strain_energy_coeffs(nexp,K0,KP0,KP20)

        press_a = 3.0*K0*(1+a1*fstrain_a + a2*fstrain_a**2)*\
            fstrain_a*(nexp*fstrain_a+1)**((nexp+3)/nexp)
        return press_a

    def calc_energy( self, V_a, eos_d ):
        # globals()['set_param']( ['nexp'], [self.nexp], eos_d )
        # energy_a = self.gen_finite_strain_mod.energy( V_a, eos_d )
        V0, K0, KP0, KP20, E0 = core.get_params( ['V0','K0','KP0','KP20','E0'], eos_d )
        nexp = +2.0

        PV_ratio, = core.get_consts( ['PV_ratio'], eos_d )

        vratio_a = 1.0*V_a/V0
        fstrain_a = 1./nexp*(vratio_a**(-nexp/3) - 1)

        a1,a2 = self.calc_strain_energy_coeffs(nexp,K0,KP0,KP20)


        energy_a = E0 + 9.0*(V0*K0/PV_ratio)*\
            ( 0.5*fstrain_a**2 + a1/3*fstrain_a**3 + a2/4*fstrain_a**4)

        return energy_a
#====================================================================
class GenFiniteStrain(CompressPathMod):
    """
    Generalized Finite Strain EOS from Jeanloz1989b

    Note: nexp=2 yields Birch Murnaghan (eulerian strain) EOS
          nexp=-2 yields lagragian strain EOS
    """

    def calc_strain_energy_coeffs(self, nexp, K0, KP0, KP20=None, KP30=None):
        a1 = 3./2*(KP0-nexp-2)
        if KP20 is None:
            return a1
        else:
            a2 = 3./2*(K0*KP20 + KP0*(KP0-2*nexp-3)+3+4*nexp+11./9*nexp**2)
            if KP30 is None:
                return a1,a2
            else:
                a3 = 1./8*(9*K0**2*KP30 + 6*(6*KP0-5*nexp-6)*K0*KP20
                           +((3*KP0-5*nexp-6)**2 +10*nexp**2 + 30*nexp + 18)*KP0
                           -(50./3*nexp**3 + 70*nexp**2 + 90*nexp + 36))
                return a1,a2,a3

    def calc_press( self, V_a, eos_d ):
        V0, K0, KP0, KP20, nexp = core.get_params( ['V0','K0','KP0','KP20','nexp'], eos_d )

        vratio_a = 1.0*V_a/V0
        fstrain_a = 1./nexp*(vratio_a**(-nexp/3) - 1)

        a1,a2 = self.calc_strain_energy_coeffs(nexp,K0,KP0,KP20=KP20)

        press_a = 3.0*K0*(1+a1*fstrain_a + a2*fstrain_a**2)*\
            fstrain_a*(nexp*fstrain_a+1)**((nexp+3)/nexp)
        return press_a

    def calc_energy( self, V_a, eos_d ):
        V0, K0, KP0, KP20, E0, nexp = core.get_params( ['V0','K0','KP0','KP20','E0','nexp'], eos_d )
        PV_ratio, = core.get_consts( ['PV_ratio'], eos_d )

        vratio_a = 1.0*V_a/V0
        fstrain_a = 1./nexp*(vratio_a**(-nexp/3) - 1)

        a1,a2 = self.calc_strain_energy_coeffs(nexp,K0,KP0,KP20=KP20)


        energy_a = E0 + 9.0*(V0*K0/PV_ratio)*\
            ( 0.5*fstrain_a**2 + a1/3*fstrain_a**3 + a2/4*fstrain_a**4)

        return energy_a
#====================================================================
class Vinet(CompressPathMod):
    def get_param_scale_sub( self, eos_d):
        """Return scale values for each parameter"""
        V0, K0, KP0 = core.get_params( ['V0','K0','KP0'], eos_d )
        PV_ratio, = core.get_consts( ['PV_ratio'], eos_d )

        paramkey_a = np.array(['V0','K0','KP0','E0'])
        scale_a = np.array([V0,K0,KP0,K0*V0/PV_ratio])

        return scale_a, paramkey_a

    def calc_press( self, V_a, eos_d ):
        V0, K0, KP0 = core.get_params( ['V0','K0','KP0'], eos_d )

        eta = 3./2*(KP0-1)
        vratio_a = 1.0*V_a/V0
        x_a = vratio_a**(1./3)

        press_a = 3*K0*(1-x_a)*x_a**(-2)*np.exp(eta*(1-x_a))

        return press_a

    def calc_energy( self, V_a, eos_d ):
        V0, K0, KP0, E0 = core.get_params( ['V0','K0','KP0','E0'], eos_d )
        # print V0
        # print K0
        # print KP0
        # print E0
        PV_ratio, = core.get_consts( ['PV_ratio'], eos_d )

        eta = 3./2*(KP0-1)
        vratio_a = 1.0*V_a/V0
        x_a = vratio_a**(1.0/3)


        energy_a = E0 + 9*K0*V0/PV_ratio/eta**2*\
            (1 + (eta*(1-x_a)-1)*np.exp(eta*(1-x_a)))

        return energy_a

    def calc_energy_perturb( self, V_a, eos_d ):
        """Returns Energy pertubation basis functions resulting from fractional changes to EOS params."""

        V0, K0, KP0, E0 = core.get_params( ['V0','K0','KP0','E0'], eos_d )
        PV_ratio, = core.get_consts( ['PV_ratio'], eos_d )

        eta = 3./2*(KP0-1)
        vratio_a = 1.0*V_a/V0
        x = vratio_a**(1./3)

        scale_a, paramkey_a = self.get_param_scale_sub( eos_d )

        # NOTE: CHECK UNITS (PV_RATIO) here
        dEdp_a = 1.0/PV_ratio*np.vstack\
            ([-3*K0*(eta**2*x*(x-1) + 3*eta*(x-1) - 3*np.exp(eta*(x-1)) + 3)\
              *np.exp(-eta*(x-1))/eta**2,
              -9*V0*(eta*(x-1) - np.exp(eta*(x-1)) + 1)*np.exp(-eta*(x-1))/eta**2,
              27*K0*V0*(2*eta*(x-1) + eta*(-x + (x-1)*(eta*(x-1) + 1) + 1)
                        -2*np.exp(eta*(x-1)) + 2)*np.exp(-eta*(x-1))/(2*eta**3),
              PV_ratio*np.ones(V_a.shape)])

        Eperturb_a = np.expand_dims(scale_a,1)*dEdp_a
        #Eperturb_a = np.expand_dims(scale_a)*dEdp_a

        return Eperturb_a, scale_a, paramkey_a
#====================================================================
class Tait(CompressPathMod):
    def __init__( self, setlogPmin=False,
                 path_const='T', level_const=300, expand_adj_mod=None,
                 expand_adj=None, supress_energy=False, supress_press=False ):
        super(Tait, self).__init__( expand_adj=None )
        self.setlogPmin = setlogPmin
        pass
    # def __init__( self, setlogPmin=False, expand_adj=False ):
    #     self.setlogPmin = setlogPmin
    #     self.expand_adj = expand_adj
    #     pass

    def get_eos_params(self, eos_d):
        V0, K0, KP0 = core.get_params( ['V0','K0','KP0'], eos_d )
        if self.setlogPmin:
            logPmin, = core.get_params( ['logPmin'], eos_d )
            Pmin = np.exp(logPmin)
            # assert Pmin>0, 'Pmin must be positive.'
            KP20 = (KP0+1)*(KP0/K0 - 1.0/Pmin)
        else:
            KP20, = core.get_params( ['KP20'], eos_d )

        return V0,K0,KP0,KP20

    def get_param_scale_sub( self, eos_d ):
        """Return scale values for each parameter"""

        V0, K0, KP0, KP20 = self.get_eos_params(eos_d)
        PV_ratio, = core.get_consts( ['PV_ratio'], eos_d )

        if self.setlogPmin:
            # [V0,K0,KP0,E0]
            paramkey_a = np.array(['V0','K0','KP0','E0'])
            scale_a = np.array([V0,K0,KP0,K0*V0/PV_ratio])
        else:
            # [V0,K0,KP0,KP20,E0]
            paramkey_a = np.array(['V0','K0','KP0','KP20','E0'])
            scale_a = np.array([V0,K0,KP0,KP0/K0,K0*V0/PV_ratio])


        return scale_a, paramkey_a

    def eos_to_abc_params(self, K0, KP0, KP20):
        a = (KP0 + 1.0)/(K0*KP20 + KP0 + 1.0)
        b = -KP20/(KP0+1.0) + KP0/K0
        c = (K0*KP20 + KP0 + 1.0)/(-K0*KP20 + KP0**2 + KP0)

        return a,b,c

    def calc_press( self, V_a, eos_d ):
        V0, K0, KP0, KP20 = self.get_eos_params(eos_d)
        a,b,c = self.eos_to_abc_params(K0,KP0,KP20)
        vratio_a = 1.0*V_a/V0

        press_a = 1.0/b*(((vratio_a + a - 1.0)/a)**(-1.0/c) - 1.0)

        return press_a

    def calc_energy( self, V_a, eos_d ):
        V0, K0, KP0, KP20 = self.get_eos_params(eos_d)
        E0, = core.get_params( ['E0'], eos_d )
        a,b,c = self.eos_to_abc_params(K0,KP0,KP20)
        PV_ratio, = core.get_consts( ['PV_ratio'], eos_d )

        vratio_a = 1.0*V_a/V0

        press_a = self.calc_press( V_a, eos_d )
        eta_a = b*press_a + 1.0
        eta_pow_a = eta_a**(-c)
        #  NOTE: Need to simplify energy expression here
        energy_a = E0 + (V0/b)/PV_ratio*(a*c/(c-1)-1)\
            - (V0/b)/PV_ratio*( a*c/(c-1)*eta_a*eta_pow_a - a*eta_pow_a + a - 1)

        return energy_a

    def calc_energy_perturb_deprecate( self, V_a, eos_d ):
        """Returns Energy pertubation basis functions resulting from fractional changes to EOS params."""
        V0, K0, KP0, KP20 = self.get_eos_params(eos_d)
        E0, = core.get_params( ['E0'], eos_d )

        a,b,c = self.eos_to_abc_params(K0,KP0,KP20)
        PV_ratio, = core.get_consts( ['PV_ratio'], eos_d )

        vratio_a = V_a/V0

        press_a = self.calc_press( V_a, eos_d )
        eta_a = b*press_a + 1.0
        eta_pow_a = eta_a**(-c)

        scale_a, paramkey_a = self.get_param_scale_sub( eos_d )

        # [V0,K0,KP0,KP20,E0]
        dEdp_a = np.ones((4, V_a.size))
        # dEdp_a[0,:] = 1.0/(PV_ratio*b*(c-1))*eta_a*(-a*eta_pow_a -1 + (1-a)*(a+c))
        dEdp_a[0,:] = 1.0/(PV_ratio*b*(c-1))*eta_a*(-a*eta_pow_a +a -1 -a*c+c) \
            + 1.0/(PV_ratio*b)*(a*c/(c-1)-1)
        dEdp_a[-1,:] = 1.0

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
                            [-KP0/K0**2, KP20/(KP0+1)**2 + 1./K0, -1.0/(KP0+1)],
                            [KP20*(KP0**2+2.*KP0+1)/(-K0*KP20+KP0**2+KP0)**2,
                             (-K0*KP20+KP0**2+KP0-(2*KP0+1)*(K0*KP20+KP0+1))/\
                             (-K0*KP20+KP0**2+KP0)**2,
                             K0*(KP0**2+2*KP0+1)/(-K0*KP20+KP0**2+KP0)**2]])

        dEdp_a[1:4,:] = 1.0/PV_ratio*np.dot(abc_jac.T,dEdabc_a)

        print dEdp_a.shape

        if self.setlogPmin:
            # [V0,K0,KP0,E0]
            print dEdp_a.shape
            dEdp_a = dEdp_a[[0,1,2,4],:]


        Eperturb_a = np.expand_dims(scale_a,1)*dEdp_a


        #Eperturb_a = np.expand_dims(scale_a)*dEdp_a

        return Eperturb_a, scale_a, paramkey_a
#====================================================================
class RosenfeldTaranzonaShiftedAdiabat(CompressPathMod):
    def get_param_scale_sub( self, eos_d):
        """Return scale values for each parameter"""
        V0, K0, KP0 = core.get_params( ['V0','K0','KP0'], eos_d )
        PV_ratio, = core.get_consts( ['PV_ratio'], eos_d )

        paramkey_a = np.array(['V0','K0','KP0','E0'])
        scale_a = np.array([V0,K0,KP0,K0*V0/PV_ratio])

        return scale_a, paramkey_a

    def calc_press( self, V_a, eos_d ):
        PV_ratio, = core.get_consts( ['PV_ratio'], eos_d )
        fac = 1e-3
        Vhi_a = V_a*(1.0 + 0.5*fac)
        Vlo_a = V_a*(1.0 - 0.5*fac)

        dV_a = Vhi_a-Vlo_a


        E0S_hi_a = self.calc_energy(Vhi_a, eos_d)
        E0S_lo_a = self.calc_energy(Vlo_a, eos_d)

        P0S_a = -PV_ratio*(E0S_hi_a - E0S_lo_a)/dV_a
        return P0S_a

    def calc_energy( self, V_a, eos_d ):
        V0, T0, mexp  = core.get_params( ['V0','T0','mexp'], eos_d )
        kB, = core.get_consts( ['kboltz'], eos_d )

        poly_blogcoef_a = core.get_array_params( 'blogcoef', eos_d )


        compress_path_mod, thermal_mod, gamma_mod = \
            core.get_modtypes( ['CompressPathMod', 'ThermalMod', 'GammaMod'],
                                 eos_d )

        free_energy_isotherm_a = compress_path_mod.energy(V_a,eos_d)

        T0S_a = gamma_mod.temp(V_a,T0,eos_d)


        bV_a = np.polyval(poly_blogcoef_a,np.log(V_a/V0))

        dS_a = -mexp/(mexp-1)*bV_a/T0*((T0S_a/T0)**(mexp-1)-1)\
            -3./2*kB*np.log(T0S_a/T0)


        energy_isotherm_a = free_energy_isotherm_a + T0*dS_a
        E0S_a = energy_isotherm_a + bV_a*((T0S_a/T0)**mexp-1)\
            +3./2*kB*(T0S_a-T0)

        return E0S_a
#====================================================================
