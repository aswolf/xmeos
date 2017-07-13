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
class ThermalMod(EosMod):
    """
    Abstract Equation of State class to describe Thermal Behavior

    generally depends on both vol and temp
    """

    __metaclass__ = ABCMeta

    # EOS property functions
    def energy( self, V_a, T_a, eos_d ):
        return self.calc_energy( V_a, T_a, eos_d )

    def heat_capacity( self, V_a, T_a, eos_d ):
        return self.calc_heat_capacity( V_a, T_a, eos_d )

    def press( self, V_a, T_a, eos_d ):
        return self.calc_press( V_a, T_a, eos_d )

    def entropy( self, V_a, T_a, eos_d ):
        return self.calc_entropy( V_a, T_a, eos_d )

    def vol( self, P_a, T_a, eos_d ):
        return self.calc_vol( P_a, T_a, eos_d )

    # Standard methods must be overridden (as needed) by implimentation model
    def calc_energy( self, V_a, T_a, eos_d ):
        """Returns Thermal Component of Energy."""
        raise NotImplementedError("'energy' function not implimented for this model")

    def calc_heat_capacity( self, V_a, T_a, eos_d ):
        """Returns Heat Capacity."""
        raise NotImplementedError("'heat_capacity' function not implimented for this model")

    def calc_entropy( self, V_a, T_a, eos_d ):
        """Returns Entropy."""
        raise NotImplementedError("'entropy' function not implimented for this model")

    def calc_press( self, V_a, T_a, eos_d ):
        """Returns thermal contribution to pressure."""
        raise NotImplementedError("'press' function not implimented for this model")

    def calc_vol( self, V_a, T_a, eos_d ):
        """Returns thermally expanded volume."""
        raise NotImplementedError("'vol' function not implimented for this model")
#====================================================================
class ThermalPathMod(ThermalMod):
    """
    Abstract Equation of State class for a reference Thermal Path

    Path can either be isobaric (P=const) or isochoric (V=const)

    For this restricted path, thermodyn properties depend only on temperature.
    """

    __metaclass__ = ABCMeta

    path_opts = ['P','V']

    def __init__( self, path_const='V', level_const=np.nan ):
        assert path_const in self.path_opts, path_const + ' is not a valid ' + \
            'path const. You must select one of: ' + path_opts
        self.path_const = path_const
        self.level_const = level_const
        pass

    def get_path_const( self ):
        return self.path_const

    def get_level_const( self ):
        return self.level_const

    # EOS property functions
    def energy( self, T_a, eos_d ):
        return self.calc_energy( T_a, eos_d )

    def heat_capacity( self, T_a, eos_d ):
        return self.calc_heat_capacity( T_a, eos_d )

    def press( self, T_a, eos_d ):
        return self.calc_press( T_a, eos_d )

    def entropy( self, T_a, eos_d ):
        return self.calc_entropy( T_a, eos_d )

    def vol( self, T_a, eos_d ):
        return self.calc_vol( T_a, eos_d )

    # Standard methods must be overridden (as needed) by implimentation model
    def calc_energy( self, T_a, eos_d ):
        """Returns Thermal Component of Energy along heating path."""
        raise NotImplementedError("'energy' function not implimented for this model")

    def calc_heat_capacity( self, T_a, eos_d ):
        """Returns Heat Capacity along heating path."""
        raise NotImplementedError("'heat_capacity' function not implimented for this model")

    def calc_entropy( self, T_a, eos_d ):
        """Returns Entropy along heating path."""
        raise NotImplementedError("'entropy' function not implimented for this model")

    def calc_press( self, T_a, eos_d ):
        """Returns thermal contribution to pressure along heating path."""
        raise NotImplementedError("'press' function not implimented for this model")

    def calc_vol( self, T_a, eos_d ):
        """Returns thermally expanded volume along heating path."""
        raise NotImplementedError("'vol' function not implimented for this model")
#====================================================================

#====================================================================
class GenRosenfeldTaranzona(ThermalPathMod):
    """
    Generalized Rosenfeld-Taranzona Equation of State Model (Rosenfeld1998)
    - Cv takes on general form of shifted power-law as in original
    Rosenfeld-Taranzona model, but the exponent and high-temp limit are
    parameters rather than fixed
    - only applicable to isochores
    - must provide a method to evaluate properties along isochore
    """
    __metaclass__ = ABCMeta

    def get_param_scale_sub( self, eos_d):
        """Return scale values for each parameter"""
        acoef, bcoef, mexp, lognfac = core.get_params\
            ( ['acoef','bcoef','mexp','lognfac'], eos_d )

        acoef_scl = 1.0 # This cannot be well-determined without more info
        # ...like a reference temp or energy variation
        bcoef_scl = np.abs(bcoef)
        mexp_scl = 3./5
        lognfac_scl = 0.01
        paramkey_a = np.array(['acoef','bcoef','mexp','lognfac'])
        scale_a = np.array([acoef_scl,bcoef_scl,mexp_scl,lognfac_scl])

        return scale_a, paramkey_a

    def get_param_override( self, paramkey, paramval, eos_d ):
        if paramval is None:
            paramval, = core.get_params( [paramkey], eos_d )

        return paramval

    def calc_therm_dev( self, T_a, eos_d ):
        """
        """
        # assert False, 'calc_thermal_dev is not yet implimented'

        T0, = core.get_params( ['T0'], eos_d )
        mexp, = core.get_params( ['mexp'], eos_d )
        # therm_dev_a = (T_a/T0)**mexp
        therm_dev_a = (T_a/T0)**mexp - 1.0

        return therm_dev_a

    def calc_therm_dev_deriv( self, T_a, eos_d ):
        """
        """
        # assert False, 'calc_thermal_dev is not yet implimented'

        T0, = core.get_params( ['T0'], eos_d )
        mexp, = core.get_params( ['mexp'], eos_d )

        dtherm_dev_a = (mexp/T0)*(T_a/T0)**(mexp-1.0)

        return dtherm_dev_a

    def calc_energy( self, T_a, eos_d, acoef_a=None, bcoef_a=None ):
        """Returns Thermal Component of Energy."""
        mexp, lognfac = core.get_params( ['mexp','lognfac'], eos_d )


        energy_pot_a = self.calc_energy_pot( T_a, eos_d, acoef_a=acoef_a,
                                            bcoef_a=bcoef_a )
        energy_kin_a = self.calc_energy_kin( T_a, eos_d )
        energy_a = energy_pot_a + energy_kin_a

        return energy_a

    def calc_energy_kin( self, T_a, eos_d ):
        """Returns Thermal Component of Energy."""
        lognfac, = core.get_params( ['lognfac'], eos_d )
        kB, = core.get_consts( ['kboltz'], eos_d )
        nfac = np.exp(lognfac)
        energy_kin_a = 3.0/2*nfac*kB*T_a

        return energy_kin_a

    def calc_energy_pot( self, T_a, eos_d, acoef_a=None, bcoef_a=None ):
        """Returns Thermal Component of Energy."""
        acoef_a = self.get_param_override( 'acoef', acoef_a, eos_d )
        energy_pot_diff_a = self.calc_energy_pot_diff( T_a, eos_d, bcoef_a=bcoef_a )
        energy_pot_a = acoef_a + energy_pot_diff_a

        return energy_pot_a

    def calc_energy_pot_diff( self, T_a, eos_d, bcoef_a=None ):
        bcoef_a = self.get_param_override( 'bcoef', bcoef_a, eos_d )
        therm_dev_a = self.calc_therm_dev( T_a, eos_d )

        energy_pot_diff_a = bcoef_a*therm_dev_a

        return energy_pot_diff_a

    def calc_heat_capacity( self, T_a, eos_d, bcoef_a=None ):
        """Calculate Heat Capacity usin."""
        heat_capacity_pot = self.calc_heat_capacity_pot( T_a, eos_d,
                                                        bcoef_a=bcoef_a )

        heat_capacity_kin = self.calc_heat_capacity_kin( T_a, eos_d )

        heat_capacity_a = heat_capacity_pot+heat_capacity_kin

        return heat_capacity_a

    def calc_heat_capacity_pot( self, T_a, eos_d, bcoef_a=None ):
        mexp, = core.get_params( ['mexp'], eos_d )

        bcoef_a = self.get_param_override( 'bcoef', bcoef_a, eos_d )
        dtherm_dev_a = self.calc_therm_dev_deriv( T_a, eos_d )

        heat_capacity_pot_a = bcoef_a*dtherm_dev_a

        return heat_capacity_pot_a

    def calc_heat_capacity_kin( self, T_a, eos_d ):
        lognfac, = core.get_params( ['lognfac'], eos_d )
        kB, = core.get_consts( ['kboltz'], eos_d )
        nfac = np.exp(lognfac)
        heat_capacity_kin_a = + 3.0/2*nfac*kB

        return heat_capacity_kin_a

    def calc_entropy_pot( self, T_a, eos_d, Tref=None, bcoef_a=None ):
        mexp, = core.get_params( ['mexp'], eos_d )

        Tref = self.get_param_override( 'T0', Tref, eos_d )

        Cv_pot = self.calc_heat_capacity_pot( T_a, eos_d, bcoef_a=bcoef_a )
        Cv_ref_pot = self.calc_heat_capacity_pot( Tref, eos_d, bcoef_a=bcoef_a )
        dSpot_a = (Cv_pot-Cv_ref_pot)/(mexp-1.0)

        return dSpot_a

    def calc_entropy_kin( self, T_a, eos_d, Tref=None ):
        Tref = self.get_param_override( 'T0', Tref, eos_d )

        Cv_kin = self.calc_heat_capacity_kin( T_a, eos_d )
        dSkin_a = Cv_kin*np.log( T_a/Tref )

        return dSkin_a

    def calc_entropy_heat( self, T_a, eos_d, Tref=None, bcoef_a=None ):
        """Calculate Entropy change upon heating at constant volume."""
        mexp, = core.get_params( ['mexp'], eos_d )

        Tref = self.get_param_override( 'T0', Tref, eos_d )

        delS_pot = self.calc_entropy_pot( T_a, eos_d, Tref=Tref,
                                         bcoef_a=bcoef_a )

        delS_kin = self.calc_entropy_kin( T_a, eos_d, Tref=Tref )

        delS_heat_a = delS_pot + delS_kin
        return delS_heat_a

    def calc_entropy( self, T_a, eos_d, Tref=None, Sref=None, bcoef_a=None ):
        """Calculate Full Entropy for isochore."""
        Sref = self.get_param_override( 'S0', Sref, eos_d )


        delS_heat_a = self.calc_entropy_heat( T_a, eos_d, Tref=Tref,
                                             bcoef_a=bcoef_a )

        entropy_a = Sref + delS_heat_a
        return entropy_a
#====================================================================
class RosenfeldTaranzonaCompress(ThermalMod):
    """
    Volume-dependent Rosenfeld-Taranzona Equation of State
      - must impliment particular volume-dependence

    """
    __metaclass__ = ABCMeta

    #========================
    #  Override Method
    #========================

    @abstractmethod
    def calc_entropy_compress( self, V_a, eos_d ):
        """
        If compress path is
        """
        return 0.0

    #========================
    # Initialization
    #========================

    def __init__( self, coef_kind='logpoly', temp_path_kind='T0', acoef_fun=None ):
        self.set_empirical_coef( coef_kind, acoef_fun=acoef_fun )
        self.set_temp_path( temp_path_kind )
        pass

    def set_empirical_coef( self, coef_kind, acoef_fun=None ):
        coef_kind_typ = ['logpoly','poly','polynorm']

        assert coef_kind in coef_kind_typ, 'coef_kind is not a valid type. '\
            'Available types = '+str(coef_kind_typ)+'.'

        self.coef_kind= coef_kind

        calc_coef = getattr(self, 'calc_coef_'+coef_kind)
        self.calc_bcoef = lambda V_a, eos_d, deriv=0: calc_coef( V_a, 'bcoef',
                                                                eos_d, deriv=deriv )

        if acoef_fun is None:
            self.calc_acoef = lambda V_a, eos_d: calc_coef( V_a, 'acoef', eos_d )
        else:
            self.calc_acoef = acoef_fun

        pass

    def set_temp_path( self, temp_path_kind ):
        temp_path_kind_typ = ['S0','T0','abszero']

        self.temp_path_kind = temp_path_kind

        assert temp_path_kind in temp_path_kind_typ, 'temp_path_kind is not a valid type. '\
            'Available types = '+str(temp_path_kind_typ)+'.'

        self.temp_path_kind= temp_path_kind
        self.calc_temp_path = getattr(self, 'calc_temp_path_'+temp_path_kind)

        pass

    #========================
    def get_param_scale_sub( self, eos_d):
        """Return scale values for each parameter"""

        bcoef_a = core.get_array_params( 'bcoef', eos_d )
        coef_param_key = ['bcoef_'+str(i) for i in range(bcoef_a.size)]
        coef_param_scale = np.ones(bcoef_a.shape)

        try:
            acoef_a = core.get_array_params( 'acoef', eos_d )
            acoef_param_key = ['acoef_'+str(i) for i in range(acoef_a.size)]
            acoef_param_scale = np.ones(acoef_a.shape)

            coef_param_key = np.append(coef_param_key,acoef_param_key)
            coef_param_scale = np.append(coef_param_scale,acoef_param_scale)
        except:
            # No bcoef
            pass

        paramkey_a = coef_param_key
        scale_a = coef_param_scale

        T0, = core.get_params( ['T0'], eos_d )
        T0_scl = T0
        mexp_scl = 3./5
        lognfac_scl = 0.01

        paramkey_a = np.append(paramkey_a,['T0','mexp','lognfac'])
        scale_a = np.append(scale_a,[T0_scl,mexp_scl,lognfac_scl])

        return scale_a, paramkey_a

    def calc_therm_dev( self, V_a, T_a, eos_d ):
        """
        Extend thermal deviation concept to take difference from reference path,
        rather than reference point
        """

        T0, = core.get_params( ['T0'], eos_d )
        T_ref_a = self.calc_temp_path( V_a, eos_d )

        therm_dev_f = GenRosenfeldTaranzona().calc_therm_dev

        therm_dev_path_a = therm_dev_f( T_a, eos_d ) -  therm_dev_f( T_ref_a, eos_d )

        return therm_dev_path_a

    def calc_energy_pot_diff( self, V_a, T_a, eos_d ):
        # T0, = core.get_params( ['T0'], eos_d )

        # gamma_mod = eos_d['modtype_d']['GammaMod']
        # T_ref_a = gamma_mod.temp( V_a, T0, eos_d )

        # del_energy_pot_a = self.calc_energy_pot( V_a, T_a, eos_d ) \
        #     - self.calc_energy_pot( V_a, T_ref_a, eos_d )

        therm_dev_a = self.calc_therm_dev( V_a, T_a, eos_d )
        bcoef_a = self.calc_bcoef( V_a, eos_d )

        del_energy_pot_a = bcoef_a*therm_dev_a
        return del_energy_pot_a

    def calc_energy_kin_diff( self, V_a, T_a, eos_d ):
        T0, = core.get_params( ['T0'], eos_d )
        T_ref_a = self.calc_temp_path( V_a, eos_d )

        del_energy_kin_a = GenRosenfeldTaranzona().calc_energy_kin( T_a, eos_d ) \
            - GenRosenfeldTaranzona().calc_energy_kin( T_ref_a, eos_d )

        return del_energy_kin_a

    def calc_energy( self, V_a, T_a, eos_d ):
        # acoef_a = self.calc_acoef( V_a, eos_d )

        dE_pot = self.calc_energy_pot_diff( V_a, T_a, eos_d )
        dE_kin = self.calc_energy_kin_diff( V_a, T_a, eos_d )
        # E_tot = acoef_a + dE_pot + dE_kin
        dE_tot = dE_pot + dE_kin

        return dE_tot

    def calc_free_energy( self, V_a, T_a, eos_d ):
        E_tot = self.calc_energy( V_a, T_a, eos_d )
        S_tot = self.calc_entropy( V_a, T_a, eos_d )
        F_tot = E_tot - T_a*S_tot

        return F_tot

    def calc_press( self, V_a, T_a, eos_d ):
        PV_ratio, = core.get_consts( ['PV_ratio'], eos_d )
        V0, = core.get_params( ['V0'], eos_d )

        # Use numerical deriv
        dV = V0*1e-5
        F_a = self.calc_free_energy( V_a, T_a, eos_d )
        F_hi_a = self.calc_free_energy( V_a+dV, T_a, eos_d )
        press_therm_a = -PV_ratio*(F_hi_a-F_a)/dV

        return press_therm_a

    def calc_RT_coef_deriv( self, V_a, eos_d ):
        V0, = core.get_params( ['V0'], eos_d )

        # Use numerical deriv
        dV = V0*1e-5

        acoef_a = self.calc_acoef( V_a, eos_d )
        bcoef_a = self.calc_bcoef( V_a, eos_d )

        acoef_hi_a = self.calc_acoef( V_a+dV, eos_d )
        bcoef_hi_a = self.calc_bcoef( V_a+dV, eos_d )

        acoef_deriv_a = (acoef_hi_a-acoef_a)/dV
        bcoef_deriv_a = (bcoef_hi_a-bcoef_a)/dV

        return acoef_deriv_a, bcoef_deriv_a

    def calc_heat_capacity( self, V_a, T_a, eos_d ):
        """Calculate Heat Capacity usin."""
        bcoef_a = self.calc_bcoef( V_a, eos_d )
        heat_capacity_a = GenRosenfeldTaranzona().calc_heat_capacity\
            ( T_a, eos_d, bcoef_a=bcoef_a )
        return heat_capacity_a

    def calc_entropy( self, V_a, T_a, eos_d ):
        """
        Entropy depends on whether reference Compress Path is isotherm or adiabat
        """

        # from IPython import embed; embed(); import ipdb; ipdb.set_trace()
        V0, S0 = core.get_params( ['V0','S0'], eos_d )
        T0, = core.get_params( ['T0'], eos_d )

        bcoef_a = self.calc_bcoef( V_a, eos_d )
        gamma_mod = eos_d['modtype_d']['GammaMod']

        Tref_a = gamma_mod.temp( V_a, T0, eos_d )

        dS_heat_a = GenRosenfeldTaranzona().calc_entropy_heat( T_a, eos_d,
                                                              Tref=Tref_a,
                                                              bcoef_a=bcoef_a )

        dS_compress_a = self.calc_entropy_compress( V_a, eos_d )
        S_a = S0 + dS_heat_a + dS_compress_a
        return S_a

    def calc_gamma( self, V_a, T_a, eos_d ):
        gamma_mod = eos_d['modtype_d']['GammaMod']
        T0, = core.get_params( ['T0'], eos_d )

        gamma_0S_a = gamma_mod.gamma(V_a,eos_d)
        T_0S_a = gamma_mod.temp(V_a,T0,eos_d)

        bcoef_a = self.calc_bcoef(V_a,eos_d)
        bcoef_der1_a = self.calc_bcoef(V_a,eos_d,deriv=1)

        CV_a = self.calc_heat_capacity( V_a, T_a, eos_d )
        CV_0S_a = self.calc_heat_capacity( V_a, T_0S_a, eos_d )

        dS_pot_a = GenRosenfeldTaranzona().calc_entropy_pot( T_a, eos_d,
                                                            Tref=T_0S_a,
                                                            bcoef_a=bcoef_a )
        gamma_a = gamma_0S_a*(CV_0S_a/CV_a) \
            + V_a*(bcoef_der1_a/bcoef_a)*(dS_pot_a/CV_a)
        return gamma_a

    #========================
    #  Empirical Coefficient Model
    #========================

    def calc_coef_poly( self, V_a, coef_key, eos_d, deriv=0 ):
        poly_coef_a = core.get_array_params( coef_key, eos_d )
        # coef_a = np.polyval(poly_coef_a[::-1], V_a)
        if deriv==0:
            coef_a = np.polyval(poly_coef_a, V_a)
        else:
            dpoly_coef_a = np.polyder(poly_coef_a,deriv)
            coef_a = np.polyval(dpoly_coef_a, V_a)

        return coef_a

    def calc_coef_logpoly( self, V_a, coef_key, eos_d, deriv=0 ):
        V0, = core.get_params( ['V0'], eos_d )
        logpoly_coef_a = core.get_array_params( coef_key, eos_d )
        # coef_a = np.polyval(logpoly_coef_a[::-1], np.log(V_a/V0))
        if deriv==0:
            coef_a = np.polyval(logpoly_coef_a, np.log(V_a/V0))
        else:
            dlogpoly_coef_a = np.polyder(logpoly_coef_a,deriv)
            coef_a = V_a**(-deriv)*np.polyval(dlogpoly_coef_a, np.log(V_a/V0))

        return coef_a

    def calc_coef_polynorm( self, V_a, coef_key, eos_d, deriv=0 ):
        V0, = core.get_params( ['V0'], eos_d )
        polynorm_coef_a = core.get_array_params( coef_key, eos_d )
        # coef_a = np.polyval(polynorm_coef_a[::-1], V_a/V0-1.0 )
        if deriv==0:
            coef_a = np.polyval(polynorm_coef_a, V_a/V0-1.0 )
        else:
            dpolynorm_coef_a = np.polyder(polynorm_coef_a,deriv)
            coef_a = V0**(-deriv)*np.polyval(dpolynorm_coef_a, V_a)
        return coef_a

    #========================
    #  Ref Temp path
    #========================

    def calc_temp_path_T0( self, V_a, eos_d ):
        T0, = core.get_params( ['T0'], eos_d )
        return T0

    def calc_temp_path_S0( self, V_a, eos_d ):
        T0, = core.get_params( ['T0'], eos_d )
        gamma_mod, = core.get_modtypes( ['GammaMod'], eos_d )

        Tref_a = gamma_mod.temp( V_a, T0, eos_d )
        return Tref_a

    def calc_temp_path_abszero( self, V_a, eos_d ):
        Tref = 0.0
        return Tref
#====================================================================
class RosenfeldTaranzonaAdiabat(RosenfeldTaranzonaCompress):
    def __init__( self, coef_kind='logpoly' ):
        temp_path_kind = 'S0'
        acoef_fun= self.calc_energy_adiabat_ref
        self.set_temp_path( temp_path_kind )
        self.set_empirical_coef( coef_kind, acoef_fun=acoef_fun )
        pass

    def calc_entropy_compress( self, V_a, eos_d ):
        dS_a = np.zeros(V_a.shape)
        return dS_a

    def calc_entropy_isotherm( self, V_a, eos_d ):
        kB, = core.get_consts( ['kboltz'], eos_d )
        T0, mexp = core.get_params( ['T0','mexp'], eos_d )

        T0S_a = self.calc_temp_path( V_a, eos_d )

        bV_a = self.calc_bcoef( V_a, eos_d )

        dS_T0_a = -1.0*mexp/(mexp-1)*bV_a/T0*((T0S_a/T0)**(mexp-1.0)-1.0)\
            -3./2*kB*np.log(T0S_a/T0)

        return dS_T0_a

    def calc_energy_adiabat_ref( self, V_a, eos_d ):

        kB, = core.get_consts( ['kboltz'], eos_d )
        T0, mexp = core.get_params( ['T0','mexp'], eos_d )

        compress_path_mod = eos_d['modtype_d']['CompressPathMod']

        energy_isotherm_a = compress_path_mod.energy(V_a,eos_d)

        if compress_path_mod.path_const == 'S':
            # compress path mod directly describes adiabat
            # internal energy is given directly by integral of compress path
            E0S_a = energy_isotherm_a

        else:
            # compress path mod describes isotherm
            # free energy is given by integral of compress path
            # adjustments are needed to obtain internal energy
            free_energy_isotherm_a = energy_isotherm_a

            T0S_a = self.calc_temp_path( V_a, eos_d )
            bV_a = self.calc_bcoef( V_a, eos_d )

            dS_T0_a = self.calc_entropy_isotherm( V_a, eos_d )

            energy_isotherm_a = free_energy_isotherm_a + T0*dS_T0_a
            E0S_a = energy_isotherm_a +bV_a*((T0S_a/T0)**mexp-1)\
                +3./2*kB*(T0S_a-T0)

        return E0S_a
#====================================================================
class RosenfeldTaranzonaIsotherm(RosenfeldTaranzonaCompress):
    def __init__( self, coef_kind='logpoly' ):
        temp_path_kind = 'T0'
        acoef_fun = None
        self.set_temp_path( temp_path_kind )
        self.set_empirical_coef( coef_kind, acoef_fun=acoef_fun )

        pass

    def calc_entropy_compress( self, V_a, eos_d ):
        dS_T0_a = self.calc_entropy_isotherm( V_a, eos_d )
        return dS_T0_a

    def calc_entropy_isotherm( self, V_a, eos_d ):
        kB, = core.get_consts( ['kboltz'], eos_d )
        T0, mexp = core.get_params( ['T0','mexp'], eos_d )

        compress_path_mod = eos_d['modtype_d']['CompressPathMod']

        free_energy_isotherm_a = compress_path_mod.energy(V_a,eos_d)


        T0S_a = self.calc_temp_path( V_a, eos_d )

        bV_a = self.calc_bcoef( V_a, eos_d )

        dS_a = -mexp/(mexp-1)*bV_a/T0*((T0S_a/T0)**(mexp-1)-1)\
            -3./2*kB*np.log(T0S_a/T0)

        return dS_a
#====================================================================
class MieGrun(ThermalMod):
    """
    Mie-Gruneisen Equation of State Model
    (requires extension to define thermal energy model)
    """
    __metaclass__ = ABCMeta

    def press( self, V_a, T_a, eos_d ):
        V_a, T_a = fill_array( V_a, T_a )

        PV_ratio, = core.get_consts( ['PV_ratio'], eos_d )
        gamma_mod, = core.get_modtypes( ['GammaMod'], eos_d )

        # Needed functions
        energy_therm_a = self.calc_energy( V_a, T_a, eos_d )
        gamma_a = gamma_mod.gamma( V_a, eos_d )

        press_therm_a = PV_ratio*(gamma_a/V_a)*energy_therm_a
        return press_therm_a

    @abstractmethod
    def calc_energy( self, V_a, T_a, eos_d ):
        """Returns Thermal Component of Energy."""
#====================================================================
class MieGrunDebye(MieGrun):
    def __init__( self ):
       super(MieGrunDebye, self).__init__()
       path_const='V'
       self.path_const = path_const

    def calc_energy( self, V_a, T_a, eos_d ):
        '''
        Thermal Energy for Debye model

        Relies on reference profile properties stored in eos_d defined by:
        * debye_temp_f( V_a, T_a )
        * ref_temp_f( V_a, T_a )
        '''
        V_a, T_a = fill_array( V_a, T_a )

        # NOTE: T0 refers to temp on ref adiabat evaluated at V0
        Cvmax, T0, thetaR = core.get_params( ['Cvmax','T0','thetaR'], eos_d )
        TS_ratio, = core.get_consts( ['TS_ratio'], eos_d )
        gamma_mod, = core.get_modtypes( ['GammaMod'], eos_d )

        theta_a = gamma_mod.temp( V_a, thetaR, eos_d )
        # Tref_a = gamma_mod.temp( V_a, T0, eos_d )
        Tref_a = self.calc_temp_path(V_a,eos_d)

        # print theta_a

        ######################
        # NOTE: Some weird issue with TS_ratio!!!
        ######################
        # energy_therm_a = (Cvmax/TS_ratio)*(
        #     + T_a*self.debye_func( theta_a/T_a )
        #     - Tref_a*self.debye_func( theta_a/Tref_a ) )
        energy_therm_a = (Cvmax)*(
            + T_a*self.debye_func( theta_a/T_a )
            - Tref_a*self.debye_func( theta_a/Tref_a ) )

        return energy_therm_a

    def calc_entropy( self, V_a, T_a, eos_d ):
        V_a, T_a = fill_array( V_a, T_a )

        Cvmax, thetaR = core.get_params( ['Cvmax','thetaR'], eos_d )
        TS_ratio, = core.get_consts( ['TS_ratio'], eos_d )
        gamma_mod, = core.get_modtypes( ['GammaMod'], eos_d )

        theta_a = gamma_mod.temp( V_a, thetaR, eos_d )
        x_a = theta_a/T_a

        # entropy_a = Cvmax*Cv_const/3. \
            #     *(4*debye_func( x_a )-3*np.log( 1-np.exp( -x_a ) ) )

        # TS_ratio????
        # entropy_a = 1.0/3*(Cvmax/TS_ratio) \
        #     *(4*self.debye_func( x_a )-3*np.log( np.exp( x_a ) - 1 ) + 3*x_a)
        entropy_a = 1.0/3*(Cvmax) \
            *(4*self.debye_func( x_a )-3*np.log( np.exp( x_a ) - 1 ) + 3*x_a)

        return entropy_a

    def calc_heat_capacity( self, V_a, T_a, eos_d ):
        V_a, T_a = fill_array( V_a, T_a )

        Cvmax, thetaR = core.get_params( ['Cvmax','thetaR'], eos_d )
        TS_ratio, = core.get_consts( ['TS_ratio'], eos_d )
        gamma_mod, = core.get_modtypes( ['GammaMod'], eos_d )

        theta_a = gamma_mod.temp( V_a, thetaR, eos_d )

        # The reference adiabat terms in the internal energy are temperature
        # independent, and thus play no role in heat capacity
        x_a = theta_a/T_a
        # heat_capacity_a = (Cvmax/TS_ratio)*\
        #     (4*self.debye_func( x_a )-3*x_a/(np.exp(x_a)-1))

        ######################
        # NOTE: Some weird issue with TS_ratio!!!
        ######################
        heat_capacity_a = (Cvmax)*\
            (4*self.debye_func( x_a )-3*x_a/(np.exp(x_a)-1))

        return heat_capacity_a

    def debye_func( self, x_a ):
        """
        Return debye integral value

        - calculation done using interpolation in a lookup table
        - interpolation done in log-space where behavior is close to linear
        - linear extrapolation is implemented manually
        """

        if np.isscalar( x_a ):
            assert x_a >= 0, 'x_a values must be greater than zero.'
        else:
            assert all( x_a >= 0 ), 'x_a values must be greater than zero.'
            # Lookup table
            # interpolate in log space where behavior is nearly linear
            debyex_a = np.array( [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                  1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8,
                                  3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0,
                                  5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0] )
            debyelogf_a = np.array( [ 0.0, -0.03770187, -0.07580279, -0.11429475,
                                     -0.15316866, -0.19241674, -0.2320279 , -0.27199378,
                                     -0.31230405, -0.35294619, -0.39390815, -0.43518026,
                                     -0.47674953, -0.51860413, -0.56072866, -0.64573892,
                                     -0.73167389, -0.81841793, -0.90586032, -0.99388207,
                                     -1.08236598, -1.17119911, -1.26026101, -1.34944183,
                                     -1.43863241, -1.52771969, -1.61660856, -1.70519469,
                                     -1.79338479, -1.88108917, -1.96822938, -2.05471771,
                                     -2.14049175, -2.35134476, -2.55643273, -2.75507892,
                                     -2.94682783, -3.13143746, -3.30880053, -3.47894273,
                                     -3.64199587, -3.79820337, -3.94785746] )
            # Create interpolation function
            logdeb_func = interpolate.interp1d( debyex_a, debyelogf_a,
                                               kind='cubic',
                                               bounds_error=False,
                                               fill_value=np.nan )
            logfval_a = logdeb_func( x_a )

            # Check for extrapolated values indicated by NaN
            #   - replace with linear extrapolation
            logfextrap_a = debyelogf_a[-1] + (x_a - debyex_a[-1]) \
                *(debyelogf_a[-1]-debyelogf_a[-2])\
                /(debyex_a[-1]-debyex_a[-2])
            logfval_a = np.where( x_a > debyex_a[-1], logfextrap_a,
                                 logfval_a )
            # exponentiate to get integral value
            return np.exp( logfval_a )

    def calc_temp_path( self, V_a, eos_d ):
        T0, = core.get_params( ['T0'], eos_d )

        gamma_mod, = core.get_modtypes( ['GammaMod'], eos_d )
        Tref_a = gamma_mod.temp( V_a, T0, eos_d )

        return Tref_a

    def calc_free_energy( self, V_a, T_a, eos_d ):
        E_tot = self.calc_energy( V_a, T_a, eos_d )
        S_tot = self.calc_entropy( V_a, T_a, eos_d )
        F_tot = E_tot - T_a*S_tot

        return F_tot

    def calc_gamma( self, V_a, T_a, eos_d ):
        gamma_mod = eos_d['modtype_d']['GammaMod']
        T0, = core.get_params( ['T0'], eos_d )

        gamma_0S_a = gamma_mod.gamma(V_a,eos_d)
        return gamma_0S_a
#====================================================================
