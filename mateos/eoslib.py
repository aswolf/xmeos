import numpy as np
import scipy as sp
from abc import ABCMeta, abstractmethod
import scipy.interpolate as interpolate

#====================================================================
#     EOSMod: Equation of State Model
#      eoslib- library of common equation of state models
#====================================================================

#====================================================================
# SECT 0: Admin functions
#====================================================================
#====================================================================
def init_const( eos_d ):
    eos_d['const_d'] = default_const()
    pass
#====================================================================
def set_const( name_l, val_l, eos_d ):
    if 'const_d' in eos_d.keys():
        const_d = eos_d['const_d']
    else:
        init_const( eos_d )

    for name, val in zip( name_l, val_l ):
        const_d[name] = val

    pass

#====================================================================
def set_param( name_l, val_l, eos_d ):
    if 'param_d' in eos_d.keys():
        param_d = eos_d['param_d']
    else:
        param_d = {}
        eos_d['param_d'] = param_d

    for name, val in zip( name_l, val_l ):
        param_d[name] = val

    pass

#====================================================================
def set_modtype( name_l, val_l, eos_d ):
    if 'modtype_d' in eos_d.keys():
        modtype_d = eos_d['modtype_d']
    else:
        modtype_d = {}
        eos_d['modtype_d'] = modtype_d

    # Should we verify match?
    for name, val in zip( name_l, val_l ):
        if globals().has_key(name):
            # modtype = globals()[name]
            # modtype_d[name] = modtype()
            modtype_d[name] = val
        else:
            print name + " is not a valid modtype object"

    pass

#====================================================================
def default_const():
    const_d = {}
    const_d['eVperHa'] = 27.211 # eV/Ha
    const_d['JperHa'] = 4.35974434e-18 # J/Ha
    const_d['JperCal'] = 4.184 # J/Cal
    const_d['Nmol'] = 6.0221413e+23 # atoms/mol
    const_d['R'] = 8.314462 # J/K/mol
    const_d['kboltz'] = 8.617332e-5 # eV/K
    const_d['ang3percc'] = 1e24 # ang^3/cm^3

    const_d['PV_ratio'] = 160.2176487 # (GPa*ang^3)/eV
    const_d['TS_ratio'] = const_d['R']/const_d['kboltz'] # (J/mol)/eV

    return const_d

#====================================================================

#def get_eos_func( prop, eos_mod ):
#    func_name = prop + '_' + eos_mod
#
#    if globals().has_key( func_name ):
#        func = globals()[func_name]
#    else:
#        raise ValueError(func_name + " is not a valid function name " +
#                         "within eoslib.py")
#    return func

#====================================================================
# SECT 1: Fitting Routines
#====================================================================
class ModFit(object):

    def __init__( self ):
        pass

    def get_resid_fun( self, eos_fun, eos_d, param_key_a, sys_state_tup,
                      val_a, err_a=1.0):

        # def resid_fun( param_in_a, eos_fun=eos_fun, eos_d=eos_d,
        #               param_key_a=param_key_a, sys_state_tup=sys_state_tup,
        #               val_a=val_a, err_a=err_a ):

        #     # Don't worry about transformations right now
        #     param_a = param_in_a

        #     # set param value in eos_d dict
        #     globals()['set_param']( param_key_a, param_a, eos_d )

        #     # Take advantage of eos model function input format
        #     #   uses tuple expansion for input arguments
        #     mod_val_a = eos_fun( *(sys_state_tup+(eos_d,)) )
        #     resid_a = (mod_val_a - val_a)/err_a
        #     return resid_a

        wrap_eos_fun = self.get_wrap_eos_fun( eos_fun, eos_d, param_key_a )

        def resid_fun( param_in_a, wrap_eos_fun=wrap_eos_fun,
                      sys_state_tup=sys_state_tup,
                      val_a=val_a, err_a=err_a ):

            mod_val_a = wrap_eos_fun( param_in_a, sys_state_tup )
            resid_a = (mod_val_a - val_a)/err_a
            return resid_a

        return resid_fun


    def get_wrap_eos_fun( self, eos_fun, eos_d, param_key_a ):

        def wrap_eos_fun(param_in_a, sys_state_tup, eos_fun=eos_fun,
                         eos_d=eos_d, param_key_a=param_key_a ):

            # Don't worry about transformations right now
            param_a = param_in_a

            # set param value in eos_d dict
            globals()['set_param']( param_key_a, param_a, eos_d )

            # Take advantage of eos model function input format
            #   uses tuple expansion for input arguments
            mod_val_a = eos_fun( *(sys_state_tup+(eos_d,)) )
            return mod_val_a

        return wrap_eos_fun


#====================================================================
# SECT 2: Thermal EOS
#====================================================================


#====================================================================
# SECT N: Code Utility Functions
#====================================================================
def fill_array( var1, var2 ):
    """
    fix fill_array such that it returns two numpy arrays of equal size

    use numpy.full_like

    """
    var1_a = np.asarray( var1 )
    var2_a = np.asarray( var2 )

    if var1_a.shape==():
        var1_a = np.asarray( [var1] )
    if var2_a.shape==():
        var2_a = np.asarray( [var2] )

    # Begin try/except block to handle all cases for filling an array
    while True:
        try:
            assert var1_a.shape == var2_a.shape
            break
        except: pass
        try:
            var1_a = np.full_like( var2_a, var1_a )
            break
        except: pass
        try:
            var2_a = np.full_like( var1_a, var2_a )
            break
        except: pass

        # If none of the cases properly handle it, throw error
        assert False, 'var1 and var2 must both be equal shape or size=1'

    return var1_a, var2_a
#====================================================================

#====================================================================
#           EOS  Objects
#====================================================================
class EosMod(object):
    """
    Abstract Equation of State class for reference thermodynamic path
    """
    __metaclass__ = ABCMeta

    def __init__( self ):
        pass

    def get_params( self, name_l, eos_d ):
        """
        Retrieve list of desired params stored in eos_d['param_d']
        """
        param_d = eos_d['param_d']
        param_l = []
        for name in name_l:
            param_l.append( param_d[name] )

        return tuple( param_l )

    def get_consts( self, name_l, eos_d ):
        """
        Retrieve list of desired consts stored in eos_d['const_d']
        """
        const_d = eos_d['const_d']
        const_l = []
        for name in name_l:
            const_l.append( const_d[name] )

        return tuple( const_l )

    def get_modtypes( self, name_l, eos_d ):
        """
        Retrieve list of desired model types stored in eos_d['modtype_d']
        """
        modtype_d = eos_d['modtype_d']
        modtype_l = []
        for name in name_l:
            modtype_l.append( modtype_d[name] )

        return tuple( modtype_l )

#====================================================================
#====================================================================
class CompressMod(EosMod):
    """
    Abstract Equation of State class for reference Compression curves
    """
    __metaclass__ = ABCMeta

    path_opts = ['T','S']
    def __init__( self, path_const='T', level_const=300 ):
        assert path_const in self.path_opts, path_const + ' is not a valid ' + \
            'path const. You must select one of: ' + path_opts
        self.path_const = path_const
        self.level_const = level_const
        pass

    def get_path_const( self ):
        return self.path_const

    def get_level_const( self ):
        return self.level_const

    # Must always impliment press function
    @abstractmethod
    def press( self, V_a, eos_d ):
        """Returns Press variation along compression curve."""


    # Standard methods, but not required for all applications, so
    # object that inherits CompressMod must override method if function needed
    def energy( self, V_a, eos_d ):
        """Returns Energy along compression curve."""
        raise NotImplementedError("'energy' function not implimented for this model")

    def bulk_mod( self, V_a, eos_d ):
        """Returns Bulk Modulus variation along compression curve."""
        raise NotImplementedError("'bulk_mod' function not implimented for this model")

    def bulk_mod_deriv( self, V_a, eos_d ):
        """Returns Bulk Modulus Deriv (K') variation along compression curve."""
        raise NotImplementedError("'bulk_mod_deriv' function not implimented for this model")

#====================================================================
class ThermMod(EosMod):
    """
    Abstract Equation of State class for thermally induced variations
    """
    __metaclass__ = ABCMeta

    path_opts = ['P','V']
    def __init__( self, path_const='V', level_const=np.nan ):
        assert path_const in self.path_opts, path_const + ' is not a valid ' + \
            'path const. You must select one of: ' + path_opts
        self.path_const = path_const
        self.level_const = level_const
        pass


    @abstractmethod
    def press( self, V_a, T_a, eos_d ):
        """Returns Press variation along compression curve."""

    # Standard methods, but not required for all applications, so
    # object that inherits CompressMod must override method if function needed
    def energy( self, V_a, T_a, eos_d ):
        """Returns Thermal Component of Energy."""
        raise NotImplementedError("'energy' function not implimented for this model")

    def entropy( self, V_a, T_a, eos_d ):
        """Returns Entropy."""
        raise NotImplementedError("'entropy' function not implimented for this model")

    def heat_capacity( self, V_a, T_a, eos_d ):
        """Returns Heat Capacity."""
        raise NotImplementedError("'heat_capacity' function not implimented for this model")

#====================================================================
class GammaMod(EosMod):
    """
    Abstract Equation of State class for Gruneisen Parameter curves
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def gamma( self, V_a, eos_d ):
        """Returns Gruneisen Param (gamma) variation due to compression."""

    @abstractmethod
    def temp( self, V_a, T0, eos_d ):
        """Returns Gruneisen Param (gamma) variation due to compression."""
#====================================================================
class FullMod(EosMod):
    """
    Abstract Equation of State class for Full Model (combines all EOS terms)
    """
    __metaclass__ = ABCMeta


    @abstractmethod
    def press( self, V_a, T_a, eos_d ):
        """Returns Press variation along compression curve."""

    # Standard methods, but not required for all applications, so
    # object that inherits CompressMod must override method if function needed
    def energy( self, V_a, T_a, eos_d ):
        """Returns Thermal Component of Energy."""
        raise NotImplementedError("'energy' function not implimented for this model")

    def therm_exp( self, V_a, T_a, eos_d ):
        """Returns Thermal Expansion."""
        raise NotImplementedError("'bulk_mod' function not implimented for this model")

    def bulk_mod( self, V_a, T_a, eos_d ):
        """Returns Bulk Modulus variation along compression curve."""
        raise NotImplementedError("'bulk_mod' function not implimented for this model")

#====================================================================

#====================================================================
class BirchMurn3(CompressMod):
    def press( self, V_a, eos_d ):
        V0, K0, KP0 = self.get_params( ['V0','K0','KP0'], eos_d )

        vratio_a = V_a/V0

        press_a = 3.0/2*K0 * (vratio_a**(-7.0/3) - vratio_a**(-5.0/3)) * \
            (1 + 3.0/4*(KP0-4)*(vratio_a**(-2.0/3)-1))

        return press_a

    def energy( self, V_a, eos_d ):
        V0, K0, KP0, E0 = self.get_params( ['V0','K0','KP0','E0'], eos_d )
        PV_ratio, = self.get_consts( ['PV_ratio'], eos_d )

        vratio_a = V_a/V0

        fstrain_a = 0.5*(vratio_a**(-2.0/3) - 1)

        energy_a = E0 + 9.0/2*(V0*K0/PV_ratio)*\
            ( KP0*fstrain_a**3 + fstrain_a**2*(1-4*fstrain_a) )

        return energy_a

#====================================================================
class GammaPowLaw(GammaMod):
    def __init__( self ):
        pass

    def gamma( self, V_a, eos_d ):
        # OLD version fixed to zero-press ref volume
        # V0, gamma0, q = self.get_params( ['V0','gamma0','q'], eos_d )
        # gamma_a = gamma0 *(V_a/V0)**q

        # generalized version
        VR, gammaR, q = self.get_params( ['VR','gammaR','q'], eos_d )
        gamma_a = gammaR *(V_a/VR)**q

        return gamma_a

    def temp( self, V_a, TR, eos_d ):
        """
        Return temperature for debye model
        V_a: sample volume array
        TR: temperature at V=VR
        """
        # OLD version fixed to zero-press ref volume
        # V0, gamma0, q = self.get_params( ['V0','gamma0','q'], eos_d )
        # gamma_a = self.gamma( V_a, eos_d )
        # T_a = T0*np.exp( -(gamma_a - gamma0)/q )

        # OLD version fixed to zero-press ref volume
        VR, gammaR, q = self.get_params( ['VR','gammaR','q'], eos_d )
        gamma_a = self.gamma( V_a, eos_d )
        T_a = TR*np.exp( -(gamma_a - gammaR)/q )

        return T_a

#====================================================================
class MieGrun(ThermMod):
    """
    Mie-Gruneisen Equation of State Model
    (requires extension to define thermal energy model)
    """
    __metaclass__ = ABCMeta

    def press( self, V_a, T_a, eos_d ):
        V_a, T_a = fill_array( V_a, T_a )

        PV_ratio, = self.get_consts( ['PV_ratio'], eos_d )
        gamma_mod, = self.get_modtypes( ['GammaMod'], eos_d )

        # Needed functions
        energy_therm_a = self.energy( V_a, T_a, eos_d )
        gamma_a = gamma_mod.gamma( V_a, eos_d )

        press_therm_a = PV_ratio*(gamma_a/V_a)*energy_therm_a
        return press_therm_a

    @abstractmethod
    def energy( self, V_a, T_a, eos_d ):
        """Returns Thermal Component of Energy."""

#====================================================================
#      get_ref_temp()
#         T0
#====================================================================
class MieGrunDebye(MieGrun):
    def __init__( self ):
       super(MieGrunDebye, self).__init__( path_const='V' )

    def energy( self, V_a, T_a, eos_d ):
        '''
        Thermal Energy for Debye model

        Relies on reference profile properties stored in eos_d defined by:
        * debye_temp_f( V_a, T_a )
        * ref_temp_f( V_a, T_a )
        '''
        V_a, T_a = fill_array( V_a, T_a )

        # NOTE: T0 refers to temp on ref adiabat evaluated at V0
        Cvmax, T0, thetaR = self.get_params( ['Cvmax','T0','thetaR'], eos_d )
        TS_ratio, = self.get_consts( ['TS_ratio'], eos_d )
        gamma_mod, = self.get_modtypes( ['GammaMod'], eos_d )

        theta_a = gamma_mod.temp( V_a, thetaR, eos_d )
        Tref_a = gamma_mod.temp( V_a, T0, eos_d )
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

    def entropy( self, V_a, T_a, eos_d ):
        V_a, T_a = fill_array( V_a, T_a )

        Cvmax, thetaR = self.get_params( ['Cvmax','thetaR'], eos_d )
        TS_ratio, = self.get_consts( ['TS_ratio'], eos_d )
        gamma_mod, = self.get_modtypes( ['GammaMod'], eos_d )

        theta_a = gamma_mod.temp( V_a, thetaR, eos_d )
        x_a = theta_a/T_a

        # entropy_a = Cvmax*Cv_const/3. \
            #     *(4*debye_func( x_a )-3*np.log( 1-np.exp( -x_a ) ) )
        entropy_a = 1.0/3*(Cvmax/TS_ratio) \
            *(4*debye_func( x_a )-3*np.log( np.exp( x_a ) - 1 ) + 3*x_a)

        return entropy_a

    def heat_capacity( self, V_a, T_a, eos_d ):
        V_a, T_a = fill_array( V_a, T_a )

        Cvmax, thetaR = self.get_params( ['Cvmax','thetaR'], eos_d )
        TS_ratio, = self.get_consts( ['TS_ratio'], eos_d )
        gamma_mod, = self.get_modtypes( ['GammaMod'], eos_d )

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

        #

#====================================================================
class ThermPressMod(FullMod):

    def press( self, V_a, T_a, eos_d ):
        """Returns Press variation along compression curve."""
        V_a, T_a = fill_array( V_a, T_a )
        compress_mod, therm_mod = self.get_modtypes( ['CompressMod', 'ThermMod'],
                                               eos_d )
        press_a = compress_mod.press( V_a, eos_d ) \
            + therm_mod.press( V_a, T_a, eos_d )
        return press_a

    def energy( self, V_a, T_a, eos_d ):
        """Returns Thermal Component of Energy."""
        V_a, T_a = fill_array( V_a, T_a )
        compress_mod, therm_mod = self.get_modtypes( ['CompressMod', 'ThermMod'],
                                               eos_d )
        energy_a = compress_mod.energy( V_a, eos_d ) \
            + therm_mod.energy( V_a, T_a, eos_d )
        return energy_a

    # Compute with finite diff
    # def therm_exp( V_a, T_a, eos_d ):
    #     """Returns Thermal Expansion."""

    # def bulk_mod( V_a, T_a, eos_d ):
    #     """Returns Bulk Modulus variation along compression curve."""

#====================================================================
