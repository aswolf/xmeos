import numpy as np
import scipy as sp

#====================================================================
#     EOSMod: Equation of State Model
#      eoslib- library of common equation of state models
#====================================================================

    # mgd_ref_model()
    # ref_model: BM3, BM4, VIN, LOG
    # ref_path: S, T, (P),
    # gamma_model: powlaw, shift_powlaw
    #
    #
    # set_therm_model( 'MGD', eos_d )
    # set_ref_model( 'MGD', FixS=True, eos_d )
    # set_ref_model( 'MGD', FixT=True, eos_d )
    #
    # energy_therm_f = eos_d['func_d']['energy_therm_f']
    # gamma_f = eos_d['func_d']['gamma_ref_f']
    # temp_debye_f = eos_d['func_d']['temp_scale_ref_f']
    # temp_ref_f = eos_d['func_d']['temp_ref_f']
#====================================================================
# SECT 0: Reference Compression Profiles
#====================================================================
#====================================================================
def set_param( name_l, val_l, eos_d ):
    if 'param_d' in eos_d.keys():
        param_d = eos_d['param_d']
    else:
        param_d = {}
        eos_d['param_d'] = param_d

    for name, val in zip( name_l, val_l ):
        param_d[name] = val

#====================================================================
def set_const( name_l, val_l, eos_d ):
    if 'const_d' in eos_d.keys():
        const_d = eos_d['const_d']
    else:
        const_d = init_const()
        eos_d['const_d'] = const_d

    for name, val in zip( name_l, val_l ):
        const_d[name] = val

#====================================================================
def set_func( name_l, val_l, eos_d ):
    if 'param_d' in eos_d.keys():
        param_d = eos_d['param_d']
    else:
        param_d = {}
        eos_d['param_d'] = param_d

    for name, val in zip( name_l, val_l ):
        param_d[name] = val

#====================================================================
def init_const():
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

#====================================================================
# SECT 1: Reference Compression Profiles
#====================================================================
#====================================================================
# BM3- Birch Murnaghan 3rd Order
#====================================================================
def press_BM3( V_a, eos_d ):
    # Retrieve parameter values
    param_d = eos_d['param_d']
    V0 = param_d['V0']
    K0 = param_d['K0']
    KP0 = param_d['KP0']

    vratio_a = V_a/V0

    press_a = 3.0/2*K0 * (vratio_a**(-7.0/3) - vratio_a**(-5.0/3)) * \
        (1 + 3.0/4*(KP0-4)*(vratio_a**(-2.0/3)-1))

    return press_a

#====================================================================
def energy_BM3( V_a, eos_d ):
    # Retrieve parameter values
    param_d = eos_d['param_d']
    V0 = param_d['V0']
    K0 = param_d['K0']
    KP0 = param_d['KP0']
    E0 = param_d['E0']

    # Retrieve unit conversion ratio
    PV_ratio = eos_d['const_d']['PV_ratio']

    vratio_a = V_a/V0

    fstrain_a = 0.5*(vratio_a**(-2.0/3) - 1)

    energy_a = E0 + 9.0/2*(V0*K0/PV_ratio)*\
        ( KP0*fstrain_a**3 + fstrain_a**2*(1-4*fstrain_a) )

    return energy_a

#====================================================================

#====================================================================
# SECT 2: Thermal EOS
#====================================================================
#====================================================================
# Gruneisen Model
#====================================================================
def press_mie_grun( V_a, T_a, eos_d ):
    V_a, T_a = fill_array( V_a, T_a )

    # units const
    PV_ratio = eos_d['const_d']['PV_ratio']
    # Needed functions
    energy_therm_f = eos_d['func_d']['energy_therm_f']
    gamma_f = eos_d['func_d']['gamma_ref_f']

    energy_therm_a = energy_therm_f( V_a, T_a, eos_d )
    gamma_a = gamma_f( V_a, func_d )

    press_therm_a = PV_ratio*(gamma_a/V_a)*energy_therm_a
    return press_therm_a

#====================================================================
def gamma_powlaw( V_a, eos_d ):
    # get parameter values
    param_d = eos_d['param_d']
    V0 = param_d['V0']
    gamma0 = param_d['gamma0']
    q = param_d['q']

    gamma_a = gamma0 *(V_a/V0)**q

    return gamma_a

#====================================================================
def temp_powlaw( V_a, T0, eos_d ):
    """
    Return temperature for debye model
    V_a: sample volume array
    T0: temperature at V=V0
    """
    # get parameter values
    param_d = eos_d['param_d']
    V0 = param_d['V0']
    gamma0 = param_d['gamma0']
    q = param_d['q']

    gamma_a = gamma_powlaw( V_a, eos_d )
    T_a = T0*np.exp( -(gamma_a - gamma0)/q )
    return T_a

#====================================================================
#====================================================================
# Debye Model
#====================================================================
def energy_debye( V_a, T_a, eos_d ):
    '''
    Thermal Energy for Debye model

    Relies on reference profile properties stored in eos_d defined by:
        * debye_temp_f( V_a, T_a )
        * ref_temp_f( V_a, T_a )

    '''

    V_a, T_a = fill_array( V_a, T_a )

    # get parameter values
    Cvmax = eos_d['param_d']['Cvmax']
    TS_ratio = eos_d['const_d']['TS_ratio']

    # get eos funcs
    temp_debye_f = eos_d['func_d']['temp_scale_ref_f']
    temp_ref_f = eos_d['func_d']['temp_ref_f']


    theta_a = temp_debye_f( V_a, eos_d )
    Tref_a = temp_ref_f( V_a, eos_d )

    energy_therm_a = (Cvmax/TS_ratio) \
        *( T_a*debye_func( theta_a/T_a ) - Tref_a*debye_func( theta_a/Tref_a ) )

    return energy_therm_a

#====================================================================
def entropy_debye( V_a, T_a, eos_d ):
    V_a, T_a = fill_array( V_a, T_a )

    # get parameter values
    param_d = eos_d['param_d']
    T0 = param_d['T0']
    theta0 = param_d['theta0']
    Cvmax = param_d['Cvmax']

    TS_ratio = eos_d['const_d']['TS_ratio']
    theta_f = eos_d['func_d']['temp_scale_ref_f']

    theta_a = theta_f( V_a, eos_d )
    x_a = theta_a/T_a

    # entropy_a = Cvmax*Cv_const/3. \
    #     *(4*debye_func( x_a )-3*np.log( 1-np.exp( -x_a ) ) )
    entropy_a = 1.0/3*(Cvmax/TS_ratio) \
        *(4*debye_func( x_a )-3*np.log( np.exp( x_a ) - 1 ) + 3*x_a)

    return entropy_a

#====================================================================
def heat_capacity_V_debye( V_a, T_a, eos_d ):
    V_a, T_a = fill_array( V_a, T_a )

    # get parameter values
    Cvmax = eos_d['param_d']['Cvmax']
    TS_ratio = eos_d['const_d']['TS_ratio']

    # get funcs
    temp_debye_f = eos_d['func_d']['temp_scale_ref_f']

    theta_a = temp_debye_f( V_a, eos_d )

    # The reference adiabat terms in the internal energy are temperature
    # independent, and thus play no role in heat capacity
    x_a = theta_a/T_a
    heat_capacity_a = (Cvmax/TS_ratio)*(4*debye_func( x_a )-3*x_a/(np.exp(x_a)-1))

    return heat_capacity_a

#====================================================================
def debye_func( x_a ):
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
    logdeb_func = interpolate.interp1d( debyex_a, debyelogf_a, kind='cubic',
                           bounds_error=False, fill_value=np.nan )
    logfval_a = logdeb_func( x_a )

    # Check for extrapolated values indicated by NaN
    #   - replace with linear extrapolation
    logfextrap_a = debyelogf_a[-1] + (x_a - debyex_a[-1]) \
        *(debyelogf_a[-1]-debyelogf_a[-2])/(debyex_a[-1]-debyex_a[-2])
    logfval_a = np.where( x_a > debyex_a[-1], logfextrap_a, logfval_a )
    # exponentiate to get integral value
    return np.exp( logfval_a )

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
