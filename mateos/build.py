import numpy as np
import scipy as sp
import eoslib
import matplotlib.pyplot as plt

#====================================================================
#     EOSMod: Equation of State Model
#      build- interface for building complete eos models
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

def mie_grun_debye_model( ref_model='BM3', ref_path_const='T',
                         gamma_model='powlaw'  ):

    # Set model type
    #  Mie Gruneisen Debye thermal pressure model  (with power law gamma)

    # Initialize EOS dict
    eos_d = {}
    eoslib.set_const( [], [], eos_d )
    thermpress_mod = eoslib.ThermPressMod()
    compress_mod = eoslib.BirchMurn3(path_const='S', level_const=2536.4 )
    therm_mod = eoslib.MieGrunDebye()
    gamma_mod = eoslib.GammaPowLaw()
    eoslib.set_modtype( ['ThermPressMod','CompressMod', 'ThermMod','GammaMod'],
                       [thermpress_mod,compress_mod,therm_mod,gamma_mod], eos_d )

    # Set initial model parameter values
    V0 = 11.0 # Ang^3 / atom
    K0 = 100.0 # GPa
    KP0 = 4.0 # 1
    T0 = 1600.0 # K
    gamma0 = 1.0 # 1
    q = 0.8 # 1
    theta0 = 1000.0 # K
    Cvmax = 3*eos_d['const_d']['kboltz'] # eV/K
    param_key_a = ['V0','K0','KP0','T0','gamma0','q','theta0','Cvmax']
    param0_a = np.array([ V0, K0, KP0, T0, gamma0, q, theta0, Cvmax ])
    eoslib.set_param( param_key_a, param0_a, eos_d )


    def press_resid( param_a, eos_d, param_key_a=param_key_a,
                    V_a=V_a, T_a=T_a, P_a=P_a, Perr_a=1.0):
        eoslib.set_param( param_key_a, param_a, eos_d )
        Pmod_a = eos_d['modtype_d']['ThermPressMod'].press( V_a, 300, eos_d)
        resid_a = (Pmod_a - P_a)/Perr_a

        return resid_a


    ThermPressMod




    eos_d['mod
    eoslib.

    eos_d['modtype_d']['GammaMod'].gamma(V_a,eos_d)


    V_fac_a = np.linspace(.6,1.1,1001)
    V_a = V_fac_a*eos_d['param_d']['V0']
    compressmod.press(V_a,eos_d)


    plt.ion()
    plt.figure()
    plt.plot( V_a, compressmod.press(V_a,eos_d),'r-' )


    # Set ref model functions
    press_ref_f = eoslib.get_eos_func( 'press', ref_model )
    energy_ref_f = eoslib.get_eos_func( 'energy', ref_model )

    eoslib.set_param( [], [], eos_d )


    press_therm_f = press_mie_grun

    energy_ref_f =
    energy_therm_f = energy_debye
    press_ref_f =

    # Define adiabatic ref compression curve
    # P( V, param_d ), E( V, param_d )
    eos_solid_d['press_ref_f'] = press_BM3
    eos_solid_d['energy_ref_f'] = energy_BM3

    eos_solid_d['gamma_ref_f'] = gamma_powlaw
    eos_solid_d['temp_ref_f'] = lambda V_a, eos_d: \
        T0*temp_factor_powlaw( V_a, eos_d )
    eos_solid_d['temp_scale_ref_f'] = lambda V_a, eos_d:\
        theta0*temp_factor_powlaw( V_a, eos_d )

    # eos_solid_d['entropy_ref_f'] = entropy_f

    # Define gruneisen parameter evolution for ref

    # Define total eos surface (total P, E, S)
    # P( V_a, T_a, eos_d ), E( V_a, T_a, eos_d ), T( V, eos_d )

    # For Debye model:
    # gamma and temp_scale (debye temp) depend only on vol
    #  * unchanged from gamma_ref and temp_scale_ref
    # eos_solid_d['gamma_f'] = lambda V_a, T_a, eos_d:
    #     eos_solid_d['gamma_ref_f']( V_a, eos_d )
    # eos_solid_d['temp_scale_f'] = lambda V_a, T_a, eos_d:
    #     eos_solid_d['temp_scale_ref_f']( V_a, eos_d )

    eos_solid_d['energy_therm_f'] = energy_therm_f
    eos_solid_d['press_therm_f'] = press_therm_f

    # total energy and press are additive
    eos_solid_d['energy_f'] = lambda V_a, T_a, eos_d:\
        energy_ref_f( V_a, eos_d ) + energy_therm_f( V_a, T_a, eos_d )
    eos_solid_d['press_f'] = lambda V_a, T_a, eos_d:\
        press_ref_f( V_a, eos_d ) + press_therm_f( V_a, T_a, eos_d )

    eos_solid_d['entropy_f'] = entropy_debye
    eos_solid_d['heat_capacity_V_f'] = heat_capacity_V_debye
    eos_solid_d['bulk_mod_T_f'] = bulk_mod_T
    eos_solid_d['therm_exp_f'] = therm_exp



def set_solid_constants( param_d ):
    """
    Set EOS parameters for solid MgSiO3 perovskite (bridgmanite)
    Corresponds to BM3S model from Mosenfelder(2009)
    """
    # All units must be per atom (to make sense for arbitrary composition)

    Nat_cell = 20

    eos_solid_d = collections.OrderedDict()

    eos_solid_d['mass_avg'] = param_d['mass_avg']
    # Corresponds to BM3S model reported in Mosenfelder(2009)
    eos_solid_d['T0'] = 300 # K

    eos_solid_d['V0'] = 162.35/Nat_cell # ang^3/atom
    eos_solid_d['K0'] = 254.7 # GPa
    eos_solid_d['K0p'] = 4.26
    eos_solid_d['E0'] = 0

    eos_solid_d['theta0'] = 736 # K
    eos_solid_d['gamma0'] = 2.23
    eos_solid_d['q'] = 1.83

    # NOTE: Mosenfelder(2009) has mislabeled units as J/K/g
    #     -> units are actually J/K/kg  ???
    # The measured 1000K heat capacity of MgSiO3 is ~125 J/K/mol
    #      (equal to Dulong Petit value for 5 atom basis)
    #     -> This value is thus ~65% of that nominal value,
    #        balancing the 30 to 40% values of gamma that are higher than other
    #        studies  (static compression only constrains Gamma*Cv
    #
    # Max const-vol heat capacity:
    eos_solid_d['Cvmax'] = (806.0/1e3)*param_d['mass_avg'] # J/mol/K

    # repeat required const here
    eos_solid_d['PV_const'] = param_d['PV_const']
    eos_solid_d['Cv_const'] = param_d['Cv_const']
    eos_solid_d['ang3percc'] = param_d['ang3percc']
    eos_solid_d['Nmol'] = param_d['Nmol']

    T0 = eos_solid_d['T0']
    theta0 = eos_solid_d['theta0']

    param_d['eos_solid_d'] = eos_solid_d

