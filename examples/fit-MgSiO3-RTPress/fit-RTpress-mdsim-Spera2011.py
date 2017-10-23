import numpy as np
import matplotlib as mpl
# mpl.use('WXAgg',warn=False, force=True)
# mpl.use('GTKAgg',warn=False, force=True) #  Gtk* backend requires pygtk to be installed.
# mpl.use('Qt4Agg',warn=False, force=True) # PyQt4
# mpl.use('GTK3Agg',warn=False, force=True) # PyQt4
# mpl.use('TKAgg',warn=False, force=True) # Works
mpl.use('Qt5Agg',warn=False, force=True) # PyQt4
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
# NOTE: due to softlink in local dir, this is the local subrepo copy of xmeos
from xmeos import models
from xmeos import datamod
from scipy import optimize
import copy
from scipy import interpolate
from mpltools import annotation
# from python-tools import mpltools.annotation as annotation
from scipy import optimize as opt
from scipy import cluster
from scipy import signal
import pandas as pd
#====================================================================
def init_params(eos_d, T0, expand_adj=False, use_4th_order=False):

    models.Control.set_consts( [], [], eos_d )




    # EOS Parameter values initially set by Mosenfelder2009
    # Set model parameter values
    mass_avg = (24.31+28.09+3*16.0)/5.0 # g/(mol atom)
    S0 = 0.0 # must adjust
    param_key_a = ['T0','S0','mass_avg']
    param_val_a = np.array([T0,S0,mass_avg])
    models.Control.set_params( param_key_a, param_val_a, eos_d )

    # V0 = (38.575*1e-5)*mass_avg/eos_d['const_d']['Nmol']/1e3*1e30 # ang^3/atom
    V0 = 13.2
    K0 = 14.6
    KP0= 7.8
    E0 = -20.4

    param_key_a = ['V0','K0','KP0','E0']
    param_val_a = np.array([V0,K0,KP0,E0])
    if use_4th_order:
        KP20= KP0/K0
        # KP20 = -2.86
        param_key_a = np.append(param_key_a,['KP20'])
        param_val_a = np.append(param_val_a,[KP20])

    if expand_adj:
        # logPmin =  np.log(2.0)
        # logPmin =  np.log(5.0)
        # logPmin =  np.log(10.0)
        # logPmin =  np.log(20.0)
        # logPmin =  np.log(100.0)
        logPmin =  np.log(1000.0)
        param_key_a = np.append(param_key_a,['logPmin'])
        param_val_a = np.append(param_val_a,[logPmin])

    models.Control.set_params( param_key_a, param_val_a, eos_d )

    # gammaR = 0.17
    # qR = -1.0
    # param_key_a = ['gamma0','q0']
    # param_val_a = np.array([gammaR,qR])

    gamma0 = 0.17
    gammap0 = -1.98
    param_key_a = ['gamma0','gammap0']
    param_val_a = np.array([gamma0,gammap0])
    models.Control.set_params( param_key_a, param_val_a, eos_d )

    lognfac = 0.0
    mexp = 3.0/5
    param_key_a = ['lognfac','mexp']
    param_val_a = np.array([lognfac,mexp])
    models.Control.set_params( param_key_a, param_val_a, eos_d )

    bcoef_a = np.array([-5.0,-5.26,0.0,0.7,1.12])
    # bcoef_a = np.array([0.0,-5.0,-5.26,0.0,0.7,1.12])
    # bcoef_a = np.array([-5.0,-5.26,0.0,0.7,1.12,0.0])
    models.Control.set_array_params( 'bcoef', bcoef_a, eos_d )

    # # Must convert energy units from kJ/g to eV/atom
    energy_conv_fac = mass_avg/eos_d['const_d']['kJ_molpereV']
    models.Control.set_consts( ['energy_conv_fac'], [energy_conv_fac],
                              eos_d )

    load_eos_mod( eos_d, expand_adj=expand_adj, use_4th_order=use_4th_order )

    #     from IPython import embed; embed(); import ipdb; ipdb.set_trace()
    return eos_d
#====================================================================
def init_solid_params(eos_d):
    """
    Set EOS parameters for solid MgSiO3 perovskite (bridgmanite)
    Corresponds to BM3S model from Mosenfelder(2009)
    """
    # All units must be per atom (to make sense for arbitrary composition)

    models.Control.set_consts( [], [], eos_d )

    const_d = eos_d['const_d']

    Nat_cell = 20
    Nat_formula = 5

    T0 = 300 # K

    # EOS Parameter values initially set by Mosenfelder2009
    # Set model parameter values
    mass_avg = (24.31+28.09+3*16.0)/5.0 # g/(mol atom)
    S0 = 0.0 # must adjust
    param_key_a = ['T0','S0','mass_avg']
    param_val_a = np.array([T0,S0,mass_avg])
    models.Control.set_params( param_key_a, param_val_a, eos_d )

    # V0 = (38.575*1e-5)*mass_avg/eos_d['const_d']['Nmol']/1e3*1e30 # ang^3/atom
    V0 = 162.35/Nat_cell # ang^3/atom
    K0 = 254.7 # GPa
    KP0= 4.26
    E0 = 0.0
    param_key_a = ['V0','K0','KP0','E0']
    param_val_a = np.array([V0,K0,KP0,E0])
    models.Control.set_params( param_key_a, param_val_a, eos_d )

    VR = V0
    thetaR = 736 # K
    gammaR = 2.23
    qR     = 1.83
    param_key_a = ['VR','thetaR','gammaR','qR']
    param_val_a = np.array([VR,thetaR,gammaR,qR])
    models.Control.set_params( param_key_a, param_val_a, eos_d )

    # NOTE: Mosenfelder(2009) has mislabeled units as J/K/g
    #     -> units are actually J/K/kg  ???
    # The measured 1000K heat capacity of MgSiO3 is ~125 J/K/mol
    #      (equal to Dulong Petit value for 5 atom basis)
    #     -> This value is thus ~65% of that nominal value,
    #        balancing the 30 to 40% values of gamma that are higher than other
    #        studies  (static compression only constrains Gamma*Cv
    #
    # Max const-vol heat capacity:
    Cvmax = (806.0/1e3)*mass_avg/const_d['kJ_molpereV']/1e3 # J/mol atoms/K -> eV/K/atom

    param_key_a = ['Cvmax']
    param_val_a = np.array([Cvmax])
    models.Control.set_params( param_key_a, param_val_a, eos_d )

    # # Must convert energy units from kJ/g to eV/atom
    energy_conv_fac = mass_avg/eos_d['const_d']['kJ_molpereV']
    models.Control.set_consts( ['energy_conv_fac'], [energy_conv_fac], eos_d )


    compress_path_mod = models.BirchMurn3(path_const='S',level_const=T0,
                                          supress_energy=False,
                                          supress_press=False,
                                          expand_adj=False)
    models.Control.set_modtypes( ['CompressPathMod'], [compress_path_mod],
                                eos_d )

    gamma_mod = models.GammaPowLaw(V0ref=False)
    models.Control.set_modtypes( ['GammaMod'], [gamma_mod], eos_d )

    thermal_mod = models.MieGrunDebye()
    models.Control.set_modtypes( ['ThermalMod'], [thermal_mod], eos_d )

    full_mod = models.ThermalPressMod()
    models.Control.set_modtypes( ['FullMod'], [full_mod], eos_d )


    return eos_d
#====================================================================
def load_eos_mod( eos_d, expand_adj=False, use_4th_order=False):

    load_compress_path_mod(eos_d, expand_adj=expand_adj, use_4th_order=use_4th_order)
    load_gamma_mod(eos_d)
    load_thermal_mod(eos_d)

    full_mod = models.ThermalPressMod()
    models.Control.set_modtypes( ['FullMod'], [full_mod], eos_d )

    pass
#====================================================================
def load_thermal_mod( eos_d):
    # thermal_mod = models.RosenfeldTaranzonaPerturb()
    thermal_mod = models.RosenfeldTaranzonaAdiabat()
    models.Control.set_modtypes( ['ThermalMod'], [thermal_mod], eos_d )

    pass
#====================================================================
def load_gamma_mod( eos_d):
    # gamma_mod = models.GammaPowLaw(V0ref=True)
    gamma_mod = models.GammaFiniteStrain(V0ref=True)
    models.Control.set_modtypes( ['GammaMod'], [gamma_mod], eos_d )

    pass
#====================================================================
def load_compress_path_mod( eos_d, expand_adj=False, use_4th_order=False ):
    # path_const = 'S'
    path_const = 'T'
    # S0, = models.Control.get_params(['S0'],eos_d)
    if expand_adj:
        # expand_adj_mod = models.Tait( setlogPmin=True )
        expand_adj_mod = models.Tait( setlogPmin=True, expand_adj=True )
    else:
        expand_adj_mod = None

    if use_4th_order:
        compress_path_mod = models.BirchMurn4(path_const=path_const,
                                              supress_energy=False,
                                              supress_press=False,
                                              expand_adj_mod=expand_adj_mod)
    else:
        compress_path_mod = models.Vinet(path_const=path_const,
                                         supress_energy=False,
                                         supress_press=False,
                                         expand_adj_mod=expand_adj_mod)
        # compress_path_mod = models.BirchMurn3(path_const='S',level_const=S0,
        #                                  supress_energy=False,
        #                                  supress_press=False,
        #                                  expand_adj_mod=expand_adj_mod)
    models.Control.set_modtypes( ['CompressPathMod'], [compress_path_mod],
                                eos_d )

    pass
#====================================================================
def load_Spera2011_data():
    # Oganov mdsim data
    oganov_mdsim_dat_nm = '../data/MgSiO3-Oganov-mdsim-Spera2011.csv'

    # extract data and store
    dat_og_a = np.loadtxt(oganov_mdsim_dat_nm,skiprows=1,delimiter=',')
    mass_avg = (24.31+28.09+3*16.0)/5.0 # g/(mol atom)

    dat_og_d = {}
    dat_og_d['V'] = dat_og_a[:,1]
    dat_og_d['T'] = dat_og_a[:,2]
    dat_og_d['Terr'] = dat_og_a[:,3]
    dat_og_d['P'] = dat_og_a[:,4]
    dat_og_d['Perr'] = dat_og_a[:,5]
    dat_og_d['Etot'] = dat_og_a[:,6]
    dat_og_d['Etoterr'] = dat_og_a[:,7]
    dat_og_d['Epot'] = dat_og_a[:,10]
    dat_og_d['Epoterr'] = dat_og_a[:,11]
    dat_og_d['Ekin'] = dat_og_a[:,8]
    dat_og_d['Ekinerr'] = dat_og_a[:,9]

    dat_og_d['V-units'] = 'cc/g'
    dat_og_d['T-units'] = 'K'
    dat_og_d['P-units'] = 'GPa'
    dat_og_d['E-units'] = 'kJ/g'

    const_d = models.Control.default_consts()
    Vconv = const_d['ang3percc']*mass_avg/const_d['Nmol'] # (ang^3/atom) / (cc/g)
    Econv = mass_avg/const_d['kJ_molpereV'] # (eV/atom) / (kJ/g)

    data_d = datamod.load_data(V=dat_og_d['V']*Vconv,T=dat_og_d['T'],
                               P=dat_og_d['P'],E=dat_og_d['Etot']*Econv,
                               Terr=dat_og_d['Terr'], Perr=dat_og_d['Perr'],
                               Eerr=dat_og_d['Etoterr']*Econv)

    return dat_og_d, data_d
#====================================================================
def load_deKoker2009_data(TTOL=30):
    # Equivalent to Stixrude2005, Stixrude2009
    Vref = 38.88 # cm^3/mol
    T0 = 3000.0
    dK09_mdsim_dat_nm = 'data/MgSiO3-deKoker2009.csv'
    Natom = 5

    # extract data and store
    dat_dk_a = np.loadtxt(dK09_mdsim_dat_nm,skiprows=1,delimiter=',')
    # mass_avg = (24.31+28.09+3*16.0)/5.0 # g/(mol atom)

    dat_dk_d = {}
    dat_dk_d['T'] = dat_dk_a[:,0]
    dat_dk_d['V'] = dat_dk_a[:,1]*Vref
    dat_dk_d['P'] = dat_dk_a[:,2]
    dat_dk_d['E'] = dat_dk_a[:,3]

    Tuniq_a = np.unique(dat_dk_d['T'])
    dat_dk_d['T']/TTOL
    Tuniq_a-TTOL



    dat_dk_d['V-units'] = 'cc/mol'
    dat_dk_d['T-units'] = 'K'
    dat_dk_d['P-units'] = 'GPa'
    dat_dk_d['E-units'] = 'kJ/mol'

    const_d = models.Control.default_consts()
    Vconv = 1./Natom*const_d['ang3percc']/const_d['Nmol'] # (ang^3/atom) / (cc/mol)
    Econv = 1./Natom/const_d['kJ_molpereV'] # (eV/atom) / (kJ/mol)

    # data_d = {}
    # data_d['V'] = data_dk_d['V']*Vconv
    # data_d['E'] = data_dk_d['E']*Vconv
    # data_d['T'] = data_dk_d['T']

    data_d = datamod.load_data(V=dat_dk_d['V']*Vconv,T=dat_dk_d['T'],
                               P=dat_dk_d['P'],E=dat_dk_d['E']*Econv)

    data_d['V0'] = Vref*Vconv
    data_d['T0'] = T0

    return data_d
#====================================================================
def load_deKoker_miegrun_eos(eos_d):
    const_d = eos_d['const_d']
    # Equivalent to Stixrude2005, Stixrude2009
    Vref = 38.88 # cm^3/mol
    Natom = 5

    Nsamp = 1001

    T0 = 3000.0

    dirnm = 'data/'
    # extract data and store

    Pref_dgtz_a = np.loadtxt(dirnm+'MgSiO3-P3000-deKoker2009.csv',skiprows=1,delimiter=',')
    Eref_dgtz_a = np.loadtxt(dirnm+'MgSiO3-E3000-deKoker2009.csv',skiprows=1,delimiter=',')
    gamma_dgtz_a = np.loadtxt(dirnm+'MgSiO3-gamma-deKoker2009.csv',skiprows=1,delimiter=',')
    Ptherm_dgtz_a = np.loadtxt(dirnm+'MgSiO3-Ptherm-deKoker2009.csv',skiprows=1,delimiter=',')

    const_d = models.Control.default_consts()
    Vconv = 1./Natom*const_d['ang3percc']/const_d['Nmol'] # (ang^3/atom) / (cc/mol)
    Econv = 1./Natom/const_d['kJ_molpereV'] # (eV/atom) / (kJ/mol)

    # Get data lims of ALL datasets
    Vfac_min = np.max((Pref_dgtz_a[0,0],Eref_dgtz_a[0,0],gamma_dgtz_a[0,0],Ptherm_dgtz_a[0,0]))
    Vfac_max = np.min((Pref_dgtz_a[-1,0],Eref_dgtz_a[-1,0],gamma_dgtz_a[-1,0],Ptherm_dgtz_a[-1,0]))

    Vfac_a = np.linspace(Vfac_min,Vfac_max,Nsamp)
    V_a = Vfac_a*Vconv*Vref

    Pref_a = interpolate.interp1d(Pref_dgtz_a[:,0],Pref_dgtz_a[:,1],kind='cubic')(Vfac_a)
    Pref_a = signal.savgol_filter(Pref_a,301,3)
    # plt.clf()
    # plt.plot(Pref_dgtz_a[:,0], Pref_dgtz_a[:,1], 'ko', Vfac_a, Pref_a, 'r-' )

    Eref_a = interpolate.interp1d(Eref_dgtz_a[:,0],Eref_dgtz_a[:,1],kind='cubic')(Vfac_a)
    Eref_a = signal.savgol_filter(Eref_a,301,3)
    # plt.clf()
    # plt.plot(Eref_dgtz_a[:,0], Eref_dgtz_a[:,1], 'ko', Vfac_a, Eref_a, 'r-' )

    gamma_a = interpolate.interp1d(gamma_dgtz_a[:,0],gamma_dgtz_a[:,1],kind='cubic')(Vfac_a)
    gamma_a = signal.savgol_filter(gamma_a,301,3)
    # plt.clf()
    # plt.plot(gamma_dgtz_a[:,0], gamma_dgtz_a[:,1], 'ko', Vfac_a, gamma_a, 'r-' )

    Ptherm_a = interpolate.interp1d(Ptherm_dgtz_a[:,0],Ptherm_dgtz_a[:,1],kind='cubic')(Vfac_a)
    Ptherm_a = signal.savgol_filter(Ptherm_a,201,3)
    # plt.clf()
    # plt.plot(Ptherm_dgtz_a[:,0], Ptherm_dgtz_a[:,1], 'ko', Vfac_a, Ptherm_a, 'r-' )

    param_d = {}
    param_d['V0'] = Vref*Vconv
    param_d['mass_avg'] = eos_d['param_d']['mass_avg']


    miegrun_d = {}
    miegrun_d['const_d'] = const_d
    miegrun_d['Vmin'] = V_a[0]
    miegrun_d['Vmax'] = V_a[-1]
    miegrun_d['T0'] = T0
    miegrun_d['V'] = V_a

    # Pref_f = interpolate.interp1d(V_a,Pref_a)
    # Eref_f = interpolate.interp1d(V_a,Econv*Eref_a)
    # gamma_f = interpolate.interp1d(V_a,gamma_a)
    # Ptherm_f = interpolate.interp1d(V_a,Ptherm_a)

    # miegrun_d['Vref_T0'] = V_a
    # miegrun_d['Pref_T0'] = Pref_a
    # miegrun_d['Eref_T0'] = Eref_a*Econv
    # miegrun_d['gamma'] = gamma_a
    # miegrun_d['Ptherm'] = Ptherm_a

    # miegrun_d['Pref_f'] = Pref_f
    # miegrun_d['Eref_f'] = Eref_f
    # miegrun_d['gamma_f'] = gamma_f
    # miegrun_d['Ptherm_f'] = Ptherm_f

    # miegrun_mod = miegrun_eos_mod()
    full_mod = miegrun_eos_mod()
    full_mod.update_lookup_tables(miegrun_d,Vref_a=V_a, Pref_a=Pref_a,
                                  Eref_a=Eref_a*Econv, gamma_a=gamma_a,
                                  Ptherm_a=Ptherm_a)


    modtype_d = {}
    modtype_d['FullMod'] = full_mod
    miegrun_d['modtype_d'] = modtype_d
    miegrun_d['param_d'] = param_d

    return miegrun_d
#====================================================================
class miegrun_eos_mod():
    def __init__(self):
        pass

    def update_lookup_tables(self, miegrun_d, Vref_a=None,
                             Pref_dev_a=None, Pref_a=None,
                             Eref_dev_a=None, Eref_a=None,
                             gamma_dev_a=None, gamma_a=None,
                             Ptherm_dev_a=None, Ptherm_a=None):
        # Everything must be in correct units
        if Vref_a is not None:
            miegrun_d['V'] = Vref_a

        V_a = miegrun_d['V']


        if Pref_dev_a is not None:
            miegrun_d['Pref_dev'] = Pref_dev_a
        else:
            miegrun_d['Pref_dev'] = 0.0

        if Pref_a is not None:
            miegrun_d['Pref_T0'] = Pref_a
            Pref_dev_a = miegrun_d['Pref_dev']
            Pref_f = interpolate.interp1d(V_a,Pref_a+Pref_dev_a)
            miegrun_d['Pref_f'] = Pref_f


        if Eref_dev_a is not None:
            miegrun_d['Eref_dev'] = Eref_dev_a
        else:
            miegrun_d['Eref_dev'] = 0.0

        if Eref_a is not None:
            miegrun_d['Eref_T0'] = Eref_a
            Eref_dev_a = miegrun_d['Eref_dev']
            Eref_f = interpolate.interp1d(V_a,Eref_a+Eref_dev_a)
            miegrun_d['Eref_f'] = Eref_f


        if gamma_dev_a is not None:
            miegrun_d['gamma_dev'] = gamma_dev_a
        else:
            miegrun_d['gamma_dev'] = 0.0

        if gamma_a is not None:
            miegrun_d['gamma'] = gamma_a
            gamma_dev_a = miegrun_d['gamma_dev']
            gamma_f = interpolate.interp1d(V_a,gamma_a+gamma_dev_a)
            miegrun_d['gamma_f'] = gamma_f


        if Ptherm_dev_a is not None:
            miegrun_d['Ptherm_dev'] = Ptherm_dev_a
        else:
            miegrun_d['Ptherm_dev'] = 0.0

        if Ptherm_a is not None:
            miegrun_d['Ptherm'] = Ptherm_a
            Ptherm_dev_a = miegrun_d['Ptherm_dev']
            Ptherm_f = interpolate.interp1d(V_a,Ptherm_a+Ptherm_dev_a)
            miegrun_d['Ptherm_f'] = Ptherm_f

        pass

    def press( self, V_a, T_a, miegrun_d ):
        T0 = miegrun_d['T0']
        Pref_a = miegrun_d['Pref_f'] (V_a)
        Eref_a = miegrun_d['Eref_f'] (V_a)
        gamma_a = miegrun_d['gamma_f'] (V_a)
        dPtherm_a = (T_a-T0)*miegrun_d['Ptherm_f'] (V_a)
        dEtherm_a = dPtherm_a/(gamma_a/V_a)/miegrun_d['const_d']['PV_ratio']

        E_a = Eref_a + dEtherm_a
        P_a = Pref_a + dPtherm_a

        return P_a

    def energy( self, V_a, T_a, miegrun_d ):
        T0 = miegrun_d['T0']
        Pref_a = miegrun_d['Pref_f'] (V_a)
        Eref_a = miegrun_d['Eref_f'] (V_a)
        gamma_a = miegrun_d['gamma_f'] (V_a)
        dPtherm_a = (T_a-T0)*miegrun_d['Ptherm_f'] (V_a)
        dEtherm_a = dPtherm_a/(gamma_a/V_a)/miegrun_d['const_d']['PV_ratio']

        E_a = Eref_a + dEtherm_a
        P_a = Pref_a + dPtherm_a

        return E_a
#====================================================================
def calc_EP_miegrun_eos(V_a,T_a,miegrun_d):
    Tref = miegrun_d['Tref']
    Pref_a = miegrun_d['Pref_f'] (V_a)
    Eref_a = miegrun_d['Eref_f'] (V_a)
    gamma_a = miegrun_d['gamma_f'] (V_a)
    dPtherm_a = (T_a-Tref)*miegrun_d['Ptherm_f'] (V_a)
    dEtherm_a = dPtherm_a/(gamma_a/V_a)/miegrun_d['const_d']['PV_ratio']

    E_a = Eref_a + dEtherm_a
    P_a = Pref_a + dPtherm_a

    return E_a, P_a
#====================================================================
def fit_RTgrid_eos(data_d, mexp=3./5, RTorder=1, showfits=True):
    P_a = data_d['P']
    E_a = data_d['E']
    V_a = data_d['V']
    T_a = data_d['T']
    T0 = data_d['T0']

    # Fit thermal pressure gradient
    # assuming quadratic (or linear with only 2 data points)
    #  temperature-dependence
    Vuniq_a = np.unique(V_a)
    Tmod_a = np.linspace(1800,1.1*np.max(T_a),301)
    Np_max = 3
    Tscl = 1000

    dPdT_a = np.zeros(V_a.size)
    dEdT_a = np.zeros(V_a.size)


    RTgrid_d = {}
    RTgrid_d['V_a'] = Vuniq_a

    RTgrid_d['mexp'] = mexp
    RTgrid_d['T0'] = T0

    NV = len(Vuniq_a)
    RT_coef_a = np.zeros((NV,RTorder+1))
    grun_poly_a = np.zeros((NV,Np_max))

    if showfits:
        f, ax = plt.subplots(nrows=1,ncols=2)

    for indV,iV in enumerate(Vuniq_a):
        ind_dat = np.where(V_a==iV)[0]
        iT_a = T_a[ind_dat]
        iP_a = P_a[ind_dat]
        iE_a = E_a[ind_dat]
        iNp = np.min([Np_max,iP_a.size])
        iRTorder = np.min([RTorder+1,iP_a.size])-1

        iTdev_a = (iT_a/T0)**mexp - 1
        iRT_coef = np.polyfit(iTdev_a, iE_a, iRTorder)
        igrun_poly = np.polyfit(iE_a, iP_a, iNp-1)

        iTdev_mod_a = (Tmod_a/T0)**mexp-1
        iEmod_a = np.polyval(iRT_coef,iTdev_mod_a)
        iPmod_a = np.polyval(igrun_poly,iEmod_a)

        if showfits:
            ax[0].cla()
            ax[0].plot(iTdev_mod_a,iEmod_a,'r-')
            ax[0].plot(iTdev_a,iE_a,'ko')

            ax[1].cla()
            #plt.caxis(Tmod_a[0],Tmod_a[-1])
            ax[1].scatter(iEmod_a,iPmod_a,c=Tmod_a,s=30,lw=0,vmin=Tmod_a[0],vmax=Tmod_a[-1])
            ax[1].scatter(iE_a,iP_a,c=iT_a,s=100,lw=0,vmin=Tmod_a[0],vmax=Tmod_a[-1])
            #plt.caxis(Tmod_a[0],Tmod_a[-1])

            plt.pause(1)

        grun_poly_a[indV, Np_max-iNp:] = igrun_poly
        RT_coef_a[indV, RTorder-iRTorder:] = iRT_coef



    RTgrid_d['grun_poly_a'] = grun_poly_a
    RTgrid_d['RT_coef_a'] = RT_coef_a

    return RTgrid_d
#====================================================================
def calc_EV_RTgrid(P, T, RTgrid_d, showplot=True):
    V_a =  RTgrid_d['V_a']
    RT_coef_a = RTgrid_d['RT_coef_a']
    grun_poly_a = RTgrid_d['grun_poly_a']

    mexp = RTgrid_d['mexp']
    T0 = RTgrid_d['T0']

    E_a = np.zeros(V_a.shape)
    P_a = np.zeros(V_a.shape)

    for indV, iV in enumerate(V_a):
        iE = np.polyval(RT_coef_a[indV,:],(T/T0)**mexp-1)
        iP = np.polyval(grun_poly_a[indV,:],iE)
        P_a[indV] = iP
        E_a[indV] = iE

    Efun = interpolate.interp1d(P_a, E_a, kind='cubic')
    Vfun = interpolate.interp1d(P_a, V_a, kind='cubic')
    E = Efun(P)
    V = Vfun(P)

    if showplot:
        f, ax = plt.subplots(nrows=1,ncols=2)
        ax[0].plot(V_a,E_a,'ko')
        ax[0].plot(V,E,'rx',ms=10,mew=2)

        ax[1].plot(V_a,P_a,'ko')
        ax[1].plot(V,P,'rx',ms=10,mew=2)
        plt.show()


    return E, V
#====================================================================
def calc_PT_RTgrid(E, V, RTgrid_d, showplot=True):
    V_a =  RTgrid_d['V_a']
    RT_coef_a = RTgrid_d['RT_coef_a']
    grun_poly_a = RTgrid_d['grun_poly_a']

    mexp = RTgrid_d['mexp']

    T_a = np.zeros(V_a.shape)
    P_a = np.zeros(V_a.shape)

    T0 = RTgrid_d['T0']

    for indV, iV in enumerate(V_a):
        logTfac = optimize.fsolve(
            lambda logTfac: E-np.polyval(RT_coef_a[indV,:],
                                         np.exp(logTfac)**mexp-1), 0 )
        iT = T0*np.exp(logTfac)
        iP = np.polyval(grun_poly_a[indV,:],E)

        P_a[indV] = iP
        T_a[indV] = iT

    Pfun = interpolate.interp1d(V_a, P_a, kind='cubic')
    Tfun = interpolate.interp1d(V_a, T_a, kind='cubic')
    P = Pfun(V)
    T = Tfun(V)

    if showplot:
        f, ax = plt.subplots(nrows=1,ncols=2)
        ax[0].plot(V_a,T_a,'ko')
        ax[0].plot(V,T,'rx',ms=10,mew=2)

        ax[1].plot(V_a,P_a,'ko')
        ax[1].plot(V,P,'rx',ms=10,mew=2)
        plt.show()


    return P, T
#====================================================================
def estimate_thermal_grad(data_d):
    P_a = data_d['P']
    E_a = data_d['E']
    V_a = data_d['V']
    T_a = data_d['T']
    T0 = data_d['T0']

    # Fit thermal pressure gradient
    # assuming quadratic (or linear with only 2 data points)
    #  temperature-dependence
    Vuniq_a = np.unique(V_a)
    # Tmod_a = np.linspace(2400,5600,101)
    Np_max = 3
    Tscl = 1000

    dPdT_a = np.zeros(V_a.size)
    dEdT_a = np.zeros(V_a.size)

    for iV in Vuniq_a:
        ind_dat = np.where(V_a==iV)[0]
        iT_a = T_a[ind_dat]
        iP_a = P_a[ind_dat]
        iE_a = E_a[ind_dat]
        iNp = np.min([Np_max,iP_a.size])

        ipoly_P = np.polyfit((iT_a-T0)/Tscl, iP_a, iNp-1)
        ipoly_P_der = np.polyder(ipoly_P)
        idPdT_a = np.polyval(ipoly_P_der,(iT_a-T0)/Tscl)
        dPdT_a[ind_dat] = idPdT_a

        ipoly_E = np.polyfit((iT_a-T0)/Tscl, iE_a, iNp-1)
        ipoly_E_der = np.polyder(ipoly_E)
        idEdT_a = np.polyval(ipoly_E_der,(iT_a-T0)/Tscl)
        dEdT_a[ind_dat] = idEdT_a

        # iPref = ipoly[-1]
        # plt.clf()
        # plt.plot(Tmod_a,np.polyval(ipoly,(Tmod_a-T0)/Tscl)-iPref,'r-',
        #          iT_a,iP_a-iPref,'ko')
        # plt.pause(.1)

    data_d['dPdT'] = dPdT_a
    data_d['dEdT'] = dEdT_a

    pass
#====================================================================
def eval_model( Vgrid_a, Tgrid_a, eos_d ):

    full_mod = eos_d['modtype_d']['FullMod']

    fit_data_type = datamod_d['fit_data_type']


    energy_mod_a = []
    press_mod_a = []
    dPdT_mod_a = []

    for iT in Tgrid_a:
        ienergy_a = full_mod.energy(Vgrid_a,iT,eos_d)
        ipress_a = full_mod.press(Vgrid_a,iT,eos_d)
        idPdT_a = full_mod.dPdT(Vgrid_a,iT,eos_d)
        energy_mod_a.append(ienergy_a)
        press_mod_a.append(ipress_a)
        dPdT_mod_a.append(idPdT_a)

    # energy_mod_a = np.array( energy_mod_a )
    energy_mod_a = np.array( energy_mod_a )
    press_mod_a = np.array( press_mod_a )
    dPdT_mod_a = np.array( dPdT_mod_a )

    return energy_mod_a, press_mod_a, dPdT_mod_a
#====================================================================
def get_adiabat_paths( Vpath_a, Tfoot_a, eos_d, indVfoot=-1, TOL=1e-6 ):


    full_mod = eos_d['modtype_d']['FullMod']
    gamma_mod = eos_d['modtype_d']['GammaMod']

    # energy_conv_fac, = models.Control.get_consts(['energy_conv_fac'],eos_d)

    Vfoot = Vpath_a[indVfoot]


    thermal_mod, = models.Control.get_modtypes( ['ThermalMod'], eos_d )
    thermal_mod.entropy( np.array(Vfoot), Tfoot_a, eos_d )

    Sfoot_a = full_mod.entropy( Vfoot, Tfoot_a, eos_d )
    Sfoot_a = full_mod.entropy( Vfoot, Tfoot_a, eos_d )

    # will need to write over Tgrid_a
    Vgrid_a, Tgrid_a = np.meshgrid( Vpath_a, Tfoot_a )
    Sgrid_a = np.zeros( Tgrid_a.shape )
    Pgrid_a = np.zeros( Tgrid_a.shape )

    Tfoot_grid_a = Tgrid_a.copy()

    iT_a = np.copy(Tfoot_a)
    for ind, iV in enumerate( Vpath_a ):
        while True:
            iS_a = full_mod.entropy( iV, iT_a, eos_d )
            iCv_a = full_mod.heat_capacity( iV, iT_a, eos_d )
            dlogT_a = (iS_a-Sfoot_a)/iCv_a
            if np.all(np.abs(dlogT_a) < TOL):
                iP_a = full_mod.press( iV, iT_a, eos_d )

                Tgrid_a[:,ind] = np.squeeze(iT_a)
                Sgrid_a[:,ind] = np.squeeze(iS_a)
                Pgrid_a[:,ind] = np.squeeze(iP_a)
                break
            else:
                iT_a *= np.exp(-dlogT_a)


    output_d = {}
    output_d['Pgrid'] = Pgrid_a
    output_d['Vgrid'] = Vgrid_a
    output_d['Tgrid'] = Tgrid_a
    output_d['Sgrid'] = Sgrid_a
    output_d['Sfoot'] = Sfoot_a
    output_d['Tfoot_grid'] = Tfoot_grid_a

    return output_d
#====================================================================
def add_shared_colorbar(label, ticks, fig=None, pos=[.85,.1,.05,.85], cmap=None,
                        orientation='vertical'):

    if fig is None:
        fig = plt.gcf()

    if cmap is None:
        cmap=plt.get_cmap('coolwarm')

    clim = [ticks[0]-0.5*(ticks[1]-ticks[0]),
            ticks[-1]+0.5*(ticks[-1]-ticks[-2])]

    axdummy = fig.add_axes([0,0,1,1])
    cax = fig.add_axes(pos)
    dummy = axdummy.imshow(np.array([[0,1]]), cmap=cmap)
    axdummy.set_visible(False)
    dummy.set_clim(clim[0],clim[1])
    cbar=plt.colorbar(dummy,orientation=orientation,ticks=ticks,label=label,cax=cax)
    plt.draw()
    pass
#====================================================================
def fit_init_model( eos_d, data_d ):
    fit_compress_ref_isotherm( eos_d, data_d )
    fit_energy_isochores( eos_d, data_d )
    infer_adiabat_temp_profile( eos_d, data_d )
    fit_adiabat_temp_profile( eos_d, data_d )
    pass
#====================================================================
def fit_energy_isochores( eos_d, data_d, Ndof=4 ):
    # Fit RT model to each isochore (to get heat capacity model)
    mexp = 3.0/5
    kB = models.Control.default_consts()['kboltz']
    V0, = models.Control.get_params(['V0'],eos_d)

    Vuniq_a = data_d['Vuniq']
    NV = Vuniq_a.size

    acoef_a = np.zeros(NV)
    bcoef_a = np.zeros(NV)
    plt_const = 1e-1

    # plt.figure()
    # plt.clf()
    for indV in np.arange(Vuniq_a.size):
        ind_a = data_d['V'] == Vuniq_a[indV]
        iV = Vuniq_a[indV]
        iT_a = data_d['T'][ind_a]
        therm_dev_a = (iT_a/T0)**mexp - 1.0
        iE_a = data_d['E'][ind_a]
        idEkin0_a = 3./2*kB*(iT_a-T0)
        idEpot0_a = iE_a-idEkin0_a

        abfit = np.polyfit(therm_dev_a,idEpot0_a,1)
        acoef_a[indV] = abfit[1]
        bcoef_a[indV] = abfit[0]
        # plt.plot(therm_dev_a,E_a-k*iV,'ko-')
        # plt.plot(therm_dev_a,idEpot0_a-plt_const*iV,'ko-')

        # plt.plot(therm_dev_a,idEpot0_a-np.polyval(abfit,therm_dev_a),'o-')
        # plt.pause(0.3)

    # plt.clf()
    # plt.plot(Vuniq_a,acoef_a,'ko-')

    # plt.clf()
    # plt.plot(Vuniq_a,bcoef_a,'ko-')

    data_d['acoef_T0'] = acoef_a
    data_d['bcoef_T0'] = bcoef_a

    # plt.plot(Vuniq_a,acoef_a,'ko-',Vref_T0_a,Eref_T0_a,'rx-')

    # bpoly_a = np.polyfit(Vuniq_a/V0-1.0,bcoef_a,Ndof)
    blogpoly_a = np.polyfit(np.log(Vuniq_a/V0),bcoef_a,Ndof)

    # plt.clf()
    # plt.plot(np.log(Vuniq_a/V0),bcoef_a,'ko-',
    #          np.log(Vmod_a/V0), np.polyval(blogpoly_a,np.log(Vmod_a/V0)),'r-')

    E0 = interpolate.interp1d(Vuniq_a,acoef_a,kind='cubic')(V0)

    models.Control.set_array_params( 'bcoef', blogpoly_a, eos_d )
    models.Control.set_params( ['E0','mexp'], [E0,mexp], eos_d )
    pass
#====================================================================
def fit_compress_ref_isotherm( eos_d, data_d ):
    PV_ratio, = models.Control.get_consts( ['PV_ratio'], eos_d )


    T0 = data_d['T0']
    models.Control.set_params( ['T0'], [T0], eos_d )

    modfit = models.ModFit()
    compress_mod = eos_d['modtype_d']['CompressPathMod']

    Vref_a = data_d['Vref_T0']
    Pref_a = data_d['Pref_T0']

    # Fit reference isotherm with Vinet
    param_key = ['V0','K0','KP0']
    param0_a = np.array( models.Control.get_params(param_key,eos_d) )

    # press_mod_f = modfit.get_wrap_eos_fun( compress_mod.press, eos_d, param_key )
    resid_press_f = modfit.get_resid_fun( compress_mod.press, eos_d, param_key,
                                         (Vref_a,), Pref_a )

    paramf_a = optimize.leastsq(resid_press_f, param0_a)[0]
    models.Control.set_params( param_key, paramf_a, eos_d )

    pass
#====================================================================
def calc_avg_isotherm( NTgrid, data_d ):
    T_a = data_d['T']
    Tavg_a, Tscatter = cluster.vq.kmeans(T_a,NTgrid,iter=1000)
    Tavg_a = np.sort(Tavg_a)

    Tdiff_a = Tavg_a-np.expand_dims(T_a,1)
    Tbin_a = np.argmin(np.abs(Tdiff_a),axis=1)

    Vuniq_a = np.unique(V_a)
    # T_a[Tbin_a]

    data_d['Tavg_bins'] = Tavg_a
    data_d['Tbin'] = Tbin_a
    data_d['Vuniq'] = Vuniq_a

    pass
#====================================================================
def def_ref_isotherm( Tbin_ref, data_d ):
    data_d['V']
    ind_ref_a = np.where(data_d['Tbin']==Tbin_ref)[0]
    ind_sort = np.argsort(data_d['V'][ind_ref_a])
    Vref_a = data_d['V'][ind_ref_a][ind_sort]
    Pref_a = data_d['P'][ind_ref_a][ind_sort]
    Eref_a = data_d['E'][ind_ref_a][ind_sort]

    # Define T0 as avg temp of ref curve (to nearest 5K)
    T0 = data_d['Tavg_bins'][Tbin_ref]
    # T0 = 5*np.round(np.mean(Tref_T0_a)/5)
    # T0 = np.mean(Tref_T0_a)
    # T0_err = np.std(Tref_T0_a)

    data_d['T0'] = T0
    data_d['Vref_T0'] = Vref_a
    data_d['Pref_T0'] = Pref_a
    data_d['Eref_T0'] = Eref_a

    pass
#====================================================================
def infer_adiabat_temp_profile( eos_d, data_d ):
    # Infer adiabatic temperature based on entropy diff for reference isotherm

    T0, = models.Control.get_params(['T0'],eos_d)
    kB = models.Control.default_consts()['kboltz']

    compress_mod = eos_d['modtype_d']['CompressPathMod']

    Vref_T0_a = data_d['Vref_T0']
    Fref_T0_a = compress_mod.energy( Vref_T0_a, eos_d )

    acoef_f = interpolate.interp1d(data_d['Vuniq'],data_d['acoef_T0'],'cubic')
    Eref_T0_a = acoef_f(Vref_T0_a)

    dSref_T0_a = (Eref_T0_a - Fref_T0_a)/(kB*T0)

    def entropy_diff_ref(T0S,V,eos_d,data_d):
        T0,V0,mexp = models.Control.get_params(['T0','V0','mexp'],eos_d)
        bcoef_T0 = data_d['bcoef_T0']
        bV = np.polyval(bcoef_T0,np.log(V/V0))
        dS = -mexp/(mexp-1)*bV/T0*((T0S/T0)**(mexp-1)-1)-3./2*kB*np.log(T0S/T0)
        return dS

    Tref_S0_a = np.zeros(Vref_T0_a.size)
    for ind,(iV,idS) in enumerate(zip(Vref_T0_a,dSref_T0_a)):
        idlogTfac = optimize.brentq(lambda dlogTfac,iV=iV:
                                   idS - entropy_diff_ref(np.exp(dlogTfac)*T0,
                                                          iV,eos_d,data_d)/kB,
                                    -3.0,3.0)
        Tref_S0_a[ind] = T0*np.exp(idlogTfac)

    data_d['dSref_T0'] = dSref_T0_a
    data_d['Tref_S0'] = Tref_S0_a

    pass
#====================================================================
def fit_adiabat_temp_profile( eos_d, data_d ):
    # Fit temperature adiabat profile using gamma model

    # from IPython import embed; embed(); import ipdb; ipdb.set_trace()
    gamma_mod = eos_d['modtype_d']['GammaMod']
    modfit = models.ModFit()

    Vref_T0_a = data_d['Vref_T0']
    Tref_S0_a = data_d['Tref_S0']

    T0, = models.Control.get_params(['T0'],eos_d)

    # param_key=['gammaR','gammapR']
    param_key=['gamma0','gammap0']
    param0_a = np.array( models.Control.get_params(param_key,eos_d) )
    # eos_d['param_d']['VR'] = V0

    temp_ref_f = modfit.get_wrap_eos_fun( gamma_mod.temp, eos_d, param_key )

    resid_temp_f = modfit.get_resid_fun( gamma_mod.temp, eos_d,
                                        param_key, (Vref_T0_a,T0), Tref_S0_a )


    paramf_a = optimize.leastsq(resid_temp_f, param0_a)[0]

    models.Control.set_params( param_key, paramf_a, eos_d )
    pass
#====================================================================
def load_liquidus_data():
    # Pthresh = 14.5
    PTliq_S09_a = np.loadtxt('../data/MgSiO3-liq-Stixrude2009.csv',delimiter=',')
    PTliq_A11_a = np.loadtxt('data/liquidus-Andrault2011.txt',delimiter=',')
    Pthresh = PTliq_A11_a[0,0]

    # mask_UM_a = PTliq_S09_a[:,0]<=Pthresh
    PTliq_UM_a = PTliq_S09_a[PTliq_S09_a[:,0]<=Pthresh]


    Tthresh_S09 = interpolate.interp1d(PTliq_S09_a[:,0],PTliq_S09_a[:,1],kind='linear')(Pthresh)
    Tthresh_A11 = interpolate.interp1d(PTliq_A11_a[:,0],PTliq_A11_a[:,1],kind='linear',fill_value='extrapolate',bounds_error=False)(Pthresh)

    liqdat_d = {}
    liqdat_d['UM'] = np.vstack((PTliq_UM_a,(Pthresh,Tthresh_S09)))
    liqdat_d['S09'] = np.vstack(((Pthresh,Tthresh_S09),PTliq_S09_a[PTliq_S09_a[:,0]>Pthresh]))
    liqdat_d['A11'] = np.vstack(((Pthresh,Tthresh_A11),PTliq_A11_a))
    return liqdat_d
#====================================================================
def plot_adiabat_melting_curve(eos_d,show_liquidus=True):

    liqdat_d = load_liquidus_data()


    V0, = models.Control.get_params(['V0'],eos_d)
    TOL = 1e-6
    Vpath_a = np.arange(0.4,1.101,.02)*V0
    # Tfoot_a = np.arange(1500.0,5100.0,500.0)
    dT = 250.0
    Tfoot_a = np.arange(2000.0,3500.1,dT)

    cmap=plt.get_cmap('coolwarm',Tfoot_a.size)

    indVfoot = np.where(np.abs(np.log(Vpath_a/V0))<TOL)[0][0]
    # Vgrid_ad_a, Tgrid_ad_a = get_adiabat_paths( Vpath_a, Tfoot_a, eos_d,
    #                                            indVfoot=indVfoot )
    adiabat_d = get_adiabat_paths( Vpath_a, Tfoot_a, eos_d, indVfoot=indVfoot )

    plt.figure()
    plt.clf()

    # Plot adiabatic paths
    cbarinfo = plt.scatter( adiabat_d['Pgrid'], adiabat_d['Tgrid'],
                           c=adiabat_d['Tfoot_grid'],s=50,lw=0,cmap=cmap)
    plt.clim(Tfoot_a[0]-0.5*dT,Tfoot_a[-1]+0.5*dT)
    plt.clf()

    col_a = cmap(1.0*(Tfoot_a-Tfoot_a[0])/np.ptp(Tfoot_a))[:,:3]
    lbl_ad = [plt.plot(ipress_a, itemp_a,'-',color=icol_a,label=iT,lw=2) \
     for ipress_a,itemp_a,icol_a,iT  in
     zip(adiabat_d['Pgrid'],adiabat_d['Tgrid'],col_a,Tfoot_a)]
    plt.xlabel(r'Pressure  [GPa]')
    plt.ylabel(r'Temperature  [K]')
    plt.xlim(0,136.0)
    plt.ylim(1250.0,5000.0)
    plt.colorbar(cbarinfo,ticks=Tfoot_a,label=r'$T_{\rm foot}$  [K]')

    if show_liquidus:
        # plt.plot(liqdat_d['UM'][:,0] ,liqdat_d['UM'][:,1] ,'k-',lw=2,color=[.3,.3,.3])
        plt.plot(liqdat_d['UM'][:,0] ,liqdat_d['UM'][:,1] ,'k-',lw=2)
        plt.plot(liqdat_d['A11'][:,0],liqdat_d['A11'][:,1],'k-.',lw=2)
        plt.plot(liqdat_d['S09'][:,0],liqdat_d['S09'][:,1],'k--',lw=2)
        # plt.plot(PTsol_a[:,0],PTsol_a[:,1],'k-')

    return lbl_ad
#====================================================================
def plot_model_mismatch((xgrid_a, ygrid_a, zgrid_a),(x_a,y_a,z_a,ymod_a),
                        (xlims,ylims),(xlbl,ylbl),zlbl_a,yresid_ticks,
                        yresid_err):
    plt.rc('text', usetex=True)
    hfig = plt.figure()
    plt.clf()

    # from IPython import embed; embed(); import ipdb; ipdb.set_trace()

    shp = ygrid_a.shape
    if xgrid_a.shape != shp:
        xgrid_tbl_a, zgrid_tbl_a = np.meshgrid(xgrid_a,zgrid_a)
        xgrid_a = xgrid_tbl_a

    ax_main = plt.subplot(211)
    ax_resid = plt.subplot(212)
    ax_main.set_xticklabels([])
    plt.draw()

    pos_main = [.125,.27,.7,.68]
    pos_resid = [.125,.1,.7,.15]

    ax_main.set_position(pos_main)
    ax_resid.set_position(pos_resid)

    cmap_cont=plt.get_cmap('coolwarm')
    col_mod_a = cmap_cont( np.linspace(0,1,zlbl_a.size) )

    cmap=plt.get_cmap('coolwarm',zlbl_a.size)



    ax_main.scatter( x_a, y_a,c=z_a,s=50,lw=0,cmap=cmap)
    [ax_main.plot(ixgrid_a, iygrid_a,'-',color=icol_a,label=izlbl) \
     for ixgrid_a, iygrid_a,icol_a,izlbl  in zip(xgrid_a,ygrid_a,col_mod_a,zgrid_a)]
    ax_main.set_xlim(xlims)
    ax_main.set_ylim(ylims)

    xrange_a = np.array([np.min(xgrid_a[:]),np.max(xgrid_a[:])])
    ax_resid.add_patch(patches.Rectangle((xlims[0],-yresid_err),
                                         xlims[1]-xlims[0],2*yresid_err,
                                         facecolor='#DCDCDC',edgecolor='none',
                                         zorder=0))
    ax_resid.plot(xrange_a,0.0*xrange_a,'k--',zorder=1)
    hlbl=ax_resid.scatter( x_a, y_a-ymod_a,c=z_a,s=50,lw=0,cmap=cmap,zorder=2)
    ax_resid.set_xlim(xlims)
    ax_resid.set_ylim([1.1*yresid_ticks[0],1.1*yresid_ticks[-1]])
    ax_resid.set_yticks(yresid_ticks)

    ax_resid.set_xlabel(xlbl)
    ax_main.set_ylabel(ylbl)


    ticks = zlbl_a
    label = 'Temperature [K]'
    add_shared_colorbar(label, ticks, pos=[.85,.1,.03,.85], cmap=cmap)

    return hfig
    pass
#====================================================================
def do_plot_model_mismatch((xgrid_a, ygrid_a, zgrid_a),(x_a,y_a,z_a,ymod_a),
                           (xlims,ylims),(xlbl,ylbl),zlbl_a,yresid_ticks,
                           yresid_err,hfig,ax_main,ax_resid,xtick_lbl=True,
                           show_colbar=True,panel_lbl='a'):

    # from IPython import embed; embed(); import ipdb; ipdb.set_trace()

    shp = ygrid_a.shape
    if xgrid_a.shape != shp:
        xgrid_tbl_a, zgrid_tbl_a = np.meshgrid(xgrid_a,zgrid_a)
        xgrid_a = xgrid_tbl_a


    cmap_cont=plt.get_cmap('coolwarm')
    col_mod_a = cmap_cont( np.linspace(0,1,zlbl_a.size) )

    cmap=plt.get_cmap('coolwarm',zlbl_a.size)



    ax_main.scatter( x_a, y_a,c=z_a,s=50,lw=0,cmap=cmap)
    [ax_main.plot(ixgrid_a, iygrid_a,'-',color=icol_a,label=izlbl) \
     for ixgrid_a, iygrid_a,icol_a,izlbl  in zip(xgrid_a,ygrid_a,col_mod_a,zgrid_a)]
    ax_main.set_xlim(xlims)
    ax_main.set_ylim(ylims)
    ax_main.set_xticklabels('')

    xpanel_lbl = 0.95*xlims[1] + 0.05*xlims[0]
    ypanel_lbl = 0.9*ylims[1] + 0.1*ylims[0]
    ax_main.text(xpanel_lbl,ypanel_lbl,panel_lbl,fontsize=14,weight='bold')


    xrange_a = np.array([np.min(xgrid_a[:]),np.max(xgrid_a[:])])
    ax_resid.add_patch(patches.Rectangle((xlims[0],-yresid_err),
                                         xlims[1]-xlims[0],2*yresid_err,
                                         facecolor='#DCDCDC',edgecolor='none',
                                         zorder=0))
    ax_resid.plot(xrange_a,0.0*xrange_a,'k--',zorder=1)
    hlbl=ax_resid.scatter( x_a, y_a-ymod_a,c=z_a,s=50,lw=0,cmap=cmap,zorder=2)
    ax_resid.set_xlim(xlims)
    ax_resid.set_ylim([1.1*yresid_ticks[0],1.1*yresid_ticks[-1]])
    ax_resid.set_yticks(yresid_ticks)
    if not xtick_lbl:
        ax_resid.set_xticklabels('')

    ax_resid.set_xlabel(xlbl)
    ax_main.set_ylabel(ylbl)


    # if show_colbar:
    #     ticks = zlbl_a
    #     label = 'Temperature [K]'
    #     add_shared_colorbar(label, ticks, pos=[.85,.1,.03,.88], cmap=cmap)

    # return hfig
    pass
#====================================================================
def setup_model_mismatch_plot(Nmodval,zlbl_a):
    plt.rc('text', usetex=True)
    tall_size = (8,10)
    if Nmodval == 1:
        hfig = plt.figure()
    else:
        # hfig = plt.figure(figsize=(4,5))
        hfig = plt.figure()
        hfig.set_size_inches(tall_size)
        hfig.set_size_inches(tall_size)

    plt.clf()

    cmap=plt.get_cmap('coolwarm',zlbl_a.size)
    # from IPython import embed; embed(); import ipdb; ipdb.set_trace()

    # gs1 = gridspec.GridSpec(6, 1)
    # gs1.update(bottom=0.05, top=0.48, hspace=0.05)
    # gs2 = gridspec.GridSpec(6, 1)
    # gs2.update(bottom=.55,top=.98,hspace=.05)


    # .55 .98
    # .1 .53
    # .1 .98
    # .1 .98

    top = .98
    bottom = .05
    if Nmodval==1:
        bottom = .1

    space = 0.02
    hspace=0.05
    right = .8

    height = (top-bottom - (Nmodval-1)*space)/Nmodval
    bottom_a = bottom+(height+space)*np.arange(Nmodval)
    top_a = bottom+height+(height+space)*np.arange(Nmodval)

    # top = 1-.02
    # bottom=.08+.02

    # gs1 = gridspec.GridSpec(6, 1)
    # gs1.update(bottom=0.1, top=0.53, hspace=0.05)
    # gs2 = gridspec.GridSpec(6, 1)
    # gs2.update(bottom=.55,top=.98,hspace=.05)

    # ax1 = plt.subplot(gs1[:-1, :])
    # ax2 = plt.subplot(gs1[-1, :])
    # ax3 = plt.subplot(gs2[:-1, :])
    # ax4 = plt.subplot(gs2[-1, :])
    ax_main_l = []
    ax_resid_l = []
    for i in range(Nmodval):
        gsi = gridspec.GridSpec(6, 1)
        gsi.update(bottom=bottom_a[i], top=top_a[i], right=right, hspace=hspace)
        ax_main_i = plt.subplot(gsi[:-1, :])
        ax_resid_i = plt.subplot(gsi[-1, :])

        ax_main_l.append(ax_main_i)
        ax_resid_l.append(ax_resid_i)



    ticks = zlbl_a
    label = 'Temperature [K]'
    add_shared_colorbar(label, ticks, pos=[.85,bottom,.03,top-bottom], cmap=cmap)


    plt.draw()

    # ax_main = plt.subplot(211)
    # ax_resid = plt.subplot(212)
    # ax_main.set_xticklabels([])
    # plt.draw()

    # pos_main = [.125,.27,.7,.68]
    # pos_resid = [.125,.1,.7,.15]

    # ax_main.set_position(pos_main)
    # ax_resid.set_position(pos_resid)


    # cmap_cont=plt.get_cmap('coolwarm')
    # col_mod_a = cmap_cont( np.linspace(0,1,zlbl_a.size) )

    # cmap=plt.get_cmap('coolwarm',zlbl_a.size)
    return hfig, ax_main_l, ax_resid_l
#====================================================================
def plot_model_fit_results(Tlbl_a,datamod_d,
                           Nsamp=101,Vbnds=[0.4,1.2]):


    eos_d = datamod_d['eos_d']
    full_mod = eos_d['modtype_d']['FullMod']

    V0 = eos_d['param_d']['V0']

    err_d = datamod_d['posterior_d']['err']

    Vgrid_a = np.linspace(Vbnds[0],Vbnds[1],Nsamp)*V0

    Tgrid_a = datamod_d['data_d']['Tavg_bins']
    dT = 500
    Tlbl_a = dT*np.round(Tgrid_a/dT)
    # Tlbl_a = np.array([2500,3000,3500,4000,4500,5000])


    E_grid_a, P_grid_a, dPdT_grid_a = eval_model( Vgrid_a, Tgrid_a, eos_d )
    E_mod_a = full_mod.energy(data_d['V'],data_d['T'],eos_d)
    P_mod_a = full_mod.press(data_d['V'],data_d['T'],eos_d)

    Plims=[-5.0,181.0]
    Vlims=[5.25,14.5]
    Elims=[-21.05,-18.95]
    Presid_ticks = [-2,0,2]
    Eresid_ticks = [-.02,0,.02]
    Plbl = r'Pressure  [GPa]'
    Elbl = r'Internal Energy  [eV / atom]'
    Vlbl = r'$V / V_0$'
    # Vlbl = r'Volume  [$\AA^3$ / atom]'

    # hfigEP = plot_model_mismatch((P_grid_a, E_grid_a, Tgrid_a, ),
    #                              (P_a,E_a,T_a,E_mod_a),
    #                              (Plims,Elims),(Plbl,Elbl),Tlbl_a,
    #                              Eresid_ticks,err_d['E'])

    hfig, ax_main_l, ax_resid_l = setup_model_mismatch_plot(2,Tlbl_a)


    do_plot_model_mismatch((Vgrid_a/V0, P_grid_a, Tgrid_a),
                           (V_a/V0,P_a,T_a,P_mod_a),
                           (Vlims/V0,Plims),('',Plbl),Tlbl_a,
                           Presid_ticks,err_d['P'],hfig,ax_main_l[1],
                           ax_resid_l[1],xtick_lbl=False,show_colbar=True,
                           panel_lbl='a')


    do_plot_model_mismatch((Vgrid_a/V0, E_grid_a, Tgrid_a),
                           (V_a/V0,E_a,T_a,E_mod_a),
                           (Vlims/V0,Elims),(Vlbl,Elbl),Tlbl_a,
                           Eresid_ticks,err_d['E'],hfig,ax_main_l[0],
                           ax_resid_l[0],show_colbar=False,
                           panel_lbl='b')

    plt.draw()



    # # hfig, ax_main_l, ax_resid_l = setup_model_mismatch_plot(1)
    # do_plot_model_mismatch((P_grid_a, E_grid_a, Tgrid_a, ),
    #                              (P_a,E_a,T_a,E_mod_a),
    #                              (Plims,Elims),(Plbl,Elbl),Tlbl_a,
    #                              Eresid_ticks,err_d['E'],hfig,ax_main_l[0],
    #                        ax_resid_l[0])


    # plt.draw()
    # plt.pause(2)

    # hfigEV = plot_model_mismatch((Vgrid_a, E_grid_a, Tgrid_a),
    #                              (V_a,E_a,T_a,E_mod_a),
    #                              (Vlims,Elims),(Vlbl,Elbl),Tlbl_a,
    #                              Eresid_ticks,err_d['E'])
    # plt.draw()
    # plt.pause(2)

    # hfigPV = plot_model_mismatch((Vgrid_a, P_grid_a, Tgrid_a),
    #                              (V_a,P_a,T_a,P_mod_a),
    #                              (Vlims,Plims),(Vlbl,Plbl),Tlbl_a,
    #                              Presid_ticks,err_d['P'])
    # plt.draw()
    # plt.pause(2)

    # return hfigPV, hfigEV, hfigEP
    # return hfig
    hfig.set_size_inches(8,10)
    hfig.set_size_inches(8,10)

    return ax_main_l, ax_resid_l
#====================================================================
def plot_model_fit_results_EP(Tlbl_a,datamod_d,
                           Nsamp=101,Vbnds=[0.4,1.2]):


    eos_d = datamod_d['eos_d']
    full_mod = eos_d['modtype_d']['FullMod']

    V0 = eos_d['param_d']['V0']

    err_d = datamod_d['posterior_d']['err']

    Vgrid_a = np.linspace(Vbnds[0],Vbnds[1],Nsamp)*V0

    Tgrid_a = datamod_d['data_d']['Tavg_bins']
    dT = 500
    Tlbl_a = dT*np.round(Tgrid_a/dT)
    # Tlbl_a = np.array([2500,3000,3500,4000,4500,5000])


    E_grid_a, P_grid_a, dPdT_grid_a = eval_model( Vgrid_a, Tgrid_a, eos_d )
    E_mod_a = full_mod.energy(data_d['V'],data_d['T'],eos_d)
    P_mod_a = full_mod.press(data_d['V'],data_d['T'],eos_d)

    Plims=[-2.0,181.0]
    Vlims=[5.25,14.5]
    Elims=[-21.05,-18.95]
    Presid_ticks = [-2,0,2]
    Eresid_ticks = [-.02,0,.02]
    Plbl = r'Pressure  [GPa]'
    Elbl = r'Internal Energy  [eV / atom]'
    Vlbl = r'$V / V_0$'
    # Vlbl = r'Volume  [$\AA^3$ / atom]'

    # hfigEP = plot_model_mismatch((P_grid_a, E_grid_a, Tgrid_a, ),
    #                              (P_a,E_a,T_a,E_mod_a),
    #                              (Plims,Elims),(Plbl,Elbl),Tlbl_a,
    #                              Eresid_ticks,err_d['E'])

    hfig, ax_main_l, ax_resid_l = setup_model_mismatch_plot(1,Tlbl_a)


    do_plot_model_mismatch((P_grid_a,E_grid_a, Tgrid_a),
                           (P_a,E_a,T_a,E_mod_a),
                           (Plims,Elims),(Plbl,Elbl),Tlbl_a,
                           Eresid_ticks,err_d['E'],hfig,ax_main_l[0],
                           ax_resid_l[0],xtick_lbl=True,show_colbar=True,
                           panel_lbl='')

    # do_plot_model_mismatch((Vgrid_a/V0, E_grid_a, Tgrid_a),
    #                        (V_a/V0,E_a,T_a,E_mod_a),
    #                        (Vlims/V0,Elims),(Vlbl,Elbl),Tlbl_a,
    #                        Eresid_ticks,err_d['E'],hfig,ax_main_l[0],
    #                        ax_resid_l[0],show_colbar=False,
    #                        panel_lbl='b')



    plt.draw()



    # # hfig, ax_main_l, ax_resid_l = setup_model_mismatch_plot(1)
    # do_plot_model_mismatch((P_grid_a, E_grid_a, Tgrid_a, ),
    #                              (P_a,E_a,T_a,E_mod_a),
    #                              (Plims,Elims),(Plbl,Elbl),Tlbl_a,
    #                              Eresid_ticks,err_d['E'],hfig,ax_main_l[0],
    #                        ax_resid_l[0])


    hfig.set_size_inches(8,10)
    hfig.set_size_inches(8,10)

    return ax_main_l, ax_resid_l
#====================================================================
def old_plot_model_fit_results(Tlbl_a,datamod_d,
                           Nsamp=101,Vbnds=[0.4,1.2]):
    eos_d = datamod_d['eos_d']
    full_mod = eos_d['modtype_d']['FullMod']

    V0 = eos_d['param_d']['V0']

    err_d = datamod_d['posterior_d']['err']

    Vgrid_a = np.linspace(Vbnds[0],Vbnds[1],Nsamp)*V0

    Tgrid_a = datamod_d['data_d']['Tavg_bins']
    dT = 500
    Tlbl_a = dT*np.round(Tgrid_a/dT)
    # Tlbl_a = np.array([2500,3000,3500,4000,4500,5000])


    E_grid_a, P_grid_a, dPdT_grid_a = eval_model( Vgrid_a, Tgrid_a, eos_d )
    E_mod_a = full_mod.energy(data_d['V'],data_d['T'],eos_d)
    P_mod_a = full_mod.press(data_d['V'],data_d['T'],eos_d)

    Plims=[-5.0,181.0]
    Vlims=[5.25,14.5]
    Elims=[-21.05,-18.95]
    Presid_ticks = [-2,0,2]
    Eresid_ticks = [-.02,0,.02]
    Plbl = r'Pressure  [GPa]'
    Elbl = r'Internal Energy  [eV / atom]'
    Vlbl = r'Volume  [$\AA^3$ / atom]'

    hfigPV = []
    hfigEV = []
    hfigEP = []
    # hfigEP = plot_model_mismatch((P_grid_a, E_grid_a, Tgrid_a, ),
    #                              (P_a,E_a,T_a,E_mod_a),
    #                              (Plims,Elims),(Plbl,Elbl),Tlbl_a,
    #                              Eresid_ticks,err_d['E'])

    hfig, ax_main_l, ax_resid_l = setup_model_mismatch_plot(2)


    do_plot_model_mismatch((Vgrid_a, P_grid_a, Tgrid_a),
                           (V_a,P_a,T_a,P_mod_a),
                           (Vlims,Plims),('',Plbl),Tlbl_a,
                           Presid_ticks,err_d['P'],hfig,ax_main_l[1],
                           ax_resid_l[1],xtick_lbl=False)

    do_plot_model_mismatch((Vgrid_a, E_grid_a, Tgrid_a),
                           (V_a,E_a,T_a,E_mod_a),
                           (Vlims,Elims),(Vlbl,Elbl),Tlbl_a,
                           Eresid_ticks,err_d['E'],hfig,ax_main_l[0],
                           ax_resid_l[0])

    plt.draw()
    plt.pause(2)



    # # hfig, ax_main_l, ax_resid_l = setup_model_mismatch_plot(1)
    # do_plot_model_mismatch((P_grid_a, E_grid_a, Tgrid_a, ),
    #                              (P_a,E_a,T_a,E_mod_a),
    #                              (Plims,Elims),(Plbl,Elbl),Tlbl_a,
    #                              Eresid_ticks,err_d['E'],hfig,ax_main_l[0],
    #                        ax_resid_l[0])


    # plt.draw()
    # plt.pause(2)

    # hfigEV = plot_model_mismatch((Vgrid_a, E_grid_a, Tgrid_a),
    #                              (V_a,E_a,T_a,E_mod_a),
    #                              (Vlims,Elims),(Vlbl,Elbl),Tlbl_a,
    #                              Eresid_ticks,err_d['E'])
    # plt.draw()
    # plt.pause(2)

    # hfigPV = plot_model_mismatch((Vgrid_a, P_grid_a, Tgrid_a),
    #                              (V_a,P_a,T_a,P_mod_a),
    #                              (Vlims,Plims),(Vlbl,Plbl),Tlbl_a,
    #                              Presid_ticks,err_d['P'])
    # plt.draw()
    # plt.pause(2)

    return hfigPV, hfigEV, hfigEP
#====================================================================
def plot_gamma_datamod(datamod_d):
    data_d = datamod_d['data_d']
    eos_d = datamod_d['eos_d']
    V_a = data_d['V']
    T_a = data_d['T']
    P_a = data_d['P']
    dPdT_a = data_d['dPdT']
    dEdT_a = data_d['dEdT']
    gamma_a = V_a*dPdT_a/dEdT_a/eos_d['const_d']['PV_ratio']

    therm_mod = eos_d['modtype_d']['ThermalMod']

    V0 = eos_d['param_d']['V0']
    Tavg_bins_a = datamod_d['data_d']['Tavg_bins']
    Vlims=np.array([datamod_d['data_d']['Vuniq'][0],datamod_d['data_d']['Vuniq'][-1]])
    Vlims=.02*(Vlims[1]-Vlims[0])*np.array([-1,1])+Vlims
    # Vlims=[5.9,14.1]
    N = 101
    Vmod_a = np.linspace(Vlims[0],Vlims[1],N)

    plt.figure()
    plt.clf()
    cmap_a = plt.cm.coolwarm(np.linspace(0,1,data_d['Tavg_bins'].size))
    cmap=plt.get_cmap('coolwarm',data_d['Tavg_bins'].size)
    plt.scatter(V_a/V0,gamma_a,c=T_a,s=50,lw=0,cmap=cmap)

    [plt.plot(Vmod_a/V0,therm_mod.calc_gamma( Vmod_a, iT, eos_d ),color=icmap)
     for (iT,icmap) in zip(Tavg_bins_a,cmap_a)]

    plt.xlim(Vlims/V0)
    plt.xlabel('$V/V_0$')
    plt.ylabel('$\gamma$')
#====================================================================
def plot_gamma_adiabat(eos_d,show_cbar=True,ax=None):
    therm_mod = eos_d['modtype_d']['ThermalMod']
    gamma_mod = eos_d['modtype_d']['GammaMod']

    T0 = eos_d['param_d']['T0']

    V0, = models.Control.get_params(['V0'],eos_d)
    TOL = 1e-6


    Vlims = [np.min(data_d['V']),np.max(data_d['V'])]

    Vpath_a = np.arange(0.4,1.101,.02)*V0
    # Tfoot_a = np.arange(1500.0,5100.0,500.0)
    dT = 250.0
    Tfoot_a = np.arange(2000.0,3500.1,dT)

    cmap_a = plt.cm.coolwarm(np.linspace(0,1,Tfoot_a.size))
    cmap=plt.get_cmap('coolwarm',Tfoot_a.size)

    indVfoot = np.where(np.abs(np.log(Vpath_a/V0))<TOL)[0][0]
    # Vgrid_ad_a, Tgrid_ad_a = get_adiabat_paths( Vpath_a, Tfoot_a, eos_d,
    #                                            indVfoot=indVfoot )
    # adiabat_d = get_adiabat_paths( Vpath_a, Tfoot_a, eos_d, indVfoot=indVfoot )
    adiabat_d = get_adiabat_paths( Vpath_a, Tfoot_a, eos_d )

    T_0S_a = gamma_mod.temp(V_a,T0,eos_d)


    if(show_cbar):
        plt.figure()
        plt.clf()
        cbarinfo = plt.scatter( adiabat_d['Pgrid'], adiabat_d['Tgrid'],
                               c=adiabat_d['Tfoot_grid'],s=50,lw=0,cmap=cmap)
        plt.clim(Tfoot_a[0]-0.5*dT,Tfoot_a[-1]+0.5*dT)
        plt.clf()

    if ax is None:
        ax = plt.gca()

    # [plt.plot(Vmod_a/V0,therm_mod.calc_gamma( Vmod_a, iT, eos_d ),color=icmap)
    #  for (iT,icmap) in zip(Tavg_bins_a,cmap_a)]

    for (iV_a,iT_a,iP_a,icmap) in zip(adiabat_d['Vgrid'], adiabat_d['Tgrid'],
                                      adiabat_d['Pgrid'],cmap_a):
        idV=iV_a[1]-iV_a[0]
        igamma_num_a = -iV_a/iT_a*np.gradient(iT_a,idV)

        # plt.plot(iV_a,therm_mod.calc_gamma( iV_a, iT_a, eos_d ),'r-',
        #          iV_a,igamma_num_a,'k--')
        #plt.plot(iV_a,therm_mod.calc_gamma( iV_a, iT_a, eos_d ),'r-')
        ax.plot(iP_a,therm_mod.calc_gamma( iV_a, iT_a, eos_d ),'-',
                 color=icmap, lw=2 )
        # plt.plot(iV_a,gamma_mod.gamma( V_a, eos_d ),'k--')

    Plims=[-1.0,181.0]
    gamlims = [-0.03, 1.43]
    ax.set_xlim(Plims)
    ax.set_ylim(gamlims)
    ax.set_xlabel('P [GPa]')
    ax.set_ylabel('$\gamma$')

    if(show_cbar):
        plt.colorbar(cbarinfo,ticks=Tfoot_a,label=r'$T_{\rm foot}$  [K]')
#====================================================================
def plot_corr_matrix(corr_a,param_a):

    Np = len(param_a)
    plt.figure()
    plt.clf()
    cmap=plt.get_cmap('coolwarm')
    corr_plt_a = np.flipud(np.ma.masked_where(np.eye(Np),corr_a))
    # corr_plt_a = np.flipud(corr_a)
    plt.pcolormesh(corr_plt_a,cmap=cmap);
    plt.xlim([0,11]);plt.ylim([0,11])

    plt.colorbar(label=r'Correlation Coefficient')
    plt.clim([-1,1])

    plt.xticks( np.arange(11)+.5, param_a, va='top')
    # plt.yticks( np.flipud(np.arange(11))+.5, param_a )
    # plt.yticks( np.flipud(np.arange(11)), param_a, verticalalignment='center' )
    plt.yticks( np.flipud(np.arange(11)+.5), param_a, va='center', ha='right' )

    for (index,val) in np.ndenumerate(np.flipud(corr_a)):
        if index[1]!=Np-1-index[0]:
            plt.text(index[1]+.5,index[0]+.5,'%+.2f'%(val),fontsize=10,
                     horizontalalignment='center', verticalalignment='center')

    plt.setp(plt.gca().get_xticklines(),visible=False)
    plt.setp(plt.gca().get_yticklines(),visible=False)

    plt.plot((0,11),(5,5),'k-',linewidth=2)
    plt.plot((0,11),(7,7),'k-',linewidth=2)
    plt.plot((4,4),(0,11),'k-',linewidth=2)
    plt.plot((6,6),(0,11),'k-',linewidth=2)
    pass
#====================================================================



#====================================================================
#  Fitting Scripts
#====================================================================

T0=2565
# set model conditions (V,T)
expand_adj = False
use_4th_order = False
# 4th order EOS only minorly improves low press regime (thermal press)
# use_4th_order = True

# from IPython import embed; embed(); import ipdb; ipdb.set_trace()
eos_d = init_params({},T0,expand_adj=expand_adj, use_4th_order=use_4th_order)

dat_og_d, data_d = load_Spera2011_data()
data_dk_d = load_deKoker2009_data()
miegrun_dk_d = load_deKoker_miegrun_eos(eos_d)


# Extract reference isotherm data

E_a = data_d['E']
P_a = data_d['P']
Perr_a = data_d['Perr']
V_a = data_d['V']
T_a = data_d['T']
Vmod_a = np.linspace(np.min(V_a),np.max(V_a),101)


NTgrid=6
calc_avg_isotherm( NTgrid, data_d )
print(data_d['Tavg_bins'])
Tbin_ref=0
def_ref_isotherm( Tbin_ref, data_d )

NTgrid_dk=5
calc_avg_isotherm( NTgrid_dk, data_dk_d )
print(data_dk_d['Tavg_bins'])
Tbin_ref=1
def_ref_isotherm( Tbin_ref, data_dk_d )


#
#
# eos_d = init_params({},T0,expand_adj=expand_adj, use_4th_order=use_4th_order)
# fit_compress_ref_isotherm( eos_d, data_d )
#
# fit_compress_ref_isotherm( eos_d, data_dk_d )
#
#
#
# plt.figure()
# Vmod_a = np.linspace(6.3,16,101)
# plt.clf()
# plt.plot(data_dk_d['Vref_T0'],data_dk_d['Pref_T0'],'ko')
# plt.plot(Vmod_a,eos_d['modtype_d']['CompressPathMod'].press(Vmod_a,eos_d),'r-')
#
# fit_compress_ref_isotherm( eos_d, miegrun_dk_d)
# plt.figure()
# Vmod_a = np.linspace(6.3,16,101)
# plt.clf()
# plt.plot(miegrun_dk_d['Vref_T0'],miegrun_dk_d['Pref_T0'],'ko')
# plt.plot(Vmod_a,eos_d['modtype_d']['CompressPathMod'].press(Vmod_a,eos_d),'r-')
#
#
# eos_d['param_d']['E0'] = -6.9
# plt.clf()
# plt.plot(data_dk_d['Vref_T0'],data_dk_d['Eref_T0'],'ko')
# plt.plot(Vmod_a,eos_d['modtype_d']['CompressPathMod'].energy(Vmod_a,eos_d),'r-')
#
# plt.plot(data_dk_d['Vref_T0'],data_dk_d['Eref_T0']\
#          -eos_d['modtype_d']['CompressPathMod'].energy(data_dk_d['Vref_T0'],eos_d),'ro')
#
#
#
#
# plt.ion()
# plt.figure()

# from IPython import embed; embed(); import ipdb; ipdb.set_trace()
estimate_thermal_grad(data_d)





# fit_init_model( eos_d, data_d )

#  eos_d['param_d']['E0']= -20.423196931407105
#  eos_d['param_d']['K0']= 14.607146602645738
#  eos_d['param_d']['KP0']= 7.798966693180998
#  eos_d['param_d']['S0']= 0.0
#  eos_d['param_d']['T0']= 3543.625
#  eos_d['param_d']['V0']= 13.181418129337608
#  eos_d['param_d']['gammaR']= 0.16721148807459488
#  eos_d['param_d']['gammapR']= -1.9751432068740147
#  eos_d['param_d']['mexp']= 0.6
#  bcoef_a = np.array([-4.9807654531761933, -5.2637856225462656,
#                      -0.0024262989460256388, 0.69597080683223489,
#                      1.1206165556997725])
#  models.Control.set_array_params( 'bcoef', bcoef_a, eos_d )
#  eos_d['param_d']['T0']= 2500.0

################################
# from IPython import embed; embed(); import ipdb; ipdb.set_trace()

#  gamma_mod, = models.Control.get_modtypes( ['GammaMod'], eos_d )
#  V0,T0 = models.Control.get_params( ['V0','T0'], eos_d )
#
#  gamma_mod.temp(V0*np.array([.6,.7,.8,.9,1.0,1.1]),np.array([T0]),eos_d)
#
#  thermal_mod, = models.Control.get_modtypes( ['ThermalMod'], eos_d )
#  thermal_mod.calc_entropy(V0*np.array([.6,.7,.8,.9,1.0,1.1]),
#                           np.array([T0]),eos_d)
#  thermal_mod.calc_entropy(V0*np.array([1.0]),
#                           T0*np.array([1.,1.5,2.0,2.5]),eos_d)
#
#  # Major issue with entropy calculation
#
#  thermal_mod.calc_entropy(V0*np.array([1.1]),
#                           np.array([2500.0,2750.0,3500.0]),eos_d)
#
#  from IPython import embed; embed(); import ipdb; ipdb.set_trace()
#  gamma_mod.temp(np.array([V0+1e-14]), np.array([2500.0]),eos_d)
#
#  thermal_mod.calc_entropy(np.array([V0+1e-10]), np.array([2500.0]),eos_d)
#  thermal_mod.calc_entropy(np.array([V0+1e-14]), np.array([2500.0,2750.0,3500.0]),eos_d)
#
#  gamma_mod.temp(np.array([V0+1e-10]), np.array([2500.0]),eos_d)
#
#  gamma_mod.temp(np.array([V0]),2500,eos_d)



# adiabat_d = get_adiabat_paths( Vpath_a, Tfoot_a, eos_d, indVfoot=indVfoot )
# eos_d['param_d']['KP20']=-1.5

V0 = eos_d['param_d']['V0']
Nsamp = 101

Vgrid_a = np.linspace(0.4,1.1,Nsamp)*V0
Tgrid_a = data_d['Tavg_bins']
Tlbl_a = np.array([2500,3000,3500,4000,4500,5000])


# Test heat capacity
fit_model_l = ['CompressPathMod','GammaMod','ThermalMod']
fix_param_l = ['T0','lognfac','mexp']
# fix_param_l = ['T0','lognfac']
prior_d = datamod.init_param_distbn( fit_model_l,eos_d, fix_param_l)
datamod_d = datamod.init_datamod( data_d, prior_d, eos_d,
                                 fit_data_type=['E','P'])


plt.ion()


# eos_init_d = copy.deepcopy(eos_d)
# datamod_d['posterior_d'] = datamod_d['prior_d']

eos_d = datamod_d['eos_d']
Tlbl_a = np.array([2500,3000,3500,4000,4500,5000])

from IPython import embed; embed(); import ipdb; ipdb.set_trace()

#
#
#   # def fit_miegrun_dat( miegrun_d, data_d ):
#   #     Edat_a = data_d['E']
#   #     Pdat_a = data_d['P']
#   #     Tdat_a = data_d['T']
#   #     Vdat_a = data_d['V']
#
#   # gam_mod = models.GammaFiniteStrain()
#
#   gam_mod = models.GammaPowLaw()
#   compress_mod = models.Vinet()
#   eos_dk_d = init_params({},T0,expand_adj=expand_adj, use_4th_order=use_4th_order)
#
#   # GammaPowLaw
#   # 2 params for gamma
#   eos_dk_d['param_d']['gamma0']=0.58
#   eos_dk_d['param_d']['q0']=-0.88
#   plt.plot(miegrun_dk_d['V'],miegrun_dk_d['gamma'],'k-')
#   plt.plot(miegrun_dk_d['V'],gam_mod.gamma(miegrun_dk_d['V'],eos_dk_d),'r-')
#
#   # log(V/V0) polynomial
#   # 3 params for Ptherm
#   pcoef_pth_a = np.polyfit(np.log(miegrun_dk_d['V']/V0),miegrun_dk_d['Ptherm'],3)
#   plt.plot(miegrun_dk_d['V'],miegrun_dk_d['Ptherm'],'k-')
#   plt.plot(miegrun_dk_d['V'],np.polyval(pcoef_pth_a,np.log(miegrun_dk_d['V']/V0)),'r-')
#
#   # pcoef_pth_a = np.polyfit(miegrun_dk_d['V'],np.log(miegrun_dk_d['Ptherm']),3)
#   # plt.plot(miegrun_dk_d['V'],miegrun_dk_d['Ptherm'],'k-')
#   # plt.plot(miegrun_dk_d['V'],np.exp(np.polyval(pcoef_pth_a,miegrun_dk_d['V'])),'r-')
#
#   # Vinet
#   # 3 params for Pref
#   eos_dk_d['param_d']['KP0']=7
#   eos_dk_d['param_d']['K0']=17
#   plt.plot(miegrun_dk_d['V'],miegrun_dk_d['Pref'],'k-')
#   plt.plot(miegrun_dk_d['V'],compress_mod.press(miegrun_dk_d['V'],eos_dk_d),'r-')
#
#
#   # Poly Eref
#   # 3 params for Eref
#   Eresid_a = miegrun_dk_d['Eref']-compress_mod.energy(miegrun_dk_d['V'],eos_dk_d)
#   pcoef_Eresid_a = np.polyfit(np.log(miegrun_dk_d['V']/V0),Eresid_a,2)
#   plt.clf()
#   plt.plot(miegrun_dk_d['V'],Eresid_a,'ko',
#            miegrun_dk_d['V'],np.polyval(pcoef_Eresid_a,np.log(miegrun_dk_d['V']/V0)),'r-')
#
#
#
#   plt.plot(miegrun_dk_d['V'],gam_mod.gamma(miegrun_dk_d['V'],eos_dk_d),'r-')
#
#
#
#   # miegrun_dk_d['modtype_d']['FullMod'].press(V_a, 2000,miegrun_dk_d)
# full_mod_dk = miegrun_dk_d['modtype_d']['FullMod']
# # full_mod_dk.press(V_a, 2000,miegrun_dk_d)
#
# V_a = miegrun_dk_d['V']
# #
# #
# plt.scatter(data_dk_d['V'],data_dk_d['P'],c=data_dk_d['T'], s=50, lw=0, cmap='viridis')
# cmap = plt.get_cmap()
#
# plt.plot(V_a,full_mod_dk.press(V_a,2000,miegrun_dk_d),'b-',
#          V_a,full_mod_dk.press(V_a,3000,miegrun_dk_d),'c-',
#          V_a,full_mod_dk.press(V_a,4000,miegrun_dk_d),'g-',
#          V_a,full_mod_dk.press(V_a,6000,miegrun_dk_d),'r-',
#          V_a,full_mod_dk.press(V_a,8000,miegrun_dk_d),'k-')
#
#
#   plt.clf()
#   plt.plot(V_a,full_mod_dk.energy(V_a,2000,miegrun_dk_d),'b-',
#            V_a,full_mod_dk.energy(V_a,3000,miegrun_dk_d),'c-',
#            V_a,full_mod_dk.energy(V_a,4000,miegrun_dk_d),'g-',
#            V_a,full_mod_dk.energy(V_a,5000,miegrun_dk_d),'r-',
#            V_a,full_mod_dk.energy(V_a,6000,miegrun_dk_d),'k-')
#
#
#   plt.plot(V_a,P3K_a,'k-',lw=2)
#
#
#   plt.clf()
#   plt.plot(V_a,E2K_a,'b-',
#            V_a,E3K_a,'c-',
#            V_a,E4K_a,'g-',
#            V_a,E5K_a,'r-',
#            V_a,E6K_a,'k-')
#   plt.scatter(data_dk_d['V'],data_dk_d['E'],c=data_dk_d['T'])
#
#   data_dk_d
#
#
#



#########################
#  First fit with T0=2565 (avg value)
#########################

# eos_d['param_d']['T0'] = 2565.0
print('T0 = '+ str(eos_d['param_d']['T0']))
datamod.fit( datamod_d )
print datamod_d['posterior_d']['err']
print('\r')
plot_model_fit_results(Tlbl_a,datamod_d)
plt.title(str(eos_d['param_d']['T0']))


def plot_rtpress_model_fig():
    plt.rc('text', usetex=True)

    lbl_siz = 16-2
    par_lbl_siz = lbl_siz-2
    par_col = np.array([43,71,20])/255.
    par_col = np.array([78,128,0])/255.

    f, ax_a = plt.subplots(2, 2, sharex='col')
    f.set_size_inches(8,6)
    # f.subplots_adjust(hspace=0.05,wspace=0.25,left=.1,right=.97,top=.95)
    f.subplots_adjust(hspace=0.03,wspace=0.18,left=.05,right=.97,top=.97,bottom=.05)
    plt.draw()

    # ax_P = plt.subplot(221)
    # ax_E = plt.subplot(222)
    # ax_T = plt.subplot(223)
    # ax_Cpv = plt.subplot(224)

    # Fix ticks
    # ax_P.set_xticklabels([])
    # ax_E.set_xticklabels([])

    # Set labels
    ax_a[0,0].set_ylabel(r'$P$')
    ax_a[1,1].set_ylabel(r'$E$')
    ax_a[0,1].set_ylabel(r'$C_{P/V}$')
    ax_a[1,0].set_ylabel(r'$T$')
    ax_a[1,0].set_xlabel(r'$V$')
    ax_a[1,1].set_xlabel(r'$T$')


    ax_a[0,0].set_yticklabels([])
    ax_a[1,0].set_yticklabels([])
    ax_a[0,1].set_yticklabels([])
    ax_a[1,1].set_yticklabels([])

    ax_a[1,0].set_xticklabels([])
    ax_a[1,1].set_xticklabels([])

    plt.draw()

    Vbnds=[0.4,1.2]
    eos_d = datamod_d['eos_d']
    full_mod = eos_d['modtype_d']['FullMod']
    gamma_mod = eos_d['modtype_d']['GammaMod']
    V0 = eos_d['param_d']['V0']

    Vgrid_a = np.linspace(Vbnds[0],Vbnds[1],Nsamp)*V0

    Tgrid_a = np.array([2500,7000])
    E_grid_a, P_grid_a, dPdT_grid_a = eval_model( Vgrid_a, Tgrid_a, eos_d )

    Vdat_a = V0*np.array([0.5,.6,.7,.8,.9,1.0])
    Pdat_a = full_mod.press(Vdat_a,Tgrid_a[0],eos_d)
    Pdat_hi_a = full_mod.press(Vdat_a,Tgrid_a[1],eos_d)

    # P-V plot
    ax_a[0,0].plot(Vgrid_a/V0,P_grid_a[0],'k-',
              Vgrid_a/V0,P_grid_a[1],'r-')
    ax_a[0,0].plot(Vdat_a/V0,Pdat_a,'ko',mew=2,markeredgecolor='k')
    ax_a[0,0].plot(Vdat_a/V0,Pdat_hi_a,'ro',mew=2,markeredgecolor='r')
    ax_a[0,0].text(.65,17-12,r'$P(V,T_0)$',horizontalalignment='right',fontsize=lbl_siz)
    # ax_a[0,0].text(.705,2,r"$p_1= \{V_0,K_0,K'_0\}$",horizontalalignment='right',
    #                fontsize=par_lbl_siz,color=par_col)
    ax_a[0,0].text(.65,85,r"1. Isothermal Compression "+"\n"+r"$\;\;\;\;\; \{V_0,K_0,K'_0\}$",
                   horizontalalignment='left', fontsize=par_lbl_siz,color=par_col,
                   bbox=dict(lw=2,boxstyle="round", fc=[.95,1,0.95], ec=par_col))


    ax_a[0,0].text(.65,60,r'$P(V,T)$',horizontalalignment='left',color='r',fontsize=lbl_siz)

    ax_a[0,0].text(.57,170+5,r"$\rm{(I)\,Isothermal\,Compression:}$"+
                   r" $P(V,T_0)$"+"\n"+r"$\rm{(II)\,Thermal\,Pressure\,Deriv:}$"+"\n"+
                   r"$\;\;\;\;\;\;\;\;\left.\frac{dP}{dT}\right|_V = \left.\frac{dS}{dV}\right|_T$",horizontalalignment='left',
                   verticalalignment='top', fontsize=par_lbl_siz,
                   bbox=dict(lw=2,boxstyle="square", fc=".9", ec='k'))
    # ax_a[0,0].text(.53,150+5,r"$\rm{(II)\,Thermal\,Pressure\,Deriv:}$"+"\n"+
    #                r"$\;\;\;\;\;\;\left.\frac{dP}{dT}\right|_V = \left.\frac{dS}{dV}\right|_T$",horizontalalignment='left',
    #                verticalalignment='top', fontsize=par_lbl_siz)
    # ax_a[0,0].text(.6,140,r"$\rm{(II)\,Thermal\,Pressure}$"+"\n"+
    #                r"$\;\;P(V,T_0)$",horizontalalignment='left',
    #                verticalalignment='top', fontsize=par_lbl_siz)

    ax_a[0,0].text(0.95,30,r"$+\Delta T$",verticalalignment='top',
                   horizontalalignment='left',fontsize=par_lbl_siz+6,color='r')

    ax_a[0,0].set_xlim(0.49,1.11)
    ax_a[0,0].set_ylim(-2,190)


    for (iV,iP0,iPT) in zip(Vdat_a/V0,Pdat_a,Pdat_hi_a):
        idy = iPT-iP0
        ax_a[0,0].annotate("", xy=(iV, iPT), xycoords='data',
                      xytext=(iV, iP0), textcoords='data',
                      arrowprops=dict( facecolor='r',shrink=0.0,
                                      width=3.,headwidth=6.,headlength=3.0,edgecolor='k'))
        plt.draw()



    T0 = eos_d['param_d']['T0']

    # T-V plot

    ax_a[1,0].plot(Vgrid_a/V0, gamma_mod.temp( Vgrid_a,T0,eos_d)/T0,'k-')
    ax_a[1,0].plot(Vdat_a/V0, gamma_mod.temp( Vdat_a,T0,eos_d)/T0,'ko')

    ax_a[1,0].set_xlim(0.49,1.11)
    ax_a[1,0].set_ylim(2400/T0,4700/T0)
    ax_a[1,0].text(.52,1.27-.22-.05,r'$T_{0S}(V)$',horizontalalignment='left',fontsize=lbl_siz)
    # ax_a[1,0].text(.52,1.18-.22,r"$p_2= \{\gamma_0,\gamma_0'\}$",color=par_col,
    #                horizontalalignment='left',fontsize=par_lbl_siz)
    ax_a[1,0].text(.72,1.35,r"2. Adiabatic Reference: "+"\n"+r"$\;\;\;\;\; \{\gamma_0,\gamma_0'\}$",color=par_col,
                   horizontalalignment='left',fontsize=par_lbl_siz,
                   bbox=dict(lw=2,boxstyle="round", fc=[.95,1,0.95], ec=par_col))

    ax_a[1,0].text(0.58,1.75,r"$\rm{(IIIb)\,Adiabatic\,Temperature}$"+
                   r"$\rm{\,Deriv:}$"+"\n"+ r" $\;\;\;\;\;\;\left.\frac{dT}{dV}\right|_{S} = -\left.\frac{dS}{dV}\right|_{T} \; / \; \left.\frac{dS}{dT}\right|_{V} $",horizontalalignment='left',
                   verticalalignment='top', fontsize=par_lbl_siz,
                   bbox=dict(lw=2,boxstyle="square", fc=".9", ec='k'))

    annotation.slope_marker((.721, 1.21), -1.15, labels=(r'$dT$',r'$dV$'),
                            ax=ax_a[1,0])


    # E-T plot
    E0 = eos_d['param_d']['E0']
    Tdat_a = T0*np.array([.6,1.0,1.4,1.8,2.2,2.6,3.0])
    Edat_a = full_mod.energy(V0,Tdat_a,eos_d)
    Tgrid_a = T0*np.linspace(.5,3.2,1001)
    Egrid_a = full_mod.energy(V0,Tgrid_a,eos_d)

    ax_a[1,1].plot(Tdat_a/T0, Edat_a-E0,'ko')
    ax_a[1,1].plot(Tgrid_a/T0, Egrid_a-E0,'k-')
    ax_a[1,1].set_xlim(.9,3.1)
    # ax_a[1,1].set_ylim(-.3,1.7)
    ax_a[1,1].set_ylim(-.4,2.0)

    ax_a[1,1].text(1.5,0.4-.65,r'$E(V_0,T)$',horizontalalignment='right',fontsize=lbl_siz)
    ax_a[1,1].text(1.2,1.5+.3,r"$\rm{(IIIc)\,Thermal\,Energy\,Deriv:}$"+
                   "\n"+r"$\;\;\;\;\;\;\left.\frac{dE}{dT}\right|_V = \left. T \, \frac{dS}{dT}\right|_V$",horizontalalignment='left',
                   verticalalignment='top', fontsize=par_lbl_siz,
                   bbox=dict(lw=2,boxstyle="square", fc=".9", ec='k'))

    annotation.slope_marker((1.5, 0.35), 0.85, labels=(r'$dE$',r'$dT$'),
                            ax=ax_a[1,1])


    # Cv-T plot
    Cv_dat_a = full_mod.heat_capacity(V0,Tdat_a,eos_d)/eos_d['const_d']['kboltz']
    Cv_dat_hiP_a = full_mod.heat_capacity(0.5*V0,Tdat_a,eos_d)/eos_d['const_d']['kboltz']
    Cv_grid_a = full_mod.heat_capacity(V0,Tgrid_a,eos_d)/eos_d['const_d']['kboltz']
    Cv_grid_hiP_a = full_mod.heat_capacity(0.5*V0,Tgrid_a,eos_d)/eos_d['const_d']['kboltz']
    ax_a[0,1].plot(Tgrid_a/T0, Cv_grid_a,'k-', Tdat_a/T0,Cv_dat_a,'ko')
    ax_a[0,1].plot(Tgrid_a/T0, Cv_grid_hiP_a,'b-')

    ax_a[0,1].set_xlim(.9,3.1)
    #ax_a[0,1].set_ylim(2.9,4.4)
    ax_a[0,1].set_ylim(3.3,5.1)

    # ax_a[0,1].text(0.75,4.15,r'$C_{P/V}(T)$',horizontalalignment='left',fontsize=lbl_siz)
    ax_a[0,1].text(1.5,3.35-.26+.3,r'$C_{P/V}(T)$',horizontalalignment='right',fontsize=lbl_siz)
    # ax_a[0,1].text(1.05,3.2-.23,r"$p_3= \{b_0,...,b_n\}$",color=par_col,
    #                horizontalalignment='left',fontsize=par_lbl_siz)
    ax_a[0,1].text(1.7,3.8+.6,r"3. RT Thermal Coeffs: "+"\n"+r"$\;\;\;\;\; \{b_0,...,b_n\}$",color=par_col,
                   horizontalalignment='left',fontsize=par_lbl_siz,
                   bbox=dict(lw=2,boxstyle="round", fc=[.95,1,0.95], ec=par_col))

    # ax_a[0,1].text(2.5,3.03,"low press",verticalalignment='top',
    #                horizontalalignment='left',fontsize=par_lbl_siz,color='k')
    # ax_a[0,1].text(2.5,3.47,"high press",verticalalignment='top',
    #                horizontalalignment='left',fontsize=par_lbl_siz,color='b')
    ax_a[0,1].text(2.6,3.45+.45,r"$+\Delta P$",verticalalignment='top',
                   horizontalalignment='left',fontsize=par_lbl_siz+6,color='b')

    ax_a[0,1].text(1.3,4.3+.6,r"$\rm{(IIIa)\,Heat\,Capacity:}$"+
                   r" $C_{P/V} = T\,\frac{dS}{dT}$",horizontalalignment='left',
                   verticalalignment='top', fontsize=par_lbl_siz,
                   bbox=dict(lw=2,boxstyle="square", fc=".9", ec='k'))

    for (iT,iCv0,iCvT) in zip(Tdat_a/T0,Cv_dat_a,Cv_dat_hiP_a):
        ax_a[0,1].annotate("", xy=(iT, iCvT), xycoords='data',
                      xytext=(iT, iCv0), textcoords='data',
                      arrowprops=dict( facecolor='b',shrink=0.0,
                                      width=3.,headwidth=6.,headlength=3.0,edgecolor='k'))
        plt.draw()

    plt.draw()
    return ax_a

# Make RTpress model construction plot
ax_a = plot_rtpress_model_fig()
plt.savefig('figs/MgSiO3-rtpress-modelfit-Spera2011.eps')
plt.savefig('figs/MgSiO3-rtpress-modelfit-Spera2011.png',dpi=450)



#########################
#  Refit with round number
#########################
eos_d['param_d']['T0'] = 3000.0
print('T0 = '+ str(eos_d['param_d']['T0']))
T0 = eos_d['param_d']['T0']
datamod.fit( datamod_d )
print('\r')


######################
#  Plot model fit
# No overplot of deKoker sims
######################
plot_model_fit_results(Tlbl_a,datamod_d)
plt.gcf().set_size_inches(8,10)
plt.gcf().set_size_inches(8,10)

plt.savefig('figs/MgSiO3-rtpress-PVEmod-Spera2011.eps')
plt.savefig('figs/MgSiO3-rtpress-PVEmod-Spera2011.png',dpi=450)

# plt.savefig('figs/MgSiO3-rtpress-powlaw-PVEmod-Spera2011.eps')

ax_main_l,ax_resid_l = plot_model_fit_results_EP(Tlbl_a,datamod_d)
plt.savefig('figs/MgSiO3-rtpress-PEmod-Spera2011.eps')



R2fit_d = datamod_d['posterior_d']['R2fit']

####################################
#          Model fit error
#  Write out R^2 values for E and P fit
####################################
print datamod_d['posterior_d']['err']
print(u'$R^2_E$ = %.5f'%np.round(R2fit_d['E'],decimals=5))
print(u'$R^2_P$ = %.5f'%np.round(R2fit_d['P'],decimals=5))


datamod_draw_d = copy.deepcopy(datamod_d)
eos_draw_d = datamod.eos_posterior_draw(datamod_d)
datamod_draw_d['eos_d'] = eos_draw_d
plot_model_fit_results(Tlbl_a,datamod_draw_d)


##################################
#  Load solidus/liquidus from Stixrude 2009
##################################

fusion_data = 'data/MgSiO3-fus-Stixrude2009.csv'
liquidus_data = 'data/MgSiO3-liq-Stixrude2009.csv'
solidus_data = 'data/MgSiO3-sol-Stixrude2009.csv'

PTfus = np.loadtxt(fusion_data,delimiter=',')
PTliq = np.loadtxt(liquidus_data,delimiter=',')
PTsol = np.loadtxt(solidus_data,delimiter=',')

indliq = np.where(PTliq[:,0]>24)[0]
indsol = np.where(PTsol[:,0]>24)[0]

#polyfus = np.polyfit(PTfus[:,0],PTfus[:,1],3)

polyliq = np.polyfit(PTliq[indliq,0],PTliq[indliq,1],3)
polysol = np.polyfit(PTsol[indsol,0],PTsol[indsol,1],3)
#polyfus = 0.5*(polyliq+polysol)


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

def infer_vol_PT( P_a, T_a, eos_d ):
    P_a, T_a = fill_array( P_a, T_a )

    TOL = 1e-6
    FAC_STEP = 0.1

    logfacshift = 0.05
    V0 = eos_d['param_d']['V0']

    full_mod = eos_d['modtype_d']['FullMod']

    V_a = []
    for P, T in zip( P_a, T_a ):
        xinit = -0.5
        # ensure that initial value has physically sensible positive value
        while( full_mod.bulk_modulus( np.exp(xinit)*V0, T, eos_d ) < 0 ):
            xinit = xinit-logfacshift
        # Pmisfit_f = lambda xadj:press_f( np.exp(xadj)*V0, T, eos_d )-P
        # delP = Pmisfit_f( xinit )
        # xadj = xinit
        # while( abs(delP) > TOL ):
        #     bulk_mod = bulk_mod_T_f( np.exp(xadj)*V0, T, eos_d )
        #     xadj += FAC_STEP*delP/bulk_mod
        #     delP = Pmisfit_f( xadj )

        # # Take full step at end to finish off
        # bulk_mod = bulk_mod_T_f( np.exp(xadj)*V0, T, eos_d )
        # xadj += 1.0*delP/bulk_mod
        # delP = Pmisfit_f( xadj )

        xadj, output_d, ier, mesg = optimize.fsolve(
            (lambda xadj:full_mod.press( np.exp(xadj)*V0, T, eos_d )-P), xinit,
            full_output=1,
            fprime=lambda xadj:-full_mod.bulk_modulus( np.exp(xadj)*V0, T, eos_d ) )
        delP = output_d['fvec']
        if np.abs( delP ) > TOL:
            xadj = np.nan
        V_a.append( np.squeeze(np.exp(xadj)*V0) )

    return np.asarray( V_a )

dTfus = 0.07
Pmod = np.linspace(0,140,100)
Pmod = np.linspace(0,200,100)

Tmod_liq_a = np.polyval(polyliq,Pmod)

#polysol = (1+dTfus/2.0)*polyfus
#polyliq = (1-dTfus/2.0)*polyfus
#Pmod,np.polyval(polyfus,Pmod),'r--',\
#Pmod, (1+dTfus/2.0)*np.polyval(polyfus,Pmod),'r-',\
#Pmod, (1-dTfus/2.0)*np.polyval(polyfus,Pmod),'r-',\
plt.plot(PTliq[:,0],PTliq[:,1],'ko',PTsol[:,0],PTsol[:,1],'kx',
         Pmod, np.polyval(polysol,Pmod),'gx-',\
         Pmod, np.polyval(polyliq,Pmod),'gx-')
plt.xlabel('P [GPa]')
plt.ylabel('T [K]')
plt.show()

full_mod = eos_d['modtype_d']['FullMod']


Pliq_a = Pmod
Tliq_a = Tmod_liq_a

# Pliq_a = PTliq[:,0]
# Tliq_a = PTliq[:,1]
Vliq_a = infer_vol_PT(Pliq_a,Tliq_a,eos_d)
Sliq_a = full_mod.entropy(Vliq_a,Tliq_a,eos_d)


const_d = eos_d['const_d']

plt.clf()
plt.plot(Pliq_a,Sliq_a/const_d['kboltz'],'k-')
plt.show()

Psol_a = Pmod
Tmod_sol_a = np.polyval(polysol,Pmod)
Tsol_a = Tmod_sol_a

eos_solid_d = {}
init_solid_params(eos_solid_d)

# should be about 65%
eos_solid_d['param_d']['Cvmax']/(eos_solid_d['const_d']['kboltz']*3)

full_solid_mod = eos_solid_d['modtype_d']['FullMod']
thermal_solid_mod = eos_solid_d['modtype_d']['ThermalMod']
V0_solid = eos_solid_d['param_d']['V0']

# thermal_mod.calc_energy(8,300,eos_solid_d)
# thermal_mod.calc_energy(8,3000,eos_solid_d)
#
# thermal_mod.press(8,300,eos_solid_d)
# thermal_mod.press(8,3000,eos_solid_d)
#
# full_solid_mod.press(8,300,eos_solid_d)
# full_solid_mod.press(8,3000,eos_solid_d)
#
# full_solid_mod.energy(8,300,eos_solid_d)
# full_solid_mod.energy(8,3000,eos_solid_d)
#
# full_solid_mod.bulk_modulus(8,300,eos_solid_d)
# full_solid_mod.bulk_modulus(8,3000,eos_solid_d)

Vsol_a = infer_vol_PT(Psol_a,Tsol_a,eos_solid_d)
Ssol_a = full_mod.entropy(Vsol_a,Tsol_a,eos_solid_d)

plt.figure()
plt.plot(Psol_a,Ssol_a/const_d['kboltz'],'k-')
plt.show()

#########################################
#     Adjust melt entropy up
#       to get reasonable dSmelt
#########################################

def eval_Sfus( Pfus, Tfus, eos_solid_d=eos_solid_d, eos_melt_d=eos_d ):

    full_solid_mod = eos_solid_d['modtype_d']['FullMod']
    full_melt_mod = eos_melt_d['modtype_d']['FullMod']

    Vfus_solid = infer_vol_PT( Pfus, Tfus, eos_solid_d )
    Vfus_melt = infer_vol_PT( Pfus, Tfus, eos_melt_d )

    Sfus = (full_melt_mod.entropy(Vfus_melt,Tfus,eos_melt_d) \
            - full_solid_mod.entropy(Vfus_solid,Tfus,eos_solid_d))

    # Sfus = (eos_melt_d['entropy_f']( Vfus_melt, Tfus, eos_melt_d ) \
    #         - eos_solid_d['entropy_f']( Vfus_solid, Tfus, eos_solid_d
    #                                    ))/param_d['Cv_const']/8.31446
    return Sfus


Pfus = 25
Tfus = 2900
eval_Sfus(Pfus,Tfus)

Vfus_solid = infer_vol_PT( Pfus, Tfus, eos_solid_d )
Vfus_melt = infer_vol_PT( Pfus, Tfus, eos_d )


dSfus = 1.5*const_d['kboltz'] #  eV/K, taken fromm Stixrude(2009)
dS0 = dSfus - eval_Sfus(Pfus,Tfus,eos_solid_d=eos_solid_d,eos_melt_d=eos_d)
eos_d['param_d']['S0'] += dS0
eval_Sfus(Pfus,Tfus,eos_solid_d=eos_solid_d, eos_melt_d=eos_d)/const_d['kboltz']



Vliq_a = infer_vol_PT(Pliq_a,Tliq_a,eos_d)
Sliq_a = full_mod.entropy(Vliq_a,Tliq_a,eos_d)

Vsol_a = infer_vol_PT(Psol_a,Tsol_a,eos_solid_d)
Ssol_a = full_mod.entropy(Vsol_a,Tsol_a,eos_solid_d)

plt.figure()
plt.plot(Pliq_a,Sliq_a/const_d['kboltz'],'r-',Psol_a,Ssol_a/const_d['kboltz'],'b-')
plt.show()

Pfus = np.linspace(0, 150, 301)
Tliq_a = np.polyval(polyliq,Pfus)
Tsol_a = np.polyval(polysol,Pfus)

Vliq_a = infer_vol_PT(Pfus,Tliq_a,eos_d)
Sliq_a = full_mod.entropy(Vliq_a,Tliq_a,eos_d)

Vsol_a = infer_vol_PT(Pfus,Tsol_a,eos_solid_d)
Ssol_a = full_mod.entropy(Vsol_a,Tsol_a,eos_solid_d)

import collections
import dataio

def set_output_constants(eos_d):
    const_d = eos_d['const_d']
    param_d = eos_d['param_d']

    output_d = collections.OrderedDict()
    output_d['header_default'] = "Pressure, Entropy, Quantity\n"\
        "column * scaling factor should be SI units\n"\
        "scaling factors (constant) for each column given on line below\n"
    #output_d['datadir'] = 'data/lookup/lookup-rough/'
    #output_d['datadir'] = 'data/lookup/lookup-hires/'
    #output_d['datadir'] = 'data/lookup/lookup-hires-RTmelt/'
    output_d['datadir'] = '../data/lookup/lookup-hires-RTpress/orig/'

    # NOTE: All extensive quantities changed from per atom to per unit mass

    # All output constants in mks units
    output_d['1'] = 1 # No unit change
    output_d['g'] = 1e-3 # mass [g] -> [kg]
    output_d['per_mass']  = const_d['Nmol']/(param_d['mass_avg']*output_d['g'])

    output_d['GPa'] = 1e9 # P:  [GPa] -> [Pa]
    output_d['GPa-1'] = 1e-9 # 1/P:  [1/GPa] -> [1/Pa]
    output_d['eV'] = 1.60217657e-19 \
        *output_d['per_mass'] # E, H, T*S, T*Cp:  [eV/atom] -> [J/kg]
    output_d['g_cc'] = 1e3 # rho:  [g/cc] -> [kg/m^3]
    eos_d['output_d'] = output_d
    pass


set_output_constants(eos_d)
set_output_constants(eos_solid_d)

dataio.write_data_table('liquidus.dat', (Pfus, Sliq_a), ('GPa', 'eV'), eos_d['output_d'] )
dataio.write_data_table('solidus.dat', (Pfus, Ssol_a), ('GPa', 'eV'), eos_solid_d['output_d'] )

##################
#  Calculate PS grids
##################
Ngrid_P = 151
Ngrid_S= 100

# Ngrid_P = 151
# Ngrid_S= 10


Pprof_a = np.linspace(0,200,Ngrid_P)

def get_adiabat_prof_grid(Pprof_a, Tpot_a, Vfac_a, eos_d, TOL=1e-6):
    V0, = models.Control.get_params(['V0'],eos_d)
    Vprof_a = Vfac_a*V0


    indVfoot = np.where(np.abs(np.log(Vprof_a/V0))<TOL)[0][0]

    adiabat_d = get_adiabat_paths( Vprof_a, Tpot_a, eos_d, indVfoot=indVfoot, TOL=1e-10 )

    Sprof_a = adiabat_d['Sfoot']
    Vprof_grid_a = np.zeros((len(Tpot_a),len(Pprof_a)))
    Tprof_grid_a = np.zeros((len(Tpot_a),len(Pprof_a)))
    for ind,(Tad_a,Pad_a) in enumerate(zip(adiabat_d['Tgrid'],adiabat_d['Pgrid'])):
        # print(np.min(Pad_a))
        # print(np.max(Pad_a))
        Vfun = interpolate.interp1d(Pad_a,Vprof_a,kind='cubic')
        Tfun = interpolate.interp1d(Pad_a,Tad_a,kind='cubic')
        Vprof_grid_a[ind] = Vfun(Pprof_a)
        Tprof_grid_a[ind] = Tfun(Pprof_a)

    adiabat_prof_grid_d = {}
    adiabat_prof_grid_d['Pprof_a'] = Pprof_a
    adiabat_prof_grid_d['Sprof_a'] = Sprof_a
    adiabat_prof_grid_d['Tpot_prof_a'] = Tpot_a

    adiabat_prof_grid_d['Vprof_grid_a'] = Vprof_grid_a
    adiabat_prof_grid_d['Tprof_grid_a'] = Tprof_grid_a
    adiabat_prof_grid_d['adiabat_VT_d'] = adiabat_d

    return adiabat_prof_grid_d

Tpot_melt_a = np.linspace(1500.,6000.,Ngrid_S)
Tpot_melt_a = np.linspace(1350.,6000.,Ngrid_S)
# Tpot_melt_a = np.linspace(1500.,3000.,Ngrid_S)

# Vfac_melt_a = np.arange(0.5,1.25,.001)
Vfac_melt_a = np.arange(0.4,1.25,.001)


TOL=1e-6
V0, = models.Control.get_params(['V0'],eos_d)
Vprof_a = Vfac_melt_a*V0
indVfoot = np.where(np.abs(np.log(Vprof_a/V0))<TOL)[0][0]
adiabat_d = get_adiabat_paths( Vprof_a, Tpot_melt_a, eos_d, indVfoot=indVfoot, TOL=1e-10 )


plt.plot(Vfac_melt_a,adiabat_d['Pgrid'][0],'k-')
plt.plot(Vfac_melt_a,adiabat_d['Pgrid'][1],'r-')



adiabat_prof_grid_melt_d = get_adiabat_prof_grid(Pprof_a, Tpot_melt_a, Vfac_melt_a, eos_d)

Tgrid=adiabat_prof_grid_melt_d['adiabat_VT_d']['Tgrid']
Pgrid=adiabat_prof_grid_melt_d['adiabat_VT_d']['Pgrid']
plt.plot(Tgrid[0],'ko')

plt.figure()
plt.imshow(adiabat_prof_grid_melt_d['adiabat_VT_d']['Tgrid'])
plt.colorbar()
plt.gca().invert_yaxis()
plt.show()


Tpot_solid_a = np.linspace(250.,5000.,Ngrid_S)
Vfac_solid_a = np.arange(0.4,1.35,.004)
adiabat_prof_grid_solid_d = get_adiabat_prof_grid(Pprof_a, Tpot_solid_a, Vfac_solid_a, eos_solid_d)

##################
#  Plot figs
##################

plt.figure()
plt.imshow(adiabat_prof_grid_melt_d['Tprof_grid_a'])
plt.colorbar()
plt.gca().invert_yaxis()
plt.show()

plt.figure()
plt.imshow(adiabat_prof_grid_melt_d['Vprof_grid_a'])
plt.colorbar()
plt.gca().invert_yaxis()
plt.show()


plt.figure()
plt.imshow(adiabat_prof_grid_solid_d['Tprof_grid_a'])
plt.colorbar()
plt.gca().invert_yaxis()
plt.show()

plt.figure()
plt.imshow(adiabat_prof_grid_solid_d['Vprof_grid_a'])
plt.colorbar()
plt.gca().invert_yaxis()
plt.show()



def eos_property_mesh(adiabat_prof_grid_d, eos_d):
    # (Vmesh_a, Tmesh_a, eos_d, Pmesh_a=None, Smesh_a=None):

    Pmesh_a, Smesh_a = np.meshgrid( adiabat_prof_grid_d['Pprof_a'],
                                   adiabat_prof_grid_d['Sprof_a'] )

    Vmesh_a = adiabat_prof_grid_d['Vprof_grid_a'].ravel()
    Tmesh_a = adiabat_prof_grid_d['Tprof_grid_a'].ravel()

    mass_avg = eos_d['param_d']['mass_avg']
    const_d = eos_d['const_d']
    rho_const = (mass_avg/const_d['Nmol'])*const_d['ang3percc']


    full_mod = eos_d['modtype_d']['FullMod']
    gamma_mod = eos_d['modtype_d']['GammaMod']
    thermal_mod = eos_d['modtype_d']['ThermalMod']

    energy_f = full_mod.energy
    press_f = full_mod.press
    entropy_f = full_mod.entropy
    heat_capacity_V_f = full_mod.heat_capacity
    gamma_f = thermal_mod.calc_gamma
    bulk_mod_T_f = full_mod.bulk_modulus
    therm_exp_f = full_mod.therm_exp


    rhomesh_a = rho_const/Vmesh_a
    Emesh_a = energy_f( Vmesh_a, Tmesh_a, eos_d )

    Pmesh_a = Pmesh_a.ravel()
    Smesh_a = Smesh_a.ravel()

    Cvmesh_a = heat_capacity_V_f( Vmesh_a, Tmesh_a, eos_d )
    gammamesh_a = gamma_f( Vmesh_a, Tmesh_a, eos_d )
    Hmesh_a = Emesh_a + Pmesh_a*Vmesh_a*const_d['PV_ratio']
    KTmesh_a = bulk_mod_T_f( Vmesh_a, Tmesh_a, eos_d )
    alphamesh_a = therm_exp_f( Vmesh_a, Tmesh_a, eos_d )

    logalphamesh_a = np.log(alphamesh_a)

    Cpmesh_a = Cvmesh_a*(1+alphamesh_a*gammamesh_a*Tmesh_a)
    KSmesh_a = KTmesh_a*(1+alphamesh_a*gammamesh_a*Tmesh_a)
    dTdP_Smesh_a = gammamesh_a*(Tmesh_a/KSmesh_a) # K/GPa

    eos_prop_d = collections.OrderedDict()
    eos_prop_d['Vmesh_a'] = Vmesh_a
    eos_prop_d['Tmesh_a'] = Tmesh_a
    eos_prop_d['Pmesh_a'] = Pmesh_a
    eos_prop_d['Emesh_a'] = Emesh_a
    eos_prop_d['Hmesh_a'] = Hmesh_a
    eos_prop_d['Smesh_a'] = Smesh_a
    eos_prop_d['rhomesh_a'] = rhomesh_a
    eos_prop_d['Cvmesh_a'] = Cvmesh_a
    eos_prop_d['Cpmesh_a'] = Cpmesh_a
    eos_prop_d['KTmesh_a'] = KTmesh_a
    eos_prop_d['KSmesh_a'] = KSmesh_a
    eos_prop_d['gammamesh_a'] = gammamesh_a
    eos_prop_d['alphamesh_a'] = alphamesh_a
    eos_prop_d['logalphamesh_a'] = logalphamesh_a
    eos_prop_d['dTdP_Smesh_a'] = dTdP_Smesh_a

    return eos_prop_d


eos_prop_melt_d = eos_property_mesh(adiabat_prof_grid_melt_d, eos_d)
eos_prop_solid_d = eos_property_mesh(adiabat_prof_grid_solid_d, eos_solid_d)


def write_all_data_tables( phasename, eos_prop_d, output_d ):
    """
    Write phase-specific data files
    """

    dataio.write_data_table( 'temperature_' + phasename + '.dat',
                            (eos_prop_d[key] for key in
                             ('Pmesh_a', 'Smesh_a', 'Tmesh_a')),
                            ('GPa', 'eV', 1), output_d )

    dataio.write_data_table( 'density_' + phasename + '.dat',
                     (eos_prop_d[key] for key in
                      ('Pmesh_a', 'Smesh_a', 'rhomesh_a')),
                     ('GPa', 'eV','g_cc'), output_d )
    dataio.write_data_table( 'heat_capacity_' + phasename + '.dat',
                     (eos_prop_d[key] for key in
                      ('Pmesh_a', 'Smesh_a', 'Cpmesh_a')),
                     ('GPa','eV','eV'), output_d )

    dataio.write_data_table( 'thermal_exp_' + phasename + '.dat',
                            (eos_prop_d[key] for key in
                             ('Pmesh_a', 'Smesh_a', 'alphamesh_a')),
                            ('GPa','eV',1), output_d )

    dataio.write_data_table( 'adiabat_temp_grad_' + phasename + '.dat',
                     (eos_prop_d[key] for key in
                      ('Pmesh_a', 'Smesh_a', 'dTdP_Smesh_a')),
                     ('GPa','eV','GPa-1'), output_d )
    pass

write_all_data_tables( 'melt', eos_prop_melt_d, eos_d['output_d'] )
write_all_data_tables( 'solid', eos_prop_solid_d, eos_solid_d['output_d'] )






##################################
#  Compare EOS properties
#
#   properties at 0GPa are somewhat less accurate with only 3rd order
#   compression curve
################################
eos_d = datamod_d['eos_d']
full_mod = eos_d['modtype_d']['FullMod']
const_d = eos_d['const_d']

V0 = eos_d['param_d']['V0']

Tref = 1773.0
Pref = 0.0


logVf = optimize.fsolve(lambda logVf:full_mod.press(np.exp(logVf)*V0,Tref,eos_d)-Pref,0)
V = V0*np.exp(logVf)
full_mod.press(V,Tref,eos_d)

Natom_pfu = 5
Natom_pfu*V/const_d['ang3percc']*const_d['Nmol']
full_mod.bulk_modulus(V,Tref,eos_d)
full_mod.heat_capacity(V,Tref,eos_d)/eos_d['const_d']['kboltz']

# ax_main_l, ax_resid_l = plot_model_fit_results(Tlbl_a,datamod_d)
# PV_dK09_3K_a = np.loadtxt('data/PV-3000-deKoker2009.csv',delimiter=',')
# PV_dK09_4K_a = np.loadtxt('data/PV-4000-deKoker2009.csv',delimiter=',')
#
# V_dK09_3K_a = PV_dK09_3K_a[:,0]*38.9*1e-5*eos_d['param_d']['mass_avg']/eos_d['const_d']['Nmol']/1e3*1e30
# V_dK09_4K_a = PV_dK09_4K_a[:,0]*38.9*1e-5*eos_d['param_d']['mass_avg']/eos_d['const_d']['Nmol']/1e3*1e30
#
#
#
# V0 = eos_d['param_d']['V0']
# full_mod = eos_d['modtype_d']['FullMod']
# P_mod_4K_a = full_mod.press(V_dK09_4K_a,4000,eos_d)
#
# plt.figure()
# plt.plot(V_dK09_4K_a/V0,P_mod_4K_a,'k-',
#          V_dK09_4K_a/V0,PV_dK09_4K_a[:,1],'k--')
#
#
# ax_main_l[1].plot(PV_dK09_3K_a[:,0],PV_dK09_3K_a[:,1],'b-')
# ax_main_l[1].plot(V_dK09_4K_a,PV_dK09_4K_a[:,1],'r-')
#
#
# ax_main_l[1].plot(PV_dK09_4K_a[:,0],PV_dK09_4K_a[:,1],'r-')
#
# plt.draw()



#########################
#   Calculate hugoniot to compare with shock data
#########################

def calc_hugoniot_RTgrid( Vlims, rhoinit, Tinit, RTgrid_d, eos_d,
                             Etrans=0, Ttrans=None, Nsamp=30):

    PV_ratio = eos_d['const_d']['PV_ratio']

    Pinit = 0
    V_a = np.linspace(Vlims[0],Vlims[1],Nsamp)
    rho_a =1/V_a*eos_d['param_d']['mass_avg']/const_d['Nmol']*const_d['ang3percc']

    Vinit =1/rhoinit*eos_d['param_d']['mass_avg']/const_d['Nmol']*const_d['ang3percc']


    if Ttrans is None:
        Ttrans = Tinit

    Emelt0, Vmelt0 = calc_EV_RTgrid(Pinit, Ttrans, RTgrid_d, showplot=False)
    Einit = Emelt0-Etrans

    hugoniot_d = {}
    hugoniot_d['rhoinit'] = rhoinit
    hugoniot_d['Tinit'] = Tinit
    hugoniot_d['Einit'] = Einit
    hugoniot_d['Ttrans'] = Ttrans
    hugoniot_d['Emelt0'] = Emelt0
    hugoniot_d['Vmelt0'] = Vmelt0
    hugoniot_d['Etrans'] = Etrans

    E_a = np.nan*np.ones(V_a.shape)
    P_a = np.nan*np.ones(V_a.shape)
    T_a = np.nan*np.ones(V_a.shape)



    for ind,V in enumerate(V_a):
        def hugoniot_dev( delE ):

            E = delE + Einit
            P, T = calc_PT_RTgrid(E, V, RTgrid_d, showplot=False)

            Pavg = 0.5*(P+Pinit)
            delV = V-Vinit
            Eshock = -delV*Pavg/PV_ratio
            Edev = delE-Eshock
            return Edev

        delE = optimize.fsolve(hugoniot_dev,1.0)
        Ehug = delE + Einit
        Phug, Thug = calc_PT_RTgrid(Ehug, V, RTgrid_d, showplot=False)

        E_a[ind] = Ehug
        P_a[ind] = Phug
        T_a[ind] = Thug


    hugoniot_d['P_a'] = P_a
    hugoniot_d['T_a'] = T_a
    hugoniot_d['E_a'] = E_a
    hugoniot_d['V_a'] = V_a
    hugoniot_d['rho_a'] = rho_a

    return hugoniot_d

def calc_hugoniot_miegrun( Vlims, rhoinit, Tinit, miegrun_eos_d, eos_d,
                             Etrans=0, Ttrans=None, Nsamp=30):

    PV_ratio = eos_d['const_d']['PV_ratio']

    Pinit = 0
    V_a = np.linspace(Vlims[0],Vlims[1],Nsamp)
    rho_a =1/V_a*eos_d['param_d']['mass_avg']/const_d['Nmol']*const_d['ang3percc']

    Vinit =1/rhoinit*eos_d['param_d']['mass_avg']/const_d['Nmol']*const_d['ang3percc']


    if Ttrans is None:
        Ttrans = Tinit

    Emelt0, Vmelt0 = calc_EV_RTgrid(Pinit, Ttrans, RTgrid_d, showplot=False)
    Einit = Emelt0-Etrans

    hugoniot_d = {}
    hugoniot_d['rhoinit'] = rhoinit
    hugoniot_d['Tinit'] = Tinit
    hugoniot_d['Einit'] = Einit
    hugoniot_d['Ttrans'] = Ttrans
    hugoniot_d['Emelt0'] = Emelt0
    hugoniot_d['Vmelt0'] = Vmelt0
    hugoniot_d['Etrans'] = Etrans

    E_a = np.nan*np.ones(V_a.shape)
    P_a = np.nan*np.ones(V_a.shape)
    T_a = np.nan*np.ones(V_a.shape)



    for ind,V in enumerate(V_a):
        def hugoniot_dev( delE ):

            E = delE + Einit
            P, T = calc_PT_RTgrid(E, V, RTgrid_d, showplot=False)

            Pavg = 0.5*(P+Pinit)
            delV = V-Vinit
            Eshock = -delV*Pavg/PV_ratio
            Edev = delE-Eshock
            return Edev

        delE = optimize.fsolve(hugoniot_dev,1.0)
        Ehug = delE + Einit
        Phug, Thug = calc_PT_RTgrid(Ehug, V, RTgrid_d, showplot=False)

        E_a[ind] = Ehug
        P_a[ind] = Phug
        T_a[ind] = Thug


    hugoniot_d['P_a'] = P_a
    hugoniot_d['T_a'] = T_a
    hugoniot_d['E_a'] = E_a
    hugoniot_d['V_a'] = V_a
    hugoniot_d['rho_a'] = rho_a

    return hugoniot_d

def calc_hugoniot_miegrun_dev( rho, rhoinit, Tinit, eos_d, Etrans=0, Ttrans=None,
                              isobar_trans=True, logTfaclim=[-1.5,1.5], Nsamp=3001 ):

    full_mod = eos_d['modtype_d']['FullMod']
    PV_ratio = eos_d['const_d']['PV_ratio']
    const_d = eos_d['const_d']

    Pinit=0

    # rho=1/V*eos_d['param_d']['mass_avg']/const_d['Nmol']*const_d['ang3percc']
    # rhoinit=1/Vinit*eos_d['param_d']['mass_avg']/const_d['Nmol']*const_d['ang3percc']

    V=1/rho*eos_d['param_d']['mass_avg']/const_d['Nmol']*const_d['ang3percc']
    Vinit=1/rhoinit*eos_d['param_d']['mass_avg']/const_d['Nmol']*const_d['ang3percc']

    T_a = Tinit*np.logspace(logTfaclim[0],logTfaclim[1],Nsamp)

    if Ttrans is None:
        Ttrans = Tinit

    if isobar_trans:
        V0 = eos_d['param_d']['V0']
        logVfac = optimize.fsolve(lambda logVfac: full_mod.press(np.exp(logVfac)*V0,Ttrans,eos_d)-Pinit,0.0)
        Vtrans = np.exp(logVfac)*V0
    else:
        Vtrans = Vinit

    Einit = full_mod.energy(Vtrans, Ttrans, eos_d)-Etrans
    # Pinit = full_mod.press(Vinit, Tinit, eos_d)

    E_a = full_mod.energy(V, T_a, eos_d)
    delE_a = E_a-Einit

    P_a = full_mod.press(V, T_a, eos_d)
    Pavg_a = 0.5*(P_a+Pinit)

    delV = V-Vinit

    Eshock_a = -delV*Pavg_a/PV_ratio
    Edev_a = delE_a-Eshock_a

    indmin = np.argmin(np.abs(Edev_a))

    Ploc_a = P_a[indmin-3:indmin+3]
    Tloc_a = T_a[indmin-3:indmin+3]
    Eloc_a = Eshock_a[indmin-3:indmin+3]
    Edevloc_a = Edev_a[indmin-3:indmin+3]

    Thug = interpolate.interp1d(Edevloc_a, Tloc_a, kind='cubic')(0)
    Phug = interpolate.interp1d(Edevloc_a, Ploc_a, kind='cubic')(0)
    Ehug = interpolate.interp1d(Edevloc_a, Eloc_a, kind='cubic')(0)

    # optimize.fsolve(

    # Phug = P_a[indmin]
    # Thug = T_a[indmin]
    # return delE_a-Eshock_a

    hugoniot_dev_d = {}
    hugoniot_dev_d['rho'] = rho
    hugoniot_dev_d['rhoinit'] = rhoinit
    hugoniot_dev_d['V'] = V
    hugoniot_dev_d['Vinit'] = Vinit
    hugoniot_dev_d['Tinit'] = Tinit
    hugoniot_dev_d['Einit'] = Einit
    hugoniot_dev_d['Etrans'] = Etrans
    hugoniot_dev_d['T_a'] = T_a
    hugoniot_dev_d['P_a'] = P_a
    hugoniot_dev_d['E_a'] = E_a
    hugoniot_dev_d['delE_a'] = delE_a
    hugoniot_dev_d['Eshock_a'] = Eshock_a
    hugoniot_dev_d['Phug'] = Phug
    hugoniot_dev_d['Thug'] = Thug
    hugoniot_dev_d['Ehug'] = Ehug
    hugoniot_dev_d['indmin'] = indmin

    return hugoniot_dev_d

def calc_hugoniot_dev( rho, rhoinit, Tinit, eos_d, Etrans=0, Ttrans=None,
                      isobar_trans=True, logTfaclim=[-1.5,1.5], Nsamp=3001,Pshft=0 ):

    full_mod = eos_d['modtype_d']['FullMod']
    PV_ratio = eos_d['const_d']['PV_ratio']
    const_d = eos_d['const_d']

    Pinit=0

    # rho=1/V*eos_d['param_d']['mass_avg']/const_d['Nmol']*const_d['ang3percc']
    # rhoinit=1/Vinit*eos_d['param_d']['mass_avg']/const_d['Nmol']*const_d['ang3percc']

    V=1/rho*eos_d['param_d']['mass_avg']/const_d['Nmol']*const_d['ang3percc']
    Vinit=1/rhoinit*eos_d['param_d']['mass_avg']/const_d['Nmol']*const_d['ang3percc']

    T_a = Tinit*np.logspace(logTfaclim[0],logTfaclim[1],Nsamp)

    if Ttrans is None:
        Ttrans = Tinit

    if isobar_trans:
        V0 = eos_d['param_d']['V0']
        logVfac = optimize.fsolve(lambda logVfac: full_mod.press(np.exp(logVfac)*V0,Ttrans,eos_d)+Pshft-Pinit,0.0)
        Vtrans = np.exp(logVfac)*V0
    else:
        Vtrans = Vinit

    Einit = full_mod.energy(Vtrans, Ttrans, eos_d)-Etrans
    # Pinit = full_mod.press(Vinit, Tinit, eos_d)

    E_a = full_mod.energy(V, T_a, eos_d)
    delE_a = E_a-Einit

    P_a = full_mod.press(V, T_a, eos_d)+Pshft
    Pavg_a = 0.5*(P_a+Pinit)

    delV = V-Vinit

    Eshock_a = -delV*Pavg_a/PV_ratio
    Edev_a = delE_a-Eshock_a

    indmin = np.argmin(np.abs(Edev_a))

    Ploc_a = P_a[indmin-3:indmin+3]
    Tloc_a = T_a[indmin-3:indmin+3]
    Eloc_a = Eshock_a[indmin-3:indmin+3]
    Edevloc_a = Edev_a[indmin-3:indmin+3]

    Thug = interpolate.interp1d(Edevloc_a, Tloc_a, kind='cubic')(0)
    Phug = interpolate.interp1d(Edevloc_a, Ploc_a, kind='cubic')(0)
    Ehug = interpolate.interp1d(Edevloc_a, Eloc_a, kind='cubic')(0)

    # optimize.fsolve(

    # Phug = P_a[indmin]
    # Thug = T_a[indmin]
    # return delE_a-Eshock_a

    hugoniot_dev_d = {}
    hugoniot_dev_d['rho'] = rho
    hugoniot_dev_d['rhoinit'] = rhoinit
    hugoniot_dev_d['V'] = V
    hugoniot_dev_d['Vinit'] = Vinit
    hugoniot_dev_d['Tinit'] = Tinit
    hugoniot_dev_d['Einit'] = Einit
    hugoniot_dev_d['Etrans'] = Etrans
    hugoniot_dev_d['T_a'] = T_a
    hugoniot_dev_d['P_a'] = P_a
    hugoniot_dev_d['E_a'] = E_a
    hugoniot_dev_d['delE_a'] = delE_a
    hugoniot_dev_d['Eshock_a'] = Eshock_a
    hugoniot_dev_d['Phug'] = Phug
    hugoniot_dev_d['Thug'] = Thug
    hugoniot_dev_d['Ehug'] = Ehug
    hugoniot_dev_d['indmin'] = indmin

    return hugoniot_dev_d

def calc_hugoniot( rhofaclims, rhoinit, Tinit, eos_d, Etrans=0, Ttrans=None,
                  isobar_trans=True, Nsamp=30,Pshft=0 ):


    rho_a = rhoinit*np.linspace(rhofaclims[0],rhofaclims[1],Nsamp)
    T_a = np.zeros(rho_a.shape)
    P_a = np.zeros(rho_a.shape)
    E_a = np.zeros(rho_a.shape)
    V_a = np.zeros(rho_a.shape)

    hugoniot_d = {}
    hugoniot_d['rhoinit'] = rhoinit
    hugoniot_d['Tinit'] = Tinit
    hugoniot_d['Etrans'] = Etrans


    for ind,rho in enumerate(rho_a):
        hugoniot_dev_d = calc_hugoniot_dev(rho,rhoinit,Tinit,eos_d,Etrans=Etrans,
                                           Ttrans=Ttrans,isobar_trans=isobar_trans,Pshft=Pshft)
        P_a[ind] = hugoniot_dev_d['Phug']
        T_a[ind] = hugoniot_dev_d['Thug']
        E_a[ind] = hugoniot_dev_d['Ehug']
        V_a[ind] = hugoniot_dev_d['V']


    hugoniot_d['P_a'] = P_a
    hugoniot_d['T_a'] = T_a
    hugoniot_d['E_a'] = E_a
    hugoniot_d['V_a'] = V_a
    hugoniot_d['rho_a'] = rho_a

    return hugoniot_d
#====================================================================

RTgrid_d = fit_RTgrid_eos(data_d,mexp=3./5,RTorder=1,showfits=False)

# RTgrid_dk_d = fit_RTgrid_eos(data_dk_d, mexp=3./5, RTorder=2, showfits=True)
RTgrid_dk_d = fit_RTgrid_eos(data_dk_d, mexp=0.866, RTorder=1, showfits=False)



const_d = eos_d['const_d']
# Vinit = 12.466366810397778
# Vinit = 1/rhoinit*eos_d['param_d']['mass_avg']/const_d['Nmol']*const_d['ang3percc']
# Tinit = 300.0
#
# Etrans = 2.192455e6*1e-3/const_d['JperHa']*const_d['eVperHa']*eos_d['param_d']['mass_avg']/const_d['Nmol']
#
# T_a = Tinit*np.logspace(-3,2,1001)
#
# Vfac=.57
# rho=1/(Vfac*Vinit)*eos_d['param_d']['mass_avg']/const_d['Nmol']*const_d['ang3percc']


#
# rhoinit=np.mean(shock_dat['rho0'][melt_en_mask])
# Tinit = 300
# Tmelt = 1816
# Efus = 73*1e3/5/const_d['JperHa']*const_d['eVperHa']/const_d['Nmol']
# Eheat = 126*(Tmelt-273)/5/const_d['JperHa']*const_d['eVperHa']/const_d['Nmol']
# Etrans = Eheat+Efus
# rhofac = 1.8
# rhofac = 1.5
# rho = rhoinit*rhofac
# hugoniot_dev_d = calc_hugoniot_dev( rho, rhoinit, Tinit, eos_d, Etrans=Etrans,Ttrans=Tmelt,isobar_trans=True)
# print(hugoniot_dev_d['Thug'])
# plt.clf()
# plt.plot(hugoniot_dev_d['T_a'],hugoniot_dev_d['delE_a'],'k--',hugoniot_dev_d['T_a'],hugoniot_dev_d['Eshock_a'],'r-')


# rhofac = 1.5
# rho = rhoinit*rhofac
# hugoniot_dev_d = calc_hugoniot_dev( rho, rhoinit, Tinit, miegrun_dk_d, Etrans=Etrans,Ttrans=Tmelt,isobar_trans=True)
#
# rhofac=1/.57
# rhofac = 1.5873
# # rhofac = 1.59 # rhofac = 1.86 # hugoniot_dev_d = calc_hugoniot_dev( rhoinit*rhofac, rhoinit, Tinit, eos_d, Etrans=Etrans) # print(np.array([hugoniot_dev_d['Phug'], hugoniot_dev_d['Thug']]))
################
#   Load data
################
shock_dat = pd.read_csv('data/shock-data-Mosenfelder2009.csv',delimiter=',')
melt_en_mask = ((shock_dat['Starting Material']=='Enstatite')&(shock_dat['Phase State']=='melt'))
melt_glass_mask = ((shock_dat['Starting Material']=='Glass')&(shock_dat['Phase State']=='melt'))

melt_enpor_mask = ((shock_dat['Starting Material']=='Porous Enstatite')&(shock_dat['Phase State']=='melt'))
melt_oxmix_mask = ((shock_dat['Starting Material']=='Oxide mix')&(shock_dat['Phase State']=='melt'))

Tinit = 300



# hugoniot_glass_d = calc_hugoniot( rhofaclims_glass, rhoinit_glass, Tinit, eos_d, Etrans=Etrans_glass)

# Enstatite Hugoniot
rhoinit_en=np.mean(shock_dat['rho0'][melt_en_mask])
rhofaclims_en = [1.59, 1.84]
# Etrans_en = 2.192455e6*1e-3/const_d['JperHa']*const_d['eVperHa']*eos_d['param_d']['mass_avg']/const_d['Nmol']
Tmelt_en = 1816
#Tmelt_en = 300
Efus_en = 73*1e3/5/const_d['JperHa']*const_d['eVperHa']/const_d['Nmol']
Eheat_en = 126*(Tmelt_en-273)/5/const_d['JperHa']*const_d['eVperHa']/const_d['Nmol']
Etrans_en = Eheat_en+Efus_en


# hugoniot_en_d = calc_hugoniot_RTgrid( [6.3,14.0], rhoinit_en, Tinit, RTgrid_d, eos_d, Etrans=Etrans_en, Ttrans=Tmelt_en )
# hugoniot_dk_en_d = calc_hugoniot_RTgrid( [6.3,14.0], rhoinit_en, Tinit, RTgrid_dk_d, eos_d, Etrans=Etrans_en, Ttrans=Tmelt_en )


# Glass Hugoniot
rhoinit_glass = np.mean(shock_dat['rho0'][melt_glass_mask])
rhofaclims_glass = [1.65, 1.94]
# Etrans_glass = 1.862455e6*1e-3/const_d['JperHa']*const_d['eVperHa']*eos_d['param_d']['mass_avg']/const_d['Nmol']
delE_glass = (2.192455e6-1.862455e6)*1e-3/const_d['JperHa']*const_d['eVperHa']*eos_d['param_d']['mass_avg']/const_d['Nmol']
Etrans_glass = Etrans_en - delE_glass


# # Porous Hugoniots
# delE_oxmix = (2.192455e6-1.952455e6)*1e-3/const_d['JperHa']*const_d['eVperHa']*eos_d['param_d']['mass_avg']/const_d['Nmol']
# rhoinit_oxmix = np.mean(shock_dat['rho0'][melt_oxmix_mask])
# Etrans_oxmix = Etrans_en - delE_oxmix

rhoinit_enpor=np.mean(shock_dat['rho0'][melt_enpor_mask])
# hugoniot_glass_d = calc_hugoniot_RTgrid( [6.3,14.0], rhoinit_glass, Tinit, RTgrid_d, eos_d, Etrans=Etrans_glass, Ttrans=Tmelt_en )
# hugoniot_dk_glass_d = calc_hugoniot_RTgrid( [6.3,14.0], rhoinit_glass, Tinit, RTgrid_dk_d, eos_d, Etrans=Etrans_glass, Ttrans=Tmelt_en )



# NOTE WARNING: something is not quite right about the hugoniot error estimation


rhofaclims_en = np.array([1.6, 1.8])
hugoniot_en_d = calc_hugoniot( rhofaclims_en, rhoinit_en, Tinit, eos_d,
                              Etrans=Etrans_en, Ttrans=Tmelt_en, isobar_trans=True)
hugoniot_dk_en_d = calc_hugoniot( rhofaclims_en, rhoinit_en, Tinit, miegrun_dk_d,
                                 Etrans=Etrans_en, Ttrans=Tmelt_en, isobar_trans=True)

# rhofaclims_enpor = np.array([1.7, 1.8])
# hugoniot_enpor_d = calc_hugoniot( rhofaclims_enpor, rhoinit_enpor, Tinit, eos_d,
#                               Etrans=Etrans_en, Ttrans=Tmelt_en, isobar_trans=True)
# hugoniot_dk_enpor_d = calc_hugoniot( rhofaclims_enpor, rhoinit_enpor, Tinit, miegrun_dk_d,
#                               Etrans=Etrans_en, Ttrans=Tmelt_en, isobar_trans=True)


rhofaclims_glass = [1.65, 1.9]
hugoniot_glass_d = calc_hugoniot( rhofaclims_glass, rhoinit_glass, Tinit, eos_d, Etrans=Etrans_glass, Ttrans=Tmelt_en, isobar_trans=True)
hugoniot_dk_glass_d = calc_hugoniot( rhofaclims_glass, rhoinit_glass, Tinit, miegrun_dk_d, Etrans=Etrans_glass, Ttrans=Tmelt_en, isobar_trans=True)


# rhofaclims_oxmix = [2.1, 2.55]
# hugoniot_oxmix_d = calc_hugoniot( rhofaclims_oxmix, rhoinit_oxmix, Tinit, eos_d, Etrans=Etrans_oxmix, Ttrans=Tmelt_en, isobar_trans=True)
# hugoniot_dk_oxmix_d = calc_hugoniot( rhofaclims_oxmix, rhoinit_oxmix, Tinit, miegrun_dk_d, Etrans=Etrans_oxmix, Ttrans=Tmelt_en, isobar_trans=True)

#############################
# Plot Hugoniot Comparison
#############################

Plim = [50,225]
col_glass = [0,0,0]
col_en = [.65,.65,.65]
col_en = [.5,.5,.5]
f, ax_a = plt.subplots(2, 1, sharex='col')

ax_a[0].errorbar(shock_dat['P'][melt_glass_mask], shock_dat['rho'][melt_glass_mask],
             xerr=shock_dat['P err'][melt_glass_mask],
             yerr=shock_dat['rho err'][melt_glass_mask],fmt='.',color=col_glass)
ax_a[0].errorbar(shock_dat['P'][melt_en_mask], shock_dat['rho'][melt_en_mask],
             xerr=shock_dat['P err'][melt_en_mask],
             yerr=shock_dat['rho err'][melt_en_mask],fmt='.',color=col_en)
# ax_a[0].errorbar(shock_dat['P'][melt_enpor_mask], shock_dat['rho'][melt_enpor_mask],
#              xerr=shock_dat['P err'][melt_enpor_mask],
#              yerr=shock_dat['rho err'][melt_enpor_mask],fmt='m.')
# ax_a[0].errorbar(shock_dat['P'][melt_oxmix_mask], shock_dat['rho'][melt_oxmix_mask],
#              xerr=shock_dat['P err'][melt_oxmix_mask],
#              yerr=shock_dat['rho err'][melt_oxmix_mask],fmt='g.')

ax_a[0].plot(hugoniot_glass_d['P_a'],hugoniot_glass_d['rho_a'],'-',color=col_glass)
ax_a[0].plot(hugoniot_en_d['P_a'],hugoniot_en_d['rho_a'],'-',color=col_en)
# ax_a[0].plot(hugoniot_oxmix_d['P_a'],hugoniot_oxmix_d['rho_a'],'g-')
# ax_a[0].plot(hugoniot_enpor_d['P_a'],hugoniot_enpor_d['rho_a'],'m-')

ax_a[0].plot(hugoniot_dk_glass_d['P_a'],hugoniot_dk_glass_d['rho_a'],'--',color=col_glass)
ax_a[0].plot(hugoniot_dk_en_d['P_a'],hugoniot_dk_en_d['rho_a'],'--',color=col_en)
# ax_a[0].plot(hugoniot_dk_oxmix_d['P_a'],hugoniot_dk_oxmix_d['rho_a'],'g--')
# ax_a[0].plot(hugoniot_dk_enpor_d['P_a'],hugoniot_dk_enpor_d['rho_a'],'m--')

ax_a[0].set_xlim(Plim[0],Plim[1])
ax_a[0].set_ylabel(u'Density [g/cm$^3$]')

ax_a[1].plot(hugoniot_glass_d['P_a'],hugoniot_glass_d['T_a'],'-',color=col_glass)
ax_a[1].plot(hugoniot_en_d['P_a'],hugoniot_en_d['T_a'],'-',color=col_en)
# ax_a[1].plot(hugoniot_oxmix_d['P_a'],hugoniot_oxmix_d['T_a'],'g-')

ax_a[1].plot(hugoniot_dk_glass_d['P_a'],hugoniot_dk_glass_d['T_a'],'--',color=col_glass)
ax_a[1].plot(hugoniot_dk_en_d['P_a'],hugoniot_dk_en_d['T_a'],'--',color=col_en)
# ax_a[1].plot(hugoniot_dk_oxmix_d['P_a'],hugoniot_dk_oxmix_d['T_a'],'g--')

ax_a[1].errorbar(shock_dat['P'][melt_glass_mask], shock_dat['TH'][melt_glass_mask],
                 xerr=shock_dat['P err'][melt_glass_mask],
                 yerr=shock_dat['TH err'][melt_glass_mask],fmt='.',color=col_glass)
ax_a[1].errorbar(shock_dat['P'][melt_en_mask], shock_dat['TH'][melt_en_mask],
             xerr=shock_dat['P err'][melt_en_mask],
             yerr=shock_dat['TH err'][melt_en_mask],fmt='.',color=col_en)

ax_a[1].set_xlim(Plim[0],Plim[1])
ax_a[1].set_xlabel('Pressure  [GPa]')
ax_a[1].set_ylabel('Temperature  [K]')

ax_a[1].set_ylim(2000,8000)



ax_a[0].text(80,4.78,'glass hugoniot',fontsize=12,color=col_glass,
             verticalalignment='center',horizontalalignment='center',rotation=20)
ax_a[0].text(150,5.5,'enstatite hugoniot',fontsize=12,color=col_en,
             verticalalignment='center',horizontalalignment='center',rotation=15)

# ax_a[0].plot(136*np.ones(2),ax_a[0].get_ylim(),'r:',lw=2)
# ax_a[1].plot(136*np.ones(2),ax_a[1].get_ylim(),'r:',lw=2)

ax_a[0].axvspan(136,Plim[1],color = [.95,.95,.95])
ax_a[1].axvspan(136,Plim[1],color = [.95,.95,.95])

# ax_a[0].axvspan(50,136,color = [1,.9,.9])
# ax_a[1].axvspan(50,136,color = [1,.9,.9])

# ax_a[0].text(90,5.75,'Terrestrial Mantle Region',fontsize=14,color='k',
#              horizontalalignment='center')
# ax_a[1].text(90,7000,'Earth-like Mantle Region',fontsize=14,color='k',
#              horizontalalignment='center')
# ax_a[0].text(90,5.75,'Terrestrial Mantle Region',fontsize=14,color='k',
#              horizontalalignment='center')
# ax_a[1].text(90,7000,'Terrestrial Mantle Region',fontsize=14,color='k',
#              horizontalalignment='center')
ax_a[1].text(90,7000,'Terrestrial Mantle\nRegion',fontsize=14,color='k',
             verticalalignment='top',horizontalalignment='center',weight='bold')

plt.draw()
plt.tight_layout(h_pad=.15)


plt.savefig('figs/MgSiO3-hugoniot-compare.eps')
plt.savefig('figs/MgSiO3-hugoniot-compare.png',dpi=450)












################################
# Random Errors on hugoniot are small
#   this proves unnecessary
################################

# for i in range(5):
#     eos_draw_d = datamod.eos_posterior_draw(datamod_d)
#     hugoniot_en_draw_d = calc_hugoniot( rhofaclims_en, rhoinit_en, Tinit,
#                                        eos_draw_d, Etrans=Etrans_en,
#                                        Ttrans=Tmelt_en, isobar_trans=True)
#     hugoniot_glass_draw_d = calc_hugoniot( rhofaclims_glass, rhoinit_glass,
#                                           Tinit, eos_draw_d, Etrans=Etrans_glass,
#                                           Ttrans=Tmelt_en, isobar_trans=True)
#
#     ax_a[0].plot(hugoniot_glass_draw_d['P_a'],hugoniot_glass_draw_d['rho_a'],'b-')
#     ax_a[0].plot(hugoniot_en_draw_d['P_a'],hugoniot_en_draw_d['rho_a'],'r-')
#
#     ax_a[1].plot(hugoniot_glass_draw_d['P_a'],hugoniot_glass_draw_d['T_a'],'b-')
#     ax_a[1].plot(hugoniot_en_draw_d['P_a'],hugoniot_en_draw_d['T_a'],'r-')
#     plt.draw()




















def load_mgsio3_xtal_eos():
    eos_xtal_d = {}
    models.Control.set_consts( [], [], eos_xtal_d )

    F0 = -1368 # kJ/mol
    T0 = 300.0
    gamma0 = 1.57
    q0 = 1.1
    gammap0 = gamma0*q0
    param_d = {}
    param_d['Cvmax'] = 3*eos_xtal_d['const_d']['kboltz']
    param_d['T0'] = T0
    param_d['S0'] = 0.0
    param_d['V0'] = 24.45/.6022/5 # ang^3/atom cm^3/mol
    param_d['K0'] = 251  # GPa
    param_d['KP0'] = 4.1 #
    param_d['E0'] = 0.0
    param_d['theta0'] = 905. # K
    param_d['gamma0'] = 1.57
    param_d['gammap0'] = gammap0
    param_d['thetaR'] = 905. # K
    param_d['gammaR'] = 1.57
    param_d['gammapR'] = gammap0

    eos_xtal_d['param_d'] = param_d

    compress_path_mod = models.BirchMurn3(path_const='T',
                                          supress_energy=False,
                                          supress_press=False)

    models.Control.set_modtypes( ['CompressPathMod'], [compress_path_mod], eos_xtal_d )
    gamma_mod = models.GammaFiniteStrain(V0ref=True)
    models.Control.set_modtypes( ['GammaMod'], [gamma_mod], eos_xtal_d )

    thermal_mod = models.MieGrunDebye()
    models.Control.set_modtypes( ['ThermalMod'], [thermal_mod], eos_xtal_d )

    return eos_xtal_d

def load_mgsio3_xtal_Oganov_eos():
    eos_xtal_d = {}
    models.Control.set_consts( [], [], eos_xtal_d )

    T0 = 2500.0
    param_d = {}
    param_d['T0'] = T0
    param_d['S0'] = 0.0
    param_d['V0'] = 174.85/20 # ang^3/atom cm^3/mol
    param_d['K0'] = 174.12  # GPa
    param_d['KP0'] = 5.10 #
    param_d['E0'] = 0.0

    eos_xtal_d['param_d'] = param_d

    compress_path_mod = models.Vinet(path_const='T',
                                          supress_energy=False,
                                          supress_press=False)

    models.Control.set_modtypes( ['CompressPathMod'], [compress_path_mod], eos_xtal_d )

    return eos_xtal_d

def calc_cold_xtal_press(V_a,eos_d):
    compress_path_mod = eos_d['modtype_d']['CompressPathMod']

    Pcold_a = compress_path_mod.press(V_a,eos_d)

    return Pcold_a

def calc_xtal_press(V_a,T_a,eos_d):
    T0 = eos_d['param_d']['T0']
    compress_path_mod = eos_d['modtype_d']['CompressPathMod']
    gamma_mod = eos_d['modtype_d']['GammaMod']
    thermal_mod = eos_d['modtype_d']['ThermalMod']

    Pcold_a = compress_path_mod.press(V_a,eos_d)
    Phot_a = thermal_mod.press(V_a,T_a,eos_d)\
        - thermal_mod.press(V_a,T0,eos_d)

    P_a = Pcold_a + Phot_a
    return P_a

def calc_vol(P,T,press_fun,Vinit=10):
    press_mismatch = lambda V, T=T:press_fun(V,T)-P
    V = opt.newton(press_mismatch,Vinit)
    return V

eos_xtal_d = load_mgsio3_xtal_eos()

press_xtal_fun = lambda V,T,eos_d=eos_xtal_d:calc_xtal_press(V,T,eos_d)
press_melt_fun = lambda V,T,eos_d=eos_d:eos_d['modtype_d']['FullMod'].press(V,T,eos_d)

eos_xtal_oganov_d = load_mgsio3_xtal_Oganov_eos()
calc_cold_xtal_press(Vgrid_a,eos_xtal_oganov_d)

###############################
# Calculate MgSiO3 melting curve from EOS
###############################

mass_mgsio3 = 24.31+28.09+3*16
Nat_mgsio3 = 5
# convert from volume [ang^3/atom] to rho [g/cc]
rho_const = mass_mgsio3/Nat_mgsio3/.6022

P_tz = 25.0
T_tz = 2900.0
dS = 1.45 # kB per atom
kT300 = 1./40 # eV

iP = P_tz
iT = T_tz
dP = 0.5

iVxtal = 10.0
iVmelt = 10.0

Pmelt_a = []
Tmelt_a = []
Pmelt_a.append(iP)
Tmelt_a.append(iT)
Pshift = -2
Pshift = 0

while iP < 136:
    iVmelt = calc_vol(iP+Pshift,iT,press_melt_fun,Vinit=iVmelt)
    iVxtal = calc_vol(iP,iT,press_xtal_fun,Vinit=iVxtal)
    idVmelt = iVmelt-iVxtal
    idTdP = 300./(dS*kT300)*idVmelt[0]/160.2

    iT +=idTdP*dP
    iP += dP
    Pmelt_a.append(iP)
    Tmelt_a.append(iT)

Pmelt_a=np.array(Pmelt_a)
Tmelt_a=np.array(Tmelt_a)

# Load other melting curves
PTmelt_S09_a = np.loadtxt('data/melting-curve-S09.txt',delimiter=',')
PTmelt_ZB93_a = np.loadtxt('data/melting-curve-ZB93.txt',delimiter=',')
PTmelt_SH96_a = np.loadtxt('data/melting-curve-SH96.txt',delimiter=',')
PTmelt_liq_A11_a = np.loadtxt('data/liquidus-Andrault2011.txt',delimiter=',')
PTmelt_sol_A11_a = np.loadtxt('data/solidus-Andrault2011.txt',delimiter=',')



# plt.plot(Pmelt_a,Tmelt_a,'m--')

plt.clf()
plt.plot(Pmelt_a,Tmelt_a,'r-',
         PTmelt_S09_a[:,0],PTmelt_S09_a[:,1],'k--',
         PTmelt_ZB93_a[:,0],PTmelt_ZB93_a[:,1],'g--',
         PTmelt_SH96_a[:,0],PTmelt_SH96_a[:,1],'b--',lw=2)
plt.legend(['this work','Stixrude09','Zerr93','Sweeney96'],
           loc='upper left')
plt.xlabel('Pressure [GPa]')
plt.ylabel('Temperature [K]')

plt.savefig('figs/MgSiO3-melt-curve-comparison.png',dpi=450)
plt.savefig('figs/MgSiO3-melt-curve-comparison.eps')
# plt.plot(PTmelt_liq_A11_a[:,0],PTmelt_liq_A11_a[:,1],'r:',
#          PTmelt_sol_A11_a[:,0],PTmelt_sol_A11_a[:,1],'b:',lw=2)

###############################




# Lower mantle pressure fraction
Pfrac_a = (Pmelt_a-25)/(135.6-25)
dP_fp_a = (1-Pfrac_a)*290 + Pfrac_a*500



# V = 20/.6022/dPdT_d['rho']

# idTdP = 300./(dS*kT300)*20/.6022*idVmelt/160.2

calc_vol(P_tz,T_tz,press_xtal_fun)




print(eos_d['param_d']['T0'])

###############################
#   Construct LaTeX param table
###############################
param_val = datamod_d['posterior_d']['param_val']
param_err = datamod_d['posterior_d']['param_err']

sci_note_exp  = np.floor(np.log10(param_err))
sci_note_base = 10**(np.log10(param_err)-sci_note_exp)
np.round(10*sci_note_base,decimals=0)

decimal_a = -sci_note_exp.astype(np.int)
param_err*10**decimal_a

param_val_tbl = 1.0*np.round( param_val*10**(1+decimal_a)).astype(np.int)\
    /10**(1+decimal_a)
param_err_tbl = np.round( param_err*10**(1+decimal_a)).astype(np.int)
param_key_tbl = datamod_d['posterior_d']['param_key']

# for (key,val,err,dec) in zip(param_key_tbl,param_val_tbl, param_err_tbl,decimal_a+1):
#     valfmt = '%.'+str(dec)+'f'
#     errfmt = '%d'
#     print((key+' '+valfmt+'('+errfmt+') \\\\')%(val,err))

print('%d $\qquad$\\\\'%eos_d['param_d']['T0'])
for (key,val,err,dec) in zip(param_key_tbl,param_val_tbl, param_err_tbl,decimal_a+1):
    valfmt = '%.'+str(dec)+'f'
    errfmt = '%d'
    print((valfmt+'('+errfmt+') \\\\')%(val,err))

# for (key,val,err) in zip(param_key_tbl,param_val_tbl, param_err_tbl):
#     print '|' + key + '|'+str(val)+'('+str(err)+')|'


################################################
#     Compare adiabats with prev EOS and melting curve
################################################

# Show mosenfelder adiabats!
Tad2540_M09_a = np.loadtxt('data/adiabat2540.txt',delimiter=',')
Tad3200_M09_a = np.loadtxt('data/adiabat3200.txt',delimiter=',')
Tad3750_M09_a = np.loadtxt('data/adiabat3750.txt',delimiter=',')
Tad2500_S09_a = np.loadtxt('data/adiabat2500-S09.txt',delimiter=',')

dT = 250.0
Tfoot_a = np.arange(2000.0,3500.1,dT)
cmap=plt.get_cmap('coolwarm',Tfoot_a.size)

ad_lbl = plot_adiabat_melting_curve(eos_d)
ad_M09_lbl = plt.plot(Tad2540_M09_a[:,0],Tad2540_M09_a[:,1],'k--',lw=1,color=cmap(2))
# plt.plot(Tad3200_M09_a[:,0],Tad3200_M09_a[:,1],'k--',lw=1,color=cmap(5))
ad_S09_lbl = plt.plot(Tad2500_S09_a[:,0],Tad2500_S09_a[:,1],'k-.',lw=1,color=cmap(2))
fig = plt.gcf()

ax_sub = fig.add_axes([.43,.21,.3,.23])
plot_gamma_adiabat(eos_d,show_cbar=False,ax=ax_sub)

plt.savefig('figs/MgSiO3-adiabat-melt-compare-Spera2011.eps')
plt.savefig('figs/MgSiO3-adiabat-melt-compare-Spera2011.png',dpi=450)


plot_gamma_datamod(datamod_d)
plt.savefig('figs/MgSiO3-gamma-Spera2011.png',dpi=450)



######################################
#  Compare adiabats to Stixrude09 and Mosenfelder09
######################################


plt.figure()

V0, = models.Control.get_params(['V0'],eos_d)
TOL = 1e-6
Vpath_a = np.arange(0.4,1.101,.02)*V0
# Tfoot_a = np.arange(1500.0,5100.0,500.0)
Tfoot= 2500.0


indVfoot = np.where(np.abs(np.log(Vpath_a/V0))<TOL)[0][0]
# Vgrid_ad_a, Tgrid_ad_a = get_adiabat_paths( Vpath_a, Tfoot_a, eos_d,
#                                            indVfoot=indVfoot )
adiabat_d = get_adiabat_paths( Vpath_a, Tfoot, eos_d, indVfoot=indVfoot )

Pad_ref_a = np.squeeze(adiabat_d['Pgrid'])
Tad_ref_a = np.squeeze(adiabat_d['Tgrid'])

Tad_fun = interpolate.interp1d(Pad_ref_a, Tad_ref_a,kind='cubic')
Tad_S09_fun = interpolate.interp1d(Tad2500_S09_a[:,0], Tad2500_S09_a[:,1],kind='cubic')
Tad_M09_fun = interpolate.interp1d(Tad2540_M09_a[:,0], Tad2540_M09_a[:,1],kind='cubic')
Pmod_a = np.linspace(0,135,1001)
dP = Pmod_a[1]-Pmod_a[0]

dTdPad_a = np.gradient(Tad_fun(Pmod_a),dP)
dTdPad_S09_a = np.gradient(Tad_S09_fun(Pmod_a),dP)
dTdPad_M09_a = np.gradient(Tad_M09_fun(Pmod_a),dP)

plt.figure()
plt.plot(Pmod_a,Tad_fun(Pmod_a),'k-',Pmod_a,Tad_S09_fun(Pmod_a),'r--',Pmod_a,Tad_M09_fun(Pmod_a),'b--')

plt.figure()
plt.clf()
plt.plot(Pmod_a,dTdPad_a,'k-',Pmod_a,dTdPad_S09_a,'r--',Pmod_a,dTdPad_M09_a,'b--')

plt.clf()
plt.plot(Pmod_a,dTdPad_S09_a/dTdPad_a,'r--',Pmod_a,dTdPad_M09_a/dTdPad_a,'b--')

print(Tad_S09_fun(135) - Tad_fun(135))


######################################
#        Correlation Table Figure
######################################

corr_a=datamod_d['posterior_d']['corr']
param_a = ["$V_0$", "$K_0$", "$K'_0$", "$E_0$", "$\gamma_0$", "$\gamma'_0$",
 "$b_0$", "$b_1$", "$b_2$", "$b_3$", "$b_4$"]
# param_a = ["$V_0$", "$K_0$", "$K'_0$", "$E_0$", "$\gamma_0$", "$\gamma'_0$",
#  "$b_0$", "$b_1$", "$b_2$", "$b_3$", "$b_4$","$m$"]

plot_corr_matrix(corr_a,param_a)
plt.savefig('figs/MgSiO3-rtpress-corr-Spera2011.png',dpi=450)
plt.savefig('figs/MgSiO3-rtpress-corr-Spera2011.eps')










# Combine with adiabat plot
plot_gamma_adiabat(eos_d)
plt.savefig('figs/MgSiO3-adiabat-gamma-Spera2011.png',dpi=450)

plot_adiabat_melting_curve(eos_d)
# Need to add curves from Mosenfelder and Stixrude
plt.savefig('figs/MgSiO3-adiabat-melt-curve-Spera2011.png',dpi=450)




# Compare with previous adiabats!





ad_lbl = plot_adiabat_melting_curve(eos_d)
ad_M09_lbl = plt.plot(Tad2540_M09_a[:,0],Tad2540_M09_a[:,1],'k--',lw=2,color=cmap(2))
# plt.plot(Tad3200_M09_a[:,0],Tad3200_M09_a[:,1],'k--',lw=1,color=cmap(5))
ad_S09_lbl = plt.plot(Tad2500_S09_a[:,0],Tad2500_S09_a[:,1],'k-.',lw=2,color=cmap(2))

plt.savefig('figs/MgSiO3-adiabat-melt-compare-Spera2011.png',dpi=450)


therm_mod = eos_d['modtype_d']['ThermalMod']
gamma_mod = eos_d['modtype_d']['GammaMod']

T0 = eos_d['param_d']['T0']

V0, = models.Control.get_params(['V0'],eos_d)
TOL = 1e-6


Vlims = [np.min(data_d['V']),np.max(data_d['V'])]

Vpath_a = np.arange(0.4,1.101,.02)*V0
# Tfoot_a = np.arange(1500.0,5100.0,500.0)
dT = 250.0
Tfoot_a = np.arange(2000.0,3500.1,dT)

cmap_a = plt.cm.coolwarm(np.linspace(0,1,Tfoot_a.size))
cmap=plt.get_cmap('coolwarm',Tfoot_a.size)

indVfoot = np.where(np.abs(np.log(Vpath_a/V0))<TOL)[0][0]
# Vgrid_ad_a, Tgrid_ad_a = get_adiabat_paths( Vpath_a, Tfoot_a, eos_d,
#                                            indVfoot=indVfoot )
# adiabat_d = get_adiabat_paths( Vpath_a, Tfoot_a, eos_d, indVfoot=indVfoot )
adiabat_d = get_adiabat_paths( Vpath_a, Tfoot_a, eos_d )

T_0S_a = gamma_mod.temp(V_a,T0,eos_d)


plt.figure()
plt.clf()
cbarinfo = plt.scatter( adiabat_d['Pgrid'], adiabat_d['Tgrid'],
                       c=adiabat_d['Tfoot_grid'],s=50,lw=0,cmap=cmap)
plt.clim(Tfoot_a[0]-0.5*dT,Tfoot_a[-1]+0.5*dT)
plt.clf()

# [plt.plot(Vmod_a/V0,therm_mod.calc_gamma( Vmod_a, iT, eos_d ),color=icmap)
#  for (iT,icmap) in zip(Tavg_bins_a,cmap_a)]

for (iV_a,iT_a,iP_a,icmap) in zip(adiabat_d['Vgrid'], adiabat_d['Tgrid'],
                                  adiabat_d['Pgrid'],cmap_a):
    idV=iV_a[1]-iV_a[0]
    igamma_num_a = -iV_a/iT_a*np.gradient(iT_a,idV)

    # plt.plot(iV_a,therm_mod.calc_gamma( iV_a, iT_a, eos_d ),'r-',
    #          iV_a,igamma_num_a,'k--')
    #plt.plot(iV_a,therm_mod.calc_gamma( iV_a, iT_a, eos_d ),'r-')
    plt.plot(iP_a,therm_mod.calc_gamma( iV_a, iT_a, eos_d ),'-',
             color=icmap, lw=2 )
    # plt.plot(iV_a,gamma_mod.gamma( V_a, eos_d ),'k--')

Plims=[-1.0,181.0]
gamlims = [-0.03, 1.43]
plt.xlim(Plims)
plt.ylim(gamlims)
plt.xlabel('P [GPa]')
plt.ylabel('$\gamma$')










leg_lbl = []
leg_lbl.append(ad_lbl[2][0])
leg_lbl.append(ad_M09_lbl[0])
leg_lbl.append(ad_S09_lbl[0])
plt.legend(leg_lbl,('this work                 (MD sim)','Mosenfelder09 (shock+DAC)','Stixrude09              (MD sim)'),loc='lower right')


plt.savefig('figs/MgSiO3-adiabat-melt-curve-compare-Spera2011.png',dpi=450)





plt.legend((ad_lbl[2],ad_M09_lbl,ad_S09_lbl),('RT-press','Mosenfelder 2009','Stixrude 2009'))



# Plot model fit results
plot_model_fit_results(Tlbl_a,datamod_d)
plt.savefig('figs/MgSiO3-rtpress-PVEmod-Spera2011.png',dpi=450)




# hfigPV.savefig('figs/MgSiO3-rtpress-PVmod-Spera2011.png',dpi=450)
# hfigEV.savefig('figs/MgSiO3-rtpress-EVmod-Spera2011.png',dpi=450)
# hfigEP.savefig('figs/MgSiO3-rtpress-EPmod-Spera2011.png',dpi=450)



##########################################
#      END
##########################################

# Make RTpress model construction plot

def old_plot_rtpress_model_fig():

    lbl_siz = 14
    par_lbl_siz = lbl_siz-2
    par_col = np.array([43,71,20])/255.
    par_col = np.array([78,128,0])/255.

    f, ax_a = plt.subplots(2, 2, sharex='col')
    f.subplots_adjust(hspace=0.05,wspace=0.25,left=.1,right=.97,top=.95)
    plt.draw()

    # ax_P = plt.subplot(221)
    # ax_E = plt.subplot(222)
    # ax_T = plt.subplot(223)
    # ax_Cpv = plt.subplot(224)

    # Fix ticks
    # ax_P.set_xticklabels([])
    # ax_E.set_xticklabels([])

    # Set labels
    ax_a[0,0].set_ylabel(r'$P \;\; [\rm{GPa}]$')
    ax_a[0,1].set_ylabel(r'$E - E_0 \;\;[\rm{eV}]$')
    ax_a[1,0].set_ylabel(r'$T/T_0$')
    ax_a[1,1].set_ylabel(r'$C_{P/V} \; / \; k_B$')
    ax_a[1,0].set_xlabel(r'$V/V_0$')
    ax_a[1,1].set_xlabel(r'$T/T_0$')

    plt.draw()

    Vbnds=[0.4,1.2]
    eos_d = datamod_d['eos_d']
    full_mod = eos_d['modtype_d']['FullMod']
    gamma_mod = eos_d['modtype_d']['GammaMod']
    V0 = eos_d['param_d']['V0']

    Vgrid_a = np.linspace(Vbnds[0],Vbnds[1],Nsamp)*V0

    Tgrid_a = np.array([2500,7000])
    E_grid_a, P_grid_a, dPdT_grid_a = eval_model( Vgrid_a, Tgrid_a, eos_d )

    Vdat_a = V0*np.array([0.5,.6,.7,.8,.9,1.0])
    Pdat_a = full_mod.press(Vdat_a,Tgrid_a[0],eos_d)
    Pdat_hi_a = full_mod.press(Vdat_a,Tgrid_a[1],eos_d)

    # P-V plot
    ax_a[0,0].plot(Vgrid_a/V0,P_grid_a[0],'k-',
              Vgrid_a/V0,P_grid_a[1],'r-')
    ax_a[0,0].plot(Vdat_a/V0,Pdat_a,'ko',mew=2,markeredgecolor='k')
    ax_a[0,0].plot(Vdat_a/V0,Pdat_hi_a,'ro',mew=2,markeredgecolor='r')
    ax_a[0,0].text(.65,20,r'$P(V,T_0)$',horizontalalignment='right',fontsize=lbl_siz)
    ax_a[0,0].text(.68,5,r"$\{V_0,K_0,K'_0\}$",horizontalalignment='right',
                   fontsize=par_lbl_siz,color=par_col)
    ax_a[0,0].text(.65,60,r'$P(V,T)$',horizontalalignment='left',color='r',fontsize=lbl_siz)

    ax_a[0,0].text(.55,170,r"$\rm{(I)\,Isothermal\,Compression:}$"+
                   r" $P(V,T_0)$",horizontalalignment='left',
                   verticalalignment='top', fontsize=par_lbl_siz)
    ax_a[0,0].text(.55,150,r"$\rm{(II)\,Thermal\,Pressure\,Deriv:}$"+"\n"+
                   r"$\;\;\;\;\;\;\left.\frac{dP}{dT}\right|_V = \left.\frac{dS}{dV}\right|_T$",horizontalalignment='left',
                   verticalalignment='top', fontsize=par_lbl_siz)
    # ax_a[0,0].text(.6,140,r"$\rm{(II)\,Thermal\,Pressure}$"+"\n"+
    #                r"$\;\;P(V,T_0)$",horizontalalignment='left',
    #                verticalalignment='top', fontsize=par_lbl_siz)

    ax_a[0,0].set_xlim(0.49,1.11)
    ax_a[0,0].set_ylim(-2,190)

    for (iV,iP0,iPT) in zip(Vdat_a/V0,Pdat_a,Pdat_hi_a):
        idy = iPT-iP0
        ax_a[0,0].annotate("", xy=(iV, iPT), xycoords='data',
                      xytext=(iV, iP0), textcoords='data',
                      arrowprops=dict( facecolor='r',shrink=0.0,
                                      width=3.,headwidth=6.,edgecolor='k'))
        plt.draw()



    T0 = eos_d['param_d']['T0']

    # T-V plot

    ax_a[1,0].plot(Vgrid_a/V0, gamma_mod.temp( Vgrid_a,T0,eos_d)/T0,'k-')
    ax_a[1,0].plot(Vdat_a/V0, gamma_mod.temp( Vdat_a,T0,eos_d)/T0,'ko')

    ax_a[1,0].set_xlim(0.49,1.11)
    ax_a[1,0].set_ylim(2400/T0,4700/T0)
    ax_a[1,0].text(.52,1.27,r'$T_{0S}(V)$',horizontalalignment='left',fontsize=lbl_siz)
    ax_a[1,0].text(.54,1.18,r"$\{\gamma_0,\gamma_0'\}$",color=par_col,
                   horizontalalignment='left',fontsize=par_lbl_siz)

    ax_a[1,0].text(0.52,1.75,r"$\rm{(IIIb)\,Adiabatic\,Temperature}$"+
                   r"$\rm{\,Deriv:}$"+"\n"+ r" $\;\;\;\;\;\;\left.\frac{dT}{dV}\right|_{S} = -\left.\frac{dS}{dV}\right|_{T} \; / \; \left.\frac{dS}{dT}\right|_{V} $",horizontalalignment='left',
                   verticalalignment='top', fontsize=par_lbl_siz,
                   bbox=dict(lw=2,boxstyle="square", fc=".9", ec='k'))

    annotation.slope_marker((.721, 1.21), -1.15, labels=(r'$dT$',r'$dV$'),
                            ax=ax_a[1,0])


    # E-T plot
    E0 = eos_d['param_d']['E0']
    Tdat_a = T0*np.array([.6,1.0,1.4,1.8,2.2,2.6,3.0])
    Edat_a = full_mod.energy(V0,Tdat_a,eos_d)
    Tgrid_a = T0*np.linspace(.5,3.2,1001)
    Egrid_a = full_mod.energy(V0,Tgrid_a,eos_d)

    ax_a[0,1].plot(Tdat_a/T0, Edat_a-E0,'ko')
    ax_a[0,1].plot(Tgrid_a/T0, Egrid_a-E0,'k-')
    ax_a[0,1].set_xlim(.9,3.1)
    ax_a[0,1].set_ylim(-.3,1.7)

    ax_a[0,1].text(1.5,0.4,r'$E(V_0,T)$',horizontalalignment='right',fontsize=lbl_siz)
    ax_a[0,1].text(1.1,1.5,r"$\rm{(IIIa)\,Thermal\,Energy\,Deriv:}$"+
                   "\n"+r"$\;\;\;\;\;\;\left.\frac{dE}{dT}\right|_V = \left. T \, \frac{dS}{dT}\right|_V$",horizontalalignment='left',
                   verticalalignment='top', fontsize=par_lbl_siz)

    annotation.slope_marker((1.5, 0.35), 0.85, labels=(r'$dE$',r'$dT$'),
                            ax=ax_a[0,1])


    # Cv-T plot
    Cv_dat_a = full_mod.heat_capacity(V0,Tdat_a,eos_d)/eos_d['const_d']['kboltz']
    Cv_dat_hiP_a = full_mod.heat_capacity(0.5*V0,Tdat_a,eos_d)/eos_d['const_d']['kboltz']
    Cv_grid_a = full_mod.heat_capacity(V0,Tgrid_a,eos_d)/eos_d['const_d']['kboltz']
    Cv_grid_hiP_a = full_mod.heat_capacity(0.5*V0,Tgrid_a,eos_d)/eos_d['const_d']['kboltz']
    ax_a[1,1].plot(Tgrid_a/T0, Cv_grid_a,'k-', Tdat_a/T0,Cv_dat_a,'ko')
    ax_a[1,1].plot(Tgrid_a/T0, Cv_grid_hiP_a,'b-')

    ax_a[1,1].set_xlim(.9,3.1)
    ax_a[1,1].set_ylim(2.9,4.4)

    # ax_a[1,1].text(0.75,4.15,r'$C_{P/V}(T)$',horizontalalignment='left',fontsize=lbl_siz)
    ax_a[1,1].text(1.5,3.35,r'$C_{P/V}(T)$',horizontalalignment='right',fontsize=lbl_siz)
    ax_a[1,1].text(1.05,3.2,r"$\{b_0,...,b_n\}$",color=par_col,
                   horizontalalignment='left',fontsize=par_lbl_siz)
    ax_a[1,1].text(2.5,3.03,"low press",verticalalignment='top',
                   horizontalalignment='left',fontsize=par_lbl_siz,color='k')
    ax_a[1,1].text(2.5,3.47,"high press",verticalalignment='top',
                   horizontalalignment='left',fontsize=par_lbl_siz,color='b')

    ax_a[1,1].text(1.1,4.3,r"$\rm{(IIIc)\,Heat\,Capacity:}$"+
                   r" $C_{P/V} = T\,\frac{dS}{dT}$",horizontalalignment='left',
                   verticalalignment='top', fontsize=par_lbl_siz)

    for (iT,iCv0,iCvT) in zip(Tdat_a/T0,Cv_dat_a,Cv_dat_hiP_a):
        ax_a[1,1].annotate("", xy=(iT, iCvT), xycoords='data',
                      xytext=(iT, iCv0), textcoords='data',
                      arrowprops=dict( facecolor='b',shrink=0.0,
                                      width=3.,headwidth=6.,edgecolor='k'))
        plt.draw()

    plt.draw()
    pass


def old2_plot_rtpress_model_fig():

    lbl_siz = 14
    par_lbl_siz = lbl_siz-2
    par_col = np.array([43,71,20])/255.
    par_col = np.array([78,128,0])/255.

    f, ax_a = plt.subplots(2, 2, sharex='col')
    # f.subplots_adjust(hspace=0.05,wspace=0.25,left=.1,right=.97,top=.95)
    f.subplots_adjust(hspace=0.03,wspace=0.18,left=.05,right=.97,top=.97,bottom=.05)
    plt.draw()

    # ax_P = plt.subplot(221)
    # ax_E = plt.subplot(222)
    # ax_T = plt.subplot(223)
    # ax_Cpv = plt.subplot(224)

    # Fix ticks
    # ax_P.set_xticklabels([])
    # ax_E.set_xticklabels([])

    # Set labels
    ax_a[0,0].set_ylabel(r'$P$')
    ax_a[1,1].set_ylabel(r'$E$')
    ax_a[0,1].set_ylabel(r'$C_{P/V}$')
    ax_a[1,0].set_ylabel(r'$T$')
    ax_a[1,0].set_xlabel(r'$V$')
    ax_a[1,1].set_xlabel(r'$T$')


    ax_a[0,0].set_yticklabels([])
    ax_a[1,0].set_yticklabels([])
    ax_a[0,1].set_yticklabels([])
    ax_a[1,1].set_yticklabels([])

    ax_a[1,0].set_xticklabels([])
    ax_a[1,1].set_xticklabels([])

    plt.draw()

    Vbnds=[0.4,1.2]
    eos_d = datamod_d['eos_d']
    full_mod = eos_d['modtype_d']['FullMod']
    gamma_mod = eos_d['modtype_d']['GammaMod']
    V0 = eos_d['param_d']['V0']

    Vgrid_a = np.linspace(Vbnds[0],Vbnds[1],Nsamp)*V0

    Tgrid_a = np.array([2500,7000])
    E_grid_a, P_grid_a, dPdT_grid_a = eval_model( Vgrid_a, Tgrid_a, eos_d )

    Vdat_a = V0*np.array([0.5,.6,.7,.8,.9,1.0])
    Pdat_a = full_mod.press(Vdat_a,Tgrid_a[0],eos_d)
    Pdat_hi_a = full_mod.press(Vdat_a,Tgrid_a[1],eos_d)

    # P-V plot
    ax_a[0,0].plot(Vgrid_a/V0,P_grid_a[0],'k-',
              Vgrid_a/V0,P_grid_a[1],'r-')
    ax_a[0,0].plot(Vdat_a/V0,Pdat_a,'ko',mew=2,markeredgecolor='k')
    ax_a[0,0].plot(Vdat_a/V0,Pdat_hi_a,'ro',mew=2,markeredgecolor='r')
    ax_a[0,0].text(.65,20,r'$P(V,T_0)$',horizontalalignment='right',fontsize=lbl_siz)
    ax_a[0,0].text(.68,5,r"$\{V_0,K_0,K'_0\}$",horizontalalignment='right',
                   fontsize=par_lbl_siz,color=par_col)
    ax_a[0,0].text(.65,60,r'$P(V,T)$',horizontalalignment='left',color='r',fontsize=lbl_siz)

    ax_a[0,0].text(.55,170,r"$\rm{(I)\,Isothermal\,Compression:}$"+
                   r" $P(V,T_0)$",horizontalalignment='left',
                   verticalalignment='top', fontsize=par_lbl_siz)
    ax_a[0,0].text(.55,150,r"$\rm{(II)\,Thermal\,Pressure\,Deriv:}$"+"\n"+
                   r"$\;\;\;\;\left.\frac{dP}{dT}\right|_V = \left.\frac{dS}{dV}\right|_T$",horizontalalignment='left',
                   verticalalignment='top', fontsize=par_lbl_siz)
    # ax_a[0,0].text(.6,140,r"$\rm{(II)\,Thermal\,Pressure}$"+"\n"+
    #                r"$\;\;P(V,T_0)$",horizontalalignment='left',
    #                verticalalignment='top', fontsize=par_lbl_siz)

    ax_a[0,0].set_xlim(0.49,1.11)
    ax_a[0,0].set_ylim(-2,190)


    for (iV,iP0,iPT) in zip(Vdat_a/V0,Pdat_a,Pdat_hi_a):
        idy = iPT-iP0
        ax_a[0,0].annotate("", xy=(iV, iPT), xycoords='data',
                      xytext=(iV, iP0), textcoords='data',
                      arrowprops=dict( facecolor='r',shrink=0.0,
                                      width=3.,headwidth=6.,edgecolor='k'))
        plt.draw()



    T0 = eos_d['param_d']['T0']

    # T-V plot

    ax_a[1,0].plot(Vgrid_a/V0, gamma_mod.temp( Vgrid_a,T0,eos_d)/T0,'k-')
    ax_a[1,0].plot(Vdat_a/V0, gamma_mod.temp( Vdat_a,T0,eos_d)/T0,'ko')

    ax_a[1,0].set_xlim(0.49,1.11)
    ax_a[1,0].set_ylim(2400/T0,4700/T0)
    ax_a[1,0].text(.52,1.27-.22,r'$T_{0S}(V)$',horizontalalignment='left',fontsize=lbl_siz)
    ax_a[1,0].text(.52,1.18-.22,r"$\{\gamma_0,\gamma_0'\}$",color=par_col,
                   horizontalalignment='left',fontsize=par_lbl_siz)

    ax_a[1,0].text(0.52,1.75,r"$\rm{(IIIb)\,Adiabatic\,Temperature}$"+
                   r"$\rm{\,Deriv:}$"+"\n"+ r" $\;\;\;\;\left.\frac{dT}{dV}\right|_{S} = -\left.\frac{dS}{dV}\right|_{T} \; / \; \left.\frac{dS}{dT}\right|_{V} $",horizontalalignment='left',
                   verticalalignment='top', fontsize=par_lbl_siz)

    annotation.slope_marker((.721, 1.21), -1.15, labels=(r'$dT$',r'$dV$'),
                            ax=ax_a[1,0])


    # E-T plot
    E0 = eos_d['param_d']['E0']
    Tdat_a = T0*np.array([.6,1.0,1.4,1.8,2.2,2.6,3.0])
    Edat_a = full_mod.energy(V0,Tdat_a,eos_d)
    Tgrid_a = T0*np.linspace(.5,3.2,1001)
    Egrid_a = full_mod.energy(V0,Tgrid_a,eos_d)

    ax_a[1,1].plot(Tdat_a/T0, Edat_a-E0,'ko')
    ax_a[1,1].plot(Tgrid_a/T0, Egrid_a-E0,'k-')
    ax_a[1,1].set_xlim(.9,3.1)
    ax_a[1,1].set_ylim(-.3,1.7)

    ax_a[1,1].text(1.5,0.4-.65,r'$E(V_0,T)$',horizontalalignment='right',fontsize=lbl_siz)
    ax_a[1,1].text(1.1,1.5,r"$\rm{(IIIc)\,Thermal\,Energy\,Deriv:}$"+
                   "\n"+r"$\;\;\;\;\left.\frac{dE}{dT}\right|_V = \left. T \, \frac{dS}{dT}\right|_V$",horizontalalignment='left',
                   verticalalignment='top', fontsize=par_lbl_siz)

    annotation.slope_marker((1.5, 0.35), 0.85, labels=(r'$dE$',r'$dT$'),
                            ax=ax_a[1,1])


    # Cv-T plot
    Cv_dat_a = full_mod.heat_capacity(V0,Tdat_a,eos_d)/eos_d['const_d']['kboltz']
    Cv_dat_hiP_a = full_mod.heat_capacity(0.5*V0,Tdat_a,eos_d)/eos_d['const_d']['kboltz']
    Cv_grid_a = full_mod.heat_capacity(V0,Tgrid_a,eos_d)/eos_d['const_d']['kboltz']
    Cv_grid_hiP_a = full_mod.heat_capacity(0.5*V0,Tgrid_a,eos_d)/eos_d['const_d']['kboltz']
    ax_a[0,1].plot(Tgrid_a/T0, Cv_grid_a,'k-', Tdat_a/T0,Cv_dat_a,'ko')
    ax_a[0,1].plot(Tgrid_a/T0, Cv_grid_hiP_a,'b-')

    ax_a[0,1].set_xlim(.9,3.1)
    # ax_a[0,1].set_ylim(2.9,4.4)
    ax_a[0,1].set_ylim(3.3,5.1)

    # ax_a[0,1].text(0.75,4.15,r'$C_{P/V}(T)$',horizontalalignment='left',fontsize=lbl_siz)
    ax_a[0,1].text(1.5,3.35-.26,r'$C_{P/V}(T)$',horizontalalignment='right',fontsize=lbl_siz)
    ax_a[0,1].text(1.05,3.2-.23,r"$\{b_0,...,b_n\}$",color=par_col,
                   horizontalalignment='left',fontsize=par_lbl_siz)
    ax_a[0,1].text(2.5,3.03,"low press",verticalalignment='top',
                   horizontalalignment='left',fontsize=par_lbl_siz,color='k')
    ax_a[0,1].text(2.5,3.47,"high press",verticalalignment='top',
                   horizontalalignment='left',fontsize=par_lbl_siz,color='b')

    ax_a[0,1].text(1.1,4.3,r"$\rm{(IIIa)\,Heat\,Capacity:}$"+
                   r" $C_{P/V} = T\,\frac{dS}{dT}$",horizontalalignment='left',
                   verticalalignment='top', fontsize=par_lbl_siz)

    for (iT,iCv0,iCvT) in zip(Tdat_a/T0,Cv_dat_a,Cv_dat_hiP_a):
        ax_a[0,1].annotate("", xy=(iT, iCvT), xycoords='data',
                      xytext=(iT, iCv0), textcoords='data',
                      arrowprops=dict( facecolor='b',shrink=0.0,
                                      width=3.,headwidth=6.,edgecolor='k'))
        plt.draw()

    plt.draw()
    pass




head_length=np.ceil(0.2*idy),
plt.draw()
ax_P.cla()

#     ax_P.annotate("", xy=(iV, iPT), xycoords='data',
#                   xytext=(iV, iP0), textcoords='data',
#                   arrowprops=dict( shrink=0.0))
#
#     ax_P.annotate("", xy=(iV, iPT), xycoords='data',
#                   xytext=(iV, iP0), textcoords='data',
#                   arrowprops=dict(arrowstyle="->",shrinkA=0.0,shrinkB=0.0,
#                                   connectionstyle="arc3"))
#
# ax_P.annotate("", xy=(iV, iPT), xycoords='data',
#               xytext=(iV, iP0), textcoords='data',
#               arrowprops=dict(arrowstyle="->",
#                               connectionstyle="arc3"),visible=True)
#
#     ax_resid = plt.subplot(212)
#     ax_main.set_xticklabels([])
#     plt.draw()
#
#     pos_main = [.125,.27,.7,.68]
#     pos_resid = [.125,.1,.7,.15]
#
#     ax_main.set_position(pos_main)
#     ax_resid.set_position(pos_resid)
#
#     cmap_cont=plt.get_cmap('coolwarm')
#     col_mod_a = cmap_cont( np.linspace(0,1,zlbl_a.size) )
#
#     cmap=plt.get_cmap('coolwarm',zlbl_a.size)
#
#
#
#     ax_main.scatter( x_a, y_a,c=z_a,s=50,lw=0,cmap=cmap)
#     [ax_main.plot(ixgrid_a, iygrid_a,'-',color=icol_a,label=izlbl) \
#      for ixgrid_a, iygrid_a,icol_a,izlbl  in zip(xgrid_a,ygrid_a,col_mod_a,zgrid_a)]
#     ax_main.set_xlim(xlims)
#     ax_main.set_ylim(ylims)
#
#     xrange_a = np.array([np.min(xgrid_a[:]),np.max(xgrid_a[:])])
#     ax_resid.add_patch(patches.Rectangle((xlims[0],-yresid_err),
#                                          xlims[1]-xlims[0],2*yresid_err,
#                                          facecolor='#DCDCDC',edgecolor='none',
#                                          zorder=0))
#     ax_resid.plot(xrange_a,0.0*xrange_a,'k--',zorder=1)
#     hlbl=ax_resid.scatter( x_a, y_a-ymod_a,c=z_a,s=50,lw=0,cmap=cmap,zorder=2)
#     ax_resid.set_xlim(xlims)
#     ax_resid.set_ylim([1.1*yresid_ticks[0],1.1*yresid_ticks[-1]])
#     ax_resid.set_yticks(yresid_ticks)




Nsamp = 101
V_a = np.linspace(0.4,1.1,Nsamp)*V0
T0 = eos_d['param_d']['T0']
T_0S_a = gamma_mod.temp(V_a,T0,eos_d)
T_a = T0
plt.clf()
plt.plot(V_a,therm_mod.calc_gamma( V_a, T_0S_a, eos_d ),'r-')
plt.plot(V_a,gamma_mod.gamma( V_a, eos_d ),'k--')


Nsamp = 101
Vgrid_a = np.linspace(0.4,1.1,Nsamp)*V0
T0 = eos_d['param_d']['T0']


gamma_mod = eos_d['modtype_d']['GammaMod']
full_mod = eos_d['modtype_d']['FullMod']
T_S0_a = gamma_mod.temp(Vgrid_a,T0,eos_d)
P_S0_a = full_mod.press(Vgrid_a,T_S0_a,eos_d)
E_S0_a = full_mod.energy(Vgrid_a,T_S0_a,eos_d)

vinet_mod = models.Vinet(path_const='S')

eos_S0_d= copy.deepcopy(eos_d)

eos_S0_d['param_d']['V0'] =12.463
eos_S0_d['param_d']['K0'] =15.0
eos_S0_d['param_d']['KP0'] =8.7

E0 = eos_S0_d['param_d']['E0']

plt.clf()
plt.plot(Vgrid_a,E_S0_a-E0,'k-',Vgrid_a,vinet_mod.energy(Vgrid_a,eos_S0_d)-E0,'r--')
#plt.ylim(-.005,.05)

plt.clf()
plt.plot(Vgrid_a,P_S0_a,'k-',Vgrid_a,vinet_mod.press(Vgrid_a,eos_S0_d),'r--')






Nsamp = 101
Vgrid_a = np.linspace(0.4,1.1,Nsamp)*V0
Tgrid_a = datamod_d['data_d']['Tavg_bins']
Tlbl_a = np.array([2500,3000,3500,4000,4500,5000])

E_grid_a, P_grid_a, dPdT_grid_a = eval_model( Vgrid_a, Tgrid_a, eos_d )

Vgrid_a,Tgrid_a,E_grid_a,Tlbl_a,datamod_d


eos_d = datamod_d['eos_d']
full_mod = eos_d['modtype_d']['FullMod']

E_mod_a = full_mod.energy(data_d['V'],data_d['T'],eos_d)
xlims=[5.25,14.5]
ylims=[-21.05,-18.95]
yresid_ticks = [-.02,0,.02]
xlbl = r'Volume  [$\AA^3$ / atom]'
ylbl = r'Internal Energy  [eV / atom]'
plot_model_mismatch((Vgrid_a, Tgrid_a, E_grid_a),(V_a,T_a,E_a,E_mod_a),
                    (xlims,ylims),(xlbl,ylbl),Tlbl_a,yresid_ticks)

P_mod_a = full_mod.press(data_d['V'],data_d['T'],eos_d)
xlims=[5.25,14.5]
ylims=[-5.0,181.0]
yresid_ticks = [-2,0,2]
xlbl = r'Volume  [$\AA^3$ / atom]'
ylbl = r'Pressure  [GPa]'
plot_model_mismatch((Vgrid_a, Tgrid_a, P_grid_a),(V_a,T_a,P_a,P_mod_a),
                    (xlims,ylims),(xlbl,ylbl),Tlbl_a,yresid_ticks)


################################################



fig=plt.figure()
ticks = Tgrid_a
label = 'Temperature [K]'

add_shared_colorbar(label, ticks, fig=fig, pos=[.85,.1,.05,.85], cmap=cmap)

add_shared_colorbar(label, ticks, fig=fig, pos=[.1,.1,.8,.05], cmap=cmap,
                    orientation='horizontal')


cbar.set_clim(2250, 5250)


2547.0, 5049.0
plt.draw()







cbar=plt.colorbar(hlbl,ticks=Tgrid_a,label='Temperature [K]',cax=ax2)


plt.axes(ax1)
plt.clim(2250,5250)





fig.colorbar(im, cax=cbar_ax)

plt.draw()

plt.draw()

ax2.set_visible(False)
plt.draw()



plt.colorbar(hlbl,ticks=Tgrid_a,label='Temperature [K]',cax=ax2)


plt.draw()





plt.xlabel(r'Volume  [$\AA^3$ / atom]')
plt.ylabel(r'Internal Energy  [eV / atom]')



# zoom

cmap=plt.get_cmap('coolwarm',6)
plt.colorbar(ticks=Tgrid_a,label='Temperature [K]')
plt.clim(2250,5250)
plt.xlabel(r'Volume  [$\AA^3$ / atom]')
plt.ylabel(r'Internal Energy  [eV / atom]')
# zoom


plt.savefig('figs/MgSiO3-energy-fit-Spera2011.png',dpi=450)



# Print vol vs press
plt.figure()
plt.clf()
cmap=plt.get_cmap('coolwarm',6)
plt.scatter( data_d['V'], data_d['P'],c=data_d['T'],s=50,lw=0,cmap=cmap)
[plt.plot(Vgrid_a, ipress_a,'-',color=icol_a,label=iT) \
 for ipress_a,ienergy_a,icol_a,iT  in zip(press_mod_a,energy_mod_a,col_mod_a,Tgrid_a)]
plt.colorbar(ticks=Tgrid_a,label='Temperature [K]')
plt.clim(2250,5250)
plt.xlabel(r'Volume  [$\AA^3$ / atom]')
plt.ylabel(r'Pressure  [GPa]')
# zoom

plt.figure()
plt.clf()
cmap=plt.get_cmap('coolwarm',6)
plt.scatter( data_d['V'], data_d['P']-full_mod.press(data_d['V'],data_d['T'],eos_d),c=data_d['T'],s=50,lw=0,cmap=cmap)

plt.clf()
plt.scatter( data_d['V'], full_mod.press(data_d['V'],data_d['T'],eos_d),c=data_d['T'],s=50,lw=0,cmap=cmap)


plt.savefig('figs/MgSiO3-press-fit-Spera2011.png',dpi=450)


# Print vol vs dpdT

plt.figure()
plt.clf()
cmap=plt.get_cmap('coolwarm',6)
plt.scatter(V_a,dPdT_a,c=T_a,s=70,cmap=cmap)
[plt.plot(Vgrid_a, idPdT_a,'-',color=icol_a,label=iT) \
 for idPdT_a,icol_a,iT  in zip(dPdT_mod_a,col_mod_a,Tgrid_a)]


plt.xlabel(r'Volume  [ $\AA^3$ / atom ]')
plt.ylabel(r'Thermal Pressure  [ GPa / 1000K ]')
plt.colorbar()


plt.figure()
plt.clf()
cmap=plt.get_cmap('coolwarm',6)
plt.scatter( data_d['V'], data_d['P'],c=data_d['T'],s=50,lw=0,cmap=cmap)
[plt.plot(Vgrid_a, ipress_a,'-',color=icol_a,label=iT) \
 for ipress_a,ienergy_a,icol_a,iT  in zip(press_mod_a,energy_mod_a,col_mod_a,Tgrid_a)]
plt.colorbar(ticks=Tgrid_a,label='Temperature [K]')
plt.clim(2250,5250)
plt.xlabel(r'Volume  [$\AA^3$ / atom]')
plt.ylabel(r'Pressure  [GPa]')
# zoom
plt.savefig('figs/MgSiO3-press-fit-Spera2011.png',dpi=450)




# Print press vs energy
plt.figure()
plt.clf()
cmap=plt.get_cmap('coolwarm',6)
plt.scatter( data_d['P'], data_d['E'],c=data_d['T'],s=50,lw=0,cmap=cmap)
[plt.plot(ipress_a, ienergy_a,'-',color=icol_a,label=iT) \
 for ipress_a,ienergy_a,icol_a,iT  in zip(press_mod_a,energy_mod_a,col_mod_a,Tgrid_a)]
plt.colorbar(ticks=Tgrid_a)
plt.clim(2250,5250)
#plt.colorbar(boundaries=[2250,5250])
plt.xlim(-5,165)
ybnd = [np.min(energy_mod_a[press_mod_a<170]),
        np.max(energy_mod_a[press_mod_a<170])]
plt.ylim(ybnd[0],ybnd[1])
plt.xlabel('P  [GPa]')
plt.ylabel('E  [eV/atom]')


# Print T^3/5 vs E
energy_fit_a = eos_d['modtype_d']['FullMod'].energy(data_d['V'], data_d['T'], eos_d )

plt.clf()
for indV in np.arange(Vuniq_a.size):
    ind_a = data_d['V'] == Vuniq_a[indV]
    shft_a = 0*data_d['V'][ind_a]*3e-1
    plt.plot(data_d['T'][ind_a]**(3./5),energy_fit_a[ind_a]-shft_a,'o',mew=3,
             markerfacecolor='white',markeredgecolor='r')
    plt.plot(data_d['T'][ind_a]**(3./5),data_d['E'][ind_a]-shft_a,'ko-')

plt.clf()
plt.plot(energy_fit_a,energy_fit_a-data_d['E'],'ko')


plt.figure()
plt.clf()
cmap=plt.get_cmap('coolwarm',6)
plt.scatter( data_d['V'], data_d['P'],c=data_d['T'],s=50,lw=0,cmap=cmap)
[plt.plot(Vgrid_a, ipress_a,'-',color=icol_a,label=iT) \
 for ipress_a,ienergy_a,icol_a,iT  in zip(press_mod_a,energy_mod_a,col_mod_a,Tgrid_a)]
plt.colorbar(ticks=Tgrid_a,label='Temperature [K]')
plt.clim(2250,5250)
plt.xlabel(r'Volume  [$\AA^3$ / atom]')
plt.ylabel(r'Pressure  [GPa]')
# zoom
plt.savefig('figs/MgSiO3-press-fit-Spera2011.png',dpi=450)










gam_mod = eos_d['modtype_d']['GammaMod']
Tref_a = gam_mod.temp(Vpath_a,eos_d['param_d']['T0'],eos_d)
gamref_a = gam_mod.gamma(Vpath_a,eos_d)

plt.figure()
plt.clf()
plt.plot(Vpath_a,Tref_a,'k-')

plt.clf()
plt.plot(Vpath_a,gamref_a,'k-')

dV = Vpath_a[1]-Vpath_a[0]
plt.clf()
plt.plot(Vpath_a,np.gradient(gamref_a,dV)*gamref_a/Vpath_a,'k-')
###################################################################



# energy_mod_a, press_mod_a = eval_model( Vgrid_a, Tgrid_a, eos_d )
#
# # Compare with Figure2b of Spera2011
# cmap=plt.get_cmap('coolwarm')
# col_a = cmap(1.0*(Tgrid_a-Tgrid_a[0])/np.ptp(Tgrid_a))[:,:3]
#
# plt.clf()
# cmap=plt.get_cmap('coolwarm',6)
# plt.scatter( data_d['V'], data_d['E'],c=data_d['T'],s=50,lw=0,cmap=cmap)
# [plt.plot(Vgrid_a, ienergy_a,'-',color=icol_a,label=iT) \
#  for ipress_a,ienergy_a,icol_a,iT  in zip(press_mod_a,energy_mod_a,col_a,Tgrid_a)]
# plt.colorbar(ticks=Tgrid_a)
# plt.clim(2250,5250)



energy_mod_a, press_mod_a = eval_model( Vgrid_a, Tgrid_a, eos_d )

# Compare with Figure2b of Spera2011
cmap=plt.get_cmap('coolwarm')
col_a = cmap(1.0*(Tgrid_a-Tgrid_a[0])/np.ptp(Tgrid_a))[:,:3]

plt.clf()
cmap=plt.get_cmap('coolwarm',6)
plt.scatter( data_d['V'], data_d['P'],c=data_d['T'],s=50,lw=0,cmap=cmap)
[plt.plot(Vgrid_a, ipress_a,'-',color=icol_a,label=iT) \
 for ipress_a,ienergy_a,icol_a,iT  in zip(press_mod_a,energy_mod_a,col_mod_a,Tgrid_a)]
plt.colorbar(ticks=Tgrid_a)
plt.clim(2250,5250)






#plt.colorbar(boundaries=[2250,5250])
plt.xlim(-5,165)
ybnd = [np.min(energy_mod_a[press_mod_a<165]),
        np.max(energy_mod_a[press_mod_a<165])]
plt.ylim(ybnd[0],ybnd[1])
plt.xlabel('P  [GPa]')
plt.ylabel('E  [eV/atom]')










plt.clf()
cmap=plt.get_cmap('coolwarm',6)
plt.scatter( dat_og_d['P'], dat_og_d['Etot'],c=dat_og_d['T'],s=50,lw=0,cmap=cmap)
[plt.plot(ipress_a, ienergy_a/eos_d['const_d']['energy_conv_fac'],'-',color=icol_a,label=iT) \
 for ipress_a,ienergy_a,icol_a,iT  in zip(press_mod_a,energy_mod_a,col_mod_a,Tgrid_a)]
plt.colorbar(ticks=Tgrid_a)
plt.clim(2250,5250)
#plt.colorbar(boundaries=[2250,5250])
plt.xlim(-5,165)
ybnd = [np.min(energy_mod_a[press_mod_a<165]/eos_d['const_d']['energy_conv_fac']),
        np.max(energy_mod_a[press_mod_a<165]/eos_d['const_d']['energy_conv_fac'])]
plt.ylim(ybnd[0],ybnd[1])
plt.xlabel('P  [GPa]')
plt.ylabel('E  [kJ/g]')


# Energy
plt.clf()
cmap=plt.get_cmap('coolwarm',6)
plt.scatter( dat_og_d['V'], dat_og_d['Etot'],c=dat_og_d['T'],s=50,lw=0,cmap=cmap)
plt.colorbar(ticks=Tgrid_a)
plt.clim(2250,5250)
#plt.colorbar(boundaries=[2250,5250])

# Press
plt.clf()
cmap=plt.get_cmap('coolwarm',6)
plt.scatter( dat_og_d['V'], dat_og_d['P'],c=dat_og_d['T'],s=50,lw=0,cmap=cmap)
plt.colorbar(ticks=Tgrid_a)
plt.clim(2250,5250)


plt.clf()
cmap=plt.get_cmap('coolwarm',6)
plt.scatter( dat_og_d['V'], dat_og_d['Ekin'],c=dat_og_d['T'],s=50,lw=0,cmap=cmap)
plt.colorbar(ticks=Tgrid_a)
plt.clim(2250,5250)









plt.ion()
plt.figure()
ax = plt.axes()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1],labels[::-1],loc='upper left')
plt.xlim(-5,165)
ybnd = [np.min(energy_mod_a[press_mod_a<165]), np.max(energy_mod_a[press_mod_a<165])]
plt.ylim(ybnd[0],ybnd[1])








# Fit RTpress EOS
eos_d = init_params({})
compress_mod = eos_d['modtype_d']['CompressMod']
param_d = eos_d['param_d']

V_a = param_d['V0']*np.linspace(0.5,1.0,11)

paramname = 'V0'

fname = 'press'
compress_mod.param_deriv( fname, paramname, V_a, eos_d, dxfrac=0.01)

fname = 'energy'
compress_mod.param_deriv( fname, paramname, V_a, eos_d, dxfrac=0.01)

paramname = 'E0'
fname = 'energy'
compress_mod.param_deriv( fname, paramname, V_a, eos_d, dxfrac=0.01)

eoslib
