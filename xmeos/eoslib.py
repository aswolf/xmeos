# -*- coding: utf-8 -*-
# from __future__ import absolute_import, print_function, division
from future.utils import with_metaclass

import numpy as np
import xmeos
from xmeos import models
from xmeos.models import core
from xmeos.models import CompositeEos

import pytest
import matplotlib.pyplot as plt
import matplotlib as mpl
from abc import ABCMeta, abstractmethod
import copy
from scipy import interpolate
from scipy import signal

__all__ = ['get_Di_melt_Thomas2013']


def get_value_array(value_d, keys=None):
    """
    get dictionary values as an array in desired order

    Parameters
    ----------
    value_d : dict of values to be retrieved
    keys : list of keys, optional
        defines desired order of values. If None is provided, all keys will be
        returned in default order. key array will also be returned.

    Returns
    -------
    value_a : array of values
    keys : array of keys, optional
        only return array of keys if None is provided

    """

    if keys is None:
        keys = np.array([key for key in value_d])
        return_key = True
    else:
        return_key = False

    value_a = []

    for key in keys:
        value_a.append(value_d.get(key, 0.0))

    value_a = np.array(value_a)

    if return_key:
        return value_a, keys
    else:
        return value_a

def set_value_array(vals, keys):
    value_d = dict([(key, val) for (key, val)
                    in zip(keys, vals)])
    return value_d

def get_oxide_data():
    """
    get oxide compositional dS_partial

    Returns
    -------
    oxide_data: dict with keys
        `oxides` : list of oxide strings
        `cations` : list of cation strings
        `molwt` : array of molecular weights
        `catnum` : array of cation numbers
        `oxynum` : array of oxygen numbers
        `oxycat_ratio` : array of oxygen/cation ratios
        `MgO`, `Al2O3`, `SiO2`, etc : dict of properties for each oxide

    """

    def make_oxide_dat(name, cation, molwt, charge, catnum, oxynum):
        oxycat_ratio = oxynum/catnum
        oxide_dat = {'name':name, 'cation':cation, 'molwt':molwt,
                     'charge':charge, 'catnum':catnum, 'oxynum':oxynum,
                     'oxycat_ratio': oxycat_ratio}
        return oxide_dat


    oxide_data_list = []
    oxide_data_list.append(make_oxide_dat( 'SiO2', 'Si',   60.0848, +4, 1, 2))
    oxide_data_list.append(make_oxide_dat( 'TiO2', 'Ti',   79.8988, +4, 1, 2))

    oxide_data_list.append(make_oxide_dat('Al2O3', 'Al', 101.96128, +3, 2, 3))
    oxide_data_list.append(make_oxide_dat('Fe2O3', 'Fe',  159.6922, +3, 2, 3))

    oxide_data_list.append(make_oxide_dat(  'MgO', 'Mg',   40.3044, +2, 1, 1))
    oxide_data_list.append(make_oxide_dat(  'FeO', 'Fe',   71.8464, +2, 1, 1))
    oxide_data_list.append(make_oxide_dat(  'CaO', 'Ca',   56.0794, +2, 1, 1))

    oxide_data_list.append(make_oxide_dat( 'Na2O', 'Na',  61.97894, +1, 2, 1))
    oxide_data_list.append(make_oxide_dat(  'K2O',  'K',   94.1954, +1, 2, 1))

    oxide_data = {}
    oxide_data['oxides'] = [idat['name'] for idat in oxide_data_list]
    oxide_data['cations'] = [idat['cation'] for idat in oxide_data_list]
    oxide_data['molwt'] = np.array([idat['molwt'] for idat in oxide_data_list])
    oxide_data['charge'] = np.array([idat['charge'] for idat in oxide_data_list])
    oxide_data['catnum'] = np.array([idat['catnum'] for idat in oxide_data_list])
    oxide_data['oxynum'] = np.array([idat['oxynum'] for idat in oxide_data_list])
    oxide_data['oxycat_ratio'] = np.array([idat['oxycat_ratio']
                                           for idat in oxide_data_list])

    for idat in oxide_data_list:
        oxide = idat['name']
        oxide_data[oxide] = idat

    return oxide_data
    # return cat_name_d, mol_wt_d, cat_num_d, oxycat_ratio_d

def calc_comp_details(comp_d, kind='wt'):
    oxide_data = get_oxide_data()
    # Determine molar and weight composition
    if kind=='wt':
        comp_wt_a = get_value_array(comp_d, keys=oxide_data['oxides'])
        comp_mol_a= comp_wt_a/oxide_data['molwt']
        comp_mol_a *= 100/np.sum(comp_mol_a)

    elif kind=='mol':
        comp_mol_a = get_value_array(comp_d, keys=oxide_data['oxides'])

    comp_wt_a = comp_mol_a*oxide_data['molwt']
    mol_mass = np.sum(comp_wt_a)/100
    comp_wt_a *= mol_mass

    cat_num_a = comp_mol_a/100*oxide_data['catnum']
    oxy_num_a = comp_mol_a/100*oxide_data['oxynum']
    # print('cat ',cat_num_a)
    # print('oxy ',oxy_num_a)
    oxy_num = np.sum(oxy_num_a)
    cat_a = 100*cat_num_a/np.sum(cat_num_a)
    atomspermol = np.sum(cat_num_a)+oxy_num
    # print('atomspermol ',atomspermol)
    oxy_frac = 100*oxy_num/atomspermol

    catoxy_ratio = cat_a/oxy_num

    comp_details = {}
    comp_details['oxides'] = oxide_data['oxides']
    comp_details['cations'] = oxide_data['cations']
    comp_details['molmass'] = mol_mass
    comp_details['wt'] = comp_wt_a
    comp_details['mol'] = comp_mol_a
    comp_details['cat'] = cat_a
    comp_details['oxy'] = oxy_frac
    comp_details['catoxy_ratio'] = catoxy_ratio
    comp_details['atomspermol'] = atomspermol

    return comp_details
#====================================================================
class CMASF_melt_Thomas2013(CompositeEos):
    def __init__(self, meltcomp):
        # self._comp_d = calc_comp_details(meltcomp, kind=kind)
        self.endmem_comp = meltcomp
        self.load_eos(meltcomp)

    def load_eos(self, meltcomp):
        endmems = {}
        for endmem_name in meltcomp:
            ieos, icomp_d = self.get_endmem_eos(endmem=endmem_name)
            endmems[endmem_name] = ieos

        self.endmem_eos = endmems

    def volume(self, P_a, T_a):
        P_a, T_a = core.fill_array(P_a, T_a)
        V_a = np.zeros(P_a.shape)

        for endmem in self.endmem_comp:
            ieos_mod = self.endmem_eos[endmem]
            icomp = self.endmem_comp[endmem]
            iV_a = ieos_mod.volume(P_a, T_a)
            V_a += icomp/100*iV_a

        return V_a

    @classmethod
    def get_endmem_eos(cls, endmem='En'):
        if endmem=='En':
            return cls._get_En()
        elif endmem=='Fo':
            return cls._get_Fo()
        elif endmem=='Fa':
            return cls._get_Fa()
        elif endmem=='Di':
            return cls._get_Di()
        elif endmem=='An':
            return cls._get_An()

    @classmethod
    def _get_En(cls):
        oxide_data = get_oxide_data()
        comp_d = calc_comp_details({'MgO':1,'SiO2':1}, kind='mol')

        rho0 = 2.617
        V0 = comp_d['molmass']/rho0*core.CONSTS['ang3percc']/core.CONSTS['Nmol']

        natom=1
        Cv_J_kg_K = 1690.53
        ndof = 6
        Cvlim = ndof/2*core.CONSTS['R']
        Cv = Cv_J_kg_K*1e-3*comp_d['molmass']/natom # J/mol/K
        Cvlimfac = Cv/Cvlim

        # Normalize to a single atom basis
        V0 /= natom

        eos_mod = models.MieGruneisenEos(
            kind_thermal='ConstHeatCap', kind_gamma='GammaPowLaw',
            kind_compress='BirchMurn4', compress_path_const='S',
            natom=natom)
        ref_state = eos_mod.refstate.ref_state
        ref_state['T0'] = 1673

        eos_mod.ndof = ndof
        eos_mod.set_param_values([V0], param_names=['V0'])
        eos_mod.set_param_values([0.365,-0.88], param_names=['gamma0','q'])
        eos_mod.set_param_values([24.66, 10.07, -2.35], param_names=['K0','KP0','KP20'])
        eos_mod.set_param_values([Cvlimfac], param_names=['Cvlimfac'])

        return eos_mod, comp_d

    @classmethod
    def _get_Fo(cls):
        oxide_data = get_oxide_data()
        comp_d = calc_comp_details({'MgO':2,'SiO2':1}, kind='mol')

        rho0 = 2.597
        V0 = comp_d['molmass']/rho0*core.CONSTS['ang3percc']/core.CONSTS['Nmol']

        natom=1
        Cv_J_kg_K = 1737.36
        ndof = 6
        Cvlim = ndof/2*core.CONSTS['R']
        Cv = Cv_J_kg_K*1e-3*comp_d['molmass']/natom # J/mol/K
        Cvlimfac = Cv/Cvlim

        # Normalize to a single atom basis
        V0 /= natom

        eos_mod = models.MieGruneisenEos(
            kind_thermal='ConstHeatCap', kind_gamma='GammaPowLaw',
            kind_compress='BirchMurn3', compress_path_const='S',
            natom=natom)
        ref_state = eos_mod.refstate.ref_state
        ref_state['T0'] = 2273

        eos_mod.ndof = ndof
        eos_mod.set_param_values([V0], param_names=['V0'])
        eos_mod.set_param_values([0.396,-2.02], param_names=['gamma0','q'])
        eos_mod.set_param_values([16.41, 7.37], param_names=['K0','KP0'])
        eos_mod.set_param_values([Cvlimfac], param_names=['Cvlimfac'])

        return eos_mod, comp_d

    @classmethod
    def _get_Fa(cls):
        oxide_data = get_oxide_data()
        comp_d = calc_comp_details({'FeO':2,'SiO2':1}, kind='mol')

        rho0 = 3.699
        V0 = comp_d['molmass']/rho0*core.CONSTS['ang3percc']/core.CONSTS['Nmol']

        natom=1
        Cv_J_kg_K = 1122.73
        ndof = 6
        Cvlim = ndof/2*core.CONSTS['R']
        Cv = Cv_J_kg_K*1e-3*comp_d['molmass']/natom # J/mol/K
        Cvlimfac = Cv/Cvlim

        # Normalize to a single atom basis
        V0 /= natom

        eos_mod = models.MieGruneisenEos(
            kind_thermal='ConstHeatCap', kind_gamma='GammaPowLaw',
            kind_compress='BirchMurn3', compress_path_const='S',
            natom=natom)
        ref_state = eos_mod.refstate.ref_state
        ref_state['T0'] = 1573

        eos_mod.ndof = ndof
        eos_mod.set_param_values([V0], param_names=['V0'])
        eos_mod.set_param_values([0.412,-0.95], param_names=['gamma0','q'])
        eos_mod.set_param_values([21.99, 7.28], param_names=['K0','KP0'])
        eos_mod.set_param_values([Cvlimfac], param_names=['Cvlimfac'])

        return eos_mod, comp_d

    @classmethod
    def _get_Di(cls):
        oxide_data = get_oxide_data()
        comp_d = calc_comp_details({'CaO':1,'MgO':1,'SiO2':2}, kind='mol')

        rho0 = 2.643
        V0 = comp_d['molmass']/rho0*core.CONSTS['ang3percc']/core.CONSTS['Nmol']

        natom=1
        Cv_J_kg_K = 1506.21
        ndof = 6
        Cvlim = ndof/2*core.CONSTS['R']
        Cv = Cv_J_kg_K*1e-3*comp_d['molmass']/natom # J/mol/K
        Cvlimfac = Cv/Cvlim

        # Normalize to a single atom basis
        V0 /= natom

        eos_mod = models.MieGruneisenEos(
            kind_thermal='ConstHeatCap', kind_gamma='GammaPowLaw',
            kind_compress='BirchMurn3', compress_path_const='S',
            natom=natom
            )
        ref_state = eos_mod.refstate.ref_state
        ref_state['T0'] = 1673
        eos_mod.ndof = ndof

        eos_mod.set_param_values([V0], param_names=['V0'])
        eos_mod.set_param_values([0.493,-1.28], param_names=['gamma0','q'])
        eos_mod.set_param_values([24.57, 6.77], param_names=['K0','KP0'])
        eos_mod.set_param_values([Cvlimfac], param_names=['Cvlimfac'])


        return eos_mod, comp_d

    @classmethod
    def _get_An(cls):
        oxide_data = get_oxide_data()
        comp_d = calc_comp_details({'CaO':1,'Al2O3':1,'SiO2':1}, kind='mol')

        rho0 = 2.560
        V0 = comp_d['molmass']/rho0*core.CONSTS['ang3percc']/core.CONSTS['Nmol']

        natom=1
        Cv_J_kg_K = 1511.28
        ndof = 6
        Cvlim = ndof/2*core.CONSTS['R']
        Cv = Cv_J_kg_K*1e-3*comp_d['molmass']/natom # J/mol/K
        Cvlimfac = Cv/Cvlim

        # Normalize to a single atom basis
        V0 /= natom

        eos_mod = models.MieGruneisenEos(
            kind_thermal='ConstHeatCap', kind_gamma='GammaPowLaw',
            kind_compress='BirchMurn4', compress_path_const='S',
            natom=natom)

        ref_state = eos_mod.refstate.ref_state
        ref_state['T0'] = 1932
        eos_mod.ndof = ndof
        eos_mod.set_param_values([V0], param_names=['V0'])
        eos_mod.set_param_values([0.174,-1.86], param_names=['gamma0','q'])
        eos_mod.set_param_values([19.77, 3.73, 0.38], param_names=['K0','KP0','KP20'])
        eos_mod.set_param_values([Cvlimfac], param_names=['Cvlimfac'])

        return eos_mod, comp_d
#====================================================================
class MgPv_Mosenfelder2009(models.MieGruneisenEos):
    def __init__(self):
        self.init_eos()
        self.load_params()

    # def init_solid_params(eos_d):
    #     """
    #     Set EOS parameters for solid MgSiO3 perovskite (bridgmanite)
    #     Corresponds to BM3S model from Mosenfelder(2009)
    #     """
    #     # All units must be per atom (to make sense for arbitrary composition)

    #     models.Control.set_consts( [], [], eos_d )

    #     const_d = eos_d['const_d']

    #     Nat_cell = 20
    #     Nat_formula = 5

    #     T0 = 300 # K

    #     # EOS Parameter values initially set by Mosenfelder2009
    #     # Set model parameter values
    #     mass_avg = (24.31+28.09+3*16.0)/5.0 # g/(mol atom)
    #     S0 = 0.0 # must adjust
    #     param_key_a = ['T0','S0','mass_avg']
    #     param_val_a = np.array([T0,S0,mass_avg])
    #     models.Control.set_params( param_key_a, param_val_a, eos_d )

    #     # V0 = (38.575*1e-5)*mass_avg/eos_d['const_d']['Nmol']/1e3*1e30 # ang^3/atom
    #     V0 = 162.35/Nat_cell # ang^3/atom
    #     K0 = 254.7 # GPa
    #     KP0= 4.26
    #     E0 = 0.0
    #     param_key_a = ['V0','K0','KP0','E0']
    #     param_val_a = np.array([V0,K0,KP0,E0])
    #     models.Control.set_params( param_key_a, param_val_a, eos_d )

    #     VR = V0
    #     thetaR = 736 # K
    #     gammaR = 2.23
    #     qR     = 1.83
    #     param_key_a = ['VR','thetaR','gammaR','qR']
    #     param_val_a = np.array([VR,thetaR,gammaR,qR])
    #     models.Control.set_params( param_key_a, param_val_a, eos_d )

    #     # NOTE: Mosenfelder(2009) has mislabeled units as J/K/g
    #     #     -> units are actually J/K/kg  ???
    #     # The measured 1000K heat capacity of MgSiO3 is ~125 J/K/mol
    #     #      (equal to Dulong Petit value for 5 atom basis)
    #     #     -> This value is thus ~65% of that nominal value,
    #     #        balancing the 30 to 40% values of gamma that are higher than other
    #     #        studies  (static compression only constrains Gamma*Cv
    #     #
    #     # Max const-vol heat capacity:
    #     Cvmax = (806.0/1e3)*mass_avg/const_d['kJ_molpereV']/1e3 # J/mol atoms/K -> eV/K/atom

    #     param_key_a = ['Cvmax']
    #     param_val_a = np.array([Cvmax])
    #     models.Control.set_params( param_key_a, param_val_a, eos_d )




    #     # # Must convert energy units from kJ/g to eV/atom
    #     energy_conv_fac = mass_avg/eos_d['const_d']['kJ_molpereV']
    #     models.Control.set_consts( ['energy_conv_fac'], [energy_conv_fac], eos_d )


    #     compress_path_mod = models.BirchMurn3(path_const='S',level_const=T0,
    #                                           supress_energy=False,
    #                                           supress_press=False,
    #                                           expand_adj=False)
    #     models.Control.set_modtypes( ['CompressPathMod'], [compress_path_mod],
    #                                 eos_d )

    #     gamma_mod = models.GammaPowLaw(V0ref=False)
    #     models.Control.set_modtypes( ['GammaMod'], [gamma_mod], eos_d )

    #     thermal_mod = models.MieGrunDebye()
    #     models.Control.set_modtypes( ['ThermalMod'], [thermal_mod], eos_d )

    #     full_mod = models.ThermalPressMod()
    #     models.Control.set_modtypes( ['FullMod'], [full_mod], eos_d )


    #     return eos_d

    def init_eos(self):
        kind_thermal = 'Debye'
        kind_gamma = 'GammaPowLaw'
        kind_compress = 'BirchMurn3'
        compress_path_const = 'S'
        natom = 1

        apply_electronic = False
        ref_energy_type = 'E0'

        molar_mass = (24.31+28.09+3*16.0)/5.0 # g/(mol atom)

        super().__init__(
            kind_thermal=kind_thermal, kind_gamma=kind_gamma,
            kind_compress=kind_compress,
            compress_path_const=compress_path_const, natom=natom,
            ref_energy_type=ref_energy_type, molar_mass=molar_mass)

        pass

    def load_params(self):
        T0 = 300 # K

        self.refstate.ref_state['T0'] = T0

        mass_avg = (24.31+28.09+3*16.0)/5.0 # g/(mol atom)
        S0 = 0.0 # must adjust

        Nat_cell = 20
        Nat_formula = 5

        V0 = 162.35/Nat_cell # ang^3/atom
        K0 = 254.7 # GPa
        KP0= 4.26
        E0 = 0.0

        # VR = V0
        theta0 = 736 # K
        gamma0 = 2.23
        q     = 1.83

        # EOS Parameter values initially set by Mosenfelder2009
        # Set model parameter values

        # NOTE: Mosenfelder(2009) has mislabeled units as J/K/g
        #     -> units are actually J/K/kg  ???
        # The measured 1000K heat capacity of MgSiO3 is ~125 J/K/mol
        #      (equal to Dulong Petit value for 5 atom basis)
        #     -> This value is thus ~65% of that nominal value,
        #        balancing the 30 to 40% values of gamma that are higher than other
        #        studies  (static compression only constrains Gamma*Cv
        #
        # Max const-vol heat capacity:
        # core.CONSTS['kJ_molpereV']
        Cvmax = (806.0/1e3)*mass_avg/core.CONSTS['kJ_molpereV']/1e3 # J/mol atoms/K -> eV/K/atom

        # # Must convert energy units from kJ/g to eV/atom
        # energy_conv_fac = mass_avg/eos_d['const_d']['kJ_molpereV']


        # compress_path_mod = models.BirchMurn3(
        #     path_const='S', level_const=T0, supress_energy=False,
        #     supress_press=False, expand_adj=False)

        self.set_param_values([S0, V0], param_names=['S0', 'V0'])
        self.set_param_values([K0, KP0, E0], param_names=['K0', 'KP0', 'E0'])

        # from IPython import embed; embed(); import ipdb; ipdb.set_trace();

        thermal_calc = self.calculators['thermal']
        Cvlim_default = thermal_calc._get_Cv_limit()
        Cvlimfac = Cvmax/Cvlim_default

        self.set_param_values(
            [Cvlimfac, theta0, gamma0, q],
            param_names=['Cvlimfac', 'theta0', 'gamma0', 'q'])

        pass
#====================================================================
class MgSiO3_RTPress(models.RTPressEos):
    def __init__(self):
        self.init_eos()
        self.load_params()

    def init_eos(self):
        kind_compress = 'Vinet'
        compress_path_const = 'T'
        kind_gamma = 'GammaFiniteStrain'
        kind_electronic = 'CvPowLaw'
        apply_electronic = True
        kind_RTpoly = 'V'
        RTpoly_order = 4
        ref_energy_type = 'E0'
        natom = 1
        molar_mass = (24.31+28.09+3*16.0)/5.0 # g/(mol atom)

        super().__init__(
            kind_compress=kind_compress,
            compress_path_const=compress_path_const,
            kind_gamma=kind_gamma, kind_RTpoly=kind_RTpoly,
            kind_electronic=kind_electronic, apply_electronic=True,
            ref_energy_type=ref_energy_type,
            RTpoly_order=RTpoly_order, natom=natom, molar_mass=molar_mass)
        pass

    def load_params(self):
        T0 = 3000
        # self.set_refstate('T0',T0)
        ref_state = self.refstate.ref_state
        ref_state['T0'] = T0

        S0 = 0.0
        V0 = 12.94925 # Ang^3/atom
        mexp = 0.6
        K0 = 13.2000
        KP0 = 8.23837
        E0 = -20.595341
        gamma0 = 0.189943
        gammap0 = -1.94024
        # NOTE: this is increasing order
        bcoef_a = np.array([+0.982133, +0.6149976, +1.3104885,
                            -3.04036, -4.1027947])
        Cvlimfac = 1.0

        CvelFac0 = 2.271332e-4
        CvelFacExp = 0.677736
        Tel0 = 2466.6
        TelExp = -0.45780
        # ndof=6
        # self.ndof = ndof


        self.set_param_values([S0,V0,mexp,Cvlimfac],
                              param_names=['S0','V0','mexp','Cvlimfac'])
        self.set_param_values([K0,KP0,E0], param_names=['K0','KP0','E0'])
        self.set_param_values([gamma0,gammap0],
                                 param_names=['gamma0','gammap0'])

        bcoef_names = self.get_array_param_names('bcoef')
        # print("***************")
        # print(bcoef_names)
        # print(bcoef_a)
        # print("***************")
        self.set_param_values(bcoef_a, param_names=bcoef_names)

        self.set_param_values([Tel0,TelExp], param_names=['Tel0','TelExp'])
        self.set_param_values([CvelFac0,CvelFacExp],
                              param_names=['CvelFac0','CvelFacExp'])
        pass
#====================================================================
class MgSiO3_deKoker2009_PTherm(models.PThermPolyEos):
    def __init__(self):
        self.init_eos()
        self.load_params()

    def init_eos(self):
        # Equivalent to Stixrude2005, Stixrude2009
        logscale = True
        kind_Pth = 'V'
        Pth_order = 5
        kind_gamma = 'GammaPowLaw'
        kind_compress = 'Vinet'

        ref_energy_type='E0'
        natom=1
        molar_mass = (24.31+28.09+3*16.0)/5.0 # g/(mol atom)

        super().__init__(logscale=logscale, kind_Pth=kind_Pth,
                         Pth_order=Pth_order, kind_gamma=kind_gamma,
                         kind_compress=kind_compress,
                         ref_energy_type=ref_energy_type,
                         natom=natom, molar_mass=molar_mass)
        pass

    def load_params(self, Nsamp=1001):
        T0 = 3000
        # self.set_refstate('T0',T0)
        ref_state = self.refstate.ref_state
        ref_state['T0'] = T0
        # Natom = 5
#
        # Vconv = 1./Natom*core.CONSTS['ang3percc']/core.CONSTS['Nmol']
        # # (ang^3/atom) / (cc/mol)
        # Econv = 1./Natom/core.CONSTS['kJ_molpereV'] # (eV/atom) / (kJ/mol)
#
        # # Equivalent to Stixrude2005, Stixrude2009
        # T0 = 3000
#
        # Vref = 38.88 # cm^3/mol
        # V0 = Vref*Vconv
#
        # dirnm = 'data/'
        # # extract data and store
#
        # Pref_dK09_a = np.loadtxt(
        #     dirnm+'MgSiO3-P3000-deKoker2009.csv', skiprows=1, delimiter=',')
        # Eref_dK09_a = np.loadtxt(
        #     dirnm+'MgSiO3-E3000-deKoker2009.csv', skiprows=1, delimiter=',')
        # gamma_dK09_a = np.loadtxt(
        #     dirnm+'MgSiO3-gamma-deKoker2009.csv', skiprows=1, delimiter=',')
        # Ptherm_dK09_a = np.loadtxt(
        #     dirnm+'MgSiO3-Ptherm-deKoker2009.csv', skiprows=1, delimiter=',')
#
        # # Get data lims of ALL datasets
        # Vfac_min = np.max((Pref_dK09_a[0,0], Eref_dK09_a[0,0],
        #                    gamma_dK09_a[0,0], Ptherm_dK09_a[0,0]))
        # Vfac_max = np.min((Pref_dK09_a[-1,0], Eref_dK09_a[-1,0],
        #                    gamma_dK09_a[-1,0], Ptherm_dK09_a[-1,0]))
#
        # Vfac_a = np.linspace(Vfac_min, Vfac_max, Nsamp)
        # V_a = Vfac_a*V0
#
        # Pref_a = interpolate.interp1d(
        #     Pref_dK09_a[:,0], Pref_dK09_a[:,1], kind='cubic')(Vfac_a)
        # Eref_a = interpolate.interp1d(
        #     Eref_dK09_a[:,0], Eref_dK09_a[:,1], kind='cubic')(Vfac_a)
        # gamma_a = interpolate.interp1d(
        #     gamma_dK09_a[:,0], gamma_dK09_a[:,1], kind='cubic')(Vfac_a)
        # Ptherm_a = interpolate.interp1d(
        #     Ptherm_dK09_a[:,0], Ptherm_dK09_a[:,1], kind='cubic')(Vfac_a)
#
        # Pref_a = signal.savgol_filter(Pref_a, 301, 3)
        # Eref_a = signal.savgol_filter(Eref_a, 301, 3)
        # gamma_a = signal.savgol_filter(gamma_a, 301, 3)
        # Ptherm_a = signal.savgol_filter(Ptherm_a, 201, 3)
#
        # # plt.clf()
        # # plt.plot(Pref_dK09_a[:,0], Pref_dK09_a[:,1], 'ko', Vfac_a, Pref_a, 'r-' )
        # # plt.clf()
        # # plt.plot(Eref_dK09_a[:,0], Eref_dK09_a[:,1], 'ko', Vfac_a, Eref_a, 'r-' )
#
        # # plt.clf()
        # # plt.plot(gamma_dK09_a[:,0], gamma_dK09_a[:,1], 'ko', Vfac_a, gamma_a, 'r-' )
#
        # # plt.clf()
        # # plt.plot(Ptherm_dK09_a[:,0], Ptherm_dK09_a[:,1], 'ko', Vfac_a, Ptherm_a, 'r-' )
#
        # param_d = {}
        # param_d['V0'] = V0
        # param_d['mass_avg'] = eos_mod.molar_mass
#
        # miegrun_d = {}
        # miegrun_d['const_d'] = const_d
        # miegrun_d['Vmin'] = V_a[0]
        # miegrun_d['Vmax'] = V_a[-1]
        # miegrun_d['T0'] = T0
        # miegrun_d['V'] = V_a
#
        # # Pref_f = interpolate.interp1d(V_a,Pref_a)
        # # Eref_f = interpolate.interp1d(V_a,Econv*Eref_a)
        # # gamma_f = interpolate.interp1d(V_a,gamma_a)
        # # Ptherm_f = interpolate.interp1d(V_a,Ptherm_a)
#
        # # miegrun_d['Vref_T0'] = V_a
        # # miegrun_d['Pref_T0'] = Pref_a
        # # miegrun_d['Eref_T0'] = Eref_a*Econv
        # # miegrun_d['gamma'] = gamma_a
        # # miegrun_d['Ptherm'] = Ptherm_a
#
        # # miegrun_d['Pref_f'] = Pref_f
        # # miegrun_d['Eref_f'] = Eref_f
        # # miegrun_d['gamma_f'] = gamma_f
        # # miegrun_d['Ptherm_f'] = Ptherm_f
#
        # # miegrun_mod = miegrun_eos_mod()
#
        # from IPython import embed; embed(); import ipdb as pdb; pdb.set_trace()
        # full_mod = miegrun_eos_mod()
        # full_mod.update_lookup_tables(miegrun_d,Vref_a=V_a, Pref_a=Pref_a,
        #                               Eref_a=Eref_a*Econv, gamma_a=gamma_a,
        #                               Ptherm_a=Ptherm_a)
#
        # modtype_d = {}
        # modtype_d['FullMod'] = full_mod
        # miegrun_d['modtype_d'] = modtype_d
        # miegrun_d['param_d'] = param_d
#
        # # return miegrun_d
#
        # self.set_param_values([S0,V0,mexp,Cvlimfac],
        #                       param_names=['S0','V0','mexp','Cvlimfac'])
        # self.set_param_values([K0,KP0,E0], param_names=['K0','KP0','E0'])
        # self.set_param_values([gamma0,gammap0],
        #                          param_names=['gamma0','gammap0'])
#
        # bcoef_names = self.get_array_param_names('bcoef')
        # # print("***************")
        # # print(bcoef_names)
        # # print("***************")
        # self.set_param_values(bcoef_a, param_names=bcoef_names)
#
        # self.set_param_values([Tel0,TelExp], param_names=['Tel0','TelExp'])
        # self.set_param_values([CvelFac0,CvelFacExp],
        #                       param_names=['CvelFac0','CvelFacExp'])
        pass
#====================================================================
