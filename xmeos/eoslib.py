# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
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

class CMASF_melt_Thomas2013(CompositeEos):
    def __init__(self, meltcomp, kind='endmem'):
        # self._comp_d = calc_comp_details(meltcomp, kind=kind)
        self.endmem_comp = meltcomp
        self.load_eos(meltcomp)

    def load_eos(self, meltcomp):
        endmems = {}
        for comp in meltcomp:
            ieos, icomp_d = self.get_endmem_eos(endmem=comp)
            endmems[comp] = ieos

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

class MgSiO3_RTPress(models.RTPressEos):
    def __init__(self):
        self.init_eos()
        self.load_params()

    def init_eos(self):
        kind_compress='Vinet'
        compress_path_const='T'
        kind_gamma='GammaFiniteStrain'
        kind_RTpoly='logV'
        RTpoly_order=4
        natom=1

        super().__init__(
            kind_compress=kind_compress,
            compress_path_const=compress_path_const,
            kind_gamma=kind_gamma, kind_RTpoly=kind_RTpoly,
            RTpoly_order=RTpoly_order, natom=natom )
        pass

    def load_params(self):
        T0 = 3000
        # self.set_refstate('T0',T0)
        ref_state = self.refstate.ref_state
        ref_state['T0'] = T0

        S0 = 0.0
        V0 = 12.970 # Ang^3/atom
        mexp = 0.6
        K0 = 12.73
        KP0 = 8.391
        F0 = -20.5985
        gamma0 = 0.134
        gammap0 = -2.113
        # NOTE: the python standard puts the highest order coefficients first
        bcoef_a = np.array([-6.97, -6.39, +0.122, +0.688, +1.0027])[::-1]
        Cvlimfac = 1.0

        # ndof=6
        # self.ndof = ndof


        self.set_param_values([S0,V0,mexp,Cvlimfac],
                              param_names=['S0','V0','mexp','Cvlimfac'])
        self.set_param_values([K0,KP0,F0], param_names=['K0','KP0','F0'])
        self.set_param_values([gamma0,gammap0],
                                 param_names=['gamma0','gammap0'])

        bcoef_names = self.get_array_param_names('bcoef')
        # print("***************")
        # print(bcoef_names)
        # print("***************")
        self.set_param_values(bcoef_a, param_names=bcoef_names)
        pass
