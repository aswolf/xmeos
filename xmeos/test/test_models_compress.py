from __future__ import absolute_import, print_function, division, with_statement
from builtins import object
import numpy as np
# from models import compress
# from models import core
import xmeos
from xmeos import models
from xmeos.models import core

import pytest
import matplotlib.pyplot as plt
import matplotlib as mpl
from abc import ABCMeta, abstractmethod
import copy

try:
   import cPickle as pickle
except:
   import pickle

#====================================================================
# Define "slow" tests
#  - indicated by @slow decorator
#  - slow tests are run only if using --runslow cmd line arg
#====================================================================
slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)


#====================================================================
# SEC:1 Abstract Test Classes
#====================================================================
class BaseTestCompressEos(object):
    @abstractmethod
    def load_compress_eos(self, eos_d):
        assert False, 'must implement load_compress_eos()'

    def init_params(self):
        # Set model parameter values
        E0 = 0.0 # eV/atom
        V0 = 38.0 # 1e-5 m^3 / kg
        K0 = 25.0 # GPa
        KP0 = 9.0 # 1
        param_names = ['V0','K0','KP0','E0']
        param_values = np.array([ V0, K0, KP0, E0 ])

        eos_mod = self.load_compress_eos()
        eos_mod.set_param_values(param_values, param_names)

        return eos_mod

    def test_press(self):

        TOL = 1e-4

        Nsamp = 10001
        eos_mod = self.init_params()

        # param_d = eos_d['param_d']
        V0, = eos_mod.get_param_values(param_names='V0')
        Vmod_a = np.linspace(.7,1.2,Nsamp)*V0
        dV = Vmod_a[1] - Vmod_a[0]

        press_a = eos_mod.press(Vmod_a)
        energy_a = eos_mod.energy(Vmod_a)

        press_num_a = -core.CONSTS['PV_ratio']*np.gradient(energy_a,dV)

        Prange = np.max(press_a)-np.min(press_a)
        press_diff_a = press_num_a-press_a
        #Exclude 1st and last points to avoid numerical derivative errors
        Perr =  np.max(np.abs(press_diff_a/Prange))

        PTOL = 3*Prange/Nsamp

        # print self
        # print PTOL*Prange


        # def plot_press_mismatch(Vmod_a,press_a,press_num_a):
        #     plt.figure()
        #     plt.ion()
        #     plt.clf()
        #     plt.plot(Vmod_a,press_num_a,'bx',Vmod_a,press_a,'r-')
        #     from IPython import embed; embed(); import ipdb; ipdb.set_trace()

        # plot_press_mismatch(Vmod_a,press_a,press_num_a)
        assert np.abs(Perr) < PTOL, '(Press error)/Prange, ' + np.str(Perr) + \
            ', must be less than PTOL'

    def test_pickle(self):
        eos_mod = self.init_params()

        data_string = pickle.dumps(eos_mod)
        eos_load_mod = pickle.loads(data_string)

        # filenm = 'test/pkl/test_pickle.pkl'
        # with open(filenm, 'w') as f:
        #     pickle.dump(eos_mod, f)

        # with open(filenm, 'r') as f:
        #     eos_loaded = pickle.load(f)

        assert repr(eos_mod)==repr(eos_load_mod), (
            'Pickled and unpickled Eos Models are not equal.')

    def do_test_energy_perturb_eval(self):
        TOL = 1e-4
        dxfrac = 1e-8

        Nsamp = 10001
        eos_mod = self.init_params()

        param_d = eos_d['param_d']
        Vmod_a = np.linspace(.7,1.3,Nsamp)*param_d['V0']
        dV = Vmod_a[1] - Vmod_a[0]

        if compress_path_mod.expand_adj:
            scale_a, paramkey_a = \
                compress_path_mod.get_param_scale( eos_d,apply_expand_adj=True )
        else:
            scale_a, paramkey_a = compress_path_mod.get_param_scale( eos_d)

        Eperturb_num_a = np.zeros((paramkey_a.size,Nsamp))
        for ind,paramkey in enumerate(paramkey_a):
            Eperturb_num_a[ind,:] = compress_path_mod.param_deriv\
                ( 'energy', paramkey, Vmod_a, eos_d, dxfrac=dxfrac)


        # dEdV0_a = compress_path_mod.param_deriv( 'energy', 'V0', Vmod_a, eos_d, dxfrac=dxfrac)
        # dEdK0_a = compress_path_mod.param_deriv( 'energy', 'K0', Vmod_a, eos_d, dxfrac=dxfrac)
        # dEdKP0_a = compress_path_mod.param_deriv( 'energy', 'KP0', Vmod_a, eos_d, dxfrac=dxfrac)
        # dEdKP20_a = compress_path_mod.param_deriv( 'energy', 'KP20', Vmod_a, eos_d, dxfrac=dxfrac)
        # dEdE0_a = compress_path_mod.param_deriv( 'energy', 'E0', Vmod_a, eos_d, dxfrac=dxfrac)

        Eperturb_a, scale_a, paramkey_a = compress_path_mod.energy_perturb(Vmod_a, eos_d)
        # print paramkey_a

        # Eperturb_num_a = np.vstack((dEdV0_a,dEdK0_a,dEdKP0_a,dEdKP20_a,dEdE0_a))
        max_error_a = np.max(np.abs(Eperturb_a-Eperturb_num_a),axis=1)

        # try:
        # except:

        # from IPython import embed; embed(); import ipdb; ipdb.set_trace()
        # plt.plot(Vmod_a,Eperturb_a.T,'-',Vmod_a, Eperturb_num_a.T,'--')

        # plt.ion()
        # plt.figure()
        # plt.clf()
        # plt.plot(Vmod_a[::100], Eperturb_num_a[:,::100].T,'x',
        #          Vmod_a[::100], Eperturb_a[3,::100].T,'r-')
        # plt.plot(Vmod_a[::100], Eperturb_num_a[:,::100].T,'x',
        #          Vmod_a, Eperturb_a.T,'-')
        # plt.plot(Vmod_a[::100], Eperturb_a[3,::100].T,'r-')

        # Eperturb_num_a-Eperturb_a
        assert np.all(max_error_a < TOL),'Error in energy perturbation must be'\
            'less than TOL.'
#====================================================================
class BaseTest4thOrdCompressEos(BaseTestCompressEos):
    def init_params(self):
        # Use parents init_params method
        eos_mod = super(BaseTest4thOrdCompressEos,self).init_params()

        V0,K0,KP0 = eos_mod.get_param_values(param_names=['V0','K0','KP0'])
        # Add K''0 param
        KP20 = -1.1*KP0/K0
        eos_mod.set_param_values([KP20], param_names=['KP20'])

        return eos_mod
#====================================================================
# class TestPickle(object):
#     def load_compress_eos(self):
#         eos_mod = models.CompressEos(
#             kind='Vinet', path_const='T', level_const=300)
#         return eos_mod
#
#     def init_params(self):
#         # Set model parameter values
#         E0 = 0.0 # eV/atom
#         V0 = 38.0 # 1e-5 m^3 / kg
#         K0 = 25.0 # GPa
#         KP0 = 9.0 # 1
#         param_names = ['V0','K0','KP0','E0']
#         param_values = np.array([ V0, K0, KP0, E0 ])
#
#         eos_mod = self.load_compress_eos()
#         eos_mod.set_param_values(param_values, param_names)
#
#         return eos_mod
#
#     def test_pickle(self):
#
#         TOL = 1e-4
#
#         Nsamp = 10001
#         eos_mod = self.init_params()
#
#         data_string = pickle.dumps(eos_mod)
#         eos_load_mod = pickle.loads(data_string)
#
#         # filenm = 'test/pkl/test_pickle.pkl'
#         # with open(filenm, 'w') as f:
#         #     pickle.dump(eos_mod, f)
#
#         # with open(filenm, 'r') as f:
#         #     eos_loaded = pickle.load(f)
#
#         assert repr(eos_mod)==repr(eos_load_mod), (
#             'Pickled and unpickled Eos Models are not equal.')
#
        # assert False, 'Try to pickle object.'
#====================================================================

#====================================================================
# SEC:2 Implimented Test Clases
#====================================================================
# 2.1: CompressEos Tests
#====================================================================
class TestVinet(BaseTestCompressEos):
    def load_compress_eos(self):
        eos_mod = models.CompressEos(
            kind='Vinet', path_const='T', level_const=300)
        return eos_mod

    # def test_energy_perturb_eval(self):
    #     self.do_test_energy_perturb_eval()
    #     pass
#====================================================================
class TestBirchMurn3(BaseTestCompressEos):
    def load_compress_eos(self):
        eos_mod = models.CompressEos(
            kind='BirchMurn3', path_const='S', level_const=0)
        return eos_mod
#====================================================================
class TestBirchMurn4(BaseTest4thOrdCompressEos):
    def load_compress_eos(self):
        eos_mod = models.CompressEos(
            kind='BirchMurn4', path_const='S', level_const=0)
        return eos_mod
#====================================================================
class TestGenFiniteStrain(BaseTest4thOrdCompressEos):
    def init_params(self):
        # Use parents init_params method
        eos_mod = super(TestGenFiniteStrain,self).init_params()

        # Add nexp param
        nexp = +2.0
        eos_mod.set_param_values([nexp], param_names=['nexp'])

        return eos_mod

    def load_compress_eos(self):
        eos_mod = models.CompressEos(
            kind='GenFiniteStrain', path_const='S', level_const=0)
        return eos_mod
#====================================================================
class TestTait(BaseTest4thOrdCompressEos):
    def load_compress_eos(self):
        eos_mod = models.CompressEos(
            kind='Tait', path_const='S', level_const=0)
        print(eos_mod)
        return eos_mod

    # def test_energy_perturb_eval(self):
    #     self.do_test_energy_perturb_eval()
    #     pass
#====================================================================
# class TestCompareCompressEos(object):
#     def init_params(self):
#         # Set model parameter values
#         E0 = 0.0 # eV/atom
#         V0 = 38.0 # 1e-5 m^3 / kg
#         K0 = 25.0 # GPa
#         KP0 = 9.0 # 1
#         param_key_a = ['V0','K0','KP0','E0']
#         param_val_a = np.array([ V0, K0, KP0, E0 ])
#
#         # core.set_consts( [], [], eos_d )
#         core.set_params( param_key_a, param_val_a, eos_d )
#
#         return eos_d
#
#     def get_eos_mods(self):
#         eos_vinet_d = self.init_params()
#         eos_tait_d = self.init_params()
#
#         core.set_modtypes( ['CompressPathMod'], [compress.Vinet(path_const='S')],
#                            eos_vinet_d )
#         core.set_modtypes( ['CompressPathMod'], [compress.Tait(path_const='S')],
#                            eos_tait_d )
#
#         return eos_vinet_d, eos_tait_d
#
#     def calc_energy_perturb( self, eos_d ):
#         dxfrac = 1e-6
#         Nsamp = 10001
#
#         param_d = eos_d['param_d']
#         Vmod_a = np.linspace(.7,1.1,Nsamp)*param_d['V0']
#         dV = Vmod_a[1] - Vmod_a[0]
#
#         compress_path_mod = eos_d['modtype_d']['CompressPathMod']
#         scale_a, paramkey_a = compress_path_mod.get_param_scale( eos_d )
#
#         Eperturb_num_a = np.zeros((paramkey_a.size,Nsamp))
#         for ind,paramkey in enumerate(paramkey_a):
#             Eperturb_num_a[ind,:] = compress_path_mod.param_deriv\
#                 ( 'energy', paramkey, Vmod_a, eos_d, dxfrac=dxfrac)
#
#         Eperturb_a, scale_a, paramkey_a = compress_path_mod.energy_perturb(Vmod_a, eos_d)
#
#         Eperturb_num_a = np.zeros((paramkey_a.size,Nsamp))
#         for ind,paramkey in enumerate(paramkey_a):
#             Eperturb_num_a[ind,:] = compress_path_mod.param_deriv\
#                 ( 'energy', paramkey, Vmod_a, eos_d, dxfrac=dxfrac)
#
#         return Eperturb_a, Eperturb_num_a, Vmod_a, scale_a, paramkey_a
#
#     def calc_energy( self, eos_d ):
#         dxfrac = 1e-6
#         Nsamp = 10001
#
#         param_d = eos_d['param_d']
#         Vmod_a = np.linspace(.7,1.1,Nsamp)*param_d['V0']
#         dV = Vmod_a[1] - Vmod_a[0]
#
#         compress_path_mod = eos_d['modtype_d']['CompressPathMod']
#         scale_a, paramkey_a = compress_path_mod.get_param_scale( eos_d )
#
#         energy_a = compress_path_mod.energy( Vmod_a, eos_d )
#
#         return energy_a, Vmod_a
#
#     def test_compare(self):
#         TOL = 1e-4
#
#         eos_vinet_d, eos_tait_d = self.get_eos_mods()
#         KP20 = -1.1*eos_tait_d['param_d']['KP0']/eos_tait_d['param_d']['K0']
#         core.set_params( ['KP20'], [KP20], eos_tait_d )
#
#         energy_vin_a, Vmod_vin_a = self.calc_energy( eos_vinet_d )
#         energy_tait_a, Vmod_tait_a = self.calc_energy( eos_tait_d )
#
#         # plt.ion()
#         # plt.figure()
#         # plt.clf()
#         # plt.plot(Vmod_vin_a, energy_vin_a,'k-',
#         #          Vmod_tait_a, energy_tait_a, 'r-')
#
#         Eperturb_vin_a, Eperturb_num_vin_a, Vmod_vin_a, scale_vin_a, \
#             paramkey_vin_a = self.calc_energy_perturb( eos_vinet_d )
#
#         Eperturb_tait_a, Eperturb_num_tait_a, Vmod_tait_a, scale_tait_a, \
#             paramkey_tait_a = self.calc_energy_perturb( eos_tait_d )
#
#         # from IPython import embed; embed(); import ipdb; ipdb.set_trace()
#
#         # plt.ion()
#         # plt.figure()
#         # plt.clf()
#         # plt.plot(Vmod_vin_a[::100], Eperturb_vin_a[:,::100].T,'x',
#         #          Vmod_tait_a, Eperturb_tait_a.T,'-')
#
#         dV = Vmod_vin_a[1] - Vmod_vin_a[0]
#         V0 = eos_tait_d['param_d']['V0']
#         indV0 = np.where(Vmod_vin_a==V0)[0][0]
#
#         Eperturb_diff = Eperturb_vin_a[:,indV0] - Eperturb_tait_a[[0,1,2,4],indV0]
#
#         assert np.all(np.abs(Eperturb_diff)<TOL), \
#             'Energy perturbations for Vinet and Tait EOS at V0 must agree to within TOL'
#
#         # Calc numerical volume derivs
#         # Some of these curves take very small values, making numerical
#         # comparison difficult, but  comparison by eye checks out
#         dE1_perturb_vin_a = np.gradient(Eperturb_vin_a,dV)[1]
#         dE2_perturb_vin_a = np.gradient(dE1_perturb_vin_a,dV)[1]
#         dE3_perturb_vin_a = np.gradient(dE2_perturb_vin_a,dV)[1]
#
#         dE1_perturb_tait_a = np.gradient(Eperturb_tait_a,dV)[1]
#         dE2_perturb_tait_a = np.gradient(dE1_perturb_tait_a,dV)[1]
#         dE3_perturb_tait_a = np.gradient(dE2_perturb_tait_a,dV)[1]
#
#         # plt.clf()
#         # plt.plot(Vmod_vin_a[::100], dE1_perturb_vin_a[:,::100].T,'x',
#         #          Vmod_tait_a, dE1_perturb_tait_a.T,'-')
#
#         # plt.clf()
#         # plt.plot(Vmod_vin_a[::100], dE2_perturb_vin_a[:,::100].T,'x',
#         #          Vmod_tait_a, dE2_perturb_tait_a.T,'-')
#
#         # Eperturb_vin_a[:,indV0]-Eperturb_tait_a[[0,1,2,4],indV0]
#         # Eperturb_vin_a[:,indV0]
#
#         # dE1_perturb_vin_a[:,indV0]-dE1_perturb_tait_a[[0,1,2,4],indV0]
#         # dE1_perturb_vin_a[:,indV0]
#
#         # plt.clf()
#         # plt.plot(Vmod_vin_a[::100], dE3_perturb_vin_a[:,::100].T,'x',
#         #          Vmod_tait_a, dE3_perturb_tait_a.T,'-')
#
#         pass
#====================================================================
# class TestExpandCompressPathMod(BaseTest4thOrdCompressEos):
#     def load_compress_eos(self, eos_d):
#         compress_path_mod   = compress.Vinet(path_const='S',expand_adj_mod=compress.Tait())
#         core.set_modtypes(['CompressPathMod'],[compress_path_mod], eos_d )
#
#         pass
#
#     def test_press_components(self):
#         TOL = 1e-4
#         dxfrac = 1e-8
#
#         Nsamp = 10001
#         eos_d = self.init_params()
#
#         param_d = eos_d['param_d']
#         Vmod_a = np.linspace(.7,1.3,Nsamp)*param_d['V0']
#         dV = Vmod_a[1] - Vmod_a[0]
#
#         compress_path_mod = eos_d['modtype_d']['CompressPathMod']
#
#         press_a = compress_path_mod.press( Vmod_a, eos_d )
#         press_pos_a = compress_path_mod.press( Vmod_a, eos_d, apply_expand_adj=False)
#         press_neg_a = compress_path_mod.expand_adj_mod.press( Vmod_a, eos_d )
#
#         # press_pos_a = expand_pos_mod.press( Vmod_a, eos_d )
#         # press_neg_a = expand_neg_mod.press( Vmod_a, eos_d )
#
#
#         ind_neg = Vmod_a>param_d['V0']
#         ind_pos = Vmod_a<param_d['V0']
#
#         assert np.all(press_a[ind_neg]==press_neg_a[ind_neg]),\
#             'The expansion corrected press must match ExpandNegMod for negative pressure values'
#         assert np.all(press_a[ind_pos]==press_pos_a[ind_pos]),\
#             'The expansion corrected press must match ExpandPosMod for positive pressure values'
#
#         # from IPython import embed; embed(); import ipdb; ipdb.set_trace()
#         # plt.ion()
#         # plt.figure()
#         # plt.clf()
#         # plt.plot(Vmod_a, press_pos_a, 'r--', Vmod_a, press_neg_a, 'b--',
#         #          Vmod_a, press_a, 'k-')
#
#         pass
#
#     def test_energy_components(self):
#         TOL = 1e-4
#         dxfrac = 1e-8
#
#         Nsamp = 10001
#         eos_d = self.init_params()
#
#         param_d = eos_d['param_d']
#         Vmod_a = np.linspace(.7,1.3,Nsamp)*param_d['V0']
#         dV = Vmod_a[1] - Vmod_a[0]
#
#         compress_path_mod = eos_d['modtype_d']['CompressPathMod']
#
#         energy_a = compress_path_mod.energy( Vmod_a, eos_d )
#         energy_pos_a = compress_path_mod.energy( Vmod_a, eos_d, apply_expand_adj=False )
#         energy_neg_a = compress_path_mod.expand_adj_mod.energy( Vmod_a, eos_d )
#
#
#         ind_neg = Vmod_a>param_d['V0']
#         ind_pos = Vmod_a<param_d['V0']
#
#         assert np.all(energy_a[ind_neg]==energy_neg_a[ind_neg]),\
#             'The expansion corrected energy must match ExpandNegMod for negative pressure values'
#         assert np.all(energy_a[ind_pos]==energy_pos_a[ind_pos]),\
#             'The expansion corrected energy must match ExpandPosMod for positive pressure values'
#
#
#         # from IPython import embed; embed(); import ipdb; ipdb.set_trace()
#         # plt.ion()
#         # plt.figure()
#         # plt.clf()
#         # plt.plot(Vmod_a, energy_pos_a, 'r--', Vmod_a, energy_neg_a, 'b--',
#         #          Vmod_a, energy_a, 'k-')
#
#         pass
#
#     def test_energy_perturb_eval(self):
#         self.do_test_energy_perturb_eval()
#         pass
#====================================================================



