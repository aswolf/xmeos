# from __future__ import absolute_import, print_function, division, with_statement
from builtins import object
import numpy as np
import xmeos
from xmeos import models
from xmeos.models import core

import pytest
import matplotlib.pyplot as plt
import matplotlib as mpl
from abc import ABCMeta, abstractmethod
import copy

import test_models

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
class BaseTestCompressEos(test_models.BaseTestEos):
    def test_press_S(self):
        self.calc_test_press(path_const='S')

    def test_press_T(self):
        self.calc_test_press(path_const='T')

    def test_press_0K(self):
        self.calc_test_press(path_const='0K')

    def calc_test_press(self, path_const='T'):

        TOL = 1e-3

        Nsamp = 10001
        eos_mod = self.load_eos(path_const=path_const)

        V0, = eos_mod.get_param_values(param_names='V0')
        V0 += -.137
        eos_mod.set_param_values(V0,param_names='V0')

        V0get, = eos_mod.get_param_values(param_names='V0')

        assert V0 == V0get, 'Must be able to store and retrieve non-integer values'

        assert np.abs(eos_mod.press(V0))<TOL/100,(
            'pressure at V0 must be zero by definition'
        )

        Vmod_a = np.linspace(.7,1.2,Nsamp)*V0
        dV = Vmod_a[1] - Vmod_a[0]

        press_a = eos_mod.press(Vmod_a)
        energy_a = eos_mod.energy(Vmod_a)

        abs_err, rel_err, range_err = self.numerical_deriv(
            Vmod_a, energy_a, press_a, scale=-core.CONSTS['PV_ratio'])

        assert range_err < TOL, 'range error in Press, ' + np.str(range_err) + \
            ', must be less than TOL, ' + np.str(TOL)

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


    #####################
    # Explicitly call these test methods from super to guarantee correct
    # behavior of decorated xfail classes
    #####################
    def test_param_getset(self):
        super(BaseTestCompressEos, self).test_param_getset()

    def test_pickle(self):
        super(BaseTestCompressEos, self).test_pickle()
#====================================================================

#====================================================================
# SEC:2 Implimented Test Clases
#====================================================================
# 2.1: CompressEos Tests
#====================================================================
class TestVinet(BaseTestCompressEos):
    def load_eos(self, path_const='T'):
        eos_mod = models.CompressEos(
            kind='Vinet', path_const=path_const)
        return eos_mod
#====================================================================
class TestBirchMurn3(BaseTestCompressEos):
    def load_eos(self, path_const='T'):
        eos_mod = models.CompressEos(kind='BirchMurn3', path_const=path_const)
        return eos_mod
#====================================================================
class TestBirchMurn4(BaseTestCompressEos):
    def load_eos(self, path_const='T'):
        eos_mod = models.CompressEos(kind='BirchMurn4', path_const=path_const)
        return eos_mod
#====================================================================
class TestGenFiniteStrain(BaseTestCompressEos):
    def load_eos(self, path_const='T'):
        eos_mod = models.CompressEos(kind='GenFiniteStrain', path_const=path_const)
        return eos_mod
#====================================================================
class TestTait(BaseTestCompressEos):
    def load_eos(self, path_const='T'):
        eos_mod = models.CompressEos(kind='Tait', path_const=path_const)
        return eos_mod
#====================================================================
notimplimented = pytest.mark.xfail(reason='PolyRho energy expressions not implimented yet.')
@notimplimented
class TestPolyRho(BaseTestCompressEos):
    def load_eos(self, path_const='T'):
        eos_mod = models.CompressEos(kind='PolyRho', path_const=path_const,
                                     order=6)
        return eos_mod

    def test_poly_scale(self):

        TOL = 1e-6

        Nsamp = 101
        eos_mod = self.load_eos()
        calc = eos_mod.calculators['compress']

        V0, = eos_mod.get_param_values(param_names='V0')
        Vmod_a = np.linspace(.7,1.2,Nsamp)*V0


        dV = Vmod_a[1] - Vmod_a[0]
        coef_a, rho0 = calc._get_poly_coef()

        rho_a = calc._vol_to_rho(Vmod_a)
        press_a = eos_mod.press(Vmod_a)
        press_direct_a = np.polyval(coef_a, rho_a-rho0)
        dev_a = press_a - press_direct_a

        assert np.all(np.abs(dev_a)<TOL), \
            'PolyRho polynomial calculation of press not consistent'
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
# class TestExpandCompressPathMod(BaseTestCompressEos):
#     def load_eos(self, eos_d):
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
