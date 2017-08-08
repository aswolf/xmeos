from __future__ import absolute_import, print_function, division, with_statement
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

class BaseTestEos(object):
    @abstractmethod
    def load_eos(self):
        pass

    def test_param_getset(self):
        eos_mod = self.load_eos()
        param_names = eos_mod.param_names
        param_values_orig = eos_mod.param_values

        param_values = []
        for name in param_names:
            value, = eos_mod.get_param_values(param_names=name)
            param_values.append(value)

        param_values = np.array(param_values)

        assert np.all(param_values==param_values_orig), \
            'param values retrieved one at a time unequal.'

        # Test scaling/unscaling parameters
        FAC = 2
        eos_mod.set_param_values(FAC*eos_mod.param_values)

        for name, value in zip(eos_mod.param_names, eos_mod.param_values):
            eos_mod.set_param_values(value/FAC, param_names=name)

        param_values_set = eos_mod.param_values

        assert np.all(param_values_set==param_values_orig), (
            'Parameter set method not working. '
            'Doubling and Halving should match original param. values.' )

        pass

    def numerical_deriv(self, x, y, dydx, scale=1):
        Nsamp = len(x)
        assert len(y)==Nsamp, 'y array must be same length as x array.'
        assert len(dydx)==Nsamp, 'dydx array must be same length as x array.'
        try:
            assert len(scale)==Nsamp, (
                'If scale is an array, it must be same length as x.' )
        except:
            pass

        dx = x[1]-x[0]
        dydx_num = scale*np.gradient(y, dx)
        dydx_range = np.max(dydx)-np.min(dydx)

        dydx_diff = dydx_num-dydx

        abs_err =  np.max(np.abs(dydx_diff))
        rel_err =  np.max(np.abs(dydx_diff/dydx))
        range_err =  np.max(np.abs(dydx_diff/dydx_range))

        return abs_err, rel_err, range_err

    def test_pickle(self):
        eos_mod = self.load_eos()

        data_string = pickle.dumps(eos_mod)
        eos_load_mod = pickle.loads(data_string)

        # filenm = 'test/pkl/test_pickle.pkl'
        # with open(filenm, 'w') as f:
        #     pickle.dump(eos_mod, f)

        # with open(filenm, 'r') as f:
        #     eos_loaded = pickle.load(f)

        assert repr(eos_mod)==repr(eos_load_mod), (
            'Pickled and unpickled Eos Models are not equal.')

    # # def do_test_energy_perturb_eval(self):
    # #     TOL = 1e-4
    # #     dxfrac = 1e-8

    # #     Nsamp = 10001
    # #     eos_mod = self.init_params()

    # #     param_d = eos_d['param_d']
    # #     Vmod_a = np.linspace(.7,1.3,Nsamp)*param_d['V0']
    # #     dV = Vmod_a[1] - Vmod_a[0]

    # #     if compress_path_mod.expand_adj:
    # #         scale_a, paramkey_a = \
    # #             compress_path_mod.get_param_scale( eos_d,apply_expand_adj=True )
    # #     else:
    # #         scale_a, paramkey_a = compress_path_mod.get_param_scale( eos_d)

    # #     Eperturb_num_a = np.zeros((paramkey_a.size,Nsamp))
    # #     for ind,paramkey in enumerate(paramkey_a):
    # #         Eperturb_num_a[ind,:] = compress_path_mod.param_deriv\
    # #             ( 'energy', paramkey, Vmod_a, eos_d, dxfrac=dxfrac)


    # #     # dEdV0_a = compress_path_mod.param_deriv( 'energy', 'V0', Vmod_a, eos_d, dxfrac=dxfrac)
    # #     # dEdK0_a = compress_path_mod.param_deriv( 'energy', 'K0', Vmod_a, eos_d, dxfrac=dxfrac)
    # #     # dEdKP0_a = compress_path_mod.param_deriv( 'energy', 'KP0', Vmod_a, eos_d, dxfrac=dxfrac)
    # #     # dEdKP20_a = compress_path_mod.param_deriv( 'energy', 'KP20', Vmod_a, eos_d, dxfrac=dxfrac)
    # #     # dEdE0_a = compress_path_mod.param_deriv( 'energy', 'E0', Vmod_a, eos_d, dxfrac=dxfrac)

    # #     Eperturb_a, scale_a, paramkey_a = compress_path_mod.energy_perturb(Vmod_a, eos_d)
    # #     # print paramkey_a

    # #     # Eperturb_num_a = np.vstack((dEdV0_a,dEdK0_a,dEdKP0_a,dEdKP20_a,dEdE0_a))
    # #     max_error_a = np.max(np.abs(Eperturb_a-Eperturb_num_a),axis=1)

    # #     # try:
    # #     # except:

    # #     # from IPython import embed; embed(); import ipdb; ipdb.set_trace()
    # #     # plt.plot(Vmod_a,Eperturb_a.T,'-',Vmod_a, Eperturb_num_a.T,'--')

    # #     # plt.ion()
    # #     # plt.figure()
    # #     # plt.clf()
    # #     # plt.plot(Vmod_a[::100], Eperturb_num_a[:,::100].T,'x',
    # #     #          Vmod_a[::100], Eperturb_a[3,::100].T,'r-')
    # #     # plt.plot(Vmod_a[::100], Eperturb_num_a[:,::100].T,'x',
    # #     #          Vmod_a, Eperturb_a.T,'-')
    # #     # plt.plot(Vmod_a[::100], Eperturb_a[3,::100].T,'r-')

    # #     # Eperturb_num_a-Eperturb_a
    # #     assert np.all(max_error_a < TOL),'Error in energy perturbation must be'\
    # #         'less than TOL.'
