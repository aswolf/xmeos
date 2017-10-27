# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from future.utils import with_metaclass

from builtins import str
from builtins import range
import numpy as np
import scipy as sp
from abc import ABCMeta, abstractmethod
from scipy import integrate
import scipy.interpolate as interpolate

from . import core
from . import _debye

__all__ = ['RefStateCalc']

_ref_compress_state_opts = ['P0', 'V0']
_ref_thermal_state_opts = ['T0', 'OK']
_ref_energy_type_opts = ['E0', 'F0', 'H0', 'G0']

#====================================================================
def validate_refstate_opt(refstate, refstate_opts):
    assert refstate in refstate_opts, (
        refstate + ' is not a valid reference state. '+
        'You must select one of: ' +  refstate_opts)
    pass
#====================================================================
def set_calculator(eos_mod, ref_compress_state='P0', ref_thermal_state='T0',
                   ref_energy_type='E0'):

    validate_refstate_opt(ref_compress_state, _ref_compress_state_opts)
    validate_refstate_opt(ref_thermal_state, _ref_thermal_state_opts)
    validate_refstate_opt(ref_energy_type, _ref_energy_type_opts)

    calc = RefStateCalc(eos_mod, ref_compress_state=ref_compress_state,
                        ref_thermal_state=ref_thermal_state,
                        ref_energy_type=ref_energy_type)

    eos_mod._add_calculator(calc, calc_type='refstate')
    pass
#====================================================================
class RefStateCalc(with_metaclass(ABCMeta, core.Calculator)):
    def __init__(self, eos_mod, ref_compress_state='P0',
                 ref_thermal_state='T0', ref_energy_type='E0'):
        self._eos_mod = eos_mod

        self._init_state_params(ref_compress_state, ref_thermal_state,
                                ref_energy_type, eos_mod.natom)
        # self._init_state_funcs()
        # self._init_internal_energy()
        self._init_required_calculators()
        pass

    def _init_state_params(self, ref_compress_state, ref_thermal_state,
                           ref_energy_type, natom):
        T0 = 29
        V0 = 10*natom
        S0_scale = 3*natom*core.CONSTS['kboltz']*T0
        energy_scale = 1*natom

        ref_state_points = []
        state_params = []

        if ref_compress_state=='P0':
            ref_state_points.append('P0')
            state_params.append('V0')
        elif ref_compress_state=='V0':
            ref_state_points.append('V0')
            state_params.append('P0')
        else:
            assert False, 'Invalid ref_compress_state.'

        if ref_thermal_state=='T0':
            ref_state_points.append('T0')
            state_params.append('S0')
        elif ref_thermal_state=='0K':
            ref_state_points.append('T0')
            T0 = 0
        else:
            assert False, 'Invalid ref_thermal_state.'

        all_ref_state_names = ['P0', 'V0', 'T0', 'S0', ref_energy_type]
        all_ref_state_units = ['GPa', 'ang^3', 'K', 'eV/K', 'eV']
        all_ref_state_defaults = [0, V0, T0, 0, 0]
        all_ref_state_scales = [1, V0/10, T0/10, S0_scale/10, energy_scale/10]

        param_names = state_params
        param_names.append(ref_energy_type)
        param_units = []
        param_defaults = []
        param_scales = []

        for name in param_names:
            inds, = np.where(np.array(all_ref_state_names) == name)
            ind = inds[0]
            param_units.append(all_ref_state_units[ind])
            param_defaults.append(all_ref_state_defaults[ind])
            param_scales.append(all_ref_state_scales[ind])

        ref_state = {}
        ref_state_units = {}

        for name in ref_state_points:
            inds, = np.where(np.array(all_ref_state_names) == name)
            ind = inds[0]
            ref_state_units[name] = all_ref_state_units[ind]
            ref_state[name] = all_ref_state_defaults[ind]

        self._set_params(param_names, param_units, param_defaults,
                         param_scales)
        self._ref_state = ref_state
        self._ref_state_units = ref_state_units

        # ref_state_names = []
        # ref_state_values = []
        # ref_state_units = []

        # for name in ref_state_points:
        #     inds, = np.where(np.array(all_ref_state_names) == name)
        #     ind = inds[0]
        #     ref_state_names.append(name)
        #     ref_state_values.append(all_ref_state_defaults[ind])
        #     ref_state_units.append(all_ref_state_units[ind])

        # self._set_params(param_names, param_units, param_defaults,
        #                  param_scales)
        # self._ref_state_names = ref_state_names
        # self._ref_state_values = ref_state_values
        # self._ref_state_units = ref_state_units

        pass

    @property
    def ref_state(self):
        return self._ref_state

    def _get_PV0(self):
        P0 = self.ref_press()
        V0 = self.ref_volume()
        PV0 = P0*V0/core.CONSTS['PV_ratio']
        return PV0

    def _get_TS0(self):
        T0 = self.ref_temp()
        S0 = self.ref_entropy()
        TS0 = T0*S0
        return TS0

    def ref_press(self):
        ref_state = self._ref_state
        if 'P0' in ref_state:
            P0 = ref_state['P0']
        else:
            P0 = self.eos_mod.get_param_values(param_names='P0')

        return P0

    def ref_volume(self):
        ref_state = self._ref_state
        if 'V0' in ref_state:
            V0 = ref_state['V0']
        else:
            V0 = self.eos_mod.get_param_values(param_names='V0')

        return V0

    def ref_temp(self):
        ref_state = self._ref_state
        if 'T0' in ref_state:
            T0 = ref_state['T0']
        else:
            T0 = self.eos_mod.get_param_values(param_names='T0')

        return T0

    def ref_entropy(self):
        ref_state = self._ref_state
        if 'S0' in ref_state:
            S0 = ref_state['S0']
        else:
            S0 = self.eos_mod.get_param_values(param_names='S0')

        return S0

    def ref_internal_energy(self):
        if 'E0' in self.param_names:
            E0 = self.eos_mod.get_param_values(param_names='E0')
        elif 'F0' in self.param_names:
            F0 = self.eos_mod.get_param_values(param_names='F0')
            TS0 = self._get_TS0()
            E0 = F0 + TS0
        elif 'H0' in self.param_names:
            H0 = self.eos_mod.get_param_values(param_names='H0')
            PV0 = self._get_PV0()
            E0 = H0 - PV0
        elif 'G0' in self.param_names:
            G0 = self.eos_mod.get_param_values(param_names='G0')
            TS0 = self._get_TS0()
            PV0 = self._get_PV0()
            E0 = F0 + TS0 - PV0
        else:
            assert False, print("That is not a valid ref_energy_type.")

        return E0

    def ref_helmholtz_energy(self):
        E0 = self.ref_internal_energy()
        TS0 = self._get_TS0()
        return E0 - TS0

    def ref_enthalpy(self):
        E0 = self.ref_internal_energy()
        PV0 = self._get_PV0()
        return E0 + PV0

    def ref_gibbs_energy(self):
        E0 = self.ref_internal_energy()
        TS0 = self._get_TS0()
        PV0 = self._get_PV0()
        return E0 + PV0 - TS0
#====================================================================
