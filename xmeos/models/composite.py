# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from future.utils import with_metaclass
import numpy as np
import scipy as sp
from abc import ABCMeta, abstractmethod
from scipy import integrate
import scipy.interpolate as interpolate

from . import core
from . import compress
from . import thermal
from . import gamma

__all__ = ['MieGruneisenEos']

# class RTPolyEos(with_metaclass(ABCMeta, core.Eos)):
# class RTPressEos(with_metaclass(ABCMeta, core.Eos)):
# class CompressPolyEos(with_metaclass(ABCMeta, core.Eos)):

#====================================================================
class MieGruneisenEos(with_metaclass(ABCMeta, core.Eos)):
    _kind_thermal_opts = ['Debye','Einstein']
    _kind_gamma_opts = ['GammaPowLaw','GammaShiftedPowLaw','GammaFiniteStrain']
    _kind_compress_opts = ['Vinet','BirchMurn3','BirchMurn4',
                           'GenFiniteStrain','Tait']

    def __init__(self, kind_thermal='Debye', kind_gamma='GammaPowLaw',
                 kind_compress='Vinet', compress_path_const='T',
                 natom=1, model_state={}):
        self._pre_init(natom=natom)

        compress.set_calculator(self, kind_compress, self._kind_compress_opts,
                                path_const=compress_path_const)
        gamma.set_calculator(self, kind_gamma, self._kind_gamma_opts)
        thermal.set_calculator(self, kind_thermal, self._kind_thermal_opts)

        self._set_ref_state()

        self._post_init(model_state=model_state)
        pass

    def __repr__(self):
        calc_thermal = self.calculators['thermal']
        calc_compress = self.calculators['compress']
        calc_gamma = self.calculators['gamma']

        return ("ThermalEos(kind_thermal={kind_thermal}, "
                "kind_gamma={kind_gamma}, "
                "kind_compress={kind_compress}, "
                "compress_path_const={compress_path_const}, "
                "natom={natom}, "
                "model_state={model_state}, "
                ")"
                .format(kind_thermal=repr(calc_thermal.name),
                        kind_gamma=repr(calc_gamma.name),
                        kind_compress=repr(calc_compress.name),
                        compress_path_const=repr(calc_compress.path_const),
                        natom=repr(self.natom),
                        model_state=self.model_state
                        )
                )

    def _set_ref_state(self):
        compress_calc = self.calculators['compress']
        compress_path_const = compress_calc.path_const

        V0, K0 = compress_calc.get_param_defaults(['V0','K0'])
        # redundant T0 declaration
        T0_scale = 300
        Cv_scale = 3*self.natom*core.CONSTS['kboltz']
        S0_scale = Cv_scale*T0_scale
        energy_scale = np.round(V0*K0/core.CONSTS['PV_ratio'],decimals=2)

        if   compress_path_const=='T':
            param_ref_names = ['T0', 'F0', 'S0']
            param_ref_units = ['K', 'eV', 'eV/K']
            param_ref_defaults = [T0_scale, 0.0, S0_scale]
            param_ref_scales = [T0_scale, energy_scale, S0_scale]

        elif compress_path_const=='S':
            param_ref_names = ['T0', 'E0', 'S0']
            param_ref_units = ['K', 'eV', 'eV/K']
            param_ref_defaults = [300, 0.0, S0_scale]
            param_ref_scales = [300, energy_scale, S0_scale]

        elif compress_path_const=='0K':
            param_ref_names = []
            param_ref_units = []
            param_ref_defaults = []
            param_ref_scales = []

        else:
            raise NotImplementedError(
                'path_const '+path_const+' is not valid for CompressEos.')

        self._param_ref_names = param_ref_names
        self._param_ref_units = param_ref_units
        self._param_ref_defaults = param_ref_defaults
        self._param_ref_scales = param_ref_scales
        pass

    def _calc_theta(self, V_a):
        gamma_calc = self.calculators['gamma']

        theta0, = self.get_param_values(['theta0'])
        theta_a = gamma_calc._calc_temp(V_a, T0=theta0)
        return theta_a

    def ref_temp_path(self, V_a):
        T0, theta0 = self.get_param_values(param_names=['T0','theta0'])

        gamma_calc = self.calculators['gamma']
        compress_calc = self.calculators['compress']
        compress_path_const = compress_calc.path_const

        if   compress_path_const=='T':
            Tref_path = T0
            theta_ref = theta0

        elif compress_path_const=='S':
            Tref_path = gamma_calc._calc_temp(V_a)
            theta_ref = self._calc_theta(V_a)

        elif compress_path_const=='0K':
            Tref_path = 0
            theta_ref = theta0

        else:
            raise NotImplementedError(
                'path_const '+path_const+' is not valid for CompressEos.')

        Tref_path, V_a = core.fill_array(Tref_path, V_a)
        Tref_path, theta_ref = core.fill_array(Tref_path, theta_ref)

        return Tref_path, theta_ref

    def thermal_energy(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)
        thermal_calc = self.calculators['thermal']

        Tref_path, theta_ref = self.ref_temp_path(V_a)

        # theta_a = self._calc_theta(V_a)
        # assert(np.abs(theta_a-theta_ref)<1e-3),'Doh!'

        # E_therm_a = thermal_calc._calc_energy(T_a, theta=theta_a,
        #                                       theta0=theta_ref, T0=Tref_path)
        E_therm_a = thermal_calc._calc_energy(T_a, theta=theta_ref,
                                              Tref=Tref_path)
        return E_therm_a

    def heat_capacity(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)
        thermal_calc = self.calculators['thermal']
        # theta_a = self._calc_theta(V_a)
        Tref_path, theta_ref = self.ref_temp_path(V_a)

        heat_capacity_a = thermal_calc._calc_heat_capacity(T_a, theta=theta_ref)
        return heat_capacity_a

    def press(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)
        gamma_calc = self.calculators['gamma']
        compress_calc = self.calculators['compress']

        PV_ratio, = core.get_consts(['PV_ratio'])

        P_ref_a = compress_calc._calc_press(V_a)
        gamma_a = gamma_calc._calc_gamma(V_a)

        E_therm_a = self.thermal_energy(V_a, T_a)
        P_therm_a = PV_ratio*gamma_a/V_a*E_therm_a

        press_a = P_ref_a + P_therm_a
        return press_a

    def entropy(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)
        S0, = self.get_param_values(param_names=['S0'])
        thermal_calc = self.calculators['thermal']

        Tref_path, theta_ref = self.ref_temp_path(V_a)

        theta_a = self._calc_theta(V_a)
        entropy_a = S0 + thermal_calc._calc_entropy(
            T_a, theta=theta_a, theta_ref=theta_ref, Tref=Tref_path)
        # entropy_a = S0 + thermal_calc._calc_entropy(
        #     T_a, theta=theta_ref, theta_ref=theta_ref, Tref=Tref_path)

        return entropy_a

    def helmholtz_energy(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)

        E_a = self.internal_energy(V_a, T_a)
        S_a = self.entropy(V_a, T_a)
        F_a = E_a - T_a*S_a

        return F_a

    def internal_energy(self, V_a, T_a):
        V_a, T_a = core.fill_array(V_a, T_a)
        compress_calc = self.calculators['compress']
        compress_path_const = compress_calc.path_const

        T0, = self.get_param_values(param_names=['T0'])
        Eth = self.thermal_energy(V_a, T_a)

        if   compress_path_const=='T':
            F0_ref = compress_calc._calc_energy(V_a)
            S0_ref = self.entropy(V_a, T0)
            E0_ref = F0_ref + T0*S0_ref

        elif (compress_path_const=='S')|(compress_path_const=='0K'):
            E0_ref = compress_calc._calc_energy(V_a)

        else:
            raise NotImplementedError(
                'path_const '+path_const+' is not valid for CompressEos.')

        E_a = E0_ref + Eth

        return E_a
#====================================================================
    # def entropy(self, V_a, T_a):
    #     S0, = calc.get_param_defaults(['S0'])
    #     thermal_calc = self.calculators['thermal']
    #     entropy_a = thermal_calc.entropy(T_a)
    #     return entropy_a

    # def helmholtz_energy(self, V_a, T_a):
    #     pass

    # def internal_energy(self, V_a, T_a):
    #     calculator = self.calculators['thermal']
    #     energy_a =  calculator._calc_energy(T_a)
    #     return energy_a

    # def gamma(self, V_a):
    #     pass

    # def adiabat_temp(self, Tpot, V_a):
    #     pass
#====================================================================
    # def bulk_mod(self, V_a, T_a):
    #     calculator = self.calculators['thermal']
    #     energy_a =  calculator._calc_energy(T_a)
    #     return energy_a
#====================================================================


# class CompressedThermalEnergyCalc(with_metaclass(ABCMeta, core.Calculator)):
#     """
#     Abstract Equation of State class for a reference Thermal Energy Path
#
#     Path can either be isothermal (T=const) or adiabatic (S=const)
#
#     For this restricted path, thermodyn properties depend only on volume
#
#     """
#
#     def __init__(self, eos_mod):
#         self._eos_mod = eos_mod
#         self._init_params()
#         self._required_calculators = None
#         pass
#
#     ####################
#     # Required Methods #
#     ####################
#     @abstractmethod
#     def _init_params(self):
#         """Initialize list of calculator parameter names."""
#         pass
#
#     @abstractmethod
#     def _init_required_calculators(self):
#         """Initialize list of other required calculators."""
#         pass
#
#     @abstractmethod
#     def _calc_heat_capacity(self, V_a, T_a):
#         """Returns heat capacity as a function of temperature."""
#         pass
#
#     @abstractmethod
#     def _calc_energy(self, V_a, T_a):
#         """Returns thermal energy as a function of temperature."""
#         pass
#====================================================================


#====================================================================
# Base Class
#====================================================================
# class CompositeMod(EosMod):
#     """
#     Abstract Equation of State class for Full Model (combines all EOS terms)
#     """
#     __metaclass__ = ABCMeta
#
#     # Standard methods must be overridden (as needed) by implimentation model
#     def press( self, V_a, T_a, eos_d ):
#         """Returns Total Press."""
#         raise NotImplementedError("'press' function not implimented for this model")
#
#     def energy( self, V_a, T_a, eos_d ):
#         """Returns Toal Energy."""
#         raise NotImplementedError("'energy' function not implimented for this model")
#
#     def therm_exp( self, V_a, T_a, eos_d ):
#         TOL = 1e-4
#         V_a, T_a = fill_array( V_a, T_a )
#
#         dlogV = 1e-4
#         S_a = self.entropy( V_a, T_a, eos_d )
#         KT_a = self.bulk_modulus( V_a, T_a, eos_d )
#
#         S_hi_a = self.entropy( np.exp(dlogV)*V_a, T_a, eos_d )
#         S_lo_a = self.entropy( np.exp(-dlogV)*V_a, T_a, eos_d )
#
#         dSdlogV_a = (S_hi_a-S_lo_a) / (2*dlogV)
#         alpha_a = dSdlogV_a/(KT_a*V_a*eos_d['const_d']['PV_ratio'])
#
#         return alpha_a
#
#
#     def bulk_mod( self, V_a, T_a, eos_d ):
#         """Returns Total Bulk Modulus."""
#         raise NotImplementedError("'bulk_mod' function not implimented for this model")
# #====================================================================
#
# #====================================================================
# # Implementations
# #====================================================================
# class ThermalPressMod(CompositeMod):
#     # Need to impliment get_param_scale_sub
#
#     def press( self, V_a, T_a, eos_d ):
#         """Returns Press variation along compression curve."""
#         V_a, T_a = fill_array( V_a, T_a )
#         # compress_path_mod, thermal_mod = Control.get_modtypes( ['CompressPathMod', 'ThermalMod'],
#         #                                        eos_d )
#         # press_a = np.squeeze( compress_path_mod.press( V_a, eos_d )
#         #                      + thermal_mod.press( V_a, T_a, eos_d ) )
#         # return press_a
#
#         TOL = 1e-4
#         PV_ratio, = Control.get_consts( ['PV_ratio'], eos_d )
#
#         F_mod_a = self.free_energy(V_a,T_a,eos_d)
#         F_hi_mod_a = self.free_energy(V_a*(1.0+TOL),T_a,eos_d)
#         P_mod_a = -PV_ratio*(F_hi_mod_a-F_mod_a)/(V_a*TOL)
#         return P_mod_a
#
#     def energy( self, V_a, T_a, eos_d ):
#         """Returns Internal Energy."""
#         V_a, T_a = fill_array( V_a, T_a )
#         compress_path_mod, thermal_mod = Control.get_modtypes( ['CompressPathMod', 'ThermalMod'],
#                                                eos_d )
#
#         energy_compress_a = compress_path_mod.energy( V_a, eos_d )
#         if compress_path_mod.path_const=='T':
#             """
#             Convert free energy to internal energy
#             """
#             free_energy_compress_a = energy_compress_a
#             T0, = Control.get_params(['T0'],eos_d)
#
#             # wrong
#             # S_a = thermal_mod.entropy( V_a, T_a, eos_d )
#             # dF_a = thermal_mod.calc_free_energy( V_a, T_a, eos_d )
#             # Ftot_a = free_energy_compress_a + dF_a
#             # energy_a = Ftot_a+T_a*S_a
#
#
#             # correct
#             # energy_a = thermal_mod.calc_energy_adiabat_ref(V_a,eos_d)\
#             #     +thermal_mod.calc_energy(V_a,T_a,eos_d)
#
#             S_T0_a = thermal_mod.entropy( V_a, T0, eos_d )
#             energy_T0_a = free_energy_compress_a + T0*S_T0_a
#             energy_S0_a = energy_T0_a - thermal_mod.calc_energy(V_a,T0,eos_d)
#
#             energy_a = energy_S0_a + thermal_mod.calc_energy(V_a,T_a,eos_d)
#
#
#
#         else:
#             energy_a = np.squeeze( energy_compress_a
#                                   + thermal_mod.energy( V_a, T_a, eos_d ) )
#
#         return energy_a
#
#     def bulk_modulus( self, V_a, T_a, eos_d ):
#         TOL = 1e-4
#
#         P_lo_a = self.press( V_a*(1.0-TOL/2), T_a, eos_d )
#         P_hi_a = self.press( V_a*(1.0+TOL/2), T_a, eos_d )
#         K_a = -V_a*(P_hi_a-P_lo_a)/(V_a*TOL)
#
#         return K_a
#
#     def dPdT( self, V_a, T_a, eos_d, Tscl=1000.0 ):
#         TOL = 1e-4
#
#         P_lo_a = self.press( V_a, T_a*(1.0-TOL/2), eos_d )
#         P_hi_a = self.press( V_a, T_a*(1.0+TOL/2), eos_d )
#         dPdT_a = (P_hi_a-P_lo_a)/(T_a*TOL)*Tscl
#
#         # # By a maxwell relation dPdT_V = dSdV_T
#         # S_lo_a = self.entropy( V_a*(1.0-TOL/2), T_a, eos_d )
#         # S_hi_a = self.entropy( V_a*(1.0+TOL/2), T_a, eos_d )
#         # dSdV_a = (S_hi_a-S_lo_a)/(V_a*TOL)
#         # dPdT_a = dSdV_a*Tscl
#
#         return dPdT_a
#
#     def free_energy( self, V_a, T_a, eos_d ):
#         """Returns Free Energy."""
#         V_a, T_a = fill_array( V_a, T_a )
#         compress_path_mod, thermal_mod = Control.get_modtypes( ['CompressPathMod', 'ThermalMod'],
#                                                eos_d )
#         T0,S0 = Control.get_params(['T0','S0'],eos_d)
#
#         energy_compress_a = compress_path_mod.energy( V_a, eos_d )
#         if compress_path_mod.path_const=='T':
#             free_energy_compress_a = energy_compress_a
#
#             S_T0_a = thermal_mod.entropy( V_a, T0, eos_d )
#             # wrong
#             # S_a = thermal_mod.entropy( V_a, T_a, eos_d )
#             # dF_a = thermal_mod.calc_free_energy( V_a, T_a, eos_d )
#             # Ftot_a = free_energy_compress_a + dF_a
#             # energy_a = Ftot_a+T_a*S_a
#
#
#             # correct
#             # energy_a = thermal_mod.calc_energy_adiabat_ref(V_a,eos_d)\
#             #     +thermal_mod.calc_energy(V_a,T_a,eos_d)
#
#             energy_T0_a = free_energy_compress_a + T0*S_T0_a
#             energy_S0_a = energy_T0_a - thermal_mod.calc_energy(V_a,T0,eos_d)
#
#         else:
#             energy_S0_a = energy_compress_a
#
#         Tref_a = thermal_mod.calc_temp_path(V_a,eos_d)
#
#         free_energy_S0_a = energy_S0_a - Tref_a*S0
#
#         # Fix bug for nonzero ref entropy values, need to subtract off reference
#         dF_a = thermal_mod.calc_free_energy( V_a, T_a, eos_d ) \
#             - thermal_mod.calc_free_energy( V_a, Tref_a, eos_d )
#         free_energy_a = free_energy_S0_a + dF_a
#
#         return free_energy_a
#
#     def entropy( self, V_a, T_a, eos_d ):
#         """Returns Free Energy."""
#         V_a, T_a = fill_array( V_a, T_a )
#         thermal_mod, = Control.get_modtypes( ['ThermalMod'], eos_d )
#         S_a = np.squeeze( thermal_mod.entropy( V_a, T_a, eos_d ) )
#
#         return S_a
#
#     def heat_capacity( self, V_a, T_a, eos_d ):
#         """Returns Free Energy."""
#         V_a, T_a = fill_array( V_a, T_a )
#         thermal_mod, = Control.get_modtypes( ['ThermalMod'], eos_d )
#         Cv_a = np.squeeze( thermal_mod.heat_capacity( V_a, T_a, eos_d ) )
#
#         return Cv_a
# #====================================================================
# class MieGrunPtherm(CompositeMod):
#     # Need to impliment get_param_scale_sub
#
#     def dPdT( self, V_a, T_a, eos_d ):
#         V_a, T_a = fill_array( V_a, T_a )
#
#         poly_blogcoef_a = Control.get_array_params( 'blogcoef', eos_d )
#         dPdT_a = eos_d['Ptherm_f'] (V_a)
#         return dPdT_a
#
#     def press( self, V_a, T_a, eos_d ):
#         V_a, T_a = fill_array( V_a, T_a )
#
#         Tref = eos_d['Tref']
#         Pref_a = eos_d['Pref_f'] (V_a)
#         dPdT_a = self.dPdT( V_a, T_a, eos_d )
#         dPtherm_a = (T_a-Tref)*dPdT_a
#         P_a = Pref_a + dPtherm_a
#
#         return P_a
#
#     def energy( self, V_a, T_a, eos_d ):
#         V_a, T_a = fill_array( V_a, T_a )
#
#         Tref = eos_d['Tref']
#         Eref_a = eos_d['Eref_f'] (V_a)
#         dPtherm_a = (T_a-Tref)*eos_d['Ptherm_f'] (V_a)
#
#         gamma_a = eos_d['gamma_f'] (V_a)
#
#         dPdT_a = self.dPdT( V_a, T_a, eos_d )
#         dPtherm_a = (T_a-Tref)*dPdT_a
#         dEtherm_a = dPtherm_a/(gamma_a/V_a)/eos_d['const_d']['PV_ratio']
#
#         E_a = Eref_a + dEtherm_a
#
#         return E_a
#
#
#     def bulk_modulus( self, V_a, T_a, eos_d ):
#         TOL = 1e-4
#
#         P_lo_a = self.press( V_a*(1.0-TOL/2), T_a, eos_d )
#         P_hi_a = self.press( V_a*(1.0+TOL/2), T_a, eos_d )
#         K_a = -V_a*(P_hi_a-P_lo_a)/(V_a*TOL)
#
#         return K_a
#
#
#     def dPdT( self, V_a, T_a, eos_d, Tscl=1000.0 ):
#         TOL = 1e-4
#
#         P_lo_a = self.press( V_a, T_a*(1.0-TOL/2), eos_d )
#         P_hi_a = self.press( V_a, T_a*(1.0+TOL/2), eos_d )
#         dPdT_a = (P_hi_a-P_lo_a)/(T_a*TOL)*Tscl
#
#         # # By a maxwell relation dPdT_V = dSdV_T
#         # S_lo_a = self.entropy( V_a*(1.0-TOL/2), T_a, eos_d )
#         # S_hi_a = self.entropy( V_a*(1.0+TOL/2), T_a, eos_d )
#         # dSdV_a = (S_hi_a-S_lo_a)/(V_a*TOL)
#         # dPdT_a = dSdV_a*Tscl
#
#         return dPdT_a
#
#     def free_energy( self, V_a, T_a, eos_d ):
#         """Returns Free Energy."""
#         V_a, T_a = fill_array( V_a, T_a )
#         compress_path_mod, thermal_mod = Control.get_modtypes( ['CompressPathMod', 'ThermalMod'],
#                                                eos_d )
#         T0,S0 = Control.get_params(['T0','S0'],eos_d)
#
#         energy_compress_a = compress_path_mod.energy( V_a, eos_d )
#         if compress_path_mod.path_const=='T':
#             free_energy_compress_a = energy_compress_a
#
#             S_T0_a = thermal_mod.entropy( V_a, T0, eos_d )
#             # wrong
#             # S_a = thermal_mod.entropy( V_a, T_a, eos_d )
#             # dF_a = thermal_mod.calc_free_energy( V_a, T_a, eos_d )
#             # Ftot_a = free_energy_compress_a + dF_a
#             # energy_a = Ftot_a+T_a*S_a
#
#
#             # correct
#             # energy_a = thermal_mod.calc_energy_adiabat_ref(V_a,eos_d)\
#             #     +thermal_mod.calc_energy(V_a,T_a,eos_d)
#
#             energy_T0_a = free_energy_compress_a + T0*S_T0_a
#             energy_S0_a = energy_T0_a - thermal_mod.calc_energy(V_a,T0,eos_d)
#
#         else:
#             energy_S0_a = energy_compress_a
#
#         Tref_a = thermal_mod.calc_temp_path(V_a,eos_d)
#
#         free_energy_S0_a = energy_S0_a - Tref_a*S0
#
#         dF_a = thermal_mod.calc_free_energy( V_a, T_a, eos_d )
#         free_energy_a = free_energy_S0_a + dF_a
#
#         return free_energy_a
#
#     def entropy( self, V_a, T_a, eos_d ):
#         """Returns Free Energy."""
#         V_a, T_a = fill_array( V_a, T_a )
#         thermal_mod, = Control.get_modtypes( ['ThermalMod'], eos_d )
#         S_a = np.squeeze( thermal_mod.entropy( V_a, T_a, eos_d ) )
#
#         return S_a
#
#     def heat_capacity( self, V_a, T_a, eos_d ):
#         """Returns Free Energy."""
#         V_a, T_a = fill_array( V_a, T_a )
#         thermal_mod, = Control.get_modtypes( ['ThermalMod'], eos_d )
#         Cv_a = np.squeeze( thermal_mod.heat_capacity( V_a, T_a, eos_d ) )
#
#         return Cv_a
# #====================================================================
#
# #====================================================================
# class RosenfeldTaranzonaShiftedAdiabat(CompressPath):
#     def get_param_scale_sub( self, eos_d):
#         """Return scale values for each parameter"""
#         V0, K0, KP0 = core.get_params( ['V0','K0','KP0'], eos_d )
#         PV_ratio, = core.get_consts( ['PV_ratio'], eos_d )
#
#         paramkey_a = np.array(['V0','K0','KP0','E0'])
#         scale_a = np.array([V0,K0,KP0,K0*V0/PV_ratio])
#
#         return scale_a, paramkey_a
#
#     def _calc_press( self, V_a, eos_d ):
#         PV_ratio, = core.get_consts( ['PV_ratio'], eos_d )
#         fac = 1e-3
#         Vhi_a = V_a*(1.0 + 0.5*fac)
#         Vlo_a = V_a*(1.0 - 0.5*fac)
#
#         dV_a = Vhi_a-Vlo_a
#
#
#         E0S_hi_a = self._calc_energy(Vhi_a, eos_d)
#         E0S_lo_a = self._calc_energy(Vlo_a, eos_d)
#
#         P0S_a = -PV_ratio*(E0S_hi_a - E0S_lo_a)/dV_a
#         return P0S_a
#
#     def _calc_energy( self, V_a, eos_d ):
#         V0, T0, mexp  = core.get_params( ['V0','T0','mexp'], eos_d )
#         kB, = core.get_consts( ['kboltz'], eos_d )
#
#         poly_blogcoef_a = core.get_array_params( 'blogcoef', eos_d )
#
#
#         compress_path_mod, thermal_mod, gamma_mod = \
#             core.get_modtypes( ['CompressPath', 'ThermalMod', 'GammaMod'],
#                                  eos_d )
#
#         free_energy_isotherm_a = compress_path_mod.energy(V_a,eos_d)
#
#         T0S_a = gamma_mod.temp(V_a,T0,eos_d)
#
#
#         bV_a = np.polyval(poly_blogcoef_a,np.log(V_a/V0))
#
#         dS_a = -mexp/(mexp-1)*bV_a/T0*((T0S_a/T0)**(mexp-1)-1)\
#             -3./2*kB*np.log(T0S_a/T0)
#
#
#         energy_isotherm_a = free_energy_isotherm_a + T0*dS_a
#         E0S_a = energy_isotherm_a + bV_a*((T0S_a/T0)**mexp-1)\
#             +3./2*kB*(T0S_a-T0)
#
#         return E0S_a
# #====================================================================

# class _CompressedDebye(CompressedThermalEnergyCalc):
#     """
#     Implimentation copied from Burnman.
#
#     """
#
#     def __init__(self, eos_mod, path_const='P', level_const=0.0):
#
#     def _init_required_calculators(self):
#         """Initialize list of other required calculators."""
#
#         self._eos_mod.
#
#         self._required_calculators = None
#
#         pass
#     def _init_params(self, theta_param=None):
#         """Initialize list of calculator parameter names."""
#
#         theta0, Cvmax = 100, 150
#         E0_scale = np.round(V0*KP0/core.CONSTS['PV_ratio'],decimals=2)
#         self._param_names = ['V0','K0','KP0','E0']
#         self._param_units = ['ang^3','GPa','1','eV']
#         self._param_defaults = [V0,K0,KP0,0]
#         self._param_scales = [V0,K0,KP0,E0_scale]
#
#         pass
#
#     def _calc_heat_capacity(self, T_a, Theta=None, Cvmax=None):
#         """Returns heat capacity as a function of temperature."""
#
#         if Theta is None:
#             Theta, = self.eos_mod.get_param_values(param_names=['Theta'])
#
#         if Cvmax is None:
#             Cvmax, = self.eos_mod.get_param_values(param_names=['Cvmax'])
#
#         x_values = Theta/np.array(T_a)
#         Cv_values = Cvmax*_debye.debye_heat_capacity_fun(x_values)
#         return Cv_values
#
#     def _calc_energy(self, T_a, Theta=None, Cvmax=None):
#         """Returns heat capacity as a function of temperature."""
#
#         if Theta is None:
#             Theta, = self.eos_mod.get_param_values(param_names=['Theta'])
#
#         if Cvmax is None:
#             Cvmax, = self.eos_mod.get_param_values(param_names=['Cvmax'])
#
#         x_values = Theta/np.array(T_a)
#         Cv_values = Cvmax*_debye.debye_heat_capacity_fun(x_values)
#         return Cv_values
#====================================================================
