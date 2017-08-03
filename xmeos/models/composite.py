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

__all__ = []

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
