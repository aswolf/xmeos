import numpy as np
from models import thermal
from models import core
from models import compress
from models import composite

import pytest
import matplotlib.pyplot as plt
import matplotlib as mpl
from abc import ABCMeta, abstractmethod
import copy
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
# class TestRosenfeldTaranzonaPoly(BaseTestThermalMod):
#     def load_thermal_mod(self, eos_d):
#         thermal_mod = thermal.RosenfeldTaranzonaPoly()
#         core.set_modtypes( ['ThermalMod'], [thermal_mod], eos_d )
#
#         pass
#
#     def load_compress_path_mod(self, eos_d):
#         T0, = core.get_params(['T0'],eos_d)
#         compress_path_mod = compress.Vinet(path_const='T',level_const=T0,
#                                          supress_energy=True,
#                                          supress_press=True)
#         # NOTE that supress press is included to impliment all terms according
#         # to Spera2011
#         #   (but the current implimentation actually uses the compress path
#         #   pressure unscaled)
#         core.set_modtypes( ['CompressPathMod'], [compress_path_mod], eos_d )
#
#         pass
#
#     def load_eos_mod(self, eos_d):
#
#         self.load_thermal_mod(eos_d)
#         self.load_compress_path_mod(eos_d)
#
#         full_mod = composite.ThermalPressMod()
#         core.set_modtypes( ['FullMod'], [full_mod], eos_d )
#
#         pass
#
#     def init_params(self,eos_d):
#
#         core.set_consts( [], [], eos_d )
#
#         # Set model parameter values
#         mexp = 3.0/5
#         T0 = 4000.0
#         V0_ccperg = 0.408031 # cc/g
#         K0 = 13.6262
#         KP0= 7.66573
#         E0 = 0.0
#         # nfac = 5.0
#         # mass = (24.31+28.09+3*16.0) # g/(mol atom)
#         # V0 = V0_ccperg
#
#         # NOTE that units are all per atom
#         # requires conversion from values reported in Spera2011
#         lognfac = 0.0
#         mass = (24.31+28.09+3*16.0)/5.0 # g/(mol atom)
#         Vconv_fac = mass*eos_d['const_d']['ang3percc']/eos_d['const_d']['Nmol']
#         V0 = V0_ccperg*Vconv_fac
#
#
#         param_key_a = ['mexp','lognfac','T0','V0','K0','KP0','E0','mass']
#         param_val_a = np.array([mexp,lognfac,T0,V0,K0,KP0,E0,mass])
#         core.set_params( param_key_a, param_val_a, eos_d )
#
#         # Set parameter values from Spera et al. (2011)
#         # for MgSiO3 melt using  (Oganov potential)
#
#         # Must convert energy units from kJ/g to eV/atom
#         energy_conv_fac = mass/eos_d['const_d']['kJ_molpereV']
#         core.set_consts( ['energy_conv_fac'], [energy_conv_fac],
#                                   eos_d )
#
#         # change coefficients to relative
#         # acoef_a = energy_conv_fac*\
#         #     np.array([127.116,-3503.98,20724.4,-60212.0,86060.5,-48520.4])
#         # bcoef_a = energy_conv_fac*\
#         #     np.array([-0.371466,7.09542,-45.7362,139.020,-201.487,112.513])
#         Vconv_a = (1.0/Vconv_fac)**np.arange(6)
#
#
#         unit_conv = energy_conv_fac*Vconv_a
#
#         # Reported vol-dependent polynomial coefficients for a and b
#         #  in Spera2011
#         acoef_unscl_a = np.array([127.116,-3503.98,20724.4,-60212.0,\
#                                   86060.5,-48520.4])
#         bcoef_unscl_a = np.array([-0.371466,7.09542,-45.7362,139.020,\
#                                   -201.487,112.513])
#
#         # Convert units and transfer to normalized version of RT model
#         acoef_a = unit_conv*(acoef_unscl_a+bcoef_unscl_a*T0**mexp)
#         bcoef_a = unit_conv*bcoef_unscl_a*T0**mexp
#
#         core.set_array_params( 'acoef', acoef_a, eos_d )
#         core.set_array_params( 'bcoef', bcoef_a, eos_d )
#
#         self.load_eos_mod( eos_d )
#
#         #     from IPython import embed; embed(); import ipdb; ipdb.set_trace()
#         return eos_d
#
#     def test_RT_potenergy_curves_Spera2011(self):
#         Nsamp = 101
#         eos_d = self.init_params({})
#
#         param_d = eos_d['param_d']
#         Vgrid_a = np.linspace(0.5,1.1,Nsamp)*param_d['V0']
#         Tgrid_a = np.linspace(100.0**(5./3),180.0**(5./3),11)
#
#         full_mod = eos_d['modtype_d']['FullMod']
#         thermal_mod = eos_d['modtype_d']['ThermalMod']
#
#         energy_conv_fac, = core.get_consts(['energy_conv_fac'],eos_d)
#
#         potenergy_mod_a = []
#
#         for iV in Vgrid_a:
#             ipotenergy_a = thermal_mod.calc_energy_pot(iV,Tgrid_a,eos_d)
#             potenergy_mod_a.append(ipotenergy_a)
#
#         # energy_mod_a = np.array( energy_mod_a )
#         potenergy_mod_a = np.array( potenergy_mod_a )
#
#         plt.ion()
#         plt.figure()
#         plt.plot(Tgrid_a**(3./5), potenergy_mod_a.T/energy_conv_fac,'-')
#         plt.xlim(100,180)
#         plt.ylim(-102,-95)
#
#         print 'Compare this plot with Spera2011 Fig 1b (Oganov potential):'
#         print 'Do the figures agree (y/n or k for keyboard)?'
#         s = raw_input('--> ')
#         if s=='k':
#             from IPython import embed; embed(); import ipdb; ipdb.set_trace()
#
#         assert s=='y', 'Figure must match published figure'
#         pass
#
#     def test_energy_curves_Spera2011(self):
#         Nsamp = 101
#         eos_d = self.init_params({})
#
#         param_d = eos_d['param_d']
#         Vgrid_a = np.linspace(0.4,1.1,Nsamp)*param_d['V0']
#         Tgrid_a = np.array([2500,3000,3500,4000,4500,5000])
#
#         full_mod = eos_d['modtype_d']['FullMod']
#
#         energy_conv_fac, = core.get_consts(['energy_conv_fac'],eos_d)
#
#         energy_mod_a = []
#         press_mod_a = []
#
#         for iT in Tgrid_a:
#             ienergy_a = full_mod.energy(Vgrid_a,iT,eos_d)
#             ipress_a = full_mod.press(Vgrid_a,iT,eos_d)
#             energy_mod_a.append(ienergy_a)
#             press_mod_a.append(ipress_a)
#
#         # energy_mod_a = np.array( energy_mod_a )
#         energy_mod_a = np.array( energy_mod_a )
#         press_mod_a = np.array( press_mod_a )
#
#         plt.ion()
#         plt.figure()
#         plt.plot(press_mod_a.T, energy_mod_a.T/energy_conv_fac,'-')
#         plt.legend(Tgrid_a,loc='lower right')
#         plt.xlim(-5,165)
#         plt.ylim(-100.5,-92)
#
#         # from IPython import embed; embed(); import ipdb; ipdb.set_trace()
#
#         print 'Compare this plot with Spera2011 Fig 2b (Oganov potential):'
#         print 'Do the figures agree (y/n or k for keyboard)?'
#         s = raw_input('--> ')
#         if s=='k':
#             from IPython import embed; embed(); import ipdb; ipdb.set_trace()
#
#         assert s=='y', 'Figure must match published figure'
#         pass
#
#     def test_heat_capacity_curves_Spera2011(self):
#         Nsamp = 101
#         eos_d = self.init_params({})
#
#         param_d = eos_d['param_d']
#         Vgrid_a = np.linspace(0.4,1.2,Nsamp)*param_d['V0']
#         Tgrid_a = np.array([2500,3000,3500,4000,4500,5000])
#
#         full_mod = eos_d['modtype_d']['FullMod']
#         thermal_mod = eos_d['modtype_d']['ThermalMod']
#
#         heat_capacity_mod_a = []
#         energy_conv_fac, = core.get_consts(['energy_conv_fac'],eos_d)
#
#         energy_mod_a = []
#         press_mod_a = []
#
#         for iT in Tgrid_a:
#             iheat_capacity_a = thermal_mod.heat_capacity(Vgrid_a,iT,eos_d)
#             ienergy_a = full_mod.energy(Vgrid_a,iT,eos_d)
#             ipress_a = full_mod.press(Vgrid_a,iT,eos_d)
#
#             heat_capacity_mod_a.append(iheat_capacity_a)
#             energy_mod_a.append(ienergy_a)
#             press_mod_a.append(ipress_a)
#
#
#         # energy_mod_a = np.array( energy_mod_a )
#         heat_capacity_mod_a = np.array( heat_capacity_mod_a )
#         energy_mod_a = np.array( energy_mod_a )
#         press_mod_a = np.array( press_mod_a )
#
#         plt.ion()
#         plt.figure()
#         plt.plot(press_mod_a.T,1e3*heat_capacity_mod_a.T/energy_conv_fac,'-')
#         plt.legend(Tgrid_a,loc='lower right')
#         # plt.ylim(1.2,1.9)
#         plt.xlim(-5,240)
#
#         print 'Compare this plot with Spera2011 Fig 2b (Oganov potential):'
#         print 'Do the figures agree (y/n or k for keyboard)?'
#         s = raw_input('--> ')
#         if s=='k':
#             from IPython import embed; embed(); import ipdb; ipdb.set_trace()
#
#         assert s=='y', 'Figure must match published figure'
#         pass
#====================================================================

