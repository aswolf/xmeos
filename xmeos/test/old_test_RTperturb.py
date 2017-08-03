import numpy as np
from models import compress
from models import thermal
from models import composite
from models import core

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
class BaseTestThermalPathMod(object):
    @abstractmethod
    def load_thermal_path_mod(self, eos_d):
        assert False, 'must implement load_thermal_path_mod()'

    @abstractmethod
    def init_params(self,eos_d):
        assert False, 'must implement init_params()'
        return eos_d

    def test_heat_capacity(self):
        Nsamp = 10001
        eos_d = self.init_params({})

        param_d = eos_d['param_d']
        Tmod_a = np.linspace(.7,1.3,Nsamp)*param_d['T0']
        dT = Tmod_a[1] - Tmod_a[0]

        # print eos_d['modtype_d']
        thermal_path_mod = eos_d['modtype_d']['ThermalPathMod']

        heat_capacity_a = thermal_path_mod.heat_capacity(Tmod_a,eos_d)
        energy_a = thermal_path_mod.energy(Tmod_a,eos_d)

        heat_capacity_num_a = np.gradient(energy_a,dT)

        E_range = np.max(energy_a)-np.min(energy_a)
        T_range = Tmod_a[-1]-Tmod_a[0]
        Cv_scl = E_range/T_range
        # Cv_range = np.max(heat_capacity_a)-np.min(heat_capacity_a)

        Cv_diff_a = heat_capacity_num_a-heat_capacity_a
        # Cverr =  np.max(np.abs(Cv_diff_a/Cv_range))
        Cverr =  np.max(np.abs(Cv_diff_a/Cv_scl))
        CVTOL = 1.0/Nsamp

        # print self
        # print PTOL*Prange


        # def plot_press_mismatch(Tmod_a,press_a,press_num_a):
        #     plt.figure()
        #     plt.ion()
        #     plt.clf()
        #     plt.plot(Tmod_a,press_num_a,'bx',Tmod_a,press_a,'r-')
        #     from IPython import embed; embed(); import ipdb; ipdb.set_trace()

        # plot_press_mismatch(Tmod_a,press_a,press_num_a)

        assert np.abs(Cverr) < CVTOL, '(Cv error)/Cv_scl, ' + np.str(Cverr) + \
            ', must be less than CVTOL, ' + np.str(CVTOL)
#====================================================================
class BaseTestThermalMod(object):
    @abstractmethod
    def load_thermal_mod(self, eos_d):
        assert False, 'must implement load_thermal_mod()'

    @abstractmethod
    def init_params(self,eos_d):
        assert False, 'must implement init_params()'
        return eos_d

    def test_heat_capacity_isochore(self):
        Nsamp = 10001
        eos_d = self.init_params({})

        param_d = eos_d['param_d']
        Viso = 0.7*param_d['V0']
        Tmod_a = np.linspace(.7,1.3,Nsamp)*param_d['T0']
        dT = Tmod_a[1] - Tmod_a[0]

        # print eos_d['modtype_d']
        thermal_mod = eos_d['modtype_d']['ThermalMod']

        heat_capacity_a = thermal_mod.heat_capacity(Viso,Tmod_a,eos_d)
        energy_a = np.squeeze( thermal_mod.energy(Viso,Tmod_a,eos_d) )

        heat_capacity_num_a = np.gradient(energy_a,dT)

        E_range = np.max(energy_a)-np.min(energy_a)
        T_range = Tmod_a[-1]-Tmod_a[0]
        Cv_scl = E_range/T_range
        # Cv_range = np.max(heat_capacity_a)-np.min(heat_capacity_a)

        Cv_diff_a = heat_capacity_num_a-heat_capacity_a
        # Cverr =  np.max(np.abs(Cv_diff_a/Cv_range))
        Cverr =  np.max(np.abs(Cv_diff_a/Cv_scl))
        CVTOL = 1.0/Nsamp

        # print self
        # print PTOL*Prange


        # def plot_press_mismatch(Tmod_a,press_a,press_num_a):
        #     plt.figure()
        #     plt.ion()
        #     plt.clf()
        #     plt.plot(Tmod_a,press_num_a,'bx',Tmod_a,press_a,'r-')
        #     from IPython import embed; embed(); import ipdb; ipdb.set_trace()

        # plot_press_mismatch(Tmod_a,press_a,press_num_a)

        assert np.abs(Cverr) < CVTOL, '(Cv error)/Cv_scl, ' + np.str(Cverr) + \
            ', must be less than CVTOL, ' + np.str(CVTOL)
#====================================================================

#====================================================================
class TestRosenfeldTaranzonaPerturb(BaseTestThermalMod):
    def load_thermal_mod(self, eos_d):
        thermal_mod = models.RosenfeldTaranzonaPerturb()
        core.set_modtypes( ['ThermalMod'], [thermal_mod], eos_d )

        pass

    def load_gamma_mod(self, eos_d):
        gamma_mod = models.GammaPowLaw()
        core.set_modtypes( ['GammaMod'], [gamma_mod], eos_d )

        pass

    def load_compress_path_mod(self, eos_d):
        S0, = core.get_params(['S0'],eos_d)
        compress_path_mod = models.Vinet(path_const='S',level_const=S0,
                                         supress_energy=False,
                                         supress_press=False)
        core.set_modtypes( ['CompressPathMod'], [compress_path_mod], eos_d )

        pass

    def load_eos_mod(self, eos_d):

        self.load_compress_path_mod(eos_d)
        self.load_gamma_mod(eos_d)
        self.load_thermal_mod(eos_d)

        full_mod = models.ThermalPressMod()
        core.set_modtypes( ['FullMod'], [full_mod], eos_d )

        pass

    def init_params(self,eos_d):

        core.set_consts( [], [], eos_d )

        # EOS Parameter values initially set by Mosenfelder2009
        # Set model parameter values
        mass_avg = (24.31+28.09+3*16.0)/5.0 # g/(mol atom)
        T0 = 1673.0
        S0 = 0.0 # must adjust
        param_key_a = ['T0','S0','mass_avg']
        param_val_a = np.array([T0,S0,mass_avg])
        core.set_params( param_key_a, param_val_a, eos_d )

        V0 = (38.575*1e-5)*mass_avg/eos_d['const_d']['Nmol']/1e3*1e30 # ang^3/atom
        K0 = 20.8
        KP0= 10.2
        # KP20 = -2.86 # Not actually used!
        E0 = 0.0
        param_key_a = ['V0','K0','KP0','E0']
        param_val_a = np.array([V0,K0,KP0,E0])
        core.set_params( param_key_a, param_val_a, eos_d )

        VR = V0
        gammaR = 0.46
        qR = -1.35
        param_key_a = ['VR','gammaR','qR']
        param_val_a = np.array([VR,gammaR,qR])
        core.set_params( param_key_a, param_val_a, eos_d )

        dE0th  = +1.0
        dV0th  = -0.02
        dK0th  = +0.1
        dKP0th = -0.00
        # dE0th  = +0.4
        # dV0th  = -0.0
        # dK0th  = -0.01
        # dKP0th = -0.03
        lognfac = 0.0
        mexp = 3.0/5
        param_key_a = ['dE0th','dV0th','dK0th','dKP0th','lognfac','mexp']
        param_val_a = np.array([dE0th,dV0th,dK0th,dKP0th,lognfac,mexp])
        core.set_params( param_key_a, param_val_a, eos_d )


        # Must convert energy units from kJ/g to eV/atom
        energy_conv_fac = mass_avg/eos_d['const_d']['kJ_molpereV']
        core.set_consts( ['energy_conv_fac'], [energy_conv_fac],
                                  eos_d )


        self.load_eos_mod( eos_d )

        #     from IPython import embed; embed(); import ipdb; ipdb.set_trace()
        return eos_d

    def test_energy_curves_Spera2011(self):
        Nsamp = 101
        eos_d = self.init_params({})

        param_d = eos_d['param_d']
        Vgrid_a = np.linspace(0.4,1.1,Nsamp)*param_d['V0']
        Tgrid_a = np.array([2500,3000,3500,4000,4500,5000])

        full_mod = eos_d['modtype_d']['FullMod']

        # energy_conv_fac, = core.get_consts(['energy_conv_fac'],eos_d)

        energy_mod_a = []
        press_mod_a = []

        for iT in Tgrid_a:
            ienergy_a = full_mod.energy(Vgrid_a,iT,eos_d)
            ipress_a = full_mod.press(Vgrid_a,iT,eos_d)
            energy_mod_a.append(ienergy_a)
            press_mod_a.append(ipress_a)

        # energy_mod_a = np.array( energy_mod_a )
        energy_mod_a = np.array( energy_mod_a )
        press_mod_a = np.array( press_mod_a )

        # from IPython import embed; embed(); import ipdb; ipdb.set_trace()
        cmap=plt.get_cmap('coolwarm')
        col_a = cmap(1.0*(Tgrid_a-Tgrid_a[0])/np.ptp(Tgrid_a))[:,:3]

        plt.ion()
        plt.figure()
        [plt.plot(ipress_a, ienergy_a,'-',color=icol_a,label=iT) \
         for ipress_a,ienergy_a,icol_a,iT  in zip(press_mod_a,energy_mod_a,col_a,Tgrid_a)]
        ax = plt.axes()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1],labels[::-1],loc='upper left')
        plt.xlim(-5,165)
        ybnd = [np.min(energy_mod_a[press_mod_a<165]), np.max(energy_mod_a[press_mod_a<165])]
        plt.ylim(ybnd[0],ybnd[1])
        # plt.ylim(-100.5,-92)


        print 'Compare this plot with Spera2011 Fig 2b (Oganov potential):'
        print 'Do the figures agree (y/n or k for keyboard)?'
        s = raw_input('--> ')
        if s=='k':
            from IPython import embed; embed(); import ipdb; ipdb.set_trace()

        assert s=='y', 'Figure must match published figure'
        pass

    def test_kinetic_contribution(self):
        Nsamp = 1001
        eos_d = self.init_params({})

        eos_d['param_d']['E0'] = -21.3
        eos_d['param_d']['dE0th'] = 0.5
        V0 = eos_d['param_d']['V0']

        Vgrid_a = V0*np.arange(0.4,1.11,0.1)
        Tgrid_a = np.linspace( 2500, 5000, Nsamp)
        dT = Tgrid_a[1]-Tgrid_a[0]

        kboltz = eos_d['const_d']['kboltz']
        # Test entropy
        TOL = 1e-4

        iV = Vgrid_a[0]
        genRT_mod = models.GenRosenfeldTaranzona()
        thermal_mod = eos_d['modtype_d']['ThermalMod']
        full_mod = eos_d['modtype_d']['FullMod']

        Cvkin_a = genRT_mod.calc_heat_capacity_kin( Tgrid_a ,eos_d )

        Ekin_a = genRT_mod.calc_energy_kin( Tgrid_a ,eos_d )
        Cvkin_dE_err_a = ( Cvkin_a - np.gradient( Ekin_a, dT ) )/kboltz
        assert np.all( np.abs(Cvkin_dE_err_a[1:-1]) < TOL ), \
            'Cvkin must match numerical energy deriv'

        Skin_a = genRT_mod.calc_entropy_kin( Tgrid_a ,eos_d, Tref=eos_d['param_d']['T0'] )
        Cvkin_dS_err_a = ( Cvkin_a - Tgrid_a*np.gradient( Skin_a, dT ) )/kboltz
        assert np.all( np.abs(Cvkin_dS_err_a[1:-1]) < TOL ), \
            'Cvkin must match numerical entropy deriv'

        Fkin_a = Ekin_a-Tgrid_a*Skin_a
        Skin_dF_err_a = ( Skin_a + np.gradient( Fkin_a, dT ) )/kboltz
        assert np.all( np.abs(Skin_dF_err_a[1:-1]) < TOL ), \
            'Skin must match numerical free energy deriv'

    def test_potential_contribution(self):
        Nsamp = 1001

        eos_d = self.init_params({})

        eos_d['param_d']['E0'] = -21.3
        eos_d['param_d']['dE0th'] = 0.5
        V0 = eos_d['param_d']['V0']

        Vgrid_a = V0*np.arange(0.4,1.11,0.1)
        Tgrid_a = np.linspace( 2500, 5000, Nsamp)
        dT = Tgrid_a[1]-Tgrid_a[0]

        kboltz = eos_d['const_d']['kboltz']
        # Test entropy
        TOL = 1e-4

        iV = Vgrid_a[0]
        genRT_mod = models.GenRosenfeldTaranzona()
        thermal_mod = eos_d['modtype_d']['ThermalMod']
        full_mod = eos_d['modtype_d']['FullMod']


        # verify potential heat capacity (energy deriv)
        acoef_a, bcoef_a = thermal_mod.calc_RT_coef( iV, eos_d )
        Cvpot_a = np.squeeze( genRT_mod.calc_heat_capacity_pot( Tgrid_a, eos_d,
                                                               bcoef_a=bcoef_a ) )

        Epot_a = np.squeeze( genRT_mod.calc_energy_pot( Tgrid_a, eos_d,
                                                       acoef_a=acoef_a,
                                                       bcoef_a=bcoef_a ) )
        Cvpot_dE_a = (Cvpot_a - np.gradient( Epot_a, dT ))/kboltz
        assert np.all( np.abs(Cvpot_dE_a[1:-1]) < TOL ), \
            'Cvpot must match numerical energy deriv'

        Spot_a = np.squeeze( genRT_mod.calc_entropy_pot( Tgrid_a, eos_d,
                                                        bcoef_a=bcoef_a ) )
        Cvpot_dS_a = ( Cvpot_a - Tgrid_a*np.gradient( Spot_a, dT ) )/kboltz
        assert np.all( np.abs(Cvpot_dS_a[1:-1]) < TOL ), \
            'Cvpot must match numerical entropy deriv'

        Fpot_a = Epot_a-Tgrid_a*Spot_a
        Spot_dF_err_a = ( Spot_a + np.gradient( Fpot_a, dT ) )/kboltz
        assert np.all( np.abs(Spot_dF_err_a[1:-1]) < TOL ), \
            'Spot must match numerical free energy deriv'

    def test_total_entropy(self):
        Nsamp = 1001

        eos_d = self.init_params({})

        eos_d['param_d']['E0'] = -21.3
        eos_d['param_d']['dE0th'] = 0.5
        V0 = eos_d['param_d']['V0']

        Vgrid_a = V0*np.arange(0.4,1.11,0.1)
        Tgrid_a = np.linspace( 2500, 5000, Nsamp)
        dT = Tgrid_a[1]-Tgrid_a[0]

        kboltz = eos_d['const_d']['kboltz']
        # Test entropy
        TOL = 1e-4

        iV = Vgrid_a[0]
        genRT_mod = models.GenRosenfeldTaranzona()
        thermal_mod = eos_d['modtype_d']['ThermalMod']
        full_mod = eos_d['modtype_d']['FullMod']

        # verify total entropy

        iFtot = np.squeeze( full_mod.free_energy( Vgrid_a[0], Tgrid_a, eos_d ) )
        iStot = np.squeeze( full_mod.entropy( Vgrid_a[0], Tgrid_a, eos_d ) )
        iSnum = -np.gradient( iFtot, dT )
        Stot_dF_err_a = ( iStot - iSnum )/kboltz
        assert np.all( np.abs(Stot_dF_err_a[1:-1]) < TOL ), \
            'Spot must match numerical free energy deriv'
#====================================================================
class TestRosenfeldTaranzonaPerturbExpand(TestRosenfeldTaranzonaPerturb):
    def load_compress_path_mod(self, eos_d):
        S0, = core.get_params(['S0'],eos_d)
        expand_adj_mod=models.Tait()
        compress_path_mod = models.Vinet(path_const='S',level_const=S0,
                                         supress_energy=False,
                                         supress_press=False,
                                         expand_adj_mod=expand_adj_mod)
        core.set_modtypes( ['CompressPathMod'], [compress_path_mod], eos_d )

        pass

    def init_params(self,eos_d):

        core.set_consts( [], [], eos_d )

        # EOS Parameter values initially set by Mosenfelder2009
        # Set model parameter values
        mass_avg = (24.31+28.09+3*16.0)/5.0 # g/(mol atom)
        T0 = 1673.0
        S0 = 0.0 # must adjust
        param_key_a = ['T0','S0','mass_avg']
        param_val_a = np.array([T0,S0,mass_avg])
        core.set_params( param_key_a, param_val_a, eos_d )

        V0 = (38.575*1e-5)*mass_avg/eos_d['const_d']['Nmol']/1e3*1e30 # ang^3/atom
        K0 = 20.8
        KP0= 10.2
        KP20 = -2.86 # Not actually used!
        E0 = 0.0
        param_key_a = ['V0','K0','KP0','KP20','E0']
        param_val_a = np.array([V0,K0,KP0,KP20,E0])
        core.set_params( param_key_a, param_val_a, eos_d )

        VR = V0
        gammaR = 0.46
        qR = -1.35
        param_key_a = ['VR','gammaR','qR']
        param_val_a = np.array([VR,gammaR,qR])
        core.set_params( param_key_a, param_val_a, eos_d )

        dE0th  = +1.0
        dV0th  = -0.02
        dK0th  = +0.1
        dKP0th = -0.00
        dKP20th = +1.0
        # dE0th  = +0.4
        # dV0th  = -0.0
        # dK0th  = -0.01
        # dKP0th = -0.03
        lognfac = 0.0
        mexp = 3.0/5
        param_key_a = ['dE0th','dV0th','dK0th','dKP0th','dKP20th','lognfac','mexp']
        param_val_a = np.array([dE0th,dV0th,dK0th,dKP0th,dKP20th,lognfac,mexp])
        core.set_params( param_key_a, param_val_a, eos_d )


        # Must convert energy units from kJ/g to eV/atom
        energy_conv_fac = mass_avg/eos_d['const_d']['kJ_molpereV']
        core.set_consts( ['energy_conv_fac'], [energy_conv_fac],
                                  eos_d )


        self.load_eos_mod( eos_d )

        #     from IPython import embed; embed(); import ipdb; ipdb.set_trace()
        return eos_d

    def test_energy_curves_Spera2011_exp(self):
        Nsamp = 101
        eos_d = self.init_params({})

        param_d = eos_d['param_d']
        Vgrid_a = np.linspace(0.4,1.1,Nsamp)*param_d['V0']
        Tgrid_a = np.array([2500,3000,3500,4000,4500,5000])

        full_mod = eos_d['modtype_d']['FullMod']
        compress_path_mod = eos_d['modtype_d']['CompressPathMod']
        thermal_mod = eos_d['modtype_d']['ThermalMod']

        # energy_conv_fac, = core.get_consts(['energy_conv_fac'],eos_d)

        energy_mod_a = []
        press_mod_a = []


        for iT in Tgrid_a:
            ienergy_a = full_mod.energy(Vgrid_a,iT,eos_d)
            ipress_a = full_mod.press(Vgrid_a,iT,eos_d)
            energy_mod_a.append(ienergy_a)
            press_mod_a.append(ipress_a)

        # energy_mod_a = np.array( energy_mod_a )
        energy_mod_a = np.array( energy_mod_a )
        press_mod_a = np.array( press_mod_a )

        # from IPython import embed; embed(); import ipdb; ipdb.set_trace()
        cmap=plt.get_cmap('coolwarm')
        col_a = cmap(1.0*(Tgrid_a-Tgrid_a[0])/np.ptp(Tgrid_a))[:,:3]

        plt.ion()
        plt.figure()
        [plt.plot(ipress_a, ienergy_a,'-',color=icol_a,label=iT) \
         for ipress_a,ienergy_a,icol_a,iT  in zip(press_mod_a,energy_mod_a,col_a,Tgrid_a)]
        ax = plt.axes()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1],labels[::-1],loc='upper left')
        plt.xlim(-5,165)
        ybnd = [np.min(energy_mod_a[press_mod_a<165]), np.max(energy_mod_a[press_mod_a<165])]
        plt.ylim(ybnd[0],ybnd[1])
        # plt.ylim(-100.5,-92)


        print 'Compare this plot with Spera2011 Fig 2b (Oganov potential):'
        print 'Do the figures agree (y/n or k for keyboard)?'
        s = raw_input('--> ')
        if s=='k':
            from IPython import embed; embed(); import ipdb; ipdb.set_trace()

        assert s=='y', 'Figure must match published figure'
        pass
#====================================================================
