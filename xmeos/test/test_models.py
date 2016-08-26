import numpy as np
import models
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
# SEC:1 Abstract Test Classes
#====================================================================
class BaseTestCompressPathMod(object):
    @abstractmethod
    def load_compress_path_mod(self, eos_d):
        assert False, 'must implement load_compress_path_mod()'

    def init_params(self,eos_d):
        # Set model parameter values
        E0 = 0.0 # eV/atom
        V0 = 38.0 # 1e-5 m^3 / kg
        K0 = 25.0 # GPa
        KP0 = 9.0 # 1
        param_key_a = ['V0','K0','KP0','E0']
        param_val_a = np.array([ V0, K0, KP0, E0 ])

        models.Control.set_consts( [], [], eos_d )

        self.load_compress_path_mod( eos_d )

        models.Control.set_params( param_key_a, param_val_a, eos_d )

        return eos_d

    def test_press(self):
        TOL = 1e-4

        Nsamp = 10001
        eos_d = self.init_params({})

        param_d = eos_d['param_d']
        Vmod_a = np.linspace(.7,1.2,Nsamp)*param_d['V0']
        dV = Vmod_a[1] - Vmod_a[0]

        # print eos_d['modtype_d']
        compress_path_mod = eos_d['modtype_d']['CompressPathMod']

        press_a = compress_path_mod.press(Vmod_a,eos_d)
        energy_a = compress_path_mod.energy(Vmod_a,eos_d)

        press_num_a = -eos_d['const_d']['PV_ratio']*np.gradient(energy_a,dV)

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

    def do_test_energy_perturb_eval(self):
        TOL = 1e-4
        dxfrac = 1e-8

        Nsamp = 10001
        eos_d = self.init_params({})

        param_d = eos_d['param_d']
        Vmod_a = np.linspace(.7,1.3,Nsamp)*param_d['V0']
        dV = Vmod_a[1] - Vmod_a[0]

        compress_path_mod = eos_d['modtype_d']['CompressPathMod']
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
class BaseTest4thOrdCompressPathMod(BaseTestCompressPathMod):
    def init_params(self,eos_d):
        # Use parents init_params method
        eos_d = super(BaseTest4thOrdCompressPathMod,self).init_params(eos_d)

        # Add K''0 param
        KP20 = -1.1*eos_d['param_d']['KP0']/eos_d['param_d']['K0']
        models.Control.set_params( ['KP20'], [KP20], eos_d )

        return eos_d
#====================================================================

#====================================================================
# SEC:2 Implimented Test Clases
#====================================================================
# 2.1: CompressPathMod Tests
#====================================================================
class TestVinetCompressPathMod(BaseTestCompressPathMod):
    def load_compress_path_mod(self, eos_d):
        compress_path_mod = models.Vinet(path_const='S')
        models.Control.set_modtypes( ['CompressPathMod'], [compress_path_mod], eos_d )
        pass

    def test_energy_perturb_eval(self):
        self.do_test_energy_perturb_eval()
        pass
#====================================================================
class TestBM3CompressPathMod(BaseTestCompressPathMod):
    def load_compress_path_mod(self, eos_d):
        compress_path_mod = models.BirchMurn3(path_const='S')
        models.Control.set_modtypes( ['CompressPathMod'], [compress_path_mod], eos_d )
        pass
#====================================================================
class TestBM4CompressPathMod(BaseTest4thOrdCompressPathMod):
    def load_compress_path_mod(self, eos_d):
        compress_path_mod = models.BirchMurn4(path_const='S')
        models.Control.set_modtypes( ['CompressPathMod'], [compress_path_mod], eos_d )
        pass
#====================================================================
class TestGenFiniteStrainCompressPathMod(BaseTest4thOrdCompressPathMod):
    def init_params(self,eos_d):
        # Use parents init_params method
        eos_d = super(TestGenFiniteStrainCompressPathMod,self).init_params(eos_d)

        # Add nexp param
        nexp = +2.0
        models.Control.set_params( ['nexp'], [nexp], eos_d )

        return eos_d

    def load_compress_path_mod(self, eos_d):
        compress_path_mod = models.GenFiniteStrain(path_const='S')
        models.Control.set_modtypes( ['CompressPathMod'], [compress_path_mod], eos_d )
        pass
#====================================================================
class TestTaitCompressPathMod(BaseTest4thOrdCompressPathMod):
    def load_compress_path_mod(self, eos_d):
        compress_path_mod = models.Tait(path_const='S')
        models.Control.set_modtypes( ['CompressPathMod'], [compress_path_mod], eos_d )
        pass

    def test_energy_perturb_eval(self):
        self.do_test_energy_perturb_eval()
        pass
#====================================================================
class TestCompareCompressPathMods(object):
    def init_params(self,eos_d):
        # Set model parameter values
        E0 = 0.0 # eV/atom
        V0 = 38.0 # 1e-5 m^3 / kg
        K0 = 25.0 # GPa
        KP0 = 9.0 # 1
        param_key_a = ['V0','K0','KP0','E0']
        param_val_a = np.array([ V0, K0, KP0, E0 ])

        models.Control.set_consts( [], [], eos_d )
        models.Control.set_params( param_key_a, param_val_a, eos_d )

        return eos_d

    def get_eos_mods(self):
        eos_vinet_d = self.init_params({})
        eos_tait_d = self.init_params({})

        models.Control.set_modtypes( ['CompressPathMod'], [models.Vinet(path_const='S')],
                           eos_vinet_d )
        models.Control.set_modtypes( ['CompressPathMod'], [models.Tait(path_const='S')],
                           eos_tait_d )

        return eos_vinet_d, eos_tait_d

    def calc_energy_perturb( self, eos_d ):
        dxfrac = 1e-6
        Nsamp = 10001

        param_d = eos_d['param_d']
        Vmod_a = np.linspace(.7,1.1,Nsamp)*param_d['V0']
        dV = Vmod_a[1] - Vmod_a[0]

        compress_path_mod = eos_d['modtype_d']['CompressPathMod']
        scale_a, paramkey_a = compress_path_mod.get_param_scale( eos_d )

        Eperturb_num_a = np.zeros((paramkey_a.size,Nsamp))
        for ind,paramkey in enumerate(paramkey_a):
            Eperturb_num_a[ind,:] = compress_path_mod.param_deriv\
                ( 'energy', paramkey, Vmod_a, eos_d, dxfrac=dxfrac)

        Eperturb_a, scale_a, paramkey_a = compress_path_mod.energy_perturb(Vmod_a, eos_d)

        Eperturb_num_a = np.zeros((paramkey_a.size,Nsamp))
        for ind,paramkey in enumerate(paramkey_a):
            Eperturb_num_a[ind,:] = compress_path_mod.param_deriv\
                ( 'energy', paramkey, Vmod_a, eos_d, dxfrac=dxfrac)

        return Eperturb_a, Eperturb_num_a, Vmod_a, scale_a, paramkey_a

    def calc_energy( self, eos_d ):
        dxfrac = 1e-6
        Nsamp = 10001

        param_d = eos_d['param_d']
        Vmod_a = np.linspace(.7,1.1,Nsamp)*param_d['V0']
        dV = Vmod_a[1] - Vmod_a[0]

        compress_path_mod = eos_d['modtype_d']['CompressPathMod']
        scale_a, paramkey_a = compress_path_mod.get_param_scale( eos_d )

        energy_a = compress_path_mod.energy( Vmod_a, eos_d )

        return energy_a, Vmod_a

    def test_compare(self):
        TOL = 1e-4

        eos_vinet_d, eos_tait_d = self.get_eos_mods()
        KP20 = -1.1*eos_tait_d['param_d']['KP0']/eos_tait_d['param_d']['K0']
        models.Control.set_params( ['KP20'], [KP20], eos_tait_d )

        energy_vin_a, Vmod_vin_a = self.calc_energy( eos_vinet_d )
        energy_tait_a, Vmod_tait_a = self.calc_energy( eos_tait_d )

        # plt.ion()
        # plt.figure()
        # plt.clf()
        # plt.plot(Vmod_vin_a, energy_vin_a,'k-',
        #          Vmod_tait_a, energy_tait_a, 'r-')

        Eperturb_vin_a, Eperturb_num_vin_a, Vmod_vin_a, scale_vin_a, \
            paramkey_vin_a = self.calc_energy_perturb( eos_vinet_d )

        Eperturb_tait_a, Eperturb_num_tait_a, Vmod_tait_a, scale_tait_a, \
            paramkey_tait_a = self.calc_energy_perturb( eos_tait_d )

        # from IPython import embed; embed(); import ipdb; ipdb.set_trace()

        # plt.ion()
        # plt.figure()
        # plt.clf()
        # plt.plot(Vmod_vin_a[::100], Eperturb_vin_a[:,::100].T,'x',
        #          Vmod_tait_a, Eperturb_tait_a.T,'-')

        dV = Vmod_vin_a[1] - Vmod_vin_a[0]
        V0 = eos_tait_d['param_d']['V0']
        indV0 = np.where(Vmod_vin_a==V0)[0][0]

        Eperturb_diff = Eperturb_vin_a[:,indV0] - Eperturb_tait_a[[0,1,2,4],indV0]

        assert np.all(np.abs(Eperturb_diff)<TOL), \
            'Energy perturbations for Vinet and Tait EOS at V0  must agree to within TOL'

        # Calc numerical volume derivs
        # Some of these curves take very small values, making numerical
        # comparison difficult, but  comparison by eye checks out
        dE1_perturb_vin_a = np.gradient(Eperturb_vin_a,dV)[1]
        dE2_perturb_vin_a = np.gradient(dE1_perturb_vin_a,dV)[1]
        dE3_perturb_vin_a = np.gradient(dE2_perturb_vin_a,dV)[1]

        dE1_perturb_tait_a = np.gradient(Eperturb_tait_a,dV)[1]
        dE2_perturb_tait_a = np.gradient(dE1_perturb_tait_a,dV)[1]
        dE3_perturb_tait_a = np.gradient(dE2_perturb_tait_a,dV)[1]

        # plt.clf()
        # plt.plot(Vmod_vin_a[::100], dE1_perturb_vin_a[:,::100].T,'x',
        #          Vmod_tait_a, dE1_perturb_tait_a.T,'-')

        # plt.clf()
        # plt.plot(Vmod_vin_a[::100], dE2_perturb_vin_a[:,::100].T,'x',
        #          Vmod_tait_a, dE2_perturb_tait_a.T,'-')

        # Eperturb_vin_a[:,indV0]-Eperturb_tait_a[[0,1,2,4],indV0]
        # Eperturb_vin_a[:,indV0]

        # dE1_perturb_vin_a[:,indV0]-dE1_perturb_tait_a[[0,1,2,4],indV0]
        # dE1_perturb_vin_a[:,indV0]

        # plt.clf()
        # plt.plot(Vmod_vin_a[::100], dE3_perturb_vin_a[:,::100].T,'x',
        #          Vmod_tait_a, dE3_perturb_tait_a.T,'-')

        pass
#====================================================================
class TestExpandCompressPathMod(BaseTest4thOrdCompressPathMod):
    def load_compress_path_mod(self, eos_d):
        compress_path_mod   = models.Vinet(path_const='S',expand_adj_mod=models.Tait())
        models.Control.set_modtypes(['CompressPathMod'],[compress_path_mod], eos_d )

        pass

    def test_press_components(self):
        TOL = 1e-4
        dxfrac = 1e-8

        Nsamp = 10001
        eos_d = self.init_params({})

        param_d = eos_d['param_d']
        Vmod_a = np.linspace(.7,1.3,Nsamp)*param_d['V0']
        dV = Vmod_a[1] - Vmod_a[0]

        compress_path_mod = eos_d['modtype_d']['CompressPathMod']

        press_a = compress_path_mod.press( Vmod_a, eos_d )
        press_pos_a = compress_path_mod.press( Vmod_a, eos_d, apply_expand_adj=False)
        press_neg_a = compress_path_mod.expand_adj_mod.press( Vmod_a, eos_d )

        # press_pos_a = expand_pos_mod.press( Vmod_a, eos_d )
        # press_neg_a = expand_neg_mod.press( Vmod_a, eos_d )


        ind_neg = Vmod_a>param_d['V0']
        ind_pos = Vmod_a<param_d['V0']

        assert np.all(press_a[ind_neg]==press_neg_a[ind_neg]),\
            'The expansion corrected press must match ExpandNegMod for negative pressure values'
        assert np.all(press_a[ind_pos]==press_pos_a[ind_pos]),\
            'The expansion corrected press must match ExpandPosMod for positive pressure values'

        # from IPython import embed; embed(); import ipdb; ipdb.set_trace()
        # plt.ion()
        # plt.figure()
        # plt.clf()
        # plt.plot(Vmod_a, press_pos_a, 'r--', Vmod_a, press_neg_a, 'b--',
        #          Vmod_a, press_a, 'k-')

        pass

    def test_energy_components(self):
        TOL = 1e-4
        dxfrac = 1e-8

        Nsamp = 10001
        eos_d = self.init_params({})

        param_d = eos_d['param_d']
        Vmod_a = np.linspace(.7,1.3,Nsamp)*param_d['V0']
        dV = Vmod_a[1] - Vmod_a[0]

        compress_path_mod = eos_d['modtype_d']['CompressPathMod']

        energy_a = compress_path_mod.energy( Vmod_a, eos_d )
        energy_pos_a = compress_path_mod.energy( Vmod_a, eos_d, apply_expand_adj=False )
        energy_neg_a = compress_path_mod.expand_adj_mod.energy( Vmod_a, eos_d )


        ind_neg = Vmod_a>param_d['V0']
        ind_pos = Vmod_a<param_d['V0']

        assert np.all(energy_a[ind_neg]==energy_neg_a[ind_neg]),\
            'The expansion corrected energy must match ExpandNegMod for negative pressure values'
        assert np.all(energy_a[ind_pos]==energy_pos_a[ind_pos]),\
            'The expansion corrected energy must match ExpandPosMod for positive pressure values'


        # from IPython import embed; embed(); import ipdb; ipdb.set_trace()
        # plt.ion()
        # plt.figure()
        # plt.clf()
        # plt.plot(Vmod_a, energy_pos_a, 'r--', Vmod_a, energy_neg_a, 'b--',
        #          Vmod_a, energy_a, 'k-')

        pass

    def test_energy_perturb_eval(self):
        self.do_test_energy_perturb_eval()
        pass
#====================================================================
# 2.2: ThermalPathMod Tests
#====================================================================
class TestGenRosenfeldTaranzona(BaseTestThermalPathMod):
    def load_thermal_path_mod(self, eos_d):
        thermal_path_mod = models.GenRosenfeldTaranzona(path_const='V')
        models.Control.set_modtypes( ['ThermalPathMod'], [thermal_path_mod], eos_d )

        pass

    def init_params(self,eos_d):
        # Set model parameter values
        acoef = -158.2
        bcoef = .042
        mexp = 3.0/5
        lognfac = 0.0
        T0 = 5000.0

        param_key_a = ['acoef','bcoef','mexp','lognfac','T0']
        param_val_a = np.array([acoef,bcoef,mexp,lognfac,T0])

        models.Control.set_consts( [], [], eos_d )
        self.load_thermal_path_mod( eos_d )

        models.Control.set_params( param_key_a, param_val_a, eos_d )

        return eos_d
#====================================================================
class TestRosenfeldTaranzonaPoly(BaseTestThermalMod):
    def load_thermal_mod(self, eos_d):
        thermal_mod = models.RosenfeldTaranzonaPoly()
        models.Control.set_modtypes( ['ThermalMod'], [thermal_mod], eos_d )

        pass

    def load_compress_path_mod(self, eos_d):
        T0, = models.Control.get_params(['T0'],eos_d)
        compress_path_mod = models.Vinet(path_const='T',level_const=T0,
                                         supress_energy=True,
                                         supress_press=True)
        # NOTE that supress press is included to impliment all terms according
        # to Spera2011
        #   (but the current implimentation actually uses the compress path
        #   pressure unscaled)
        models.Control.set_modtypes( ['CompressPathMod'], [compress_path_mod], eos_d )

        pass

    def load_eos_mod(self, eos_d):

        self.load_thermal_mod(eos_d)
        self.load_compress_path_mod(eos_d)

        full_mod = models.ThermalPressMod()
        models.Control.set_modtypes( ['FullMod'], [full_mod], eos_d )

        pass

    def init_params(self,eos_d):

        models.Control.set_consts( [], [], eos_d )

        # Set model parameter values
        mexp = 3.0/5
        T0 = 4000.0
        V0_ccperg = 0.408031 # cc/g
        K0 = 13.6262
        KP0= 7.66573
        E0 = 0.0
        # nfac = 5.0
        # mass = (24.31+28.09+3*16.0) # g/(mol atom)
        # V0 = V0_ccperg

        # NOTE that units are all per atom
        # requires conversion from values reported in Spera2011
        lognfac = 0.0
        mass = (24.31+28.09+3*16.0)/5.0 # g/(mol atom)
        Vconv_fac = mass*eos_d['const_d']['ang3percc']/eos_d['const_d']['Nmol']
        V0 = V0_ccperg*Vconv_fac


        param_key_a = ['mexp','lognfac','T0','V0','K0','KP0','E0','mass']
        param_val_a = np.array([mexp,lognfac,T0,V0,K0,KP0,E0,mass])
        models.Control.set_params( param_key_a, param_val_a, eos_d )

        # Set parameter values from Spera et al. (2011)
        # for MgSiO3 melt using  (Oganov potential)

        # Must convert energy units from kJ/g to eV/atom
        energy_conv_fac = mass/eos_d['const_d']['kJ_molpereV']
        models.Control.set_consts( ['energy_conv_fac'], [energy_conv_fac],
                                  eos_d )

        # change coefficients to relative
        # acoef_a = energy_conv_fac*\
        #     np.array([127.116,-3503.98,20724.4,-60212.0,86060.5,-48520.4])
        # bcoef_a = energy_conv_fac*\
        #     np.array([-0.371466,7.09542,-45.7362,139.020,-201.487,112.513])
        Vconv_a = (1.0/Vconv_fac)**np.arange(6)


        unit_conv = energy_conv_fac*Vconv_a

        # Reported vol-dependent polynomial coefficients for a and b
        #  in Spera2011
        acoef_unscl_a = np.array([127.116,-3503.98,20724.4,-60212.0,\
                                  86060.5,-48520.4])
        bcoef_unscl_a = np.array([-0.371466,7.09542,-45.7362,139.020,\
                                  -201.487,112.513])

        # Convert units and transfer to normalized version of RT model
        acoef_a = unit_conv*(acoef_unscl_a+bcoef_unscl_a*T0**mexp)
        bcoef_a = unit_conv*bcoef_unscl_a*T0**mexp

        models.Control.set_array_params( 'acoef', acoef_a, eos_d )
        models.Control.set_array_params( 'bcoef', bcoef_a, eos_d )

        self.load_eos_mod( eos_d )

        #     from IPython import embed; embed(); import ipdb; ipdb.set_trace()
        return eos_d

    def test_RT_potenergy_curves_Spera2011(self):
        Nsamp = 101
        eos_d = self.init_params({})

        param_d = eos_d['param_d']
        Vgrid_a = np.linspace(0.5,1.1,Nsamp)*param_d['V0']
        Tgrid_a = np.linspace(100.0**(5./3),180.0**(5./3),11)

        full_mod = eos_d['modtype_d']['FullMod']
        thermal_mod = eos_d['modtype_d']['ThermalMod']

        energy_conv_fac, = models.Control.get_consts(['energy_conv_fac'],eos_d)

        potenergy_mod_a = []

        for iV in Vgrid_a:
            ipotenergy_a = thermal_mod.calc_energy_pot(iV,Tgrid_a,eos_d)
            potenergy_mod_a.append(ipotenergy_a)

        # energy_mod_a = np.array( energy_mod_a )
        potenergy_mod_a = np.array( potenergy_mod_a )

        plt.ion()
        plt.figure()
        plt.plot(Tgrid_a**(3./5), potenergy_mod_a.T/energy_conv_fac,'-')
        plt.xlim(100,180)
        plt.ylim(-102,-95)

        print 'Compare this plot with Spera2011 Fig 1b (Oganov potential):'
        print 'Do the figures agree (y/n or k for keyboard)?'
        s = raw_input('--> ')
        if s=='k':
            from IPython import embed; embed(); import ipdb; ipdb.set_trace()

        assert s=='y', 'Figure must match published figure'
        pass

    def test_energy_curves_Spera2011(self):
        Nsamp = 101
        eos_d = self.init_params({})

        param_d = eos_d['param_d']
        Vgrid_a = np.linspace(0.4,1.1,Nsamp)*param_d['V0']
        Tgrid_a = np.array([2500,3000,3500,4000,4500,5000])

        full_mod = eos_d['modtype_d']['FullMod']

        energy_conv_fac, = models.Control.get_consts(['energy_conv_fac'],eos_d)

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

        plt.ion()
        plt.figure()
        plt.plot(press_mod_a.T, energy_mod_a.T/energy_conv_fac,'-')
        plt.legend(Tgrid_a,loc='lower right')
        plt.xlim(-5,165)
        plt.ylim(-100.5,-92)

        # from IPython import embed; embed(); import ipdb; ipdb.set_trace()

        print 'Compare this plot with Spera2011 Fig 2b (Oganov potential):'
        print 'Do the figures agree (y/n or k for keyboard)?'
        s = raw_input('--> ')
        if s=='k':
            from IPython import embed; embed(); import ipdb; ipdb.set_trace()

        assert s=='y', 'Figure must match published figure'
        pass

    def test_heat_capacity_curves_Spera2011(self):
        Nsamp = 101
        eos_d = self.init_params({})

        param_d = eos_d['param_d']
        Vgrid_a = np.linspace(0.4,1.2,Nsamp)*param_d['V0']
        Tgrid_a = np.array([2500,3000,3500,4000,4500,5000])

        full_mod = eos_d['modtype_d']['FullMod']
        thermal_mod = eos_d['modtype_d']['ThermalMod']

        heat_capacity_mod_a = []
        energy_conv_fac, = models.Control.get_consts(['energy_conv_fac'],eos_d)

        energy_mod_a = []
        press_mod_a = []

        for iT in Tgrid_a:
            iheat_capacity_a = thermal_mod.heat_capacity(Vgrid_a,iT,eos_d)
            ienergy_a = full_mod.energy(Vgrid_a,iT,eos_d)
            ipress_a = full_mod.press(Vgrid_a,iT,eos_d)

            heat_capacity_mod_a.append(iheat_capacity_a)
            energy_mod_a.append(ienergy_a)
            press_mod_a.append(ipress_a)


        # energy_mod_a = np.array( energy_mod_a )
        heat_capacity_mod_a = np.array( heat_capacity_mod_a )
        energy_mod_a = np.array( energy_mod_a )
        press_mod_a = np.array( press_mod_a )

        plt.ion()
        plt.figure()
        plt.plot(press_mod_a.T,1e3*heat_capacity_mod_a.T/energy_conv_fac,'-')
        plt.legend(Tgrid_a,loc='lower right')
        # plt.ylim(1.2,1.9)
        plt.xlim(-5,240)

        print 'Compare this plot with Spera2011 Fig 2b (Oganov potential):'
        print 'Do the figures agree (y/n or k for keyboard)?'
        s = raw_input('--> ')
        if s=='k':
            from IPython import embed; embed(); import ipdb; ipdb.set_trace()

        assert s=='y', 'Figure must match published figure'
        pass
#====================================================================
class TestRosenfeldTaranzonaPerturb(BaseTestThermalMod):
    def load_thermal_mod(self, eos_d):
        thermal_mod = models.RosenfeldTaranzonaPerturb()
        models.Control.set_modtypes( ['ThermalMod'], [thermal_mod], eos_d )

        pass

    def load_gamma_mod(self, eos_d):
        gamma_mod = models.GammaPowLaw()
        models.Control.set_modtypes( ['GammaMod'], [gamma_mod], eos_d )

        pass

    def load_compress_path_mod(self, eos_d):
        S0, = models.Control.get_params(['S0'],eos_d)
        compress_path_mod = models.Vinet(path_const='S',level_const=S0,
                                         supress_energy=False,
                                         supress_press=False)
        models.Control.set_modtypes( ['CompressPathMod'], [compress_path_mod], eos_d )

        pass

    def load_eos_mod(self, eos_d):

        self.load_compress_path_mod(eos_d)
        self.load_gamma_mod(eos_d)
        self.load_thermal_mod(eos_d)

        full_mod = models.ThermalPressMod()
        models.Control.set_modtypes( ['FullMod'], [full_mod], eos_d )

        pass

    def init_params(self,eos_d):

        models.Control.set_consts( [], [], eos_d )

        # EOS Parameter values initially set by Mosenfelder2009
        # Set model parameter values
        mass_avg = (24.31+28.09+3*16.0)/5.0 # g/(mol atom)
        T0 = 1673.0
        S0 = 0.0 # must adjust
        param_key_a = ['T0','S0','mass_avg']
        param_val_a = np.array([T0,S0,mass_avg])
        models.Control.set_params( param_key_a, param_val_a, eos_d )

        V0 = (38.575*1e-5)*mass_avg/eos_d['const_d']['Nmol']/1e3*1e30 # ang^3/atom
        K0 = 20.8
        KP0= 10.2
        # KP20 = -2.86 # Not actually used!
        E0 = 0.0
        param_key_a = ['V0','K0','KP0','E0']
        param_val_a = np.array([V0,K0,KP0,E0])
        models.Control.set_params( param_key_a, param_val_a, eos_d )

        VR = V0
        gammaR = 0.46
        qR = -1.35
        param_key_a = ['VR','gammaR','qR']
        param_val_a = np.array([VR,gammaR,qR])
        models.Control.set_params( param_key_a, param_val_a, eos_d )

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
        models.Control.set_params( param_key_a, param_val_a, eos_d )


        # Must convert energy units from kJ/g to eV/atom
        energy_conv_fac = mass_avg/eos_d['const_d']['kJ_molpereV']
        models.Control.set_consts( ['energy_conv_fac'], [energy_conv_fac],
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

        # energy_conv_fac, = models.Control.get_consts(['energy_conv_fac'],eos_d)

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
class TestGammaComparison():
    def init_params(self,eos_d):
        VR = 1.0
        gammaR = 1.0
        gammapR = -1.0
        qR = gammapR/gammaR
        # qR = +1.0
        # qR = +0.5

        param_key_a = ['VR','gammaR','gammapR','qR']
        param_val_a = np.array([VR,gammaR,gammapR,qR])
        models.Control.set_params( param_key_a, param_val_a, eos_d )

        return eos_d

    def load_gamma_mod(self, eos_d):
        gamma_mod = models.GammaPowLaw()
        models.Control.set_modtypes( ['GammaMod'], [gamma_mod], eos_d )

        pass

    def test(self):
        eos_d = self.init_params({})
        VR = eos_d['param_d']['VR']
        TR = 1000.0

        eos_pow_d = copy.deepcopy(eos_d)
        eos_str_d = copy.deepcopy(eos_d)


        models.Control.set_modtypes( ['GammaMod'], [models.GammaPowLaw],
                                    eos_pow_d )
        models.Control.set_modtypes( ['GammaMod'], [models.GammaFiniteStrain],
                                    eos_str_d )

        gammaR = eos_d['param_d']['gammaR']
        qR = eos_d['param_d']['qR']


        N = 1001
        V_a = VR*np.linspace(0.4,1.3,N)
        dV = V_a[1]-V_a[0]

        gam_pow_mod = eos_pow_d['modtype_d']['GammaMod']()
        gam_str_mod = eos_str_d['modtype_d']['GammaMod']()


        gam_pow_a = gam_pow_mod.gamma(V_a,eos_pow_d)
        gam_str_a = gam_str_mod.gamma(V_a,eos_str_d)

        temp_pow_a = gam_pow_mod.temp(V_a,TR,eos_pow_d)
        temp_str_a = gam_str_mod.temp(V_a,TR,eos_str_d)

        q_pow_a = V_a/gam_pow_a*np.gradient(gam_pow_a,dV)
        q_str_a = V_a/gam_str_a*np.gradient(gam_str_a,dV)


        # mpl.rcParams(fontsize=16)
        plt.ion()
        plt.figure()

        plt.clf()
        hleg = plt.plot(V_a,q_pow_a,'k--',V_a,q_str_a,'r-',lw=2)
        plt.legend(hleg,['Power-Law','Finite Strain'], loc='upper right',fontsize=16)
        plt.xlabel('$V / V_0$',fontsize=16)
        plt.ylabel('$q$',fontsize=16)
        plt.text(.9,1.1*qR,'$(\gamma_0,q_0) = ('+np.str(gammaR)+','+np.str(qR)+')$',fontsize=20)

        from IPython import embed; embed(); import ipdb; ipdb.set_trace()

        plt.clf()
        hleg = plt.plot(1.0/V_a,gam_str_a,'r-',lw=2)

        eos_str_d['param_d']['gammapR'] = -0.5
        eos_str_d['param_d']['gammapR'] = -2

        eos_str_d['param_d']['gammapR'] = -1.0
        eos_str_d['param_d']['gammaR'] = 0.5
        eos_str_d['param_d']['gammapR'] = -2.0
        eos_str_d['param_d']['gammapR'] = -10.0

        eos_str_d['param_d']['gammaR'] = 0.75
        eos_str_d['param_d']['gammapR'] = -10.0
        eos_str_d['param_d']['gammapR'] = -30.0
        gam_str_a = gam_str_mod.gamma(V_a,eos_str_d)
        eos_str_d['param_d']['gammapR'] = -0.5

        plt.clf()
        hleg = plt.plot(V_a,gam_pow_a,'k--',V_a,gam_str_a,'r-',lw=2)
        plt.legend(hleg,['Power-Law','Finite Strain'], loc='upper right',fontsize=16)
        plt.xlabel('$V / V_0$',fontsize=16)
        plt.ylabel('$\gamma$',fontsize=16)

        plt.text(.9,1.1*gammaR,'$(\gamma_0,q_0) = ('+np.str(gammaR)+','+np.str(qR)+')$',fontsize=20)

        plt.savefig('test/figs/gamma-comparison.png',dpi=450)




        plt.clf()
        hleg = plt.plot(V_a,temp_pow_a,'k--',V_a,temp_str_a,'r-',lw=2)
        plt.legend(hleg,['Power-Law','Finite Strain'], loc='upper right',
                   fontsize=16)
        plt.xlabel('$V / V_0$',fontsize=16)
        plt.ylabel('$T\; [K]$',fontsize=16)
        plt.text(.9,1.1*TR,'$(\gamma_0,q_0) = ('+np.str(gammaR)+','+np.str(qR)+')$',fontsize=20)
        plt.savefig('test/figs/gamma-temp-comparison.png',dpi=450)



#====================================================================
class TestRosenfeldTaranzonaPerturbExpand(TestRosenfeldTaranzonaPerturb):
    def load_compress_path_mod(self, eos_d):
        S0, = models.Control.get_params(['S0'],eos_d)
        expand_adj_mod=models.Tait()
        compress_path_mod = models.Vinet(path_const='S',level_const=S0,
                                         supress_energy=False,
                                         supress_press=False,
                                         expand_adj_mod=expand_adj_mod)
        models.Control.set_modtypes( ['CompressPathMod'], [compress_path_mod], eos_d )

        pass

    def init_params(self,eos_d):

        models.Control.set_consts( [], [], eos_d )

        # EOS Parameter values initially set by Mosenfelder2009
        # Set model parameter values
        mass_avg = (24.31+28.09+3*16.0)/5.0 # g/(mol atom)
        T0 = 1673.0
        S0 = 0.0 # must adjust
        param_key_a = ['T0','S0','mass_avg']
        param_val_a = np.array([T0,S0,mass_avg])
        models.Control.set_params( param_key_a, param_val_a, eos_d )

        V0 = (38.575*1e-5)*mass_avg/eos_d['const_d']['Nmol']/1e3*1e30 # ang^3/atom
        K0 = 20.8
        KP0= 10.2
        KP20 = -2.86 # Not actually used!
        E0 = 0.0
        param_key_a = ['V0','K0','KP0','KP20','E0']
        param_val_a = np.array([V0,K0,KP0,KP20,E0])
        models.Control.set_params( param_key_a, param_val_a, eos_d )

        VR = V0
        gammaR = 0.46
        qR = -1.35
        param_key_a = ['VR','gammaR','qR']
        param_val_a = np.array([VR,gammaR,qR])
        models.Control.set_params( param_key_a, param_val_a, eos_d )

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
        models.Control.set_params( param_key_a, param_val_a, eos_d )


        # Must convert energy units from kJ/g to eV/atom
        energy_conv_fac = mass_avg/eos_d['const_d']['kJ_molpereV']
        models.Control.set_consts( ['energy_conv_fac'], [energy_conv_fac],
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

        # energy_conv_fac, = models.Control.get_consts(['energy_conv_fac'],eos_d)

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
#         energy_conv_fac, = models.Control.get_consts(['energy_conv_fac'],eos_d)
#
#         potenergy_mod_a = []
#
#         for iV in Vgrid_a:
#             ipotenergy_a = thermal_mod.calc_potential_energy(iV,Tgrid_a,eos_d)
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
#         energy_conv_fac, = models.Control.get_consts(['energy_conv_fac'],eos_d)
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


#====================================================================
# SEC:3 Test Admin Funcs
#====================================================================
class TestControl(object):
    def test_get_array_params(self):
        TOL = 1e-6
        eos_d, acoef_a = self.init_params()
        param_a = models.Control.get_array_params('acoef',eos_d)

        assert np.all(np.abs(param_a-acoef_a)<TOL), 'Stored and retrieved parameter array do not match within TOL'

        param_a = models.Control.get_array_params('V0',eos_d)
        assert param_a.size==0, 'non-array parameter should not be retrievable with get_array_params()'

        pass

    def test_set_array_params(self):
        TOL = 1e-6
        eos_d = {}
        # Set model parameter values
        E0 = 0.0 # eV/atom
        V0 = 38.0 # 1e-5 m^3 / kg
        K0 = 25.0 # GPa
        KP0 = 9.0 # 1
        acoef_a = np.array([1.3,-.23,9.99,-88])

        param_key_a = ['V0','K0','KP0','E0']
        param_val_a = np.array([ V0, K0, KP0, E0 ])
        models.Control.set_params( param_key_a, param_val_a, eos_d )

        models.Control.set_array_params( 'acoef', acoef_a, eos_d )
        models.Control.set_consts( [], [], eos_d )

        param_a =  models.Control.get_array_params( 'acoef', eos_d )

        assert np.all(np.abs(param_a-acoef_a)<TOL), 'Stored and retrieved parameter array do not match within TOL'

        pass

    def init_params(self):
        eos_d = {}
        # Set model parameter values
        E0 = 0.0 # eV/atom
        V0 = 38.0 # 1e-5 m^3 / kg
        K0 = 25.0 # GPa
        KP0 = 9.0 # 1
        acoef = np.array([1.3,-.23,9.99,-88])

        param_key_a = ['V0','K0','KP0','E0','acoef_0','acoef_1','acoef_2','acoef_3']
        param_val_a = np.array([ V0, K0, KP0, E0, acoef[0], acoef[1], acoef[2], acoef[3] ])

        models.Control.set_consts( [], [], eos_d )

        models.Control.set_params( param_key_a, param_val_a, eos_d )
        return eos_d, acoef
#====================================================================
