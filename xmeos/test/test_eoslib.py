import numpy as np
import eoslib
import pytest
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod

#====================================================================
# Define "slow" tests
#  - indicated by @slow decorator
#  - slow tests are run only if using --runslow cmd line arg
#====================================================================
slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)


# def test_true():
#     assert True, 'test_true'
#
# def test_false():
#     assert False, 'test_false'

# class Test4thOrdCompressMod(TestCompressMod):

#====================================================================
class BaseTestCompressMod(object):

    # def __init__(self):
    #     self.init_params(eos_d)

    @abstractmethod
    def load_compress_mod(self, eos_d):
        assert False, 'must implement load_compress_mod()'

    def init_params(self,eos_d):
        # Set model parameter values
        E0 = 0.0 # eV/atom
        V0 = 38.0 # 1e-5 m^3 / kg
        K0 = 25.0 # GPa
        KP0 = 9.0 # 1
        param_key_a = ['V0','K0','KP0','E0']
        param_val_a = np.array([ V0, K0, KP0, E0 ])

        eoslib.set_const( [], [], eos_d )
        self.load_compress_mod( eos_d )

        eoslib.set_param( param_key_a, param_val_a, eos_d )

        return eos_d

    def test_press(self):
        TOL = 1e-4

        Nsamp = 10001
        eos_d = self.init_params({})

        param_d = eos_d['param_d']
        Vmod_a = np.linspace(.7,1.1,Nsamp)*param_d['V0']
        dV = Vmod_a[1] - Vmod_a[0]

        compress_mod = eos_d['modtype_d']['CompressMod']

        press_a = compress_mod.press(Vmod_a,eos_d)
        energy_a = compress_mod.energy(Vmod_a,eos_d)

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
        dxfrac = 1e-6

        Nsamp = 10001
        eos_d = self.init_params({})

        param_d = eos_d['param_d']
        Vmod_a = np.linspace(.7,1.1,Nsamp)*param_d['V0']
        dV = Vmod_a[1] - Vmod_a[0]

        compress_mod = eos_d['modtype_d']['CompressMod']
        scale_a, param_a = compress_mod.get_param_scale( eos_d)

        Eperturb_num_a = np.zeros((param_a.size,Nsamp))
        for ind,param in enumerate(param_a):
            Eperturb_num_a[ind,:] = compress_mod.param_deriv\
                ( 'energy', param, Vmod_a, eos_d, dxfrac=dxfrac)


        # dEdV0_a = compress_mod.param_deriv( 'energy', 'V0', Vmod_a, eos_d, dxfrac=dxfrac)
        # dEdK0_a = compress_mod.param_deriv( 'energy', 'K0', Vmod_a, eos_d, dxfrac=dxfrac)
        # dEdKP0_a = compress_mod.param_deriv( 'energy', 'KP0', Vmod_a, eos_d, dxfrac=dxfrac)
        # dEdKP20_a = compress_mod.param_deriv( 'energy', 'KP20', Vmod_a, eos_d, dxfrac=dxfrac)
        # dEdE0_a = compress_mod.param_deriv( 'energy', 'E0', Vmod_a, eos_d, dxfrac=dxfrac)

        Eperturb_a, scale_a, param_a = compress_mod.energy_perturb(Vmod_a, eos_d)

        # Eperturb_num_a = np.vstack((dEdV0_a,dEdK0_a,dEdKP0_a,dEdKP20_a,dEdE0_a))
        max_error_a = np.max(np.abs(Eperturb_a-Eperturb_num_a),axis=1)

        # try:
        # except:
        # from IPython import embed; embed(); import ipdb; ipdb.set_trace()

        assert np.all(max_error_a < TOL),'Error in energy perturbation must be'\
            'less than TOL.'

#        eoslib.set_param( ['V0'], [1.01*param_d['V0']], eos_d )
#        energy_dV_a = compress_mod.energy(Vmod_a,eos_d)
#        dEdV_a = (energy_dV_a-energy_0_a)/(.01*param_d['V0'])
#
#        eos_d = self.init_params(eos_d)
#        eoslib.set_param( ['K0'], [1.01*param_d['K0']], eos_d )
#        energy_dK_a = compress_mod.energy(Vmod_a,eos_d)
#        dEdK_a = (energy_dK_a-energy_0_a)/(.01*param_d['K0'])
#
#        eos_d = self.init_params(eos_d)
#        eoslib.set_param( ['KP0'], [1.01*param_d['KP0']], eos_d )
#        energy_dKP_a = compress_mod.energy(Vmod_a,eos_d)
#        dEdKP_a = (energy_dKP_a-energy_0_a)/(.01*param_d['KP0'])
#
#        eos_d = self.init_params(eos_d)
#        dEdE_a = np.ones(energy_0_a.shape)
#
#        basis_a = np.vstack((dEdE_a/np.mean(dEdE_a),
#                             dEdV_a/np.mean(dEdV_a),
#                             dEdK_a/np.mean(dEdK_a),
#                             dEdKP_a/np.mean(dEdKP_a)))
#
#        from IPython import embed; embed(); import ipdb; ipdb.set_trace()
#
#
#        plt.clf()
#        plt.rc('text', usetex=True)
#        for i in range(10):
#            rcoeff_a = np.random.randn(4)
#            rmod_a = np.dot(rcoeff_a,basis_a)
#            rmod_a /= np.sqrt(np.mean(rmod_a**2))
#            plt.plot(Vmod_a/param_d['V0'],rmod_a,'-')
#
#        plt.plot(Vmod_a/param_d['V0'],0.0*Vmod_a,'k--')
#        plt.xlim(.7,1.1)
#
#        plt.xlabel('$V / V0$')
#        plt.ylabel('Relative Energy Shift')
#        plt.savefig('test/compress-eos-energy-random-perturb.png',dpi=350)
#
#        plt.clf()
#        plt.rc('text', usetex=True)
#        hlbl = plt.plot(Vmod_a/param_d['V0'], dEdE_a/np.sqrt(np.mean(dEdE_a**2)),'k-',
#                        Vmod_a/param_d['V0'], dEdV_a/np.sqrt(np.mean(dEdV_a**2)),'r-',
#                        Vmod_a/param_d['V0'], dEdK_a/np.sqrt(np.mean(dEdK_a**2)),'b-',
#                        Vmod_a/param_d['V0'], dEdKP_a/np.sqrt(np.mean(dEdKP_a**2)),'g-',
#                        Vmod_a/param_d['V0'], 0.0*Vmod_a, 'k--')
#        plt.xlim(.7,1.1)
#        plt.ylim(-.5,+3)
#        # plt.rc('font', family='serif')
#
#        plt.legend(hlbl[:-1],[r'$\delta E_0$',r'$\delta V_0$',r'$\delta K_0$',
#                              r"$\delta K'_0$"])
#        # assert False, 'test_press_eval'
#        plt.xlabel('$V / V_0$')
#        plt.ylabel('Scaled Relative Energy Shift')
#        plt.savefig('test/compress-eos-energy-perturb-basis.png',dpi=350)

#====================================================================
class BaseTest4thOrdCompressMod(BaseTestCompressMod):
    def init_params(self,eos_d):
        # Use parents init_params method
        eos_d = super(BaseTest4thOrdCompressMod,self).init_params(eos_d)

        # Add K''0 param
        KP20 = -1.1*eos_d['param_d']['KP0']/eos_d['param_d']['K0']
        eoslib.set_param( ['KP20'], [KP20], eos_d )

        return eos_d

#====================================================================
class TestVinetCompressMod(BaseTestCompressMod):
    def load_compress_mod(self, eos_d):
        compress_mod = eoslib.Vinet(path_const='S')
        eoslib.set_modtype( ['CompressMod'], [compress_mod], eos_d )
        pass

    def test_energy_perturb_eval(self):
        self.do_test_energy_perturb_eval()
        pass

#====================================================================
class TestBM3CompressMod(BaseTestCompressMod):
    def load_compress_mod(self, eos_d):
        compress_mod = eoslib.BirchMurn3(path_const='S')
        eoslib.set_modtype( ['CompressMod'], [compress_mod], eos_d )
        pass

#====================================================================
class TestBM4CompressMod(BaseTest4thOrdCompressMod):
    def load_compress_mod(self, eos_d):
        compress_mod = eoslib.BirchMurn4(path_const='S')
        eoslib.set_modtype( ['CompressMod'], [compress_mod], eos_d )
        pass

#====================================================================
class TestGenFiniteStrainCompressMod(BaseTest4thOrdCompressMod):
    def init_params(self,eos_d):
        # Use parents init_params method
        eos_d = super(TestGenFiniteStrainCompressMod,self).init_params(eos_d)

        # Add nexp param
        nexp = +2.0
        eoslib.set_param( ['nexp'], [nexp], eos_d )

        return eos_d

    def load_compress_mod(self, eos_d):
        compress_mod = eoslib.GenFiniteStrain(path_const='S')
        eoslib.set_modtype( ['CompressMod'], [compress_mod], eos_d )
        pass

#====================================================================
class TestTaitCompressMod(BaseTest4thOrdCompressMod):
    def load_compress_mod(self, eos_d):
        compress_mod = eoslib.Tait(path_const='S')
        eoslib.set_modtype( ['CompressMod'], [compress_mod], eos_d )
        pass

    def test_energy_perturb_eval(self):
        self.do_test_energy_perturb_eval()
        pass
#====================================================================
# class TestTaitCompressMod(Test4thOrdCompressMod):
#     def load_compress_mod(self, eos_d):
#         compress_mod = eoslib.Tait(path_const='S')
#         eoslib.set_modtype( ['CompressMod'], [compress_mod], eos_d )
#         pass
#
#     def test_press(self):
#         self.do_test_press()



#     def press_ad_resid( param_a, eos_d, V_a, P_a, Perr_a=1.0,
#                        param_key_a=param_ad_key_a ):
#         # param_conv_a = param_conv_f(param_a)
#         # eoslib.set_param( param_key_a, param_conv_a, eos_d )
#         eoslib.set_param( param_key_a, param_a, eos_d )
#         Pmod_a = eos_d['modtype_d']['CompressMod'].press( V_a, eos_d)
#         resid_a = (Pmod_a - P_a)/Perr_a
#
#         return resid_a
#
#
#     press_ad_resid(param0_ad_UM_a,eos_d,vol_ad_UM_a,PT_ad_UM_a[:,0])
#
#     press_ad_resid_UM_f = lambda param_a, eos_d=eos_d, V_a=vol_ad_UM_a,\
#         P_a=PT_ad_UM_a[:,0]: press_ad_resid(param_a,eos_d,V_a, P_a)
#
#     press_ad_resid_UM_f(param0_ad_UM_a)
#
#     paramf_ad_UM_a = optimize.leastsq(press_ad_resid_UM_f,param0_ad_a)[0]
#
#     eoslib.set_param( param_ad_key_a, paramf_ad_UM_a, eos_d )
#
#
#     plt.clf()
#     plt.figure()
#     plt.plot(vol_ad_UM_a,PT_ad_UM_a[:,0],'rx',
#              vol_ad_UM_a,eos_d['modtype_d']['CompressMod'].press( vol_ad_UM_a, eos_d),'kx')
#
#
#     # Now fit thermal properties
#
#     param_therm_key_a = ['gamma0','q','theta0']
#     param0_therm_UM_a = np.array([ gamma0, q, theta0])
#
#
#     def eval_press_therm_mod( param_in_a, eos_d, VT_a, param_conv_f=None,
#                              param_key_a=param_key_a ):
#         if param_conv_f is None:
#             param_a = np.copy(param_in_a)
#         else:
#             param_a =  param_conv_f( param_in_a )
#
#         V_a = VT_a[:,0]
#         T_a = VT_a[:,1]
#         eoslib.set_param( param_key_a, param_a, eos_d )
#         Pmod_a = eos_d['modtype_d']['ThermPressMod'].press( V_a, T_a, eos_d)
#         return Pmod_a
#
#     def eval_press_ref_mod( param_in_a, eos_d, V_a, param_conv_f=None,
#                              param_key_a=param_key_a ):
#         if param_conv_f is None:
#             param_a = np.copy(param_in_a)
#         else:
#             param_a =  param_conv_f( param_in_a )
#
#         eoslib.set_param( param_key_a, param_a, eos_d )
#         Pmod_a = eos_d['modtype_d']['CompressMod'].press( V_a, eos_d)
#         return Pmod_a
#
#     def press_resid( param_in_a, sys_state_a, P_a, eval_press_mod_f,
#                     Perr_a=1.0):
#         Pmod_a = eval_press_mod_f( param_in_a, sys_state_a )
#         resid_a = (Pmod_a - P_a)/Perr_a
#         return resid_a
#
#     # def press_resid( param_in_a, eos_d, sys_state_a, P_a, eval_press_mod_f,
#     #                 Perr_a=1.0):
#     #     Pmod_a = eval_press_mod_f( param_in_a, eos_d, sys_state_a )
#     #     resid_a = (Pmod_a - P_a)/Perr_a
#     #     return resid_a
#
#     # param_conv_f = lambda param_in_a, eos_d, VT_a, param_conv_f=eval_press_therm_mod
#
#
#
#
#     # param_conv_null_f=None
#     # param0_in_a = param0_a
#
#     # param_conv_exp_f = lambda param_in_a: np.exp(param_in_a)
#     # param0_in_a = np.log(param0_a)
#
#     # scl_a = np.copy(param0_a)
#     # param_conv_scl_f = lambda param_in_a, scl_a=scl_a: scl_a*param_in_a
#     # param0_in_a = param0_a/scl_a
#
#
#     param_therm_key_a = ['gamma0','q','theta0']
#     param0_therm_UM_a = np.array([ gamma0, q, theta0])
#
#     param_conv_exp_f = lambda param_in_a: np.exp(param_in_a)
#     param0_inv_therm_UM_a = np.log(param0_therm_UM_a)
#     param_conv_exp_f(param0_inv_therm_UM_a)
#
#     VT_UM_a=np.vstack((vol_UM_a, PT_UM_a[:,1])).T
#
#     # def eval_press_therm_mod( param_in_a, eos_d, VT_a, param_conv_f=None,
#     #                          param_key_a=param_key_a ):
#     eval_press_therm_mod_UM_f = lambda param_in_a, VT_a, eos_d=eos_d, \
#         param_conv_f=param_conv_exp_f, param_key_a=param_therm_key_a:\
#         eval_press_therm_mod(param_in_a,eos_d,VT_a, param_conv_f=param_conv_f,
#                              param_key_a=param_key_a)
#
#     eval_press_therm_mod_UM_f(param0_inv_therm_UM_a,VT_UM_a)
#     eos_d['param_d']
#
#     press_resid( param_in_a, eos_d, sys_state_a, P_a, eval_press_mod_f,
#                     Perr_a=1.0):
#
#     press_therm_resid_UM_f
#
#     param_conv_f(param0_inv_therm_UM_a)
#     param_conv_f(param0_inv_therm_UM_a)
#
#     press_therm_resid_UM_f(param0_therm_UM_a)
#
#     press_therm_resid_UM_f(param0_therm_UM_a)
#     paramf_therm_UM_a = optimize.leastsq(press_therm_resid_UM_f,param0_therm_UM_a)[0]
#
#     press_resid(
#
#
#
#
#     scl_a = np.copy(param0_a)
#     param_conv_f = lambda param_a,scl_a=scl_a: scl_a*param_a
#     param0_inv_a = param0_a/scl_a
#
#
#     def press_resid( param_a, eos_d, V_a, T_a, P_a, Perr_a=1.0,
#                     param_conv_f=param_conv_f, param_key_a=param_key_a ):
#         param_conv_a = param_conv_f(param_a)
#         print param_conv_a
#         eoslib.set_param( param_key_a, param_conv_a, eos_d )
#         Pmod_a = eos_d['modtype_d']['ThermPressMod'].press( V_a, T_a, eos_d)
#         resid_a = (Pmod_a - P_a)/Perr_a
#
#         return resid_a
#
#     #scl_a = np.copy(param0_a)
#     press_resid_UM_f = lambda param_a, eos_d=eos_d, V_a=vol_UM_a,\
#         T_a=PT_UM_a[:,1],P_a=PT_UM_a[:,0],scl_a=scl_a: \
#         press_resid(param_a,eos_d,V_a,T_a, P_a)
#
#     sumsqr_UM_f = lambda param_a, eos_d=eos_d, V_a=vol_UM_a,\
#         T_a=PT_UM_a[:,1],P_a=PT_UM_a[:,0],scl_a=scl_a: \
#         np.sum(press_resid(param_a,eos_d,V_a,T_a, P_a)**2)
#
#     sumsqr_UM_f(param0_inv_a)
#     optimize.fmin_bfgs(sumsqr_UM_f,param0_inv_a)
#
#     press_resid_UM_f(param0_inv_a)
#     press_resid_UM_f(param0_inv_a+10)
#     press_resid_UM_f(param0_inv_a-10)
#     optimize.leastsq(press_resid_UM_f,param0_inv_a)
#
#     press_resid_UM_f(np.ones(param0_a.shape))
#     press_resid_UM_f(np.array([ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ]))
#     press_resid_UM_f(np.array([ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ]))
#
#     optimize.leastsq(press_resid_UM_f,np.ones(param0_a.shape))
#     optimize.fmin_bfgs(press_resid_UM_f,param0_a)
#     paramf_UM_a =
#
#
#
#     # express on per atom basis
#
#     # natom_avg
#
#     param0_d = {'Tref':1600,'V0':V0,'K0':150,'K0p':4.0,'gamma0':1.,'q':0.3,'theta0':1000}
#
#     eos_solid_d['mass_avg'] = param_d['mass_avg']
#     # Corresponds to BM3S model reported in Mosenfelder(2009)
#     eos_solid_d['T0'] = 300 # K
#
#     eos_solid_d['V0'] = 162.35/Nat_cell # ang^3/atom
#     eos_solid_d['K0'] = 254.7 # GPa
#     eos_solid_d['K0p'] = 4.26
#     eos_solid_d['E0'] = 0
#
#     eos_solid_d['theta0'] = 736 # K
#     eos_solid_d['gamma0'] = 2.23
#     eos_solid_d['q'] = 1.83
#
#
#
#     V0 = eos_d['V0']
#     gamma0 = eos_d['gamma0']
#     q = eos_d['q']
#
#     eos.press_mie_grun( V_a, T_a, eos_d )
#
#     plt.clf()
#     plt.scatter(vol_UM_a,PT_UM_a[:,1],c=PT_UM_a[:,0],vmin=0.0,vmax=136.0,lw=0.0)
#     plt.scatter(vol_TZ_a,PT_TZ_a[:,1],c=PT_TZ_a[:,0],vmin=0.0,vmax=136.0,lw=0.0)
#     plt.scatter(vol_LM_a,PT_LM_a[:,1],c=PT_LM_a[:,0],vmin=0.0,vmax=136.0,lw=0.0)
#
#     Vfac_a = np.linspace(.65,1.0,30)
#
#     plt.clf()
#     plt.plot(vol_ad_UM_a/V0_UM,PT_ad_UM_a[:,1]/T0_UM,'k-o',
#              vol_ad_TZ_a/V0_TZ,PT_ad_TZ_a[:,1]/T0_TZ,'r-o',
#              vol_ad_LM_a/V0_LM,PT_ad_LM_a[:,1]/T0_LM,'b-o',
#              Vfac_a,0.98*np.exp(-param0_d['gamma0']/param0_d['q']*(Vfac_a**param0_d['q']-1.0)),'r-')
#
#     np.exp(-param0_d['gamma0']/param0_d['q']*(Vfac_a**param0_d['q']-1.0))
#
#
#     # plt.clf()
#     # plt.plot(P_ad_UM_a, T_ad_UM_a,'k-',P_ad_TZ_a, T_ad_TZ_a,'r-',
#     #          P_ad_LM_a, T_ad_LM_a,'b-')
#
#     # Infer vol
#
#
#
#     plt.clf()
#     plt.plot(Tgeo_a[:,0],Tgeo_a[:,1],'-',lw=2,color=[1,.7,.7])
#     plt.plot(
#
#     #############################
#     # Shift 1600K isentrope in each phase region to remove discontinuities
#     # Fit this thermal profile with a single Gruneisen powerlaw expression
#     # Reapply shifts to obtain adiabat for each phase region
#     #############################
#
#     mask_all_a = mask_UM_a | mask_TZ_a | mask_LM_a
#     plt.figure()
#     plt.scatter(PT_a[mask_all_a,0],PT_a[mask_all_a,1],c=S_a[mask_all_a],lw=0,s=10)
#     plt.plot(Tgeo_a[:,0],Tgeo_a[:,1],'-',lw=2,color=[1,.7,.7])
#
#     S_func = interpolate.CloughTocher2DInterpolator(PT_a, S_a)
#     # Get 1600 K ref entropy level
#     Sref = S_func([0.0,1600])
#
#     sz=5
#     indr = np.argsort(np.random.rand(S_a.size))
#     plt.scatter(PT_a[indr,0],PT_a[indr,1],c=S_a[indr],lw=0,s=sz)
#     plt.plot(np.polyval(p_UM_TZ_bnd,Tlin_a),Tlin_a,'k-')
#     plt.plot(np.polyval(p_TZ_LM_bnd,Tlin_a),Tlin_a,'k-')
#     plt.clim(Sref-100,Sref+100)
#
#     # replace 1bar with 0.0 GPa for simplicity
#     PT_a[PT_a[:,0]==1e-4,0] = 0.0
#
#     Pbnds_UM_a = [0.0,13.5]
#     Pbnds_TZ_a = [13.8,18.8]
#     Pbnds_TZ2_a = [19.5,22.5]
#     Pbnds_LM0_a = [22.5,29.0]
#     Pbnds_LM_a = [29.0,120.0]
#     Pbnds_LM2_a = [121.0,200.0]
#
#     # Mesh grid creates a very noisy adiabat
#     Tgrid_a = np.linspace(1500,3100,300)
#     Pgrid_a = np.linspace(0,200,301)
#     Pmesh_a,Tmesh_a = np.meshgrid(Pgrid_a,Tgrid_a)
#     Smesh_a = S_func(Pmesh_a,Tmesh_a)
#     plt.scatter(Pmesh_a,Tmesh_a,c=Smesh_a,s=30,lw=0)
#     plt.clim(Sref-50,Sref+50)
#
#     plt.contour(PT_a[mask_all_a,0],PT_a[mask_all_a,1],S_a[mask_all_a],levels=[2540])
#     plt.contour(PT_a,PT_a,S_a,levels=[2540])
#
#     plt.scatter(xij[indr],yij[indr],c=zij[indr],lw=0,s=sz)
#     pass
#
#
#
#
#
#
#     def __init__(self):
#         eoslib.init_const( self.eos_d )
#         eoslib.set_modtype( [], [], self.eos_d )
#         eoslib.set_param( param_name_l, param_val_a, self.eos_d)
