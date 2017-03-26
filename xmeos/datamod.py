import numpy as np
import scipy as sp
import emcee
from abc import ABCMeta, abstractmethod
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from scipy import optimize
from xmeos import models
import copy

#====================================================================
#     xmeos: Xtal-Melt Equation of State package
#      data - library for storing equation of state models
#====================================================================

#====================================================================
# SECT 0: Admin functions
#====================================================================
def load_data(V=None, T=None, P=None, E=None,
              Verr=None, Terr=None, Perr=None, Eerr=None,
              mass_avg=None, groupID=None, mask=None ):
    data_d = {}

    if V is not None:
        data_d['V'] = V
    if T is not None:
        data_d['T'] = T
    if P is not None:
        data_d['P'] = P
    if E is not None:
        data_d['E'] = E

    if Verr is not None:
        data_d['Verr'] = Verr
    if Terr is not None:
        data_d['Terr'] = Terr
    if Perr is not None:
        data_d['Perr'] = Perr
    if Eerr is not None:
        data_d['Eerr'] = Eerr

    if groupID is not None:
        data_d['groupID'] = groupID
    if mask is not None:
        data_d['mask'] = mask
    if mass_avg is not None:
        data_d['mass_avg'] = mass_avg

    return data_d
#====================================================================
def init_param_distbn(fit_model_l, eos_d, fix_param_l=[]):
    param_distbn_d = {}

    scale_full_a = []
    paramkey_full_a = []
    for fit_model_type in fit_model_l:
        fit_mod = eos_d['modtype_d'][fit_model_type]
        # use sub?
        # iscale_a, iparamkey_a = fit_mod.get_param_scale_sub(eos_d)
        iscale_a, iparamkey_a = fit_mod.get_param_scale(eos_d)
        scale_full_a = np.concatenate((scale_full_a,iscale_a))
        paramkey_full_a = np.concatenate((paramkey_full_a,iparamkey_a))

    scale_a = []
    paramkey_a = []
    usedkey = []
    for iparamkey,iscale in zip(paramkey_full_a,scale_full_a):
        if (iparamkey not in paramkey_a) and \
                (iparamkey not in fix_param_l):
            paramkey_a.append(iparamkey)
            scale_a.append(iscale)
            usedkey.append(iparamkey)

    paramkey_a = np.array(paramkey_a)
    scale_a = np.array(scale_a)

    paramval_a = []
    for key in paramkey_a:
        paramval_a.append(eos_d['param_d'][key])

    paramval_a = np.array( paramval_a )

    param_distbn_d['param_key'] = paramkey_a
    param_distbn_d['param_scale'] = scale_a
    param_distbn_d['param_val'] = paramval_a

    return param_distbn_d
#====================================================================
def set_param_distbn_err( param_err_a, param_distbn_d, param_corr_a=None ):
    param_distbn_d['param_err'] = param_err_a

    if param_corr_a is None:
        param_corr_a = np.eye(param_err_a.size)

    param_distbn_d['param_corr'] = param_corr_a
    pass
#====================================================================
def init_datamod( data_d, prior_d, eos_d, fit_data_type=['P'] ):
    datamod_d = {}

    datamod_d['data_d'] = data_d
    datamod_d['prior_d'] = prior_d
    datamod_d['posterior_d'] = None
    datamod_d['eos_d'] = eos_d
    datamod_d['fit_data_type'] = fit_data_type

    return datamod_d
#====================================================================
def model_eval( data_type, V_a, T_a, datamod_d ):
    eos_d = datamod_d['eos_d']

    full_mod = eos_d['modtype_d']['FullMod']

    # print(data_type)
    # print(V_a)
    # print(T_a)

    if data_type == 'P':
        model_val_a = full_mod.press(V_a, T_a, eos_d)
    elif data_type == 'E':
        model_val_a = full_mod.energy(V_a, T_a, eos_d)
    elif data_type == 'dPdT':
        model_val_a = full_mod.dPdT(V_a, T_a, eos_d)
    else:
        assert False, data_type + ' is not a valid data type.'

    # print(model_val_a)
    # print('------')

    #elif data_type == 'dPdT':

    return model_val_a
#====================================================================
def calc_isotherm_press_min( Vinit, Tiso, eos_d ):
    TOL=1e-2
    full_mod = eos_d['modtype_d']['FullMod']

    V = Vinit

    V_a = V*(1.0+np.array([-TOL,0.0,+TOL]))
    P_a = full_mod.press( V_a, Tiso, eos_d )
    coef_a = np.polyfit(V_a,P_a,2)
    Vext = -coef_a[1]/(2*coef_a[0])
    Pext = np.polyval(coef_a,Vext)
    # if coef_a[0]>0:
    #     Vext = -coef_a[1]/(2*coef_a[0])
    # else:
    #     Vext = V+3*


    while (coef_a[0]>0) and (np.abs(Vext/V-1.0) > 2*TOL):
        V = Vext
        V_a = V*(1.0+np.array([-TOL,0.0,+TOL]))
        P_a = full_mod.press( V_a, Tiso, eos_d )
        coef_a = np.polyfit(V_a,P_a,2)
        if coef_a[0]>0:
            Vext = -coef_a[1]/(2*coef_a[0])
            Pext = np.polyval(coef_a,Vext)

    press_min = Pext
    return press_min
#====================================================================
def calc_resid_datamod( datamod_d, err_d={}, unweighted=False,
                       mask_neg_press=False ):
    """
    Error is a fraction of absolute scatter in datatype (std of P or E)
    """
    scale = 0.01

    eos_d = datamod_d['eos_d']
    full_mod = eos_d['modtype_d']['FullMod']
    # set parameter values
    # for key,val in zip(param_key,param_a):
    #     eos_d['param_d'][key] = val

    P_a = datamod_d['data_d']['P']
    V_a = datamod_d['data_d']['V']
    T_a = datamod_d['data_d']['T']

    ###########
    # protect from unphysical zero values
    ###########

    # Vmax = np.max(V_a)
    # P_phys_bnd_scale = 0.1
    # T_phys_bnd = np.array([1500.0,5000.0])
    # P_phys_bnd = np.array([calc_isotherm_press_min( Vmax, T_phys_bnd[0], eos_d ),
    #                        calc_isotherm_press_min( Vmax, T_phys_bnd[1], eos_d )])
    # phys_bnd_resid = np.exp(5*P_phys_bnd/P_phys_bnd_scale)


    # K constraints are not sufficient to ensure physical zero press values
    # K_mod_a = full_mod.bulk_modulus( V_a, T_a, eos_d )

    resid_a = []
    for data_type in datamod_d['fit_data_type']:
        idat_val_a = datamod_d['data_d'][data_type]

        # Use fixed error or Infer from spread in data
        if unweighted:
            ierr = 1.0
        elif err_d.has_key(data_type):
            ierr = err_d[data_type]
        else:
            ierr = scale*np.std(idat_val_a)

        if not unweighted:
            if np.size(ierr)==1:
                ierr = ierr*np.ones(idat_val_a.size)

            # TEMPORARILY remove overweighting of low-press points
            # since it forces an unknown error earlier
            # ierr[P_a<0] *= 1e8


            # if mask_neg_press:
            #     ierr[P_a<0] *= 1e8

        # ierr[K_mod_a < 0] *= 1e-10

        # Fitting dPdT may require too much accuracy
        # if data_type == 'dPdT':
        #     ierr[P_a>0] *= 1e+2


        # evaluate model
        imod_val_a = model_eval( data_type, V_a, T_a, datamod_d )

        # Cannot simple penalize negative thermal press
        if data_type == 'dPdT':
            dPdT_mod_a = imod_val_a
            dPdT_scl = 1e-1 #GPa/1000K
            mask_neg_dPdT = dPdT_mod_a<0
            ierr[mask_neg_dPdT] *= np.exp(-5*dPdT_mod_a[mask_neg_dPdT]/dPdT_scl)

        iresid_a = (imod_val_a-idat_val_a)/ierr

        # if data_type == 'dPdT':
        #     ierr[P_a>0] *= 1e+2
        resid_a = np.concatenate((resid_a,iresid_a))

    # resid_a = np.append(resid_a,phys_bnd_resid)
    return resid_a
#====================================================================
def model_eval_list( V_a, T_a, param_a, datamod_d ):
    """
    Error is a fraction of peak-to-peak difference
    """
    eos_d = datamod_d['eos_d']
    param_key = datamod_d['prior_d']['param_key']
    models.Control.set_params( param_key, param_a, eos_d )
    # set parameter values
    # for key,val in zip(param_key,param_a):
    #     eos_d['param_d'][key] = val

    model_val_l = []
    for data_type in datamod_d['fit_data_type']:
        imodel_val_a = model_eval( data_type, V_a, T_a, datamod_d )
        model_val_l.append( imodel_val_a )

    return model_val_l
#====================================================================
def set_fit_params( param_a, datamod_d ):
    eos_d = datamod_d['eos_d']
    param_key = datamod_d['prior_d']['param_key']
    models.Control.set_params( param_key, param_a, eos_d )
    pass
#====================================================================
def eval_resid_datamod( param_a, datamod_d, err_d={}, unweighted=False ):
    # print(param_a)
    set_fit_params( param_a, datamod_d )
    resid_a = calc_resid_datamod(datamod_d, err_d=err_d)
    return resid_a
#====================================================================
def eval_cost_fun( param_a, datamod_d, err_d ):
    set_fit_params( param_a, datamod_d )
    resid_a = calc_resid_datamod(datamod_d, err_d=err_d)
    costval = np.sum(resid_a**2)
    if np.isnan(costval):
        from IPython import embed; embed(); import ipdb; ipdb.set_trace()

    return costval
def lnprob( param_a, datamod_d, err_d ):
    return -0.5*eval_cost_fun( param_a , datamod_d, err_d )
#====================================================================
def residual_model_error( datamod_d ):
    eos_d = datamod_d['eos_d']
    param_key = datamod_d['prior_d']['param_key']

    Nparam = datamod_d['prior_d']['param_key'].size
    # calculate unweighted residuals
    resid_a = calc_resid_datamod(datamod_d,unweighted=True)
    N = datamod_d['data_d']['V'].size

    Nparam = len(datamod_d['prior_d']['param_key'])
    Ndat_typ = len(datamod_d['fit_data_type'])

    err_d={}
    for ind,data_type in enumerate(datamod_d['fit_data_type']):
        iresid_a = resid_a[ind*N:(ind+1)*N]
        Ndof = iresid_a.size-Nparam/Ndat_typ
        err_d[data_type] = np.sqrt(np.sum(iresid_a**2)/Ndof)
        # err_d[data_type] = np.sqrt(np.mean(iresid_a**2))

    return err_d
#====================================================================
def fit( datamod_d, nrepeat=6 ):

    eos_d = datamod_d['eos_d']

    param0_a = datamod_d['prior_d']['param_val']
    # from IPython import embed; embed(); import ipdb; ipdb.set_trace()

    # V_a = datamod_d['data_d']['V']
    # T_a = datamod_d['data_d']['T']
    # P_a = datamod_d['data_d']['P']
    # E_a = datamod_d['data_d']['E']

    # plt.clf()
    # plt.plot(V_a,P_a,'ko',V_a,model_eval( param0_a, 'P', datamod_d ),'rx')

    err_d = {}
    # plt.figure()
    # plt.clf()

    for i in np.arange(nrepeat):
        # fit_tup = optimize.leastsq(lambda param_a,
        #                            datamod_d=datamod_d,param0_a=param0_a:\
        #                            calc_resid_datamod( param_a, datamod_d ),
        #                            param0_a)
        fit_tup = optimize.leastsq(lambda param_a, datamod_d=datamod_d,param0_a=param0_a:\
                                   eval_resid_datamod( param_a, datamod_d, err_d=err_d ),
                                   param0_a, full_output=True)

        # Update error estimate from residuals
        err_d = residual_model_error( datamod_d )


        # plt.plot(fit_tup[2]['fvec'],'ro')
        # plt.draw()
        # plt.pause(.1)
        # plt.plot(fit_tup[2]['fvec'],'ko')

        paramf_a = fit_tup[0]
        param0_a = paramf_a

    fvec_a = fit_tup[2]['fvec']

    # Set final fit
    param_key = datamod_d['prior_d']['param_key']
    models.Control.set_params( param_key, paramf_a, eos_d )

    cov_scl = fit_tup[1]
    resid_var = np.var(fit_tup[2]['fvec'])
    # plt.figure()
    # plt.plot(fit_tup[2]['fvec'],'ko')
    # plt.pause(2)


    cov = resid_var*cov_scl
    paramerr_a = np.sqrt( np.diag(cov) )

    corr = cov/(np.expand_dims(paramerr_a,1)*paramerr_a)


    err_d = residual_model_error( datamod_d )

    posterior_d = copy.deepcopy(datamod_d['prior_d'])
    posterior_d['param_val'] = paramf_a
    posterior_d['param_err'] = paramerr_a
    posterior_d['corr'] = corr
    posterior_d['err'] = err_d

    datamod_d['posterior_d'] = posterior_d
    pass
#====================================================================
def eos_posterior_draw( datamod_d ):

    eos_draw_d = copy.deepcopy(datamod_d['eos_d'])
    posterior_d = datamod_d['posterior_d']
    param_val_a = posterior_d['param_val']
    param_err_a = posterior_d['param_err']
    corr_a = posterior_d['corr']

    cov_a= corr_a*(param_err_a*np.expand_dims(param_err_a,1))
    param_draw_a = sp.random.multivariate_normal(param_val_a,cov_a)


    param_key = posterior_d['param_key']
    models.Control.set_params( param_key, param_draw_a, eos_draw_d )

    return eos_draw_d
#====================================================================
def runmcmc( datamod_d, nwalkers_fac=3 ):
    from IPython import embed; embed(); import ipdb; ipdb.set_trace()

    posterior_d = datamod_d['posterior_d']
    param_val = posterior_d['param_val']
    param_err = posterior_d['param_err']
    corr = posterior_d['corr']
    Eerr = posterior_d['Eerr']
    Perr = posterior_d['Perr']

    cov = corr*(np.expand_dims(param_err,1)*param_err)

    ndim = param_val.size
    nwalkers = np.int(np.ceil(nwalkers_fac*ndim))

    # calc_resid_datamod( datamod_d, Eerr=Eerr, Perr=Perr )
    # lnprob(paramf_a, datamod_d, Eerr, Perr )

    param_key = posterior_d['param_key']


    # Initialize using approximate Laplace covariance matrix
    # Shrink by factor of 3 around best-fit to avoid getting stuck in improbable
    # places
    p0 = np.random.multivariate_normal(param_val,0.1*cov,nwalkers)

    sampler = emcee.EnsembleSampler( nwalkers, ndim, lnprob,
                                    args=[datamod_d, Eerr, Perr] )

    pos_warm = p0
    Nwarm = 100

    pos_warm, prob_warm, state_warm = sampler.run_mcmc(pos_warm,Nwarm)

    # sampler4 = emcee.EnsembleSampler( nwalkers, ndim, lnprob,
    #                                 args=[datamod_d, Eerr, Perr],threads=4 )
    # pos_warm, prob_warm, state_warm = sampler4.run_mcmc(pos_warm,Nwarm)


    # plt.figure()
    # plt.clf()

    # plt.clf()
    # plt.plot(sampler.lnprobability.T,'-')


    # twait=1.0
    # for i in np.arange(ndim):
    #     plt.clf()
    #     plt.plot(sampler.chain[:,:,i].T,'-')
    #     plt.ylabel(param_key[i])
    #     plt.xlabel('iter')
    #     plt.pause(twait)


    Ndraw = 301
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos_warm,Ndraw)

    # twait=1.0
    # for i in np.arange(ndim):
    #     plt.clf()
    #     plt.plot(sampler.chain[:,:,i].T,'-')
    #     plt.ylabel(param_key[i])
    #     plt.xlabel('iter')
    #     plt.pause(twait)

    # twait=1.0
    # for i in np.arange(ndim):
    #     plt.clf()
    #     plt.hist(sampler.chain[:,:,i].reshape((-1,)),30)
    #     plt.xlabel(param_key[i])
    #     plt.pause(twait)


    # samp_a = sampler.chain[:,10:,:].reshape((-1,ndim))
    # fig = corner.corner(samp_a[:,0:6], labels=param_key[0:6])
    # fig = corner.corner(samp_a[:,6:], labels=param_key[6:])

    datamod_d['posterior_d']['mcmc'] = sampler
    pass
#====================================================================

# datamod_d: prior_d, posterior_d, param_fix_l, data_d, eos_d, fit_data_type =
# ['E']
#
# fitopt_d: param_name_l, param_bnds_a,
# set_priors: covar matrix, fix_param,

# store data, adjust errors, multiple datasets, datagroups, mask
# load_data:
# store data, adjust errors, multiple datasets, datagroups, mask
#
#
class store_data(object):
    def __init__( self ):
        pass

    def get_resid_fun( self, eos_fun, eos_d, param_key_a, sys_state_tup,
                      val_a, err_a=1.0):

        # def resid_fun( param_in_a, eos_fun=eos_fun, eos_d=eos_d,
        #               param_key_a=param_key_a, sys_state_tup=sys_state_tup,
        #               val_a=val_a, err_a=err_a ):

        #     # Don't worry about transformations right now
        #     param_a = param_in_a

        #     # set param value in eos_d dict
        #     globals()['set_param']( param_key_a, param_a, eos_d )

        #     # Take advantage of eos model function input format
        #     #   uses tuple expansion for input arguments
        #     mod_val_a = eos_fun( *(sys_state_tup+(eos_d,)) )
        #     resid_a = (mod_val_a - val_a)/err_a
        #     return resid_a

        wrap_eos_fun = self.get_wrap_eos_fun( eos_fun, eos_d, param_key_a )

        def resid_fun( param_in_a, wrap_eos_fun=wrap_eos_fun,
                      sys_state_tup=sys_state_tup,
                      val_a=val_a, err_a=err_a ):

            mod_val_a = wrap_eos_fun( param_in_a, sys_state_tup )
            resid_a = (mod_val_a - val_a)/err_a
            return resid_a

        return resid_fun

    def get_wrap_eos_fun( self, eos_fun, eos_d, param_key_a ):

        def wrap_eos_fun(param_in_a, sys_state_tup, eos_fun=eos_fun,
                         eos_d=eos_d, param_key_a=param_key_a ):

            # Don't worry about transformations right now
            param_a = param_in_a

            # set param value in eos_d dict
            models.Control.set_params( param_key_a, param_a, eos_d )

            # Take advantage of eos model function input format
            #   uses tuple expansion for input arguments
            mod_val_a = eos_fun( *(sys_state_tup+(eos_d,)) )
            return mod_val_a

        return wrap_eos_fun

#====================================================================
# SECT 1: Fitting Routines
#====================================================================
class ModFit(object):
    def __init__( self ):
        pass

    def get_resid_fun( self, eos_fun, eos_d, param_key_a, sys_state_tup,
                      val_a, err_a=1.0):

        # def resid_fun( param_in_a, eos_fun=eos_fun, eos_d=eos_d,
        #               param_key_a=param_key_a, sys_state_tup=sys_state_tup,
        #               val_a=val_a, err_a=err_a ):

        #     # Don't worry about transformations right now
        #     param_a = param_in_a

        #     # set param value in eos_d dict
        #     globals()['set_param']( param_key_a, param_a, eos_d )

        #     # Take advantage of eos model function input format
        #     #   uses tuple expansion for input arguments
        #     mod_val_a = eos_fun( *(sys_state_tup+(eos_d,)) )
        #     resid_a = (mod_val_a - val_a)/err_a
        #     return resid_a

        wrap_eos_fun = self.get_wrap_eos_fun( eos_fun, eos_d, param_key_a )

        def resid_fun( param_in_a, wrap_eos_fun=wrap_eos_fun,
                      sys_state_tup=sys_state_tup,
                      val_a=val_a, err_a=err_a ):

            mod_val_a = wrap_eos_fun( param_in_a, sys_state_tup )
            resid_a = (mod_val_a - val_a)/err_a
            return resid_a

        return resid_fun

    def get_wrap_eos_fun( self, eos_fun, eos_d, param_key_a ):

        def wrap_eos_fun(param_in_a, sys_state_tup, eos_fun=eos_fun,
                         eos_d=eos_d, param_key_a=param_key_a ):

            # Don't worry about transformations right now
            param_a = param_in_a

            # set param value in eos_d dict
            Control.set_params( param_key_a, param_a, eos_d )

            # Take advantage of eos model function input format
            #   uses tuple expansion for input arguments
            mod_val_a = eos_fun( *(sys_state_tup+(eos_d,)) )
            return mod_val_a

        return wrap_eos_fun
#====================================================================
# SECT N: Code Utility Functions
#====================================================================
def fill_array( var1, var2 ):
    """
    fix fill_array such that it returns two numpy arrays of equal size

    use numpy.full_like

    """
    var1_a = np.asarray( var1 )
    var2_a = np.asarray( var2 )

    if var1_a.shape==():
        var1_a = np.asarray( [var1] )
    if var2_a.shape==():
        var2_a = np.asarray( [var2] )

    # Begin try/except block to handle all cases for filling an array
    while True:
        try:
            assert var1_a.shape == var2_a.shape
            break
        except: pass
        try:
            var1_a = np.full_like( var2_a, var1_a )
            break
        except: pass
        try:
            var2_a = np.full_like( var1_a, var2_a )
            break
        except: pass

        # If none of the cases properly handle it, throw error
        assert False, 'var1 and var2 must both be equal shape or size=1'

    return var1_a, var2_a
#====================================================================

