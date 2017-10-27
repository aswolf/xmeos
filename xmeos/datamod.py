import numpy as np
import pandas as pd
import scipy as sp
import emcee
from abc import ABCMeta, abstractmethod
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from scipy import optimize
from xmeos import models
import copy

from scipy import optimize

#====================================================================
#     xmeos: Xtal-Melt Equation of State package
#      data - library for storing equation of state models
#====================================================================

#====================================================================
# SECT 0: Admin functions
#====================================================================
def load_data(title=None, datasource=None,
              V=None, T=None, P=None, E=None,
              Verr=None, Terr=None, Perr=None, Eerr=None,
              Vconv=1, Tconv=1, Pconv=1, Econv=1,
              mass_avg=None, groupID=None, mask=None):
    data = {}
    data['table'] = pd.DataFrame()

    if title is not None:
        data['title'] = title
    if datasource is not None:
        data['datasource'] = datasource

    if V is not None:
        data['table']['V'] = V*Vconv
    if T is not None:
        data['table']['T'] = T*Tconv
    if P is not None:
        data['table']['P'] = P*Pconv
    if E is not None:
        data['table']['E'] = E*Econv

    if Verr is not None:
        data['table']['Verr'] = Verr*Vconv
    else:
        data['table']['Verr'] = 0
    if Terr is not None:
        data['table']['Terr'] = Terr*Tconv
    if Perr is not None:
        data['table']['Perr'] = Perr*Pconv
    if Eerr is not None:
        data['table']['Eerr'] = Eerr*Econv

    if groupID is not None:
        data['groupID'] = groupID
    if mask is not None:
        data['mask'] = mask
    if mass_avg is not None:
        data['mass_avg'] = mass_avg

    return data
#====================================================================
def _get_err_scale(data):
    err_scale = {}
    err_scale['P'] = np.std(data['table']['P'])
    err_scale['E'] = np.std(data['table']['E'])
    err_scale['V'] = np.std(data['table']['V'])
    err_scale['T'] = np.std(data['table']['T'])
    return err_scale
#====================================================================
def init_datamodel(data, eos_mod):
    datamodel = {}
    datamodel['data'] = data
    datamodel['eos_mod'] = eos_mod
    datamodel['err_scale'] = _get_err_scale(data)
    param_names = eos_mod.param_names
    datamodel['param_names'] = param_names
    datamodel['param_isfree'] = np.tile(False, len(param_names))
    datamodel['fit_params'] = []
    datamodel['bulk_mod_wt'] = None
    datamodel['posterior'] = None
    return datamodel
#====================================================================
def select_fit_params(datamodel, fit_calcs, fix_params=[]):
    eos_mod = datamodel['eos_mod']

    if fit_calcs=='all':
        fit_calcs = eos_mod.calculators.keys()

    calc_params = eos_mod.get_calc_params()
    fit_params = []
    for calc_name in fit_calcs:
        for param in calc_params[calc_name]:
            if param not in fit_params:
                fit_params.append(param)

    for param in fit_params:
        if param in fix_params:
            fit_params.remove(param)

    param_names = eos_mod.param_names
    param_isfree = np.tile(False, len(param_names))
    for ind, name in enumerate(datamodel['param_names']):
        if name in fit_params:
            param_isfree[ind] = True

    datamodel['param_isfree'] = param_isfree
    datamodel['fit_params'] = fit_params
    # datamodel['fit_param_values'] = get_fit_params(datamodel)
    pass
#====================================================================
def update_bulk_mod_wt(datamodel):
    eos_mod = datamodel['eos_mod']
    data = datamodel['data']

    V_a = data['table']['V']
    T_a = data['table']['T']

    K_a = eos_mod.bulk_modulus(V_a, T_a)
    datamodel['bulk_mod_wt'] = K_a
    pass
#====================================================================
def calc_resid(datamodel, detail_output=False):
    """
    Calculate model residuals

    - By default, return P and E residuals (at given V and T)
    - if bulk_mod_wt is set, then return approximate V and E residuals (at given P and T)
        - this uses approximate reweighting by bulk modulus to transform P residuals into volume residuals
    """

    output = {}

    tbl = datamodel['data']['table']
    eos_mod = datamodel['eos_mod']
    err_scale = datamodel['err_scale']

    # mask = np.array(tbl['P']>0)
    V_a = np.array(tbl['V'])
    T_a = np.array(tbl['T'])
    P_a = np.array(tbl['P'])
    Perr_a = np.array(tbl['Perr'])
    E_a = np.array(tbl['E'])
    Eerr_a = np.array(tbl['Eerr'])
    if datamodel['bulk_mod_wt'] is None:
        bulk_mod_wt = None
    else:
        bulk_mod_wt = datamodel['bulk_mod_wt']

    Pmod = eos_mod.press(V_a, T_a)
    delP = Pmod - P_a
    output['P'] = delP

    if bulk_mod_wt is not None:
        delV = - V_a*delP/bulk_mod_wt
        output['V'] = delV
        Vadj_a = V_a - delV
        resid_P = delV/err_scale['V']
    else:
        Vadj_a = V_a
        resid_P = delP/err_scale['P']

    Emod = eos_mod.internal_energy(Vadj_a, T_a)
    delE = Emod - E_a
    output['E'] = delE
    resid_E = delE/err_scale['E']

    resid_a = np.concatenate((resid_P,resid_E))

    if detail_output==True:
        return output
    else:
        return resid_a
#====================================================================
def set_fit_params(param_a, datamodel):
    eos_mod = datamodel['eos_mod']
    fit_params = datamodel['fit_params']
    eos_mod.set_param_values(param_a, param_names=fit_params)
    pass
#====================================================================
def get_fit_params(datamodel):
    eos_mod = datamodel['eos_mod']
    fit_params = datamodel['fit_params']
    return eos_mod.get_param_values(param_names=fit_params)
#====================================================================
def fit(datamodel, nrepeat=6, apply_bulk_mod_wt=False):
    param0_a = get_fit_params(datamodel)

    if apply_bulk_mod_wt:
        update_bulk_mod_wt(datamodel)
    else:
        datamodel['bulk_mod_wt'] = None

    for i in np.arange(nrepeat):
        def resid_fun(param_a, datamodel=datamodel):
            set_fit_params(param_a, datamodel)
            resid_a = calc_resid(datamodel)
            return resid_a

        fit_tup = optimize.leastsq(resid_fun, param0_a, full_output=True)

        paramf_a = fit_tup[0]
        cov_scl = fit_tup[1]
        info = fit_tup[2]

        param0_a = paramf_a

        if apply_bulk_mod_wt:
            update_bulk_mod_wt(datamodel)
        else:
            datamodel['bulk_mod_wt'] = None

    resid_a = info['fvec']
    resid_var = np.var(resid_a)
    cov = resid_var*cov_scl
    param_err = np.sqrt(np.diag(cov))
    corr = cov/(np.expand_dims(param_err,1)*param_err)

    error, R2fit = residual_model_error(datamodel, apply_bulk_mod_wt)

    posterior = {}
    posterior['param_names'] = datamodel['fit_params']
    posterior['param_val'] = paramf_a
    posterior['param_err'] = param_err
    posterior['corr'] = corr
    posterior['err'] = error
    posterior['R2fit'] = R2fit

    datamodel['posterior'] = posterior
    pass
#====================================================================
def residual_model_error(datamodel, apply_bulk_mod_wt):
    eos_mod = datamodel['eos_mod']
    err_scale = datamodel['err_scale']

    update_bulk_mod_wt(datamodel)
    output = calc_resid(datamodel, detail_output=True)

    Nparam = len(datamodel['fit_params'])
    # calculate unweighted residuals

    Ndat = output['V'].size
    # Ndat = datamodel['data']['table'].shape[0]
    # Ndat = resid_a.size/2
    Ndof = Ndat - Nparam/2 # free parameters are shared between 2 data types

    error = {}
    R2fit = {}
    for datatype in output:
        abs_resid = output[datatype]
        error[datatype] = np.sqrt(np.sum(abs_resid**2)/Ndof)
        R2fit[datatype] = 1 - np.var(abs_resid)/err_scale[datatype]

    if not apply_bulk_mod_wt:
        datamodel['bulk_mod_wt'] = None

    return error, R2fit
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
def eval_resid_datamod( param_a, datamod_d, err_d={}, unweighted=False ):
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
#====================================================================
def lnprob( param_a, datamod_d, err_d ):
    return -0.5*eval_cost_fun( param_a , datamod_d, err_d )
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
