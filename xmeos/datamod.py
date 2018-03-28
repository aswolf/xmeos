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
from collections import OrderedDict

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
              mass_avg=None, groupID=None, trust=None):
    data = {}
    data['table'] = pd.DataFrame()

    if title is not None:
        data['title'] = title
    if datasource is not None:
        data['datasource'] = datasource

    data['exp_constraint'] = None

    if V is not None:
        data['table']['V'] = V*Vconv
    if T is not None:
        data['table']['T'] = T*Tconv
    if P is not None:
        data['table']['P'] = P*Pconv
    if E is not None:
        data['table']['E'] = E*Econv
    if trust is not None:
       data['table']['trust'] = trust
    else:
       data['table']['trust'] = np.tile(True, V.size)

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
    if mass_avg is not None:
        data['mass_avg'] = mass_avg

    return data
#====================================================================
def _get_err_scale(data):
    err_scale = {}
    err_scale['V'] = np.std(data['table']['V'])
    err_scale['P'] = np.std(data['table']['P'])
    err_scale['PV'] = None
    if 'E' in data['table'].columns:
        err_scale['E'] = np.std(data['table']['E'])
    else:
        err_scale['E'] = 1
    if 'T' in data['table'].columns:
        err_scale['T'] = np.std(data['table']['T'])
    else:
        err_scale['T'] = 1
    return err_scale
#====================================================================
def init_datamodel(data, eos_mod):
    datamodel = {}
    datamodel['data'] = data
    datamodel['eos_mod'] = eos_mod

    if 'thermal' in eos_mod.calculators:
        isthermal=True
    else:
        isthermal=False
    datamodel['isthermal'] = isthermal
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
            if (param not in fit_params) and (param not in fix_params):
                fit_params.append(param)

    # for param in fit_params:
    #     if param in fix_params:
    #         fit_params.remove(param)

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
def update_bulk_mod_wt(datamodel, wt_vol=0.5):
    eos_mod = datamodel['eos_mod']
    data = datamodel['data']

    V_a = data['table']['V']

    if datamodel['isthermal']:
        T_a = data['table']['T']
        K_a = eos_mod.bulk_mod(V_a, T_a)
    else:
        K_a = eos_mod.bulk_mod(V_a)

    datamodel['bulk_mod_wt'] = K_a
    err_scale = datamodel['err_scale']

    err_scale['PV'] = 1/np.sqrt(
        (1-wt_vol)/err_scale['P']**2 + wt_vol*(V_a/K_a/err_scale['V'])**2)
    datamodel['bulk_mod_wt'] = K_a
    pass
#====================================================================
def set_exp_constraint(data, V, T, P, KT=None, alpha=None, wt=1e3):
    exp_constraint = {}
    exp_constraint['V'] = np.array(V)
    exp_constraint['T'] = np.array(T)
    exp_constraint['P'] = np.array(P)
    exp_constraint['KT'] = KT
    exp_constraint['alpha'] = alpha
    exp_constraint['wt'] = wt
    data['exp_constraint'] = exp_constraint
    pass
#====================================================================
def calc_resid(datamodel, detail_output=False, apply_prior_wt=False):
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
    exp_constraint = datamodel['data']['exp_constraint']

    # mask = np.array(tbl['P']>0)
    trust = tbl['trust']
    V_a = np.array(tbl['V'][trust])
    P_a = np.array(tbl['P'][trust])

    if datamodel['isthermal']:
        T_a = np.array(tbl['T'][trust])

    if 'E' in tbl.columns:
        hasenergy = True
        E_a = np.array(tbl['E'][trust])
    else:
        hasenergy = False

    # Perr_a = np.array(tbl['Perr'])
    # Eerr_a = np.array(tbl['Eerr'])
    if datamodel['bulk_mod_wt'] is None:
        bulk_mod_wt = None
    else:
        bulk_mod_wt = datamodel['bulk_mod_wt'][trust]

    if datamodel['isthermal']:
        Pmod = eos_mod.press(V_a, T_a)
    else:
        Pmod = eos_mod.press(V_a)

    delP = Pmod - P_a
    output['P'] = delP

    if bulk_mod_wt is not None:
        delV = - V_a*delP/bulk_mod_wt
        output['V'] = delV
        # Vadj_a = V_a - delV
        # resid_P = delV/err_scale['V']
        resid_P = delP/err_scale['PV'][trust]
    else:
        # Vadj_a = V_a
        resid_P = delP/err_scale['P']

    delE = np.zeros(V_a.shape)
    resid_E = np.zeros(V_a.shape)
    if hasenergy:
        try:
            if datamodel['isthermal']:
                Emod = eos_mod.internal_energy(V_a, T_a)
            else:
                Emod = eos_mod.internal_energy(V_a)

            delE = Emod - E_a
            resid_E = delE/err_scale['E']
            resid_E[np.isnan(E_a)] = 0
        except:
            pass



    output['E'] = delE
    resid_all = [resid_P,resid_E]

    if exp_constraint is not None:
        Vexp = exp_constraint['V']
        Texp = exp_constraint['T']
        Pexp = exp_constraint['P']
        wt = exp_constraint['wt']
        KTexp = exp_constraint['KT']
        alphaexp = exp_constraint['alpha']
        Pexp_mod = eos_mod.press(Vexp, Texp)


        residPexp = wt*(Pexp_mod-Pexp)/err_scale['P']
        resid_all.append(residPexp)

        if KTexp is not None:
            KT_mod = eos_mod.bulk_mod(Vexp, Texp)
            residKT = wt*(KT_mod-KTexp)/err_scale['P']
            resid_all.append(residKT)

        if alphaexp is not None:
            alpha_mod = eos_mod.thermal_exp(Vexp, Texp)
            residalpha = wt*np.log(alpha_mod/alphaexp)*err_scale['T']/1e4
            resid_all.append(residalpha)


    # if apply_prior_wt:
    #     prior = datamodel['prior']
    #     corr = prior['corr']
    #     param_err = prior['param_err']
    #     cov = np.dot(param_err[:,np.newaxis],param_err[np.newaxis,:])*corr

    resid_a = np.concatenate(resid_all)
    # from IPython import embed;embed();import ipdb as pdb; pdb.set_trace()

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
def fit(datamodel, nrepeat=6, apply_bulk_mod_wt=False, wt_vol=0.5,
        apply_prior_wt=False):
    if not datamodel['fit_params']:
        assert False, 'fit_params is currently empty. Use select_fit_params to set the fit parameters.'

    # from IPython import embed;embed();import ipdb as pdb;pdb.set_trace()

    param0_a = get_fit_params(datamodel)

    if apply_bulk_mod_wt:
        update_bulk_mod_wt(datamodel, wt_vol=wt_vol)
    else:
        datamodel['bulk_mod_wt'] = None

    for i in np.arange(nrepeat):
        def resid_fun(param_a, datamodel=datamodel,
                      apply_prior_wt=apply_prior_wt):
            set_fit_params(param_a, datamodel)
            resid_a = calc_resid(datamodel, apply_prior_wt=apply_prior_wt)
            return resid_a

        fit_tup = optimize.leastsq(resid_fun, param0_a, full_output=True)

        paramf_a = fit_tup[0]
        cov_scl = fit_tup[1]
        info = fit_tup[2]

        param0_a = paramf_a

        if apply_bulk_mod_wt:
            update_bulk_mod_wt(datamodel, wt_vol=wt_vol )
        else:
            datamodel['bulk_mod_wt'] = None

    resid_a = info['fvec']
    resid_var = np.var(resid_a)
    try:
        cov = resid_var*cov_scl
        param_err = np.sqrt(np.diag(cov))
        corr = cov/(np.expand_dims(param_err,1)*param_err)
    except:
        cov = None
        param_err = np.nan*paramf_a
        corr = None

    model_error, R2fit = residual_model_error(datamodel,
                                              apply_bulk_mod_wt, wt_vol)
    # print(param_err)
    # print(paramf_a)



    param_tbl = pd.DataFrame(columns=['name','value','error'])
    for ind,(name, value, error) in enumerate(
        zip(datamodel['fit_params'], paramf_a, param_err)):
        param_tbl.loc[ind] = [name, value, error]


    posterior = OrderedDict()
    posterior['param_names'] = datamodel['fit_params']
    posterior['param_val'] = paramf_a
    posterior['param_err'] = param_err
    posterior['param_tbl'] = param_tbl
    posterior['corr'] = corr
    posterior['fit_err'] = model_error
    posterior['R2fit'] = R2fit

    datamodel['posterior'] = posterior
    pass
#====================================================================
def residual_model_error(datamodel, apply_bulk_mod_wt, wt_vol):
    eos_mod = datamodel['eos_mod']
    err_scale = datamodel['err_scale']

    update_bulk_mod_wt(datamodel, wt_vol=wt_vol)
    output = calc_resid(datamodel, detail_output=True)

    Nparam = len(datamodel['fit_params'])
    # calculate unweighted residuals

    Ndat = output['V'].size
    # Ndat = datamodel['data']['table'].shape[0]
    # Ndat = resid_a.size/2
    Ndof = Ndat - Nparam/2 # free parameters are shared between 2 data types

    model_error = {}
    R2fit = {}
    for datatype in output:
        abs_resid = output[datatype]
        model_error[datatype] = np.sqrt(np.sum(abs_resid**2)/Ndof)
        R2fit[datatype] = 1 - np.var(abs_resid)/err_scale[datatype]**2

    if not apply_bulk_mod_wt:
        datamodel['bulk_mod_wt'] = None

    return model_error, R2fit
#====================================================================
def draw_from_posterior(datamodel, Ndraw=100):
    posterior = datamodel['posterior']
    param_names = posterior['param_names']
    param_val = posterior['param_val']
    param_err = posterior['param_err']
    corr = posterior['corr']
    cov= corr*(param_err*np.expand_dims(param_err,1))
    param_draw = sp.random.multivariate_normal(param_val, cov, Ndraw)

    return param_draw, param_names
#====================================================================
def posterior_prediction(V, T, fun_name, datamodel, Ndraw=100,
                         percentile=[16, 50, 84]):
    eos_draw = copy.deepcopy(datamodel['eos_mod'])
    param_draw, param_names = draw_from_posterior(datamodel, Ndraw=Ndraw)

    V, T = models.fill_array(V, T)

    val = np.zeros((Ndraw, V.size))
    fun = eos_draw.__getattribute__(fun_name)
    for ind, iparam_draw in enumerate(param_draw):
        eos_draw.set_param_values(iparam_draw, param_names=param_names)
        val[ind, :] = fun(V, T)


    bounds = np.percentile(val, percentile, axis=0)

    return bounds, val
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
