import numpy as np
import scipy as sp
from abc import ABCMeta, abstractmethod
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

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

    datamod_d['prior'] = prior_d
    datamod_d['posterior'] = None
    datamod_d['data_d'] = data_d
    datamod_d['prior_d'] = prior_d
    datamod_d['eos_d'] = eos_d
    datamod_d['fit_data_type'] = fit_data_type

    return datamod_d
#====================================================================
def calc_resid_datamod( param_a, datamod_d ):
    eos_d = datamod_d['eos_d']
    param_key = datamod_d['prior_d']['param_key']
    prior_val = datamod_d['prior_d']['param_val']
    # set parameter values
    for key,val in zip(param_key,param_a):
        eos_d['param_d'][key] = val

    resid_a = []
    for data_type in datamod_d['fit_data_type']:
        idat_val_a = datamod_d['data_d'][data_type]
        ierr = np.ptp(idat_val_a)*np.ones(idat_val_a.shape)
        # ierr = datamod_d['data_d'][data_type+'err']
        # Ttr = 3100
        # ierr[datamod_d['data_d']['T']<=Ttr] *= 0.3
        V0 = eos_d['param_d']['V0']
        ierr[datamod_d['data_d']['V']>=0.99*V0] *= 0.6
        if data_type == 'P':
            # ierr *= 1.0
            # Ptr = 15.0
            # ierr[idat_val_a <= Ptr] *= 0.1
            # ierr[datamod_d['data_d']['T']<=Ttr] *= 0.1

            imod_val_a = eos_d['modtype_d']['FullMod'].press\
                (datamod_d['data_d']['V'], datamod_d['data_d']['T'], eos_d)
        elif data_type == 'E':
            imod_val_a = eos_d['modtype_d']['FullMod'].energy\
                (datamod_d['data_d']['V'], datamod_d['data_d']['T'], eos_d)

        iresid_a = (imod_val_a-idat_val_a)/ierr

        resid_a = np.concatenate((resid_a,iresid_a))


    return resid_a
#====================================================================
def calc_cost_fun( param_a, datamod_d ):
    resid_a = calc_resid_datamod(param_a, datamod_d)
    costval = np.sum(resid_a**2)
    return costval
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
            Control.set_params( param_key_a, param_a, eos_d )

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

