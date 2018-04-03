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
#      modfit: A bayesian model-fitting module
#====================================================================

#====================================================================
class ModelPDF():
    def __init__(self, param_names, param_values, param_errors,
                 param_corr=None, priors=None):

        Nparam = len(param_names)
        if param_corr is None:
            param_corr = np.eye(Nparam)

        self._validate_param_props(Nparam, param_names, param_values,
                                   param_errors, param_corr)
        self._init_pdf_summary(param_names, param_values, param_errors,
                               param_corr)
        self._init_normal_props(param_errors, param_corr)
        self._init_priors(priors)
        return

    def _init_priors(self, priors):
        if priors is None:
            priors = []

        self._priors = priors
        return

    def _validate_param_props(self, Nparam, param_names, param_values,
                              param_errors, param_corr):

        assert len(param_values)==Nparam, (
            'param_value must be given for every parameter'
        )
        assert len(param_errors)==Nparam, (
            'param_errors must be given for every parameter'
        )

        assert param_corr.shape == (Nparam, Nparam), (
            'param_corr must be a square matrix defining correlations for '
            'every parameter pair.'
        )
        return

    def _init_pdf_summary(self, param_names, param_values, param_errors,
                          param_corr):
        param_tbl = pd.DataFrame(columns=['name','value','error'])
        for ind,(name, value, error) in enumerate(
            zip(param_names, param_values, param_errors)):
            param_tbl.loc[ind] = [name, value, error]

        self._param_names = param_names
        self._param_values = param_values
        self._param_errors = param_errors
        self._param_table = param_tbl
        self._param_corr = param_corr

    def _init_normal_props(self, param_errors, param_corr):
        param_cov, param_hess = self._calc_normal_covariance(
            param_errors, param_corr)

        eig_vec, eig_scale = self._diagonalize_pdf(param_cov)
        self._param_cov = param_cov
        self._param_hess = param_hess
        self._ortho_vectors = eig_vec
        self._ortho_scales = eig_scale
        return

    def _calc_normal_covariance(self, param_errors, param_corr):
        param_cov= param_corr*(param_errors*np.expand_dims(param_errors,1))
        param_hess = np.linalg.pinv(param_cov)
        return param_cov, param_hess

    def _calc_normal_correlation(self, param_cov):
        param_errors = np.sqrt(np.diag(param_cov))
        param_cov_scl = np.dot(param_errors[:, np.newaxis],
                               param_errors[np.newaxis, :])

        param_corr = param_cov/param_cov_scl
        return param_errors, param_corr

    def _diagonalize_pdf(self, param_cov):
        u, s, vh = np.linalg.svd(param_cov)
        eig_scale = np.sqrt(s)
        eig_vec = vh.T
        return eig_vec, eig_scale

    #=========================================

    @property
    def param_table(self):
        return self._param_table

    @property
    def notes(self):
        return self._notes

    @property
    def priors(self):
        return self._priors

    @property
    def param_names(self):
        return self._param_names

    @property
    def param_values(self):
        return self._param_values

    @property
    def param_errors(self):
        return self._param_errors

    @property
    def param_corr(self):
        return self._param_corr

    @property
    def param_cov(self):
        return self._param_cov

    @property
    def param_hess(self):
        return self._param_hess

    @property
    def ortho_vectors(self):
        return self._ortho_vectors

    @property
    def ortho_scales(self):
        return self._ortho_scales


    @property
    def fitness_metrics(self):
        return self._fitness_metrics

    def draw(self, Ndraw=1):
        param_draw = np.squeeze(sp.random.multivariate_normal(
            self.param_values, self.param_cov, Ndraw))
        return param_draw

    def cost_fun(self, param_values):
        means = self.param_values
        hess = self.param_hess

        param_dev = param_values - means
        costval = np.dot(param_dev,np.dot(hess,param_dev))
        return costval

    # def prob(self, param_values, param_names=None):
    #     return

    # def log_prob(self, param_values, param_names=None):
    #     return

    def ortho_transform(self, param_values):
        param_values = np.array(param_values)
        means = self.param_values
        param_dev = param_values - means
        ortho_values = np.dot(param_dev, self.ortho_vectors)/self.ortho_scales
        return ortho_values

    def expand(self, model2):
        return

    def reorder(self, param_names):
        return

    def constrain(self, constraint_pdf):
        mean_1 = self.param_values
        hess_1 = self.param_hess

        mean_2 = constraint_pdf.param_values
        hess_2 = constraint_pdf.param_hess

        hess_joint = hess_1 + hess_2
        cov_joint = np.linalg.inv(hess_joint)
        mean_joint = np.dot(
            cov_joint, np.dot(hess_1, mean_1) + np.dot(hess_2, mean_2))

        errors_joint, corr_joint = self._calc_normal_correlation(cov_joint)

        priors = [self, constraint_pdf]
        posterior = ModelPDF(self.param_names, mean_joint, errors_joint,
                             param_corr=corr_joint, priors=priors)

        return posterior

    def fit(self, fitness_fun, update_fitness_fun=None,
            fitness_metrics_fun=None, method='leastsq', nrepeat=6):
        assert method=='leastsq', (
            'Only leastsq method is supported for now.'
        )

        param_init = self.param_values

        for i in np.arange(nrepeat):
            if update_fitness_fun is not None:
                fitness_fun = update_fitness_fun(param_init)

            def posterior_fun(param_values):
                likelihood_resid = fitness_fun(param_values)
                prior_resid = self.ortho_transform(param_values)
                posterior_resid = np.hstack(
                    (likelihood_resid,prior_resid))
                return posterior_resid

            fit_tup = optimize.leastsq(posterior_fun, param_init,
                                       full_output=True)

            param_fit = fit_tup[0]
            cov_scl = fit_tup[1]
            info = fit_tup[2]
            param_init = param_fit


        resid = info['fvec']
        resid_var = np.var(resid)
        try:
            cov = resid_var*cov_scl
            param_errors = np.sqrt(np.diag(cov))
            corr = cov/(np.expand_dims(param_errors,1)*param_errors)
        except:
            cov = None
            param_errors = np.nan*param_fit
            corr = None


        # model_error, R2fit = residual_model_error(
        #     datamodel, apply_bulk_mod_wt, wt_vol,
        #     ignore_datatypes=ignore_datatypes)
        # print(param_err)
        # print(paramf_a)
        priors = [self]
        posterior = ModelPDF(self.param_names, param_fit, param_errors,
                          param_corr=corr, priors=priors)

        if fitness_metrics_fun is not None:
            posterior._fitness_metrics = fitness_metrics_fun(param_fit)


        # posterior = make_model_pdf(datamodel['fit_params'],
        #                           paramf_a, param_err,
        #                           param_corr=corr, fit_error=model_error,
        #                           R2fit=R2fit)
        # datamodel['posterior'] = posterior
        return posterior
#====================================================================
