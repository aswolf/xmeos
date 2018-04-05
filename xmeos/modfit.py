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

        self._param_names = np.array(param_names)
        self._param_values = np.array(param_values)
        self._param_errors = np.array(param_errors)
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
        if np.all(np.isnan(param_cov)):
            Nparam = param_cov.shape[0]
            param_errors = np.nan*np.ones(Nparam)
            param_corr = np.nan*np.ones((Nparam, Nparam))

        else:
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

    def expand(self, model2_pdf):
        param_names1 = self.param_names
        param_names2 = model2_pdf.param_names
        Nparam1 = len(param_names1)
        Nparam2 = len(param_names2)

        param_cov1 = self.param_cov
        param_cov2 = model2_pdf.param_cov

        param_means1 = self.param_values
        param_means2 = model2_pdf.param_values

        repeat_param = np.array([name2 in param_names1
                                for name2 in param_names2])
        assert np.all(~repeat_param), (
                           'Currently new pdf cannot repeat any parameters (only new parameters allowed).'
                       )

        param_names_all = np.hstack((param_names1, param_names2))
        param_means_all = np.hstack((param_means1, param_means2))

        Nparam = param_names_all.size
        param_cov_all = np.zeros((Nparam, Nparam))
        param_cov_all[:Nparam1, :Nparam1] = param_cov1
        param_cov_all[Nparam1:, Nparam1:] = param_cov2

        param_errors_all, param_corr_all = self._calc_normal_correlation(
            param_cov_all)

        priors = [self, model2_pdf]

        posterior_all = ModelPDF(param_names_all, param_means_all,
                                 param_errors_all, param_corr=param_corr_all,
                                 priors=priors)

        return posterior_all

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

    def _get_fitness_param_ind(self, fitness_params):
        if fitness_params is None:
            fitness_params = self.param_names

        fitness_params = np.array(fitness_params)

        fitness_param_ind = np.squeeze(np.array(
            [np.where(self.param_names==name)[0]
            for name in fitness_params], dtype=int))

        return fitness_param_ind

    def fit(self, fitness_fun, fitness_params=None,
            update_fitness_fun=None, fitness_metrics_fun=None,
            method='leastsq', nrepeat=6):

        fitness_param_ind = self._get_fitness_param_ind(fitness_params)

        assert method=='leastsq', (
            'Only leastsq method is supported for now.'
        )

        param_init = self.param_values
        Nparam = len(param_init)
        print(param_init[fitness_param_ind])
        print(param_init)

        print(self.ortho_transform(param_init))
        # from IPython import embed;embed();import ipdb;ipdb.set_trace()

        for i in np.arange(nrepeat):
            if update_fitness_fun is not None:
                fitness_fun = update_fitness_fun(param_init[fitness_param_ind])

            def posterior_fun(param_values,
                              fitness_param_ind=fitness_param_ind):
                likelihood_resid = fitness_fun(param_values[fitness_param_ind])
                prior_resid = self.ortho_transform(param_values)
                print(prior_resid)
                posterior_resid = np.hstack(
                    (likelihood_resid, prior_resid))
                return posterior_resid

            fit_tup = optimize.leastsq(posterior_fun, param_init,
                                       full_output=True)

            param_fit = fit_tup[0]
            cov_scl = fit_tup[1]
            info = fit_tup[2]
            param_init = param_fit

        print(posterior_fun(param_fit))
        print(self.ortho_transform(param_fit))

        resid = info['fvec']

        resid_var = np.var(resid)
        try:
            param_cov = resid_var*cov_scl
        except:
            param_cov = np.nan*np.ones((Nparam, Nparam))

        param_errors, param_corr = self._calc_normal_correlation(
            param_cov)

        # model_error, R2fit = residual_model_error(
        #     datamodel, apply_bulk_mod_wt, wt_vol,
        #     ignore_datatypes=ignore_datatypes)
        # print(param_err)
        # print(paramf_a)

        priors = [self]
        posterior = ModelPDF(self.param_names, param_fit, param_errors,
                             param_corr=param_corr, priors=priors)

        if fitness_metrics_fun is not None:
            posterior._fitness_metrics = fitness_metrics_fun(
                param_fit[fitness_param_ind])


        return posterior
#====================================================================
