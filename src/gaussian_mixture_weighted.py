#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 201802 LIU Wangsheng; Email: awang.signup@gmail.com
import numpy as np
from scipy import linalg
from scipy.special import logsumexp
from sklearn import cluster
import warnings


def _estimate_gaussian_covariances_full(resp, X, w, nk, means, reg_covar):
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(w * resp[:, k] * diff.T, diff) / nk[k]
        covariances[k].flat[::n_features + 1] += reg_covar
    return covariances


def _estimate_gaussian_parameters(X, w, resp, reg_covar, covariance_type):
    n_samples, n_components = resp.shape
    _, n_features = X.shape
    nk = np.sum(resp * np.tile(w, (n_components, 1)).T, axis=0)
    X_weighted = X * np.tile(w, (n_features, 1)).T
    means = np.dot(resp.T, X_weighted) / np.tile(nk, (n_features, 1)).T
    covariances = {"full": _estimate_gaussian_covariances_full
                   }[covariance_type](resp, X, w, nk, means, reg_covar)
    return nk, means, covariances


def _compute_precision_cholesky(covariances, covariance_type):
    if covariance_type in 'full':
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = linalg.cholesky(covariance, lower=True)
            except linalg.LinAlgError:
                raise ValueError('linalg.LinAlgError')
            precisions_chol[k] = linalg.solve_triangular(cov_chol,
                                                         np.eye(n_features),
                                                         lower=True).T
        return precisions_chol


def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    if covariance_type == 'full':
        n_components, _, _ = matrix_chol.shape
        log_det_chol = (np.sum(np.log(
            matrix_chol.reshape(
                n_components, -1)[:, ::n_features + 1]), 1))
    return log_det_chol


def _estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type):
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    # det(precision_chol) is half of det(precision)
    log_det = _compute_log_det_cholesky(
        precisions_chol, covariance_type, n_features)

    if covariance_type == 'full':
        log_prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    return -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det


class GaussianMixtureW:
    def __init__(self, n_components=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        # Non-negative regularization added to the diagonal of covariance.
        # Allows to assure that the covariance matrices are all positive.
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init

    def _initialize_parameters(self, X, w, random_state):
        n_samples, _ = X.shape
        resp = np.zeros((n_samples, self.n_components))
        label = cluster.KMeans(n_clusters=self.n_components, n_init=1,
                               random_state=random_state).fit(X).labels_
        resp[np.arange(n_samples), label] = 1
        self._initialize(X, w, resp)

    def _initialize(self, X, w, resp):
        n_samples, _ = X.shape
        weights, means, covariances = _estimate_gaussian_parameters(
            X, w, resp, self.reg_covar, self.covariance_type)
        weights /= n_samples

        self.weights_ = (weights if self.weights_init is None
        else self.weights_init)
        self.means_ = means if self.means_init is None else self.means_init
        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type)
        elif self.covariance_type == 'full':
            self.precisions_cholesky_ = np.array(
                [linalg.cholesky(prec_init, lower=True)
                 for prec_init in self.precisions_init])

    def _estimate_log_prob(self, X):
        return _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type)

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _estimate_weighted_log_prob(self, X):
        return self._estimate_log_prob(X) + self._estimate_log_weights()

    def _estimate_log_prob_resp(self, X):
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under='ignore'):
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _get_parameters(self):
        return (self.weights_, self.means_, self.covariances_,
                self.precisions_cholesky_)

    def _set_parameters(self, params):
        (self.weights_, self.means_, self.covariances_,
         self.precisions_cholesky_) = params

        _, n_features = self.means_.shape

        if self.covariance_type == 'full':
            self.precisions_ = np.empty(self.precisions_cholesky_.shape)
            for k, prec_chol in enumerate(self.precisions_cholesky_):
                self.precisions_[k] = np.dot(prec_chol, prec_chol.T)

    def _e_step(self, X, w):
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X, w)
        return np.mean(log_prob_norm), log_resp

    def _m_step(self, X, log_resp):
        n_samples, _ = X.shape
        self.weights_, self.means_, self.covariances_ = (
            _estimate_gaussian_parameters(X, np.exp(log_resp), self.reg_covar,
                                          self.covariance_type))
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type)

    def fit(self, X, w):
        max_lower_bound = -np.infty
        self.converged_ = False
        random_state = np.random.RandomState(self.random_state)
        n_samples, _ = X.shape
        # Do initialization
        self._initialize_parameters(X, w, random_state)
        self.lower_bound_ = -np.infty

        for n_iter in range(self.max_iter):
            prev_lower_bound = self.lower_bound_
            log_prob_norm, log_resp = self._e_step(X, w)
            self._m_step(X, log_resp)
            self.lower_bound_ = self._compute_lower_bound(
                log_resp, log_prob_norm)
            change = self.lower_bound_ - prev_lower_bound
            if abs(change) < self.tol:
                self.converged_ = True
                break

        if self.lower_bound_ > max_lower_bound:
            max_lower_bound = self.lower_bound_
            best_params = self._get_parameters()
            best_n_iter = n_iter

        if not self.converged_:
            warnings.warn('GMM Initialization did not converge. ')

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        return self

    def sample(self, n_samples=1):
        _, n_features = self.means_.shape
        rng = np.random.RandomState(self.random_state)
        n_samples_comp = rng.multinomial(n_samples, self.weights_)

        if self.covariance_type == 'full':
            X = np.vstack([
                rng.multivariate_normal(mean, covariance, int(sample))
                for (mean, covariance, sample) in zip(
                    self.means_, self.covariances_, n_samples_comp)])

        y = np.concatenate([j * np.ones(sample, dtype=int)
                            for j, sample in enumerate(n_samples_comp)])
        return (X, y)

    def score_samples(self, X):
        return logsumexp(self._estimate_weighted_log_prob(X), axis=1)




