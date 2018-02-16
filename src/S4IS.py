#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 201801 LIU Wangsheng; Email: awang.signup@gmail.com
import numpy as np
import openturns as ot
import sys
from datetime import datetime
from scipy.stats import gaussian_kde
from sklearn import mixture
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt

from src.Surrogate import Surrogate
from src.DensityEstimates import DensityEstimates

def lsf_multimodal_function(theta):
    """
    Define and evaluate the limit state functions that may be computationally expensive
    Args:
        theta: numpy array_like
            n by d Numpy array where n is the size of points and d is the dimension of each point;

    Returns:
        outputs: array-like, shape = (n, d)
            n by num_lsf where num_lsf is the number of limit state functions

    """
    y_out = -((theta[:, 0] ** 2 + 4) * (theta[:, 1] - 1) / 20 - np.sin(5 * theta[:, 0] / 2) - 2)
    return y_out.reshape(y_out.shape[0], -1)


base = 'K:/NTU_thesis/MyAlgorithms/S4IS/'
sys.path.append(base)
random_seed = 2018

# Hard model
hard_model_hparams = {
    'model': lsf_multimodal_function,
    'series': True  # if series or parallel
}

# Soft model
soft_model_hparams = {
    'name': 'GP',
    'kernel': None,
    'n_restarts_optimizer': 0,
    'normalize': False
}

# Density estimator
de_hparams = {
    'name': 'GM-sklearn',
    'n_components': 5,
    'max_iter': 100
}
# Distributions of random variables
marginals = [ot.Normal(1.5, 1.0), ot.Normal(2.5, 1.0)]
dist = ot.ComposedDistribution(marginals)

# Analysis settings
analysis_hparams = {
    'num_rep': 10,
    'num_pnt_init': 12,
    'num_pnt_mcs': 100000,
    'num_pnt_is': 10000,
    'num_pnt_candidate': 10000,
    'num_feval_max': 1000,
    'epsilon_pf': 0.01,
    'num_splits': 5,
    'n_top': 1
}

# Initial support points
ot.RandomGenerator.SetSeed(random_seed)
n_top = analysis_hparams['n_top']
num_feval_total = 0
num_it = 0

if 'num_pnt_init' in analysis_hparams:
    num_pnt_init = analysis_hparams['num_pnt_init']
else:
    d = dist.getDimension()
    num_pnt_init = int((d + 1) * (d + 2) / 2)
num_pnt_is = analysis_hparams['num_pnt_is']
num_pnt_candidate = analysis_hparams['num_pnt_candidate']

experiment_init = ot.LHSExperiment(dist, num_pnt_init)
X_sp = np.array(experiment_init.generate())
y_sp = hard_model_hparams['model'](X_sp)

# Fit the surrogate model
surrogate = Surrogate(soft_model_hparams).fit(X_sp, y_sp)

# Generate the candidate samples
X_candidate = np.array(ot.LHSExperiment(dist, num_pnt_candidate).generate())

y_candidate = surrogate.predict(X_candidate, False, False)
if hard_model_hparams['series']:
    y_t_candidate = np.min(y_candidate, axis=1)
else:
    y_t_candidate = np.max(y_candidate, axis=1)
idx_f_is = y_t_candidate <= 0
ratio_f = np.mean(idx_f_is)

_, min_distance = pairwise_distances_argmin_min(X_candidate, X_sp)
# obj_value = -np.abs(y_t_candidate) + logw_candidate + min_distance
obj_value = -np.abs(y_t_candidate) + min_distance
X_next = X_candidate[np.argpartition(obj_value, -n_top)[-n_top:], :]
y_next = hard_model_hparams['model'](X_next)

X_sp = np.append(X_sp, X_next, axis=0)
y_sp = np.append(y_sp, y_next, axis=0)
surrogate = Surrogate(soft_model_hparams).fit(X_sp, y_sp)

x = np.arange(-4, 7, 0.1)
y = (2 + np.sin(5 * x / 2)) * 20 / (x ** 2 + 4) + 1
y_range = [-3, 9]
plt.plot(x, y)
plt.figure(1, figsize=[8, 6])
# plt.scatter(X_candidate[idx_f_is, 0], X_candidate[idx_f_is, 1], c='r', alpha=0.5)
plt.scatter(X_next[:, 0], X_next[:, 1], c='r', alpha=0.5)
plt.show()
# Approximate the instrumental PDF (down sample the candidates)
# X_f_0 = X_candidate[idx_f, :]
# num_pnt_de = np.min([analysis_hparams['num_pnt_de'], X_f_0.shape[0]])
# X_f_1 = X_f_0[np.random.choice(X_f_0.shape[0], num_pnt_de, replace=False), :]
# dist_pseudo_best = DensityEstimates(de_hparams).fit(X_candidate[y_t <= 0, :])
# dist_pseudo_best = gaussian_kde([X_candidate[y_t <= 0, 0], X_candidate[y_t <= 0, 1]])
dist_pseudo_best = mixture.GaussianMixture(n_components=5).fit(X_candidate[y_t <= 0, :])

# Estimate the failure probability
X_is, _ = dist_pseudo_best.sample(num_pnt_is)
y_is = surrogate.predict(X_is, False, False)
if hard_model_hparams['series']:
    y_t_is = np.min(y_is, axis=1)
else:
    y_t_is = np.max(y_is, axis=1)
logw_is = np.array(dist.computeLogPDF(X_is)).flatten() - dist_pseudo_best.score_samples(X_is)
idx_f_is = y_t_is <= 0
p_f = np.mean(np.exp(logw_is) * idx_f_is)


# Generate the next support points
idx_candidate = np.random.choice(X_is.shape[0], num_pnt_candidate, replace=False)
X_candidate, y_t_candidate, logw_candidate = X_is[idx_candidate], y_t_is[idx_candidate], logw_is[idx_candidate]

_, min_distance = pairwise_distances_argmin_min(X_candidate, X_sp)
# obj_value = -np.abs(y_t_candidate) + logw_candidate + min_distance
obj_value = -np.abs(y_t_candidate) + min_distance
X_next = X_candidate[np.argpartition(obj_value, -n_top)[-n_top:], :]
y_next = hard_model_hparams['model'](X_next)


# logw_candidate = np.array(dist.computeLogPDF(X_candidate)).flatten() - dist_pseudo_opt.score_samples(X_candidate)
# _, min_distance = pairwise_distances_argmin_min(X_candidate, X_sp)
# obj_value = logw_candidate + min_distance

flag_if_stop = False
while not flag_if_stop:
    num_it += 1
    print('Starting {}-th iteration: {}'.format(num_it, datetime.now().strftime("%Y-%m-%d %H:%M")))
    # Update support points and surrogates
    X_sp = np.append(X_sp, X_next, axis=0)
    y_sp = np.append(y_sp, y_next, axis=0)
    surrogate = Surrogate(soft_model_hparams).fit(X_sp, y_sp)

    y_candidate = surrogate.predict(X_candidate, False, False)
    if hard_model_hparams['series']:
        y_t = np.min(y_candidate, axis=1)
    else:
        y_t = np.max(y_candidate, axis=1)
    dist_pseudo_opt = mixture.GaussianMixture(n_components=1).fit(X_candidate[y_t <= 0, :])
    # Estimate the failure probability
    X_is, _ = dist_pseudo_opt.sample(num_pnt_is)
    y_is = surrogate.predict(X_is, False, False)
    if hard_model_hparams['series']:
        y_t_is = np.min(y_is, axis=1)
    else:
        y_t_is = np.max(y_is, axis=1)
    idx_f_is = y_t_is <= 0
    logw_is = np.array(dist.computeLogPDF(X_is)).flatten() - dist_pseudo_best.score_samples(X_is)
    p_f = np.mean(np.exp(logw_is) * idx_f_is)
    print('p_f = {}'.format(p_f))

    if num_it > 50:
        flag_if_stop = True
    else:
        # Generate the next support points
        idx_candidate = np.random.choice(X_is.shape[0], num_pnt_candidate, replace=False)
        X_candidate, y_t_candidate, logw_candidate = X_is[idx_candidate], y_t_is[idx_candidate], logw_is[idx_candidate]
        _, min_distance = pairwise_distances_argmin_min(X_candidate, X_sp)
        c_decay = np.exp(-0.1 * num_it)
        obj_value = (1 - c_decay) * (- np.abs(y_t_candidate)) + c_decay * min_distance
        X_next = X_is[np.argpartition(obj_value, -n_top)[-n_top:], :]
        y_next = hard_model_hparams['model'](X_next)
        # logw_candidate = np.array(dist.computeLogPDF(X_candidate)).flatten() - dist_pseudo_opt.score_samples(
        #     X_candidate)
        # _, min_distance = pairwise_distances_argmin_min(X_candidate, X_sp)
        # c_decay = np.exp(-0.05 * num_it)
        # obj_value = (1 - c_decay) * logw_candidate + c_decay * min_distance
        # X_next = X_candidate[np.argpartition(obj_value, -n_top)[-n_top:], :]


# Plot

x = np.arange(-4, 7, 0.1)
y = (2 + np.sin(5 * x / 2)) * 20 / (x ** 2 + 4) + 1
y_range = [-3, 9]
plt.plot(x, y)
plt.figure(1, figsize=[8, 6])
plt.scatter(X_sp[:, 0], X_sp[:, 1], c='r', alpha=0.5, label='Inital support points')
plt.show()



