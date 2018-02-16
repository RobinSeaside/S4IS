#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 201802 LIU Wangsheng; Email: awang.signup@gmail.com
import psutil
import numpy as np
import openturns as ot
import sys
from datetime import datetime
from scipy.stats import gaussian_kde
from sklearn import mixture
from sklearn.preprocessing import StandardScaler
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
x2u = dist.getIsoProbabilisticTransformation()
u2x = dist.getInverseIsoProbabilisticTransformation()

# Analysis settings
analysis_hparams = {
    'num_rep': 10,
    'num_pnt_init': 12,
    'num_pnt_mcs': 100000,
    'num_pnt_is': 50000,
    'num_pnt_candidate': 50000,
    'num_feval_max': 1000,
    'epsilon_pf': 0.01,
    'num_splits': 5,
    'n_top': 1
}

# Find MPPs

# Initial support points
ot.RandomGenerator.SetSeed(random_seed)
n_top = analysis_hparams['n_top']
num_feval_total = 0

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

X_candidate = np.array(ot.LHSExperiment(dist, num_pnt_candidate).generate())

ratio_f_list = []
num_it_stage1 = 0
while True:
    surrogate = Surrogate(soft_model_hparams).fit(X_sp, y_sp)
    y_candidate = surrogate.predict(X_candidate, False, False)
    if hard_model_hparams['series']:
        y_t_candidate = np.min(y_candidate, axis=1)
    else:
        y_t_candidate = np.max(y_candidate, axis=1)
    idx_f = y_t_candidate <= 0
    ratio_f = np.mean(idx_f)
    # Determine if stop here
    print(ratio_f)
    ratio_f_list.append(ratio_f)
    if num_it_stage1 != 0:
        m_ratio_f = np.mean(ratio_f_list[max(0, len(ratio_f_list) - 5):-1])
        if np.abs(ratio_f - m_ratio_f) / m_ratio_f <= 0.001 or num_it_stage1 >= 50:
            break
    # Find the next support point
    # if num_it_stage1 % 2 == 0 and sum(idx_f_is) != 0:
    if False:
        logden_f = np.array(dist.computeLogPDF(X_candidate[idx_f])).flatten()
        idx_X_next = np.argpartition(-logden_f, n_top)[:n_top]
        X_next = X_candidate[idx_f][idx_X_next]
        # X_candidate = np.delete(X_candidate, idx_X_next, axis=1)
    else:
        _, min_distance = pairwise_distances_argmin_min(np.array(x2u(X_candidate)), np.array(x2u(X_sp)))
        y_t_scaled = StandardScaler().fit_transform(np.abs(y_t_candidate).reshape(-1, 1)).flatten()
        obj_value = y_t_scaled - min_distance
        idx_X_next = np.argpartition(obj_value, n_top)[:n_top]
        X_next = X_candidate[idx_X_next]
        # X_candidate = np.delete(X_candidate, idx_X_next, axis=1)

    y_next = hard_model_hparams['model'](X_next)
    # Update the dataset of support points
    X_sp = np.append(X_sp, X_next, axis=0)
    y_sp = np.append(y_sp, y_next, axis=0)
    num_it_stage1 += 1

# We have identified the regions where failure samples exist by refining the surrogate for Monte Carlo Simulation
# Estimate the pseudo optimal instrumental PDF
dist_pseudo_best = mixture.GaussianMixture(n_components=5).fit(X_candidate[idx_f])
X_is, _ = dist_pseudo_best.sample(num_pnt_is)

p_f_list = []
num_it_stage2 = 0
while True:
    surrogate = Surrogate(soft_model_hparams).fit(X_sp, y_sp)
    y_is = surrogate.predict(X_is, False, False)
    if hard_model_hparams['series']:
        y_t_is = np.min(y_is, axis=1)
    else:
        y_t_is = np.max(y_is, axis=1)
    idx_f_is = y_t_is <= 0
    logw_is = np.array(dist.computeLogPDF(X_is)).flatten() - dist_pseudo_best.score_samples(X_is)
    p_f = np.mean(np.exp(logw_is) * idx_f_is)
    print('p_f = {}'.format(p_f))
    p_f_list.append(ratio_f)
    if num_it_stage2 != 0:
        m_p_f = np.mean(p_f_list[max(0, len(p_f_list) - 5):-1])
        if np.abs(p_f - m_p_f) / m_p_f <= 0.001 or num_it_stage2 >= 50:
            break

    # Find the next support point


    # Update the dataset of support points
    X_sp = np.append(X_sp, X_next, axis=0)
    y_sp = np.append(y_sp, y_next, axis=0)
    num_it_stage2 += 1


x = np.arange(-4, 7, 0.1)
y = (2 + np.sin(5 * x / 2)) * 20 / (x ** 2 + 4) + 1
y_range = [-3, 9]
plt.plot(x, y)
plt.figure(1, figsize=[8, 6])
plt.scatter(X_is[:, 0], X_is[:, 1], c='r', alpha=0.5)
# plt.scatter(X_is[idx_f_is, 0], X_is[idx_f_is, 1], c='r', alpha=0.5)
# plt.scatter(X_sp[:, 0], X_sp[:, 1], c='b', alpha=0.5)
plt.show()
