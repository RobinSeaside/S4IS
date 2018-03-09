#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 201802 LIU Wangsheng; Email: awang.signup@gmail.com

import numpy as np
import openturns as ot
from datetime import datetime
import pickle

import sys
base = 'K:/NTU_thesis/MyAlgorithms/S4IS/'
sys.path.insert(0, base)

from src.S4IS_d import S4IS_d
from src.utilities import calc_pf_statistics


def lsf_mdp(theta):
    """
    Define and evaluate the limit state functions that may be computationally expensive
    Args:
        theta: numpy array_like
            n by d Numpy array where n is the size of points and d is the dimension of each point;

    Returns:
        outputs: array-like, shape = (n, d)
            n by num_lsf where num_lsf is the number of limit state functions

    """
    c = hard_model_hparams['constant']
    y1 = c - 1 - theta[:, 1] + np.exp(-theta[:, 0] ** 2 / 10) + (theta[:, 0] / 5) ** 4
    y2 = c ** 2 / 2 - theta[:, 0] * theta[:, 1]
    y_out = np.min(np.column_stack((y1, y2)), axis=1)
    return y_out.reshape(y_out.shape[0], -1)


# ********************************* Configurations ***********************************
# Basic settings
dir_demo = base + 'examples/reliability_level/'
verbose = 0
random_seed = 2018
# Hard model
hard_model_hparams = {
    'model': lsf_mdp,
    'constant': 5,
    'series': True  # if series or parallel
}
# Surrogate model
soft_model_hparams = {
    'name': 'GP',
    'kernel': None,
    'n_restarts_optimizer': 0,
    'normalize': False
}
# Distributions of random variables
dist = ot.Normal(2)

# Infill strategy
infill_params_1 = {
    'candidate': 'uniform',    # 'uniform'
    'name': 'conv_comb',   # support points in the boundary;
    'metric': 'euclidean',
    'n_top': 1,
    'delta_pf': 0.01,
    'decay_rate': None,
    'num_pnt_init': 12,
    'num_pnt_cand': 10000,
    'min_it': 60,
    'max_it': 10000
}
infill_params_2 = {
    'name': 'conv_comb_w',    # 'conv_comb'
    'metric': 'euclidean',
    'n_top': 1,
    'delta_pf': 0.001,
    'decay_rate': None,
    'num_pnt_cand': 10000,
    'min_it': 40,
    'max_it': 10000
}
# Density estimation
de_params = {
    'name': 'GM-mpp',
    'n_components': 5,
    'max_iter': 1000,
    'n_init': 1
}
# de_params = {
#     'name': 'GM-w',
#     'n_components': 10,
#     'max_iter': 100,
#     'n_init': 2
# }

# Analysis settings
analysis_hparams = {
    'num_rep': 10
}

# ********************************** Main **********************************
# 'SVR', 'PC', 'RF', 'XGB', 'NN'
soft_model_list = [
    # {
    #     'name': 'SVR',
    #     'normalize': False
    # },
    # {
    #     'name': 'RF',
    #     'n_estimators': 10,
    #     'normalize': False
    # },
    {
        'name': 'GP',
        'kernel': None,
        'n_restarts_optimizer': 0,
        'normalize': False
    }
]
infill_obj = ['conv_comb_w']  # 'conv_comb_w'

if __name__ == '__main__':
    for tmp_model in soft_model_list:
        soft_model_hparams = tmp_model
        for tmp_infill_obj in infill_obj:
            infill_params_2['name'] = tmp_infill_obj
            pf_list = []
            for i in range(analysis_hparams['num_rep']):
                print('Running {}/{}-th replicate ...'.format(i + 1, analysis_hparams['num_rep']))
                tmp_random_seed = random_seed + i
                S4IS_results = S4IS_d(hard_model_hparams, soft_model_hparams, dist, infill_params_1,
                                      infill_params_2, de_params, analysis_hparams, tmp_random_seed, verbose)
                pf_list.append(S4IS_results)
            pf_mean1, cov1, num_feval_total_mean1, pf_mean2, cov2, num_feval_total_mean2 = calc_pf_statistics(pf_list)
            print('pf_mean1={}, cov1={}, num_feval_total1={}'.format(pf_mean1, cov1, num_feval_total_mean1))
            print('pf_mean2={}, cov2={}, num_feval_total2={}'.format(pf_mean2, cov2, num_feval_total_mean2))
            tmp_save_dir = dir_demo + 'data/{}_{}_rep{}_{}.pkl'.format(soft_model_hparams['name'],
                                                                       infill_params_2['name'],
                                                                       analysis_hparams['num_rep'],
                                                                       datetime.now().strftime('%m%d'))
            with open(tmp_save_dir, 'wb') as f:
                pickle.dump(pf_list, f)

# num_rep = 10
# pf_mcs = np.zeros(num_rep)
# num_samples = 400000000
# for i in range(num_rep):
#     print('Running Monte Carlo Simulation for {}-th replicate.'.format(i+1))
#     experiment_init = ot.LHSExperiment(dist, 100000000)
#     X = np.array(experiment_init.generate())
#     y = lsf_mdp(X)
#     if hard_model_hparams['series']:
#         y_t = np.min(y, axis=1)
#     else:
#         y_t = np.max(y, axis=1)
#     pf_mcs[i] = sum(y_t <= 0) / len(y_t)
# pf_mean = np.mean(pf_mcs)
# cov = np.std(pf_mcs, ddof=1) / pf_mean
