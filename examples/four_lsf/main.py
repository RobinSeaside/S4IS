#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 201802 LIU Wangsheng; Email: awang.signup@gmail.com

import numpy as np
import openturns as ot
from datetime import datetime
import pickle

from src.S4IS_d import S4IS_d
from src.utilities import calc_pf_statistics


def four_lsf(theta):
    """
    Define and evaluate the limit state functions that may be computationally expensive
    Args:
        theta: numpy array_like
            n by d Numpy array where n is the size of points and d is the dimension of each point;

    Returns:
        outputs: numpy array_like
            n by num_lsf where num_lsf is the number of limit state functions
        num_feval: int
            n * num_lsf, number of function evaluations

    """
    y0 = 3 + 0.1 * (theta[:, 0] - theta[:, 1]) ** 2 - (theta[:, 0] + theta[:, 1]) / 2 ** 0.5
    y1 = 3 + 0.1 * (theta[:, 0] - theta[:, 1]) ** 2 + (theta[:, 0] + theta[:, 1]) / 2 ** 0.5
    y2 = (theta[:, 0] - theta[:, 1]) + 6 / 2 ** 0.5
    y3 = (theta[:, 1] - theta[:, 0]) + 6 / 2 ** 0.5
    y_out = np.transpose(np.array([y0, y1, y2, y3]))
    return y_out


# ********************************* Configurations ***********************************
# Basic settings
base = 'K:/NTU_thesis/MyAlgorithms/S4IS/'
dir_demo = base + 'examples/four_lsf/'
verbose = 0
random_seed = 2018
# Hard model
hard_model_hparams = {
    'model': four_lsf,
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
    'name': 'conv_comb',
    'n_top': 1,
    'decay_rate': None
}
infill_params_2 = {
    'name': 'conv_comb',    # 'conv_comb'
    'n_top': 1,
    'decay_rate': None
}
# Density estimation
de_params = {
    'name': 'GM-sklearn',
    'n_components': 10,
    'max_iter': 1000,
    'n_init': 2
}
# Analysis settings
analysis_hparams = {
    'num_rep': 10,
    'num_pnt_init': 12,
    'num_pnt_is': 100000,
    'num_pnt_candidate': 100000,
    'max_it_stage1': 100,
    'delta_pf_1': 0.001,
    'max_it_stage2': 100,
    'delta_pf_2': 0.001,
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
            pf_mean, cov, num_feval_total_mean = calc_pf_statistics(pf_list)
            print('pf_mean={}, cov={}, num_feval_total={}'.format(pf_mean, cov, num_feval_total_mean))
            tmp_save_dir = dir_demo + 'data/{}_{}_rep{}_{}.pkl'.format(soft_model_hparams['name'],
                                                                       infill_params_2['name'],
                                                                       analysis_hparams['num_rep'],
                                                                       datetime.now().strftime('%m%d'))
            with open(tmp_save_dir, 'wb') as f:
                pickle.dump(pf_list, f)

# num_rep = 10
# pf_mcs = np.zeros(num_rep)
# for i in range(num_rep):
#     print('Running Monte Carlo Simulation for {}-th replicate.'.format(i+1))
#     experiment_init = ot.LHSExperiment(dist, 1000000)
#     X = np.array(experiment_init.generate())
#     y = four_lsf(X)
#     if hard_model_hparams['series']:
#         y_t = np.min(y, axis=1)
#     else:
#         y_t = np.max(y, axis=1)
#     pf_mcs[i] = sum(y_t <= 0) / len(y_t)
# pf_mean = np.mean(pf_mcs)
# cov = np.std(pf_mcs, ddof=1) / pf_mean
