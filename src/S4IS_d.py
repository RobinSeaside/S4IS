#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 201802 LIU Wangsheng; Email: awang.signup@gmail.com

import numpy as np
import openturns as ot

from src.Surrogate import Surrogate
from src.DensityEstimates import DensityEstimates
from src.Infiller import Infiller
from src.utilities import resample_WSBRVG


def S4IS_d(h_params, s_params, dist, infill_params_1, infill_params_2, de_params, analysis_hparams,
           random_seed=2018, verbose=0):
    # Initialization
    ot.RandomGenerator.SetSeed(random_seed)
    x2u = dist.getIsoProbabilisticTransformation()
    # u2x = dist.getInverseIsoProbabilisticTransformation()
    delta_pf_1 = analysis_hparams['delta_pf_1']
    delta_pf_2 = analysis_hparams['delta_pf_2']
    if 'num_pnt_init' in analysis_hparams:
        num_pnt_init = analysis_hparams['num_pnt_init']
    else:
        d = dist.getDimension()
        num_pnt_init = int((d + 1) * (d + 2) / 2)
    num_pnt_is = analysis_hparams['num_pnt_is']
    num_pnt_candidate = analysis_hparams['num_pnt_candidate']

    # Initial experiment
    experiment_init = ot.LHSExperiment(dist, num_pnt_init)
    X_sp = np.array(experiment_init.generate())
    y_sp = h_params['model'](X_sp)

    # Stage 1: train a naive classifier
    if infill_params_1['candidate'] == 'uniform':
        dist_marg_mean = dist.getMean()
        dist_marg_sd = dist.getStandardDeviation()
        margs = []
        for i in range(dist.getDimension()):
            margs.append(ot.Uniform(dist_marg_mean[i]-5*dist_marg_sd[i],
                                    dist_marg_mean[i]+5*dist_marg_sd[i]))
        dist_stage1 = ot.ComposedDistribution(margs)
        X_candidate = np.array(ot.LHSExperiment(dist_stage1, num_pnt_candidate).generate())
    else:
        X_candidate = np.array(ot.LHSExperiment(dist, num_pnt_candidate).generate())
    logw_LHS = np.array(dist.computeLogPDF(X_candidate)).flatten()
    w = np.array(dist.computePDF(X_candidate)).flatten()
    infiller1 = Infiller(infill_params_1, h_params, x2u, X_candidate)

    ratio_f_list = []
    num_it_stage1 = 0
    min_it_stage1 = infill_params_1['min_it']
    max_it_stage1 = infill_params_1['max_it']
    while True:
        surrogate = Surrogate(s_params).fit(X_sp, y_sp)
        y_candidate = surrogate.predict(X_candidate, False, False)
        if h_params['series']:
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
            if (np.abs(ratio_f - m_ratio_f) / m_ratio_f <= delta_pf_1 or num_it_stage1 >= max_it_stage1) \
                    and num_it_stage1 >= min_it_stage1:
                break
        # Find the next support point
        X_next, y_next = infiller1.gen_next(X_sp, y_t_candidate, logw=logw_LHS)

        # Update the dataset of support points
        X_sp = np.append(X_sp, X_next, axis=0)
        y_sp = np.append(y_sp, y_next, axis=0)
        num_it_stage1 += 1

    # Stage 2: refine the surrogate based on samples from the pseudo-optimal PDF
    if infill_params_1['candidate'] == 'uniform':
        num_pnt_re = infill_params_1['num_pnt_re']
        num_neighbors = infill_params_1['num_neighbors']
        X_re = resample_WSBRVG(X_candidate[idx_f], w[idx_f], num_pnt_re, num_neighbors)
        dist_pseudo_best = DensityEstimates(de_params, random_seed).fit(X_re)
    else:
        dist_pseudo_best = DensityEstimates(de_params, random_seed).fit(X_candidate[idx_f])
    X_is = dist_pseudo_best.sample(num_pnt_is)
    logw_is = np.array(dist.computeLogPDF(X_is)).flatten() - dist_pseudo_best.score_samples(X_is)
    infiller2 = Infiller(infill_params_2, h_params, x2u, X_is)

    p_f_list = []
    num_it_stage2 = 0
    min_it_stage2 = infill_params_2['min_it']
    max_it_stage2 = infill_params_2['max_it']
    while True:
        y_is = surrogate.predict(X_is, False, False)
        if h_params['series']:
            y_t_is = np.min(y_is, axis=1)
        else:
            y_t_is = np.max(y_is, axis=1)
        idx_f_is = y_t_is <= 0
        p_f = np.mean(np.exp(logw_is) * idx_f_is)
        print('p_f = {}'.format(p_f))
        p_f_list.append(p_f)
        if num_it_stage2 != 0:
            m_p_f = np.mean(p_f_list[max(0, len(p_f_list) - 5):-1])
            if (np.abs(p_f - m_p_f) / m_p_f <= delta_pf_2 or num_it_stage2 >= max_it_stage2) \
                    and num_it_stage2 >= min_it_stage2:
                break
        # Find the next support point
        X_next, y_next = infiller2.gen_next(X_sp, y_t_is, logw=logw_is)
        print('X_next = {}'.format(X_next))

        # Update the dataset of support points
        X_sp = np.append(X_sp, X_next, axis=0)
        y_sp = np.append(y_sp, y_next, axis=0)
        surrogate = Surrogate(s_params).fit(X_sp, y_sp)
        num_it_stage2 += 1

    # Output
    num_feval_total = num_pnt_init + num_it_stage1 + num_it_stage2
    print('Done!')
    return {'pf_hat': p_f, 'X': X_sp, 'y': y_sp, 'num_it_stage1': num_it_stage1, 'num_it_stage2': num_it_stage2,
            'surrogate': surrogate, 'num_pnt_init': num_pnt_init, 'num_feval_total': num_feval_total}
