#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 201802 LIU Wangsheng; Email: awang.signup@gmail.com

import numpy as np
import openturns as ot

from src.Surrogate import Surrogate
from src.DensityEstimates import DensityEstimates
from src.Infiller import Infiller


def S4IS_d(h_params, s_params, dist, infill_params_1, infill_params_2, de_params, analysis_hparams,
           random_seed=2018, verbose=0):
    # Initialization
    ot.RandomGenerator.SetSeed(random_seed)
    d = dist.getDimension()
    x2u = dist.getIsoProbabilisticTransformation()
    u2x = dist.getInverseIsoProbabilisticTransformation()
    delta_pf_1 = infill_params_1['delta_pf']
    delta_pf_2 = infill_params_2['delta_pf']
    if 'num_pnt_init' in infill_params_1:
        num_pnt_init = infill_params_1['num_pnt_init']
    else:
        num_pnt_init = int((d + 1) * (d + 2) / 2)

    # Stage 1
    num_pnt_cand1 = infill_params_1['num_pnt_cand']
    dist_u = ot.Normal(d)
    if infill_params_1['candidate'] == 'original':
        U_candidate = np.array(ot.LHSExperiment(dist_u, num_pnt_cand1).generate())
        X_candidate = np.array(u2x(U_candidate))
    elif infill_params_1['candidate'] == 'uniform':
        margs = []
        for i in range(dist.getDimension()):
            margs.append(ot.Uniform(-5, 5))
        dist_stage1 = ot.ComposedDistribution(margs)
        U_candidate = np.array(ot.LHSExperiment(dist_stage1, num_pnt_cand1).generate())
        X_candidate = np.array(u2x(U_candidate))

    # Get the infiller for stage 1
    infiller1 = Infiller(infill_params_1, h_params, u2x, U_candidate)
    # Initial support points
    U_sp = infiller1.select_init_sp(num_pnt_init)
    X_sp = np.array(u2x(U_sp))
    y_sp = h_params['model'](X_sp)
    print('Number of initial support points: {}'.format(num_pnt_init))

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
        if infill_params_1['candidate'] == 'original':
            ratio_f = np.mean(idx_f)
        elif infill_params_1['candidate'] == 'uniform':
            logw = np.array(dist_u.computeLogPDF(U_candidate)).flatten() - np.array(dist_stage1.computeLogPDF(U_candidate)).flatten()
            ratio_f = np.mean(idx_f * np.exp(logw))

        # Determine if stop here
        print(ratio_f)
        ratio_f_list.append(ratio_f)
        if num_it_stage1 != 0:
            m_ratio_f = np.mean(ratio_f_list[max(0, len(ratio_f_list) - 5):-1])
            if (np.abs(ratio_f - m_ratio_f) / m_ratio_f <= delta_pf_1 or num_it_stage1 >= max_it_stage1) \
                    and num_it_stage1 >= min_it_stage1:
                break

        # Find the next support point
        u_next = infiller1.gen_next(y_t_candidate, U_sp=U_sp)
        X_next = np.array(u2x(u_next))
        y_next = h_params['model'](X_next)

        # Update the support points
        U_sp = np.append(U_sp, u_next, axis=0)
        X_sp = np.append(X_sp, X_next, axis=0)
        y_sp = np.append(y_sp, y_next, axis=0)
        num_it_stage1 += 1
    print('Number of infill support points in Stage 1: {}'.format(num_it_stage1 * infill_params_1['n_top']))

    # Stage 2
    num_pnt_cand2 = infill_params_2['num_pnt_cand']
    U_f = U_candidate[idx_f]
    dist_pseudo_best = DensityEstimates(de_params, random_seed).fit(U_f)
    U_is = dist_pseudo_best.sample(num_pnt_cand2)
    X_is = np.array(u2x(U_is))
    logw_is = np.array(dist_u.computeLogPDF(U_is)).flatten() - dist_pseudo_best.score_samples(U_is)
    # Get the infiller for stage 2
    infiller2 = Infiller(infill_params_2, h_params, u2x, U_is)

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
        u_next = infiller2.gen_next(y_t_is, logw=logw_is, U_sp=U_sp)
        X_next = np.array(u2x(u_next))
        y_next = h_params['model'](X_next)
        print('X_next = {}'.format(X_next))

        # Update the dataset of support points
        U_sp = np.append(U_sp, u_next, axis=0)
        X_sp = np.append(X_sp, X_next, axis=0)
        y_sp = np.append(y_sp, y_next, axis=0)
        surrogate = Surrogate(s_params).fit(X_sp, y_sp)
        num_it_stage2 += 1

    print('Number of infill support points in Stage 2: {}'.format(num_it_stage2 * infill_params_2['n_top']))

    # Output
    num_feval_total1 = (num_pnt_init + num_it_stage1 * infill_params_1['n_top'])
    num_feval_total2 = (num_pnt_init + num_it_stage1 * infill_params_1['n_top']
                        + num_it_stage2 * infill_params_2['n_top'])
    print('Done!')
    return {'pf_hat1': ratio_f, 'pf_hat2': p_f, 'X': X_sp, 'y': y_sp, 'num_it_stage1': num_it_stage1,
            'num_it_stage2': num_it_stage2, 'surrogate': surrogate, 'num_pnt_init': num_pnt_init,
            'num_feval_total1': num_feval_total1, 'num_feval_total2': num_feval_total2,
            'infill_params_1': infill_params_1,
            'infill_params_2': infill_params_2}
