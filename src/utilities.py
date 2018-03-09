#!/usr/bin/env python3
import numpy as np
import openturns as ot
from sklearn.neighbors import NearestNeighbors


def calc_pf_statistics(pf_list):
    num_rep = len(pf_list)
    pf_hat1 = np.zeros(num_rep)
    pf_hat2 = np.zeros(num_rep)
    num_sim1 = np.zeros(num_rep)
    num_sim2 = np.zeros(num_rep)
    for i in range(num_rep):
        tmp_pf = pf_list[i]
        pf_hat1[i] = tmp_pf['pf_hat1']
        pf_hat2[i] = tmp_pf['pf_hat2']
        num_sim1[i] = tmp_pf['num_feval_total1']
        num_sim2[i] = tmp_pf['num_feval_total2']
    return (np.mean(pf_hat1), np.std(pf_hat1, ddof=1)/np.mean(pf_hat1), np.mean(num_sim1),
            np.mean(pf_hat2), np.std(pf_hat2, ddof=1) / np.mean(pf_hat2), np.mean(num_sim2))


def calc_pf_mcs(y, if_series):
    if if_series:
        y_min = np.min(y, axis=1)
        return np.mean(y_min <= 0)
    else:
        y_max = np.max(y, axis=1)
        return np.mean(y_max <= 0)


def check_if_stop_iteration(dist, series, model_all, model_fold, num_feval_total, analysis_hparams):
    flag_stop = False
    num_pnt_mcs = analysis_hparams['num_pnt_mcs']
    num_feval_max = analysis_hparams['num_feval_max']
    epsilon_pf = analysis_hparams['epsilon_pf']
    X = np.array(ot.LHSExperiment(dist, num_pnt_mcs).generate())
    y_hat_all = model_all.predict(X, False, False)
    pf_hat = calc_pf_mcs(y_hat_all, series)
    if pf_hat == 0:
        return None, None, flag_stop
    else:
        pf_hat_fold = []
        for tmp_model in model_fold:
            tmp_y_hat = tmp_model.predict(X, False, False)
            pf_hat_fold.append(calc_pf_mcs(tmp_y_hat, series))
        if max(abs(np.array(pf_hat_fold)-pf_hat)) / pf_hat <= epsilon_pf:
            print('Stopped. Epsilon of the failure probability is small.')
            flag_stop = True
        elif num_feval_total > num_feval_max:
            print('Stopped. The total number of function evaluations exceeds num_feval_max={}'.format(num_feval_max))
            flag_stop = True
        return pf_hat, pf_hat_fold, flag_stop


def resample_WSBRVG(X, w, num_pnt_re, num_neighbors):
    num_s, dim = X.shape
    idx = np.random.randint(0, num_s, size=num_pnt_re)
    X_selected, w_selected = X[idx], w[idx]
    # Identify num_neighbors nearest neighbors
    if num_neighbors is None:
        num_neighbors = 10
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(X)
    _, idx_neighbors = nbrs.kneighbors(X_selected)
    # Normalize weights and calculate the means of selected samples
    w_expand = w[idx_neighbors]
    w_expand_norm = w_expand / np.sum(w_expand, axis=1, keepdims=1)
    u = np.random.uniform(-1, 1, size=(num_pnt_re, num_neighbors))
    X_expand = X[idx_neighbors]
    Z = np.zeros(shape=(num_pnt_re, dim))
    for i in range(num_pnt_re):
        tmp_X_mean = np.sum(X_expand[i] * np.tile(w_expand_norm[i], (dim, 1)).T, axis=0, keepdims=1)
        tmp_Z = np.sum(np.tile(u[i] * (3 * w_expand_norm[i]) ** 0.5, (dim, 1)).T
                       * (X_expand[i] - tmp_X_mean), axis=0) + tmp_X_mean
        Z[i] = tmp_Z[0]
    return Z



