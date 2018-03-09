#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 201802 LIU Wangsheng; Email: awang.signup@gmail.com
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler


class Infiller:
    def __init__(self, params, hard_model, u2x, u_c):
        self.params = params
        self.metric = params['metric']
        self.n_top = params['n_top']
        self.u2x = u2x
        self.model_series = hard_model['series']
        self.hard_m = hard_model['model']
        self.infill_level = 0
        self.u_c = u_c
        self.idx_sp_all = []

    def select_init_sp(self, num_pnt_init):
        # Select from non support points
        idx_selected = np.random.choice([i for i in range(self.u_c.shape[0]) if i not in self.idx_sp_all],
                                        num_pnt_init)
        self.idx_sp_all.extend(idx_selected)
        return self.u_c[idx_selected]

    def gen_next(self, y_c, **kwargs):
        self.infill_level += 1
        if self.params['name'] == 'b_pnt':
            y_c[self.idx_sp_all] = np.inf
            idx_next = np.argpartition(np.abs(y_c), self.n_top)[:self.n_top]
            self.idx_sp_all.extend(idx_next)
            return self.u_c[idx_next]

        if self.params['name'] == 'max_w':
            w_negative = -kwargs['logw']
            w_negative[self.idx_sp_all] = np.inf
            w_negative[y_c > 0] = np.inf
            idx_next = np.argpartition(w_negative, self.n_top)[:self.n_top]
            self.idx_sp_all.extend(idx_next)
            return self.u_c[idx_next]

        if self.params['name'] == 'conv_comb':
            U_sp = kwargs['U_sp']
            _, min_distance = pairwise_distances_argmin_min(self.u_c, U_sp, metric=self.metric)
            y_t_scaled = StandardScaler().fit_transform(np.abs(y_c).reshape(-1, 1)).flatten()
            if self.params['decay_rate'] is None:
                obj_value = y_t_scaled - min_distance
            else:
                c_decay = np.exp(-self.params['decay_rate'] * self.infill_level)
                obj_value = (1 - c_decay) * y_t_scaled + c_decay * (- min_distance)
            obj_value[self.idx_sp_all] = np.inf
            idx_next = np.argpartition(obj_value, self.n_top)[:self.n_top]
            self.idx_sp_all.extend(idx_next)
            return self.u_c[idx_next]

        if self.params['name'] == 'conv_comb_w':
            U_sp = kwargs['U_sp']
            _, min_distance = pairwise_distances_argmin_min(self.u_c, U_sp, metric=self.metric)
            logw_scaled = StandardScaler().fit_transform(kwargs['logw'].reshape(-1, 1)).flatten()
            y_t_scaled = StandardScaler().fit_transform(np.abs(y_c).reshape(-1, 1)).flatten()
            if self.params['decay_rate'] is None:
                # obj_value = y_t_scaled - min_distance
                obj_value = y_t_scaled - logw_scaled - min_distance
            else:
                c_decay = np.exp(-self.params['decay_rate'] * self.infill_level)
                # obj_value = (1 - c_decay) * y_t_scaled + c_decay * (- min_distance)
                obj_value = 0.5 * (1 - c_decay) * y_t_scaled + 0.5 * (1 - c_decay) * logw_scaled + \
                            c_decay * (- min_distance)
            obj_value[self.idx_sp_all] = np.inf
            idx_next = np.argpartition(obj_value, self.n_top)[:self.n_top]
            self.idx_sp_all.extend(idx_next)
            return self.u_c[idx_next]
