#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 201802 LIU Wangsheng; Email: awang.signup@gmail.com
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler


class Infiller:
    def __init__(self, params, hard_model, x2u, X_c):
        self.n_top = params['n_top']
        self.metric = params['metric']
        self.x2u = x2u
        self.params = params
        self.model_series = hard_model['series']
        self.hard_m = hard_model['model']
        self.infill_level = 0
        self.X_c = X_c
        self.idx_X_next_list = []

    def gen_next(self, X_sp, y_c, **kwargs):
        self.infill_level += 1
        if self.params['name'] == 'conv_comb':
            _, min_distance = pairwise_distances_argmin_min(np.array(self.x2u(self.X_c)),
                                                            np.array(self.x2u(X_sp)),
                                                            metric=self.metric)
            y_t_scaled = StandardScaler().fit_transform(np.abs(y_c).reshape(-1, 1)).flatten()
            if self.params['decay_rate'] is None:
                obj_value = y_t_scaled - min_distance
            else:
                c_decay = np.exp(-self.params['decay_rate'] * self.infill_level)
                obj_value = (1 - c_decay) * y_t_scaled + c_decay * (- min_distance)
            obj_value[self.idx_X_next_list] = np.inf
            idx_X_next = np.argpartition(obj_value, self.n_top)[:self.n_top]
            self.idx_X_next_list.extend(idx_X_next)
            X_next = self.X_c[idx_X_next]
            y_next = self.hard_m(X_next)
            return X_next, y_next

        if self.params['name'] == 'conv_comb_w':
            _, min_distance = pairwise_distances_argmin_min(np.array(self.x2u(self.X_c)),
                                                            np.array(self.x2u(X_sp)),
                                                            metric=self.metric)
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
            obj_value[self.idx_X_next_list] = np.inf
            idx_X_next = np.argpartition(obj_value, self.n_top)[:self.n_top]
            self.idx_X_next_list.extend(idx_X_next)
            X_next = self.X_c[idx_X_next]
            y_next = self.hard_m(X_next)
            return X_next, y_next
