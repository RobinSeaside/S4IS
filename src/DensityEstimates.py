#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 201801 LIU Wangsheng; Email: awang.signup@gmail.com
from sklearn import mixture
from src.gaussian_mixture_weighted import GaussianMixtureW

class DensityEstimates:
    def __init__(self, hparams, random_seed):
        self.random_seed = random_seed
        self.hparams = hparams
        self.name = self.hparams['name']
        if self.name == 'GM-sklearn':
            n_components = self.hparams['n_components']
            max_iter = self.hparams['max_iter']
            n_init = self.hparams['n_init']
            self.model = mixture.GaussianMixture(covariance_type='full', n_components=n_components,
                                                 n_init=n_init, max_iter=max_iter, random_state=self.random_seed)
        if self.name == 'GM-w':
            n_components = self.hparams['n_components']
            max_iter = self.hparams['max_iter']
            n_init = self.hparams['n_init']
            self.model = GaussianMixtureW(covariance_type='full', n_components=n_components,
                                           n_init=n_init, max_iter=max_iter, random_state=self.random_seed)

    def fit(self, X, **kwargs):
        if self.name == 'GM-sklearn':
            self.model.fit(X)
            return self
        elif self.name == 'GM-w':
            self.model.fit(X, kwargs['weights'])
            return self

    def sample(self, num_samples):
        if self.name == 'GM-sklearn':
            X, _ = self.model.sample(num_samples)
            return X

    def score_samples(self, X):
        return self.model.score_samples(X)
