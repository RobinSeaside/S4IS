#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 201802 LIU Wangsheng; Email: awang.signup@gmail.com
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
plt.figure(1, figsize=[8, 6])
plt.scatter(X_candidate[idx_f, 0], X_candidate[idx_f, 1], c='r', alpha=0.5)
ax = plt.gca()
ax.set_xlim([x[0], x[-1]])
ax.set_ylim([y[0], y[-1]])
plt.show()

x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
plt.figure(1, figsize=[8, 6])
plt.scatter(X_re[:, 0], X_re[:, 1], c='r', alpha=0.5)
ax = plt.gca()
ax.set_xlim([x[0], x[-1]])
ax.set_ylim([y[0], y[-1]])
plt.show()

plt.figure(1, figsize=[8, 6])
plt.scatter(U_f[:, 0], U_f[:, 1], c='r', alpha=0.5)
plt.scatter(means[0][0], means[0][1], c='b', alpha=0.5)
plt.scatter(means[1][0], means[1][1], c='b', alpha=0.5)
plt.scatter(means[2][0], means[2][1], c='b', alpha=0.5)
plt.scatter(means[3][0], means[3][1], c='b', alpha=0.5)
plt.scatter(means[4][0], means[4][1], c='b', alpha=0.5)
plt.show()
