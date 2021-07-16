#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/16 20:54
# @file    : restructure.py
# @project : fepy
# software : PyCharm

import numpy as np

from fepy.error.posteriori import L2FError


class Adaptive1D(object):
    def __init__(self, fem, f):
        self.f = f
        self.fem = fem

    def adaptive(self):
        err = L2FError(self.fem, self.f)
        error = err.error_simplices

        index, = np.where(error > np.average(error))

        insert_points = np.zeros(index.size, dtype=float)

        for k, v in enumerate(self.fem.mesh.simplices(index)):
            insert_points[k] = np.mean(self.fem.mesh.points[v])

        new_points = np.sort(np.hstack([
            self.fem.mesh.points,
            insert_points
        ]))

        self.fem.mesh.create(new_points)
