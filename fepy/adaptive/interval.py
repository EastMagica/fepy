#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/16 20:54
# @file    : interval.py
# @project : fepy
# software : PyCharm

import numpy as np


class Adaptive1D(object):
    def __init__(self, err, fem, f):
        self.f = f
        self.fem = fem
        self.err = err

    def adaptive(self):
        err = self.err(self.fem, self.f)
        error = err.error_simplices

        index, = np.where(error > np.average(error))

        insert_points = np.zeros(
            (index.size, 1), dtype=float
        )

        for k, v in enumerate(self.fem.mesh.simplices[index]):
            insert_points[k] = np.mean(self.fem.mesh.points[v])

        new_points = np.vstack([
            self.fem.mesh.points,
            insert_points
        ])
        new_points = np.sort(new_points, axis=0)

        self.fem.mesh.create(new_points)
        self.fem.init_values()
