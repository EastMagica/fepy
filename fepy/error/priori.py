#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/10 14:08
# @file    : priori.py
# @project : fepy
# software : PyCharm

import abc

import numpy as np


# Meta Class
# ----------

class MetaPrioriError(metaclass=abc.ABCMeta):
    def __init__(self, fem, u_true):
        self.fem = fem
        self.u = u_true
        self._error = None
        self.error_simplices = None

    @property
    @abc.abstractmethod
    def error(self, *args, **kwargs):
        raise NotImplementedError


# Classes
# -------

class L2Error(MetaPrioriError):
    def __init__(self, fem, u_true):
        super().__init__(fem, u_true)
        self.error_simplices = np.zeros(
            self.fem.mesh.nsimplices,
            dtype=float
        )
        self.get_error()

    def get_simplices_error(self, unit_p, unit_v):
        area = self.fem.mesh.area(unit_p)
        gauss_p, gauss_w = self.fem.gaussian.local_to_global(
            unit_p, area
        )
        basis_v = self.fem.basis_value(
            gauss_p, unit_p
        )
        value_true = self.u(*gauss_p.T)
        value_calc = np.dot(
            unit_v, basis_v.T
        )
        return np.sum((value_true - value_calc) ** 2 * gauss_w)

    def get_error(self):
        for k, v in enumerate(self.fem.mesh.simplices):
            unit_p = self.fem.mesh.points[v]
            unit_v = self.fem.mesh.values[v]
            self.error_simplices[k] = self.get_simplices_error(
                unit_p, unit_v
            )

    @property
    def error(self):
        if self._error is None:
            self._error = np.sqrt(np.sum(self.error_simplices))
        return self._error
