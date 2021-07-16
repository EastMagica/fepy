#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/10 14:15
# @file    : posteriori.py
# @project : fepy
# software : PyCharm

import abc

import numpy as np


# Meta Class
# ----------

class MetaPosterioriError(metaclass=abc.ABCMeta):
    def __init__(self, fem, f):
        self.f = f
        self.fem = fem
        self._error = None
        self.error_simplices = None

    @property
    @abc.abstractmethod
    def error(self, *args, **kwargs):
        raise NotImplementedError


# Classes
# -------

class L2FError(MetaPosterioriError):
    def __init__(self, fem, f):
        super().__init__(fem, f)
        self.error_simplices = np.zeros(
            self.fem.mesh.nsimplices
        )
        self.get_error()

    def get_simplices_error(self, unit_p):
        area = self.fem.mesh.area(unit_p)
        gauss_p, gauss_w = self.fem.gaussian.local_to_global(
            unit_p, area
        )
        value_f = self.f(*gauss_p.T)
        return np.sum(value_f ** 2 @ gauss_w)

    def get_error(self):
        for k, v in enumerate(self.fem.mesh.simplices):
            unit_p = self.fem.mesh.points[v]
            self.error_simplices[k] = self.get_simplices_error(unit_p)

    @property
    def error(self):
        if self._error is None:
            self._error = np.sqrt(np.sum(self.error_simplices))
        return self._error
