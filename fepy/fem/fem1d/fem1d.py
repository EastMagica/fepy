#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/10 14:59
# @file    : fem1d.py
# @project : fepy
# software : PyCharm

import numpy as np

from fepy.basic import is_ndarray
from fepy.basic.gaussian import load_gaussian
from fepy.fem.basic import FEM


# Functions
# ---------

def area(*points):
    [pl, pr] = is_ndarray(points)
    if (pl.ndim > 1) or (pr.ndim > 1):
        raise ValueError('Input a needs to be a (N, 1) Matrix.')
    elif pl.size != pr.size:
        raise ValueError('Input matrices need to have same raw.')
    return pr - pl


# Classes
# -------

class LinearBasisMixIn(object):
    """
    一维线性基函数
    """
    @staticmethod
    def basis_grid(p, v):
        p = is_ndarray(p).squeeze()
        value = np.array([v[1] - p, p - v[0]]) / area(*v)
        return value.T

    @staticmethod
    def basis_value(p, v):
        grid = np.array([[1], [-1]]) / area(*v)
        return grid.T


class LinearFEM1D(LinearBasisMixIn, FEM):
    def __init__(self, variation, mesh, boundary, gaussian=3):
        super().__init__(variation, mesh, boundary)
        self.gaussian = load_gaussian(gaussian, self.ndim)


