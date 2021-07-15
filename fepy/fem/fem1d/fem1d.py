#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/10 14:59
# @file    : fem1d.py
# @project : fepy
# software : PyCharm

import numpy as np

from fepy.fem.basic import FEM
from fepy.basic.gaussian import Gaussian1D


# Classes
# -------

class LinearBasisMixIn(object):
    """
    一维线性基函数
    """

    def basis_value(self, p, v):
        r"""

        .. math::

            \phi_{L} =

        Parameters
        ----------
        p
        v

        Returns
        -------

        """
        p = np.asarray(p).squeeze()
        value = np.array([v[1] - p, p - v[0]]) / self.mesh.area(*v)
        return value.T

    def basis_grid(self, p, v):
        grid = np.array([[1], [-1]]) / self.mesh.area(*v)
        return grid.T


class LinearFEM1D(LinearBasisMixIn, FEM):
    def __init__(self, variation, mesh, boundary, gaussian_n=3):
        super().__init__(variation, mesh, boundary)
        self.ndim = 1
        print("Load Gaussian in Linear FEM1D...")
        self.gaussian = Gaussian1D(gaussian_n, self.ndim)
