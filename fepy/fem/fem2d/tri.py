#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/14 18:57
# @file    : tri.py
# @project : fepy
# software : PyCharm

import numpy as np

from fepy.fem.basic import FEM
from fepy.basic.gaussian import Gaussian2D


# Classes
# -------

class TriLinearBasisMixIn(object):
    """
    一维线性基函数
    """

    def basis_value(self, p, v):
        r"""单元基函数.

        由单元基函数的定义, 可以得到单元基函数在点:math:`p`处取值.

        .. math::
            \varphi_1 = \frac{\Delta_{p23}}{\Delta} \\
            \varphi_2 = \frac{\Delta_{p31}}{\Delta} \\
            \varphi_3 = \frac{\Delta_{p12}}{\Delta} \\

        Parameters
        ----------
        p : array_like, (N, 2)
        v : array_like, (3, 2)

        Returns
        -------
        basis_v : ndarray, (N, 3)
        """
        area_v = self.mesh.area(
            v[0], v[1], v[2]
        )
        basis_v = np.array([
            self.mesh.area(p, v[1], v[2]),
            self.mesh.area(p, v[2], v[0]),
            self.mesh.area(p, v[0], v[1])
        ]) / area_v
        return basis_v.T

    def basis_grid(self, p, v):
        r"""单元基函数的导数.

        由单元基函数的定义, 可以得到单元基函数的导数,
        且三角单元基函数的导数为常数.

        .. math::

            \nabla \varphi_1 =
                \left[\begin{matrix}
                (y_2 - y_3) / \Delta \\ (x_3 - x_2) / 2\Delta
                \end{matrix}\right],\quad
            \nabla \varphi_1 =
                \left[\begin{matrix}
                (y_3 - y_1) / \Delta \\ (x_1 - x_3) / 2\Delta
                \end{matrix}\right],\quad
            \nabla \varphi_1 =
                \left[\begin{matrix}
                (y_1 - y_2) / \Delta \\ (x_2 - x_1) / 2\Delta
                \end{matrix}\right]

        Parameters
        ----------
        p : array_like, (3, 2)
        v : array_like, (N, 2)

        Returns
        -------
        basis_g : ndarray, (2, 3)
        """
        area_v = self.mesh.area(v[0], v[1], v[2])

        basis_g = np.array([
            v[[1, 2, 0], 1] - v[[2, 0, 1], 1],
            v[[2, 0, 1], 0] - v[[1, 2, 0], 0]
        ])

        basis_g = basis_g / (2 * area_v)
        return basis_g


class TriLinearFEM2D(TriLinearBasisMixIn, FEM):
    def __init__(self, variation, mesh, boundary, gaussian_n=3):
        super().__init__(variation, mesh, boundary)
        self.ndim = 1
        print("Load Gaussian in LInear FEM1D")
        self.gaussian = Gaussian2D(gaussian_n, self.ndim)
