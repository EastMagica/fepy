#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/2/6 16:12
# @file    : gaussian.py
# @project : fepy
# software : PyCharm

import abc

import numpy as np

from fepy.basic.gaussian.gauss1d import gauss_1d
from fepy.basic.gaussian.gauss2d import gauss_2d


# Meta Class
# ----------

class MetaGaussian(metaclass=abc.ABCMeta):
    """
        Methods
        ----------
        points : ndarray, (N, dim)
        weight : ndarray, (N, )
        area : float
        """

    def __init__(self, n, dim):
        self._dim = dim
        self._gauss_n = n
        self._points, self._weight, self._area = load_gaussian(n, dim)

    @property
    def gauss_n(self):
        return self._gauss_n

    @property
    def dim(self):
        return self._dim

    @property
    def points(self):
        return self._points

    @property
    def weight(self):
        return self._weight

    @property
    def area(self):
        return self._area

    @abc.abstractmethod
    def local_to_global(self, global_v, global_area):
        raise NotImplementedError


# Classes
# -------

class Gaussian1D(MetaGaussian):
    def local_to_global(self, global_v, global_area):
        global_p = self.points * (global_v[1] - global_v[0]) + global_v[0]
        global_w = global_area / self.area * self.weight
        return global_p, global_w


class Gaussian2D(MetaGaussian):
    def local_to_global(self, global_v, global_area):
        """

        Parameters
        ----------
        global_area
        global_v : ndarray, (3, 2)

        Returns
        -------
        global_p : ndarray, (3, 2)
        global_w : ndarray, (3,  )
        """
        local_ex = extend_points(self.points, 'natural', axis=0)
        global_p = np.transpose(np.dot(global_v.T, local_ex))
        global_w = global_area / self.area * self.weight
        return global_p, global_w


# Functions
# ---------

def load_gaussian(n, dim):
    r"""
    加载高斯积分参数.

    Parameters
    ----------
    n : int, str
    dim : int, str

    Returns
    -------
    points : ndarray, (N, )
    weight : ndarray, (N, )
    area : float
    """
    n = str(n) if not isinstance(n, str) else n
    dim = str(dim) if not isinstance(dim, str) else dim

    if dim == '1':
        gaussian = gauss_1d[n]
        area = gauss_1d['area']
    elif dim == '2':
        gaussian = gauss_2d[n]
        area = gauss_2d['area']
    else:
        raise ValueError(str(n) + ' Gauss Point does not exists!')

    points, weight = gaussian['points'], gaussian['weights']

    return points, weight, area


def extend_points(points, opts='global', axis=1):
    r"""扩充矩阵
    扩充全局(global)或自然(natural)坐标矩阵.
    .. math:
        \left[\begin{matrix}
            x_1, x_2, \cdots, x_n \\
            y_1, y_2, \cdots, y_n \\
        \end{matrix}\right] \rightarrow
        \left[\begin{matrix}
            x_1, x_2, \cdots, x_n \\
            y_1, y_2, \cdots, y_n \\
            1  , 1  , \cdots, 1   \\
        \end{matrix}\right] \\
        \left[\begin{matrix}
            \lambda_{11}, \cdots, \lambda_{1n} \\
            \lambda_{21}, \cdots, \lambda_{2n} \\
        \end{matrix}\right] \rightarrow
        \left[\begin{matrix}
            \lambda_{11}, \cdots, \lambda_{n1} \\
            \lambda_{12}, \cdots, \lambda_{n2} \\
            \lambda_{13}, \cdots, \lambda_{n3} \\
        \end{matrix}\right]
    Parameters
    ----------
    points : array_like, (N, 2)
    opts   : {'global', 'natural'}, optional
        扩充矩阵类型(全局或自然坐标), 默认全局坐标.
    axis   : {0, 1}, optional
        坐标矩阵方向, 默认为列坐标(axis=1).
    Returns
    -------
    """
    points = _is_ndarray(points)
    # 坐标矩阵转置为列向量
    if axis == 0:
        points = points.transpose()
    elif axis != 1:
        raise ValueError('Axis values can only be 0 or 1.')
    # 构造不同的扩展分量
    if opts.lower() == 'golbal':
        extend_vector = np.zeros(points.shape[1])
    elif opts.lower() == 'natural':
        extend_vector = 1 - np.sum(points, axis=0)
    else:
        raise ValueError(opts + ' coordinates are not supported.')
    # 合并分量
    point_extend = np.vstack((points, extend_vector))
    return point_extend


def area(*points):
    r"""计算三角形面积.
    Parameters
    ----------
    points : list, array_like, [(N, 2), (2, ), (2, )]
        三角单元顶点坐标.
    Returns
    -------
    s : array_like, (N, )
        三角单元面积
    Note
    ----
    若``points``为(3,2)数组, 则可使用``area(*points)``
    直接计算, 无需为对应参数而拆开数组.
    """
    [p0, p1, p2] = _is_ndarray(*points)
    p0 = p0.T if p0.ndim == 2 else p0
    s = ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])) / 2
    return s


def global_to_local(global_p, global_v):
    r"""局部(Local)坐标变换到全局(Global)坐标.
    .. math::
        \vec{\lambda} = X^{-1} \vec{x} \\
        \left[\begin{matrix}
            \lambda_1 \\ \lambda_2 \\ \lambda_3 \\
        \end{matrix}\right] =
        \left[\begin{matrix}
            x_1 & x_2 & x_3 \\
            y_1 & y_2 & y_3 \\
            1   & 1   & 1   \\
        \end{matrix}\right]^{-1}
        \left[\begin{matrix}
            x \\ y \\ 1 \\
        \end{matrix}\right]
    Parameters
    ----------
    global_p : array_like, (N, 2)
        全局坐标中的点坐标.
    global_v : array_like, (3, 2)
        全局坐标中的三角形顶点坐标.
    Returns
    -------
    Global Point in Local Area
    """
    global_vex = extend_points(global_v, 'global', axis=0)
    global_pex = extend_points(global_p, 'global', axis=0)
    global_inv = np.linalg.inv(global_vex)
    local_p = np.dot(global_inv, global_pex)
    return local_p
