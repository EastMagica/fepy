#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/2/6 16:12
# @file    : gaussian.py
# @project : fepy
# software : PyCharm

import abc

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
        # TODO: gaussian 2d L2G
        ...


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

