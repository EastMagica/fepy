#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/2/5 13:16
# @file    : basic.py
# @project : fepy
# software : PyCharm

import abc

import numpy as np
from scipy.spatial import Delaunay


# Functions
# ---------

def parse_box(box, dim):
    """
    解析 Box 参数.

    Parameters
    ----------
    box
    dim

    Returns
    -------

    """
    if box is None and dim == 1:
        return np.array([0, 1])
    elif box is None and dim == 2:
        return np.array([[0, 1], [0, 1]])
    box = np.array(box)
    if dim == 1:
        # box == [0, 1]
        if box.shape == (2,):
            return box
        elif box.shape == () and box > 0:
            return np.array([0, box])
    elif dim == 2:
        # box == [[0, 1], [0, 1]]
        if box.shape == (2, 2):
            return box
        # box == [[0, 1]] or box == [0, 1]
        elif box.shape == (1, 2) or box.shape == (2,):
            return np.tile(box, (2, 1))
    # 3D case is not supported temporarily.
    # box == [[0, 1], [0, 1], [0, 1]]
    raise ValueError('Box: {} is out of Specification.'.format(box))


def parse_n(n, dim):
    """
    解析 n 参数.

    Parameters
    ----------
    n
    dim

    Returns
    -------

    """
    n = np.array(n)
    # n = 100
    if dim == 1 and n.ndim == 0:
        return n
    elif dim == 2:
        # n = 100
        if n.shape == ():
            return n.repeat(2)
        # n = [100, 100]
        elif n.shape == (2,):
            return n
    raise ValueError('n: {} is out of Specification.'.format(n))


def split_range(box, n):
    """
    规则网格剖分.

    Parameters
    ----------
    box
    n

    Returns
    -------

    """
    dim = box.ndim
    if dim == 1:
        points = np.linspace(*box, num=n)
        simplices = np.transpose([np.arange(n-1), np.arange(1, n)])
        return points, simplices
    if dim == 2:
        x = np.linspace(*box[0], num=n[0])
        y = np.linspace(*box[1], num=n[1])
        x, y = np.meshgrid(x, y)
        points = np.transpose([x.flatten(), y.flatten()])
        return Delaunay(points)
    raise ValueError('Splitting range Error.')


# Classes
# -------

class IntervalMesh(metaclass=abc.ABCMeta):
    def __init__(self):
        self._ndim = 1
        self._points = None
        self._simplices = None

    @property
    def ndim(self):
        return self._ndim

    @property
    def npoints(self):
        return self._points.shape[0]

    @property
    def points(self):
        """
        获取节点坐标列表.

        Returns
        -------
        self._points: np.ndarray, (N, Dim)
            节点坐标列表.
        """
        return self._points

    @property
    def nsimplices(self):
        return self._simplices.shape[0]

    @property
    def simplices(self):
        """
        获取单纯形节点编号列表.

        Returns
        -------
        self._simplices: np.ndarray, (M, Dim+1)
            单纯形节点编号列表.
        """
        return self._simplices

    @abc.abstractmethod
    def create(self, *args, **kwargs):
        """
        更新节点及单纯形.
        """
        raise NotImplementedError


class TriangularMesh(metaclass=abc.ABCMeta):
    def __init__(self):
        self._ndim = 2
        self._stri = None

    @property
    def ndim(self):
        return self._ndim

    @property
    def npoints(self):
        return self._stri.npoints

    @property
    def points(self):
        """
        获取节点坐标列表.

        Returns
        -------
        self._points: np.ndarray, (N, Dim)
            节点坐标列表.
        """
        return self._stri.points

    @property
    def nsimplices(self):
        return self._stri.nsimplex

    @property
    def simplices(self):
        """
        获取三角形节点编号列表.

        Returns
        -------
        self._simplices: np.ndarray, (M, Dim+1)
            单纯形节点编号列表.
        """
        return self._stri.simplices

    @abc.abstractmethod
    def create(self, *args, **kwargs):
        """
        更新节点及单纯形.
        """
        raise NotImplementedError

