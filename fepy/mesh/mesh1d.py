#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/10 15:08
# @file    : mesh1d.py
# @project : fepy
# software : PyCharm

import abc

import numpy as np

from fepy.mesh.basic import parse_n, parse_box, uniform_space


# Meta Class
# ----------

class IntervalMesh(metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        self._ndim = 1
        self._points = None
        self._simplices = None
        self._boundary = None

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


# Classes
# -------

class UniformIntervalMesh(IntervalMesh):
    """
    一维规则网格.
    """
    def __init__(self, box=None, n=100):
        super().__init__()
        if box is None:
            box = [0, 1]
        self.create(box, n)

    def create(self, box, n):
        self._points = uniform_space(box, n+1)
        self._simplices = np.vstack([
            np.arange(n), np.arange(1, n+1)
        ])
        self._boundary = np.array([0, n])





