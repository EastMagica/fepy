#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/10 15:08
# @file    : interval.py
# @project : fepy
# software : PyCharm

import abc

import numpy as np

from fepy.mesh.basic import uniform_space, MetaMesh


# Meta Class
# ----------

class MetaIntervalMesh(MetaMesh, metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._ndim = 1
        self._points = None
        self._simplices = None
        self._neighbors = None
        self._boundary_points = None

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

    @property
    def neighbors(self):
        return self._neighbors

    @property
    def boundary_index(self):
        return self._boundary_points

    @staticmethod
    def area(v):
        return np.abs(v[1] - v[0])

    @abc.abstractmethod
    def create(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def init_create(self, *args, **kwargs):
        raise NotImplementedError


# Classes
# -------

class UniformIntervalMesh(MetaIntervalMesh):
    """
    一维规则网格.
    """
    def __init__(self, box=None, n=100):
        super().__init__()
        if box is None:
            box = [0, 1]
        self.option = self.init_create(box, n)

    def __str__(self):
        return (
            "<UniformIntervalMesh "
            f"box:{self.option['box'][0]} "
            f"n:{self.option['n'][0]}>"
        )

    def __repr__(self):
        return self.__str__()

    def get_format_value(self):
        return self._values

    def create(self, points, sort=False):
        if sort is True:
            points = np.sort(points)
        n = points.size

        self._points = points
        self._simplices = np.vstack([
            np.arange(0, n - 1),
            np.arange(1, n)
        ]).T
        self._neighbors = np.vstack([
            np.arange(-1, n - 1),
            np.hstack([
                np.arange(1, n),
                -1
            ]),
        ]).T
        self._boundary_points = np.array([
            0, n - 1
        ])

    def init_create(self, box, n):
        points, option = uniform_space(
            box, n, opt_out=True
        )
        self.create(points)
        return option
