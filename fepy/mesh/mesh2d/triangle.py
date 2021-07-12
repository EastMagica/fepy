#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/12 21:01
# @file    : triangle.py
# @project : fepy
# software : PyCharm

import abc

from fepy.mesh.basic import parse_box, parse_n, uniform_space


# Meta Class
# ----------

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


# Classes
# -------

class UnitSquareMesh(TriangularMesh):
    """
    二维规则三角网格.
    """
    def __init__(self, box=None, n=100):
        super().__init__()
        self.create(box, n)

    def create(self, box, n):
        # TODO: Implemented method
        box = parse_box(box)
        n = parse_n(n)
        self._stri = uniform_space(box, n)
