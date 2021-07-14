#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/12 21:01
# @file    : triangle.py
# @project : fepy
# software : PyCharm

import abc

import numpy as np

from fepy.mesh.basic import uniform_space, MetaMesh


# Meta Class
# ----------

class TriangularMesh(MetaMesh, metaclass=abc.ABCMeta):
    """

    Attributes
    ----------
    _stri: Delaunay
        scipy.spatial.qull.Delaunay

    """
    def __init__(self):
        super().__init__()
        self._ndim = 2
        self._stri = None
        self._boundary_points = None

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

    @property
    def neighbors(self):
        return self._stri.neighbors

    @property
    def boundary_index(self):
        if self._boundary_points is None:
            find_tri_boundary(self._stri)
        return self._boundary_points

    @staticmethod
    def area(v):
        r"""计算三角形面积.

        Parameters
        ----------
        v : array_like
            三角单元顶点坐标.

        Returns
        -------
        s : array_like, (N, )
            三角单元面积

        Notes
        -----
        若``points``为(3,2)数组, 则可使用``area(*points)``
        直接计算, 无需为对应参数而拆开数组.
        """
        p0, p1, p2 = np.asarray(v)

        s = ((p1[0] - p0[0]) * (p2[1] - p0[1]) -
             (p1[1] - p0[1]) * (p2[0] - p0[0])) / 2

        return s

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
        # TODO: tri mesh create method
        self._stri = uniform_space(box, n)


# Functions
# ---------

def find_tri_boundary(stri):
    pnt_2_bnd = np.array([
        [1, 2], [2, 0], [0, 1]
    ], dtype=int)

    bnd_simplices_index = np.any(stri.neighbors < 0, axis=1)
    bnd_simplices = stri.simplices[bnd_simplices_index]
    bnd_neighbors = stri.neighbors[bnd_simplices_index]

    bnd_index = []

    for nitem, sitem in zip(bnd_neighbors, bnd_simplices):
        for item in pnt_2_bnd[nitem < 0]:
            bnd_index.append(sitem[item])

    bnd_index = np.asarray(bnd_index)

    return bnd_index
