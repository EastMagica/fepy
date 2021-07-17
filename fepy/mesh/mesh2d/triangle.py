#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/12 21:01
# @file    : triangle.py
# @project : fepy
# software : PyCharm

import abc

import numpy as np
from scipy.spatial import Delaunay
from matplotlib.tri import Triangulation

from fepy.mesh.basic import MetaMesh, uniform_space, uniform_circle
from fepy.basic.time import run_time


# Meta Class
# ----------

class MetaTriangularMesh(MetaMesh, metaclass=abc.ABCMeta):
    """

    Attributes
    ----------
    _stri: Delaunay
        scipy.spatial.qull.Delaunay

    """

    __classname__ = "MetaTriangularMesh"

    def __init__(self):
        super().__init__()
        self._ndim = 2
        self._stri = None
        self.option = None
        self._boundary_edges = None
        self._boundary_points = None

    def __str__(self):
        return (
            f"< {self.__classname__}"
            f"box:{self.option['box']} "
            f"n:{self.option['n']}>"
        )

    def __repr__(self):
        return self.__str__()

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
            self.update_boundary_index()
        return self._boundary_points

    def update_boundary_index(self):
        edges = self.boundary_edge_index.flatten()
        self._boundary_points = np.unique(edges)

    @property
    def boundary_edge_index(self):
        if self._boundary_edges is None:
            self.update_boundary_edge_index()
        return self._boundary_edges

    def update_boundary_edge_index(self):
        self._boundary_edges = find_tri_boundary(self._stri)

    @property
    def stri(self):
        return self._stri

    @property
    def mtri(self):
        return Triangulation(
            x=self.points[:, 0],
            y=self.points[:, 1],
            triangles=self.simplices
        )

    @staticmethod
    def area(*v):
        r"""计算三角形面积.

        Parameters
        ----------
        v : array_like, (3, 2)
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
        if len(v) == 1:
            p0, p1, p2 = np.asarray(v[0])
        elif len(v) == 3:
            p0, p1, p2 = [np.asarray(item) for item in v]
            p0 = p0.T
        else:
            raise ValueError

        s = ((p1[0] - p0[0]) * (p2[1] - p0[1]) -
             (p1[1] - p0[1]) * (p2[0] - p0[0])) / 2

        return s

    def create(self, points):
        self._stri = Delaunay(points)
        self.update_boundary_edge_index()
        self.update_boundary_index()

    @abc.abstractmethod
    def init_create(self, *args, **kwargs):
        """
        更新节点及单纯形.
        """
        raise NotImplementedError


# Classes
# -------

class UniformSquareTriMesh(MetaTriangularMesh):
    """
    二维规则三角网格.
    """
    __classname__ = "UniformSquareTriMesh"

    def __init__(self, box=None, n=100, module='linspace'):
        super().__init__()
        if box is None:
            box = [[0, 1], [0, 1]]
        self.option = self.init_create(box, n, module)

    def get_format_value(self):
        return self._values.reshape(self.option['n'][0], self.option['n'][1])

    @run_time("Create UniformSquareTriMesh")
    def init_create(self, box, n, module):
        points, option = uniform_space(
            box, n, opt_out=True, module=module
        )
        self.create(points)
        return option


class UniformCircleTriMesh(MetaTriangularMesh):
    """
    二维规则圆形网格
    """
    __classname__ = "UniformCircleTriMesh"

    def __init__(self, radian, radius=1, center_point=None):
        super().__init__()
        if center_point is None:
            center_point = [0, 0]
        self.option = self.init_create(radian, radius, center_point)

    def get_format_value(self):
        return self.values

    @run_time("Create UniformCircleTriMesh")
    def init_create(self, radian, radius, center_point):
        points, option = uniform_circle(
            radian, radius, center_point, opt_out=True
        )
        self.create(points)
        return option


class RandomSquareTriMesh(MetaTriangularMesh):
    """
    二维随机矩形网格
    """
    __classname__ = "RandomSquareTriMesh"

    def __init__(self, box=None, n_points=25):
        super().__init__()
        if box is None:
            box = [[0, 1], [0, 1]]
        self.option = self.init_create(box, n_points)

    def get_format_value(self):
        return self.values

    @run_time("Create UniformCircleTriMesh")
    def init_create(self, box, n_points):
        option = {
            'box': box
        }
        points = np.hstack([
            np.random.uniform(*box[0], (n_points, 1)),
            np.random.uniform(*box[1], (n_points, 1))
        ])
        bnd0 = np.random.uniform(*box[0], (n_points // 10, 1))
        bnd1 = np.random.uniform(*box[1], (n_points // 10, 1))
        points = np.vstack([
            points,
            np.hstack([
                bnd0,
                np.full((n_points // 10, 1), fill_value=box[1][0])
            ]),
            np.hstack([
                bnd0,
                np.full((n_points // 10, 1), fill_value=box[1][1])
            ]),
            np.hstack([
                bnd1,
                np.full((n_points // 10, 1), fill_value=box[0][0])
            ]),
            np.hstack([
                bnd1,
                np.full((n_points // 10, 1), fill_value=box[0][1])
            ])

        ])
        self.create(points)
        return option


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
