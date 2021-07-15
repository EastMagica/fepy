#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/2/5 13:15
# @file    : interval.py
# @project : fepy
# software : PyCharm

import abc

import numpy as np
from scipy.linalg import solve
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import spsolve

from fepy.basic.time import run_time


# Functions
# ---------

@run_time("FSolve")
def fslove(a_mat, f_lst):
    """线性方程组求解器

    TODO: 稀疏矩阵线性代数解法, 减少内存占用

    Parameters
    ----------
    a_mat
    f_lst

    Returns
    -------
    ndarray
    """
    return spsolve(a_mat.tocsr(), f_lst.tocsr())


# Example: Laplace variation
# --------------------------
# def variation(basis_v, basis_g, gauss_p, gauss_w):
#     f_v = f(*gauss_p.T)
#     f_elem = np.dot(f_v * gauss_w, basis_v)
#     a_elem = np.dot(basis_g.T, basis_g) * np.sum(gauss_w)
#     return a_elem, f_elem


# Classes
# -------

class FEM(metaclass=abc.ABCMeta):
    def __init__(self, variation, mesh, boundary):
        """

        Parameters
        ----------
        mesh: MetaMesh
        boundary: Boundary
        variation: Callable
        """
        self.ndim = None
        self.mesh = mesh
        self.gaussian = None
        self.boundary = boundary
        self.variation = variation
        self.f = dok_matrix((self.mesh.npoints, 1), dtype=float)
        self.a = dok_matrix((self.mesh.npoints, self.mesh.npoints), dtype=float)

    @property
    def values(self):
        return self.mesh.get_format_value()

    @abc.abstractmethod
    def basis_value(self, p, v):
        """
        basis function.

        Parameters
        ----------
        p: any point in element
        v: element point

        Returns
        -------

        """
        raise NotImplementedError

    @abc.abstractmethod
    def basis_grid(self, p, v):
        """
        gradiant of basis function.

        Parameters
        ----------
        p: any point in element
        v: element point

        Returns
        -------

        """
        raise NotImplementedError

    def construct_af(self, unit_v):
        r"""构造单元矩阵

        .. math::
            \iint_K f\cdot v\mathrm{d}x\mathrm{d}y =
            \sum_{i=1}^n f(p_i) \cdot \phi(p_i) w_i =
            (f, \phi_i)

         .. math::
            \iint_K \nabla u\cdot\nabla v\mathrm{d}x\mathrm{d}y =
            \sum_{i=1}^n (\nabla u(p_i), \nabla u(p_i)) u_j

        Parameters
        ----------
        unit_v : array_like
            单元顶点坐标.

        Returns
        -------
        a_elem : ndarray
            单元刚度矩阵.
        f_elem : ndarray
            单元载荷矩阵.

        Notes
        -----
        此处:math:`A_i`的计算方法只针对
        右端项为:math:`\Delta u`的情形.
        """
        area = self.mesh.area(unit_v)
        gauss_p, gauss_w = self.gaussian.local_to_global(unit_v, area)
        basis_v = self.basis_value(
            gauss_p, unit_v
        )
        basis_g = self.basis_grid(
            gauss_p, unit_v
        )
        a_elem, f_elem = self.variation(
            basis_v, basis_g, gauss_p, gauss_w
        )
        return a_elem, f_elem

    @run_time("Assembly AF")
    def assembly_af(self):
        r"""组装总矩阵"""
        for k, v in enumerate(self.mesh.simplices):
            xm, ym = np.meshgrid(v, v)
            a_elem, f_elem = self.construct_af(self.mesh.points[v])
            self.f[v] += f_elem[np.newaxis, ...].T
            self.a[xm, ym] += a_elem

    @run_time("RUN FEM")
    def run(self):
        r"""
        计算.

        Returns
        -------

        """
        # print("> Assembly Matrix A and F...")
        self.assembly_af()
        # print("> Apply Boundary Conditions...")
        self.boundary.process(self.a, self.f, self.mesh)
        # print("> Solve Matrix U...")
        self.mesh.values = fslove(self.a, self.f)

    def save(self, name='data.xml'):
        self.mesh.save(name)
