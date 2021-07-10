#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/2/5 13:15
# @file    : basic.py
# @project : fepy
# software : PyCharm

import abc

import numpy as np
from scipy.linalg import solve


# Functions
# ---------

def fslove(a_mat, f_lst):
    """线性方程组求解器

    Parameters
    ----------
    a_mat
    f_lst

    Returns
    -------

    Notes
    -----
    针对不同矩阵选择最优解法.

    """
    return solve(a_mat, f_lst)


# Classes
# -------

class FEM(metaclass=abc.ABCMeta):
    def __init__(self, variation, mesh, u_true=None):
        self.mesh = mesh
        self.u_true = u_true
        self.variation = variation
        self.gaussian = None
        self.f = np.zeros(self.mesh.npoints)
        self.a = np.zeros((self.mesh.npoints, self.mesh.npoints))

    @staticmethod
    @abc.abstractmethod
    def basis_value(p, v):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def basis_grid(p, v):
        raise NotImplementedError

    def assembly_af(self):
        r"""组装总矩阵"""
        for k, v in enumerate(self.mesh.simplices):
            xm, ym = np.meshgrid(v, v)
            a_elem, f_elem = self.construct_af(self.mesh.points[v])
            self.f[v] += f_elem
            self.a[xm, ym] += a_elem
            # 显示组装状态
            # self.calc_progress.refresh(xm, ym, v, k)
        # self.calc_progress.save_fig('test.png')

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

        Note
        ----
        此处:math:`A_i`的计算方法只针对
        右端项为:math:`\Delta u`的情形.
        """
        gauss_p, gauss_w = self.gaussian.local_to_global(unit_v)
        basis_v = self.basis_value(gauss_p, unit_v)
        basis_g = self.basis_grid(gauss_p, unit_v)
        a_elem, f_elem = self.variation(basis_v, basis_g, gauss_p, gauss_w)
        return a_elem, f_elem

    def boundary(self):
        r"""边界条件

        Returns
        -------

        Note
        ----
        此处边界条件仅为第一类(Dirichlet)边界条件.
        """
        boundary = self.mesh.boundary_index
        values = self.mesh.boundary_values
        self.f[boundary] = values
        self.a[:, boundary] = 0
        self.a[boundary, :] = 0
        self.a[boundary, boundary] = 1

    def run(self):
        r"""
        计算.

        Returns
        -------

        """
        print("> Assembly Matrix A and F...")
        self.assembly_af()
        # print(f'> a_mat0:\n{self.a_mat}\n> f_lst0:\n{self.f_lst}')
        print("> Apply Boundary Conditions...")
        self.boundary()
        # print(f'> a_mat:\n{self.a_mat}\n> f_lst:\n{self.f_lst}')
        print("> Solve Matrix U...")
        self.mesh.values = fslove(self.a, self.f)
        if self.u_true:
            print("> Solve L2 Error...")
            self.error_l2()

    def error_simplices(self, u_true=None):
        r"""单元误差估计"""
        if u_true:
            error_lst = np.zeros(self.mesh.nsimplices)
            for k, v in enumerate(self.mesh.simplices):
                unit_v = self.mesh.points[v]
                gauss_p, gauss_w = self.gaussian.local_to_global(unit_v)
                basis_v = self.basis_value(gauss_p, unit_v)
                value_true = u_true(*gauss_p.T)
                value_calc = np.dot(self.mesh.values[v], basis_v.T)
                error_lst[k] = np.sum((value_true - value_calc) ** 2 * gauss_w)
        else:
            # TODO: 未知解析结果, 估计误差.
            raise NotImplementedError
        return error_lst

    def error_l2(self):
        r"""L2误差估计

        Returns
        -------

        """
        error2 = np.sum(self.error_simplices(self.u_true))
        return np.sqrt(error2)

    def save(self, name='data.xml'):
        self.mesh.save(name)
