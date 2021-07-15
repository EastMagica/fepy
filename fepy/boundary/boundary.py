#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/2/6 14:50
# @file    : boundary.py
# @project : fepy
# software : PyCharm

import abc


class Boundary(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def process(self, mat_a, mat_f, mesh):
        raise NotImplementedError


class Dirichlet(Boundary):
    def __init__(self, values):
        self.values = values

    def process(self, mat_a, mat_f, mesh):
        r"""边界条件处理

        Notes
        -----
        第一类(Dirichlet)边界条件.
        """
        bd_ind = mesh.boundary_index
        mat_f[bd_ind, ...] = self.values
        mat_a[:, bd_ind] = 0
        mat_a[bd_ind, :] = 0
        mat_a[bd_ind, bd_ind] = 1


class Neumann(Boundary):
    def process(self, mat_a, mat_f):
        ...


class Robin(Boundary):
    def process(self, mat_a, mat_f):
        ...


# def boundary(self):
#     r"""边界条件
#
#     Returns
#     -------
#
#     Notes
#     -----
#     此处边界条件仅为第一类(Dirichlet)边界条件.
#     """
#     boundary = self.mesh.boundary_index
#     values = self.mesh.boundary_values
#     self.f[boundary] = values
#     self.a[:, boundary] = 0
#     self.a[boundary, :] = 0
#     self.a[boundary, boundary] = 1
