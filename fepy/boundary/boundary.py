#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/2/6 14:50
# @file    : boundary.py
# @project : fepy
# software : PyCharm

import abc


class Boundary(metaclass=abc.ABCMeta):
    def __init__(self, mesh):
        self.mesh = mesh

    @abc.abstractmethod
    def process(self, mat_a, mat_f):
        raise NotImplementedError


class Dirichlet(Boundary):
    def process(self, mat_a, mat_f):
        r"""边界条件处理

        Notes
        -----
        第一类(Dirichlet)边界条件.
        """
        boundary = self.mesh.boundary_index
        values = self.mesh.boundary_values
        mat_f[boundary] = values
        mat_a[:, boundary] = 0
        mat_a[boundary, :] = 0
        mat_a[boundary, boundary] = 1


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
