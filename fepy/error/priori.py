#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/10 14:08
# @file    : priori.py
# @project : fepy
# software : PyCharm

import abc


class ErrorEstimate(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def simplices(self):
        raise NotImplementedError

    @abc.abstractmethod
    def error(self):
        raise NotImplementedError


class L2Error(ErrorEstimate):
    def simplices(self):
        ...

    def error(self):
        ...

    # def error_simplices(self, u_true=None):
    #     r"""单元误差估计"""
    #     if u_true:
    #         error_lst = np.zeros(self.mesh.nsimplices)
    #         for k, v in enumerate(self.mesh.simplices):
    #             unit_v = self.mesh.points[v]
    #             gauss_p, gauss_w = self.gaussian.local_to_global(unit_v)
    #             basis_v = self.basis_value(gauss_p, unit_v)
    #             value_true = u_true(*gauss_p.T)
    #             value_calc = np.dot(self.mesh.values[v], basis_v.T)
    #             error_lst[k] = np.sum((value_true - value_calc) ** 2 * gauss_w)
    #     else:
    #         # TODO: 未知解析结果, 估计误差.
    #         raise NotImplementedError
    #     return error_lst
    #
    # def error_l2(self):
    #     r"""L2误差估计
    #
    #     Returns
    #     -------
    #
    #     """
    #     error2 = np.sum(self.error_simplices(self.u_true))
    #     return np.sqrt(error2)
