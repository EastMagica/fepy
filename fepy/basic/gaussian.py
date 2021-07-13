#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/2/6 16:12
# @file    : gaussian.py
# @project : fepy
# software : PyCharm

import os
import abc

import numpy as np


# Constant
# --------

__project_name__ = "fepy"

_gaussian_list = {
    '1': [
        '1', '2', '3', '4', '5'
    ],
    '2': [
        '1', '3', '4', '6', '7', '12', '13', '25', '33'
    ]
}
_gaussian_area = {'1': 2, '2': 1 / 2}


# Classes
# -------

class MetaGaussian(metaclass=abc.ABCMeta):
    """
        Methods
        ----------
        points : ndarray, (N, dim)
        weight : ndarray, (N, )
        area : float
        """

    def __init__(self, n, dim):
        self.__dim = dim
        self.__gauss_n = n
        load_gaussian(n, dim)
        self.__points, self.__weight, self.__area = load_gaussian(n, dim)

    @property
    def gauss_n(self):
        return self.__gauss_n

    @property
    def dim(self):
        return self.__dim

    @property
    def points(self):
        return self.__points

    @property
    def weight(self):
        return self.__weight

    @property
    def area(self):
        return self.__area

    @abc.abstractmethod
    def local_to_global(self, global_v):
        raise NotImplementedError


# Functions
# ---------

def find_gaussian(n, dim):
    r"""
    解析高斯积分参数文件路径.

    Parameters
    ----------
    n
    dim

    Returns
    -------
    """
    project_path = os.path.abspath(__file__).split(__project_name__)[0]
    relative_path = 'res/gauss/gauss' + dim + 'd/gauss' + str(n) + '.npy'
    absolute_path = os.path.join(project_path, relative_path)
    print('> load gaussian: ', absolute_path)
    return absolute_path


def load_gaussian(n, dim):
    r"""
    加载高斯积分参数.

    Parameters
    ----------
    n : int, str
    dim : int, str

    Returns
    -------
    points : ndarray, (N, )
    weight : ndarray, (N, )
    area : float
    """
    n = str(n) if not isinstance(n, str) else n
    dim = str(dim) if not isinstance(dim, str) else dim

    if n not in _gaussian_list[dim]:
        raise ValueError(str(n) + ' Gauss Point does not exists!')

    file_path = find_gaussian(n, dim)
    if not os.path.exists(file_path):
        raise ValueError('gauss' + str(n) + '.npy is not exist.')

    gaussian = np.load(file_path)
    points, weight = gaussian[:-1], gaussian[-1]
    area = _gaussian_area[dim]

    if dim == '2':
        points = points.T
    else:
        points = points.reshape(-1, 1)

    return points, weight, area

