#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/2/5 13:16
# @file    : interval.py
# @project : fepy
# software : PyCharm

import abc

import numpy as np


# Functions
# ---------

def parse_box(box=None):
    """
    解析 Box 参数.

    Parameters
    ----------
    box: array_like
        shape is (ndim, 2),
        but (2, ) in 1-dim

    Returns
    -------
    ndarray
        shape is (ndim, 2)
    """
    box = np.asarray(box)

    if box.ndim == 1 and box.size == 2:
        return box.reshape(1, 2)
    elif box.ndim == 2 and box.shape[1] == 2:
        return box
    else:
        raise ValueError('Box: {} is out of Specification.'.format(box))


def parse_n(n):
    """
    解析 n 参数.

    Parameters
    ----------
    n: array_like

    Returns
    -------

    """
    n = np.asarray(n, dtype=int)

    if n.size == 1:
        return n.flatten()
    return n.squeeze()


def uniform_space(box, n, opt_out=False, module='linspace'):
    """
    规则网格剖分.

    Parameters
    ----------
    box: array_like
    n: array_like
    opt_out: bool, optional
        whether return options
    module: str, optional
        linspace, uniform

    Returns
    -------

    """
    box = parse_box(box)
    n = parse_n(n)

    if module == 'linspace':
        axis_space = [
            np.linspace(*interval, n0)
            for k, (interval, n0) in enumerate(zip(box, n))
        ]
    elif module == 'uniform':
        axis_space = [
            np.sort(np.random.uniform(*interval, n0))
            for k, (interval, n0) in enumerate(zip(box, n))
        ]
    else:
        raise ValueError

    points = np.array(
        np.meshgrid(*axis_space, indexing='ij')
    ).reshape(n.size, -1)

    opt = {
        'box': box,
        'n': n,
        'axis': axis_space
    }

    if opt_out is False:
        return points.T
    return points.T, opt


# Meta Classes
# ------------

class MetaMesh(metaclass=abc.ABCMeta):
    def __init__(self):
        self._values = None

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, data):
        self._values = data

    @abc.abstractmethod
    def get_format_value(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def ndim(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def npoints(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def points(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def nsimplices(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def simplices(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def neighbors(self):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def area(v):
        raise NotImplementedError

    def save(self):
        # TODO: mesh save method
        ...
