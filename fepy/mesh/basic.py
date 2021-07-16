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


def uniform_circle(n_radian, r_circle, center_point, opt_out=True):
    """

    Parameters
    ----------
    n_radian: array_like
    r_circle: float
    center_point: array_like
    opt_out: bool

    Returns
    -------

    """
    n_radian = np.asarray(n_radian)
    center_point = np.asarray(center_point)

    print(f"{center_point=}")

    pointer = 0
    points = np.zeros(
        (np.sum(n_radian) + 1, 2),
        dtype=float
    )

    n_layer = n_radian.size
    if isinstance(r_circle, (float, int)):
        n_radius = np.linspace(r_circle, 0, n_layer+1)[:-1]
    else:
        n_radius = np.asarray(r_circle)

    for k_layer, k_radian in enumerate(n_radian):
        k_radius = n_radius[k_layer]
        radians = np.linspace(0, 2 * np.pi, k_radian, endpoint=False)
        points[pointer:pointer+k_radian, ...] = np.transpose([
            np.cos(radians) * k_radius + center_point[0],
            np.sin(radians) * k_radius + center_point[1]
        ])
        pointer += k_radian

    points[-1, ...] = center_point[0]

    opt = {
        'r': r_circle,
        'radian': n_radian,
    }

    if opt_out is False:
        return points
    return points, opt


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

    @abc.abstractmethod
    def get_format_value(self):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def area(v):
        raise NotImplementedError

    def save(self):
        # TODO: mesh save method
        ...
