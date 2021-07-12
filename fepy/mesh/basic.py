#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/2/5 13:16
# @file    : basic.py
# @project : fepy
# software : PyCharm

import numpy as np
from scipy.spatial import Delaunay


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


def uniform_space(box, n):
    """
    规则网格剖分.

    Parameters
    ----------
    box
    n

    Returns
    -------

    """
    box = parse_box(box)
    n = parse_n(n)

    axis_array = np.zeros((n.size, n))

    for k, (interval, n0) in enumerate(zip(box, n)):
        axis_array[k, ...] = np.linspace(*interval, n0)

    points = np.array(
        np.meshgrid(*axis_array, indexing='ij')
    ).reshape(n.size, -1)

    return points.T

    # if dim == 1:
    #     points = np.linspace(*box, num=n)
    #     simplices = np.transpose([np.arange(n-1), np.arange(1, n)])
    #     return points, simplices
    # if dim == 2:
    #     x = np.linspace(*box[0], num=n[0])
    #     y = np.linspace(*box[1], num=n[1])
    #     x, y = np.meshgrid(x, y)
    #     points = np.transpose([x.flatten(), y.flatten()])
    #     return Delaunay(points)
    # raise ValueError('Splitting range Error.')

