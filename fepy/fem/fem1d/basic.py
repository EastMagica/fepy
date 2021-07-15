#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/13 10:34
# @file    : interval.py
# @project : fepy
# software : PyCharm

import numpy as np


# Functions
# ---------

def area(*points):
    [pl, pr] = np.asarray(points)
    if (pl.ndim > 1) or (pr.ndim > 1):
        raise ValueError('Input a needs to be a (N, 1) Matrix.')
    elif pl.size != pr.size:
        raise ValueError('Input matrices need to have same raw.')
    return pr - pl



