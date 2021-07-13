#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/13 10:34
# @file    : basic.py
# @project : fepy
# software : PyCharm

import numpy as np
from fepy.basic.gaussian import MetaGaussian


# Functions
# ---------

def area(*points):
    [pl, pr] = np.asarray(points)
    if (pl.ndim > 1) or (pr.ndim > 1):
        raise ValueError('Input a needs to be a (N, 1) Matrix.')
    elif pl.size != pr.size:
        raise ValueError('Input matrices need to have same raw.')
    return pr - pl


class Gaussian(MetaGaussian):
    def local_to_global(self, global_v):
        global_p = self.points * (global_v[1] - global_v[0]) + global_v[0]
        global_w = area(*global_v) / self.area * self.weight
        return global_p, global_w
