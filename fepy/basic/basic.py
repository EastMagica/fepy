#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/2/5 13:12
# @file    : basic.py
# @project : fepy
# software : PyCharm

import numpy as np


# Functions
# ---------

def is_ndarray(*args):
    """类型转换: ndarray类型"""
    ndarray_list = [np.array(arr) for arr in args]
    if len(ndarray_list) == 1:
        return ndarray_list[0]
    return ndarray_list

