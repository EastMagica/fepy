#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2019/10/7 0:23
# @file    : gauss1d.py
# @project : fem
# software : PyCharm

import numpy as np

"""
Data from Wikipedia [Gaussian quadrature](https://en.wikipedia.org/wiki/Gaussian_quadrature).
"""


# Constant
# --------

gauss_1d = {
    'area': 2.,
    '1': {
        'points': np.array([
            0
        ], dtype=np.float64).reshape(-1, 1),
        'weights': np.array([
            2
        ], dtype=np.float64),
    },
    '2': {
        'points': np.array([
            -1 / np.sqrt(3),
            1 / np.sqrt(3)
        ], dtype=np.float64).reshape(-1, 1),
        'weights': np.array([
            1,
            1
        ], dtype=np.float64),
    },
    '3': {
        'points': np.array([
            -np.sqrt(3 / 5),
            0,
            np.sqrt(3 / 5)
        ], dtype=np.float64).reshape(-1, 1),
        'weights': np.array([
            5 / 9,
            8 / 9,
            5 / 9
        ], dtype=np.float64),
    },
    '4': {
        'points': np.array([
            -np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5)),
            -np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)),
            np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)),
            np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5))
        ], dtype=np.float64).reshape(-1, 1),
        'weights': np.array([
            (18 - np.sqrt(30)) / 36,
            (18 + np.sqrt(30)) / 36,
            (18 + np.sqrt(30)) / 36,
            (18 - np.sqrt(30)) / 36
        ], dtype=np.float64),
    },
    '5': {
        'points': np.array([
            -np.sqrt(5 + 2 * np.sqrt(10 / 7)) / 3,
            -np.sqrt(5 - 2 * np.sqrt(10 / 7)) / 3,
            0,
            np.sqrt(5 - 2 * np.sqrt(10 / 7)) / 3,
            np.sqrt(5 + 2 * np.sqrt(10 / 7)) / 3
        ], dtype=np.float64).reshape(-1, 1),
        'weights': np.array([
            (322 - 13 * np.sqrt(70)) / 900,
            (322 + 13 * np.sqrt(70)) / 900,
            128 / 225,
            (322 + 13 * np.sqrt(70)) / 900,
            (322 - 13 * np.sqrt(70)) / 900
        ], dtype=np.float64),
    },
}
