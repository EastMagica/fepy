#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/2/5 13:34
# @file    : generator.py
# @project : fepy
# software : PyCharm

from fepy.mesh.basic import (
    IntervalMesh, TriangularMesh, parse_box, parse_n, split_range
)


# Classes
# -------

class UnitIntervalMesh(IntervalMesh):
    """
    一维规则网格.
    """
    def __init__(self, box=None, n=100):
        super().__init__()
        self.create(box, n)

    def create(self, box, n):
        box = parse_box(box, self.ndim)
        n = parse_n(n, self.ndim)
        self._points, self._simplices = split_range(box, n)


class UnitSquareMesh(TriangularMesh):
    """
    二维规则三角网格.
    """
    def __init__(self, box=None, n=100):
        super().__init__()
        self.create(box, n)

    def create(self, box, n):
        box = parse_box(box, self.ndim)
        n = parse_n(n, self.ndim)
        self._stri = split_range(box, n)
