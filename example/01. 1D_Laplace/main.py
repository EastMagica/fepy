#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/10 15:17
# @file    : main.py
# @project : fepy
# software : PyCharm

from fepy.fem.fem1d import FEM1D
from fepy.mesh.generator import UnitSquareMesh
from fepy.boundary.boundary import Dirichlet

FEM1D(
    variation=None,
    mesh=UnitSquareMesh(),
    boundary=None
)

