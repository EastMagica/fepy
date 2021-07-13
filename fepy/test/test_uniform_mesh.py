#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/13 9:47
# @file    : test_uniform_mesh.py
# @project : fepy
# software : PyCharm

from fepy.mesh.basic import uniform_space
from fepy.mesh.mesh1d import UniformIntervalMesh

# mesh = uniform_space([0, 1], 3)
# print(f"{mesh=}")
#
# print("-"*36)
#
# mesh = uniform_space([[0, 1], [0, 1]], [3, 3])
# print(f"{mesh=}")

mesh = UniformIntervalMesh([0, 1], 8)
