#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/14 12:39
# @file    : order.py.py
# @project : fepy
# software : PyCharm

import numpy as np

import matplotlib.pyplot as plt

from fepy.fem.fem1d import LinearFEM1D
from fepy.mesh.mesh1d import UniformIntervalMesh
from fepy.boundary.boundary import Dirichlet

r"""
1D Laplace Equations Example

.. math::

    \begin{cases}
        -\Delta u = \pi^2 \sin(\pi x) \\
        u(0) = u(\pi) = 0
    \end{cases}

true solution is

.. math::

    u = \sin(\pi x)

"""

def u_true(x):
    return np.sin(np.pi * x)


def f(x):
    return np.pi ** 2 * np.sin(np.pi * x)


def variation(basis_v, basis_g, gauss_p, gauss_w):
    f_v = f(gauss_p.squeeze())
    f_elem = np.dot(f_v * gauss_w, basis_v)
    a_elem = np.dot(basis_g.T, basis_g) * np.sum(gauss_w)
    return a_elem, f_elem


error_list = []

for i in range(1, 12):
    fem = LinearFEM1D(
        variation=variation,
        mesh=UniformIntervalMesh(
            box=[0, 1], n=2**i + 1
        ),
        boundary=Dirichlet(
            np.array([0., 0.])
        ),
        gaussian_n=5
    )
    fem.run()

    error_list.append(
        np.max(np.abs(u_true(fem.mesh.points.squeeze()) - fem.mesh.values))
    )

print(f"    error\t   order")
print(f"{error_list[0]: .4e}\t")
for i in range(1, len(error_list)):
    print(
        f"{error_list[i]: .4e}\t  {np.log2(error_list[i-1] / error_list[i]):.4f}"
    )
