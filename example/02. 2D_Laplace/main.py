#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/15 10:34
# @file    : main.py
# @project : fepy
# software : PyCharm

import time

import numpy as np
import matplotlib.pyplot as plt

from fepy.fem.fem2d import TriLinearFEM2D
from fepy.mesh.mesh2d import UniformSquareTriMesh
from fepy.boundary.boundary import Dirichlet

r"""
\nabla^2 u(x, y) + \nabla u(x, y) + u(x, y) = f(x, y)


\nabla^2 u
----------

(\nabla^2 u, \phi_j)
= -(\nabla^2 u, \nabla \phi_j)
= -(\sum^n_{i=1} u_i \nabla \phi_i, \nabla \phi_j)
= -\sum^n_{i=1} \left(u_i (\nabla \phi_i, \nabla \phi_j) \right)


\partial u
----------

(\nabla u, \phi_j)
= (\sum^n_{i=1} u_i \partial \phi_i, \phi_j)
= \sum^n_{i=1} \left(u_i (\partial \phi_i, \phi_j) \right)


u
----------

(u, \phi_j)
= (\sum^n_{i=1} u_i \phi_i, \phi_j)
= \sum^n_{i=1} \left(u_i (\phi_i, \phi_j) \right)


matrix compute
--------------

2 dim, triangular mesh, linear basis function

.. Tips: basis grad is constant in linear basis function.

gauss_n = int: N
basis_v = ndarray, (N, 3)
-------------------------
[[basis_0(p[0]), basis_1(p[0]), basis_2(p[0])],
 .............., ............., ..............,
 [basis_0(p[n]), basis_1(p[n]), basis_2(p[n])],]

basis_g = ndarray, (2, 3)
-------------------------
[[basis_0_x(p[0]), basis_1_x(p[0]), basis_2_x(p[0])],
 [basis_0_y(p[0]), basis_1_y(p[0]), basis_2_y(p[0])]]

basis_g = ndarray, (2, N, 3)
----------------------------
[[[basis_0_x(p[0]), basis_1_x(p[0]), basis_2_x(p[0])],
  ................, ..............., ................,
  [basis_0_x(p[n]), basis_1_x(p[n]), basis_2_x(p[n])],],
 [[basis_0_y(p[0]), basis_1_y(p[0]), basis_2_y(p[0])],
  ................, ..............., ................,
  [basis_0_y(p[n]), basis_1_y(p[n]), basis_2_y(p[n])]]]


\nabla^2 u
----------
-\sum^n_{i=1} \left(u_i (\nabla \phi_i, \nabla \phi_j) \right)

if grad is constant
^^^^^^^^^^^^^^^^^^^

> np.dot(basis_g.T, basis_g) * np.sum(gauss_w)

[[u_0_x, u_0_y],
 [u_1_x, u_1_y],
 [u_2_x, u_2_y],]

[[u_0_x, u_1_x, u_2_x, ],
 [u_0_y, u_1_y, u_2_y, ]]


if grad is not constant
^^^^^^^^^^^^^^^^^^^^^^^

> basis_g_x, basis_g_y = basis_g
> np.dot(basis_g_x.T * gauss_w, basis_g_x) + np.dot(basis_g_y.T * gauss_w, basis_g_y)

.. (3, N), (3, N) = (2, 3, N)
.. (3, N) .* (N, ) * (N, 3) + (3, N) .* (N, ) * (N, 3) -> (3, 3)


\partial u
--------
\sum^n_{i=1} \left(u_i (\partial \phi_i, \phi_j) \right)

if grad is constant
^^^^^^^^^^^^^^^^^^^

> basis_g_x, basis_g_y = basis_g.transpose()
> np.dot(basis_v.T, gauss_w) * basis_g_x

.. (N, 3).T * (N,  ) -> (3, 1) * (1, 3) -> (3, 3)

if grad is not constant
^^^^^^^^^^^^^^^^^^^^^^^

> basis_g_x, basis_g_y = basis_g.transpose((0, 1, 2))
> np.dot(basis_v.T * gauss_w, basis_g_x)

u
---

\sum^n_{i=1} \left(u_i (\phi_i, \phi_j) \right)

> np.dot(basis_v.T, basis_v) * gauss_w

.. (3, N) * (N, 3) -> (3, 3)


"""

r"""
A simple example of **2D Laplace equation**. 

.. math::
    \begin{cases}
    \nabla^2 u = 2\pi^2 \sin(\pi x) \sin(\pi y) \\
    u(x, 0) = u(0, y) = u(x, 1) = u(1, y) = 0
    \end{cases}

and the variational formulation is

.. math::
    \begin{align}
    (-\nabla^2 u, v) &= (f, v) \\
    (\nabla u, \nabla v) &= (f, v) \\
    (\sum^n_{i=0} u_i \nabla \phi_i, \nabla \phi_j) &= (f, \phi_j),\quad j=1,\cdots,n \\
    \sum^n_{i=0} u_i (\nabla \phi_i, \nabla phi_j) &= (f, \phi_j),\quad j=1,\cdots,n \\
    \end{align}

and the *exact solution* is :math:`u = \sin(\pi x) \sin(\pi y)`.

The default is the **linear basis functions**.
"""

__test_name__ = '02.2d_laplace'


# Init condition
# --------------

def f(x, y):
    return 2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)


def u_true(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def variation(basis_v, basis_g, gauss_p, gauss_w):
    f_v = f(*gauss_p.T)
    f_elem = np.dot(f_v * gauss_w, basis_v)
    a_elem = np.dot(basis_g.T, basis_g) * np.sum(gauss_w)
    return a_elem, f_elem


# compute numerical solution
# --------------------------

fem = TriLinearFEM2D(
    variation=variation,
    mesh=UniformSquareTriMesh(
        [[0, 1], [0, 1]], [32 + 1, 32 + 1], 'linspace'
    ),
    boundary=Dirichlet(0.),
    gaussian_n=3
)

fem.run()

# Plots
# -----

# ut = u_true(fem.mesh.points[:, 0], fem.mesh.points[:, 1]).reshape(64 + 1, -1)
# error = (
#     ut - fem.values
# )
# print(f"{error=}")
# print("abs max=", np.max(np.abs(error)))

# fig, ax = plt.subplots(
#     subplot_kw={
#         'aspect': 'equal'
#     }
# )
# ax.tripcolor(
#     fem.mesh.mtri,
#     fem.mesh.values,
#     edgecolors='black',
# )
# ax.set_xlim(
#     np.min(fem.mesh.points[:, 0]),
#     np.max(fem.mesh.points[:, 0])
# )
# ax.set_ylim(
#     np.min(fem.mesh.points[:, 1]),
#     np.max(fem.mesh.points[:, 1])
# )


fig, ax = plt.subplots(
    subplot_kw={
        'proj_type': 'ortho',
        'projection': '3d'
    }
)
ax.plot_trisurf(
    fem.mesh.mtri,
    fem.mesh.values,
    cmap='YlGnBu_r',
    linewidth=0.2,
    antialiased=True
)

plt.show()
