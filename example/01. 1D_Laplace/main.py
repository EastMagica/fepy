#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/10 15:17
# @file    : main.py
# @project : fepy
# software : PyCharm

import numpy as np

import matplotlib.pyplot as plt

from fepy.fem.fem1d import LinearFEM1D
from fepy.mesh.mesh1d import UniformIntervalMesh
from fepy.boundary.boundary import Dirichlet
from fepy.error.priori import L2Error
from fepy.error.posteriori import L2FError

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


fem = LinearFEM1D(
    variation=variation,
    mesh=UniformIntervalMesh(
        box=[0, 1], n=32 + 1
    ),
    boundary=Dirichlet(
        np.array([0., 0.])
    ),
    gaussian_n=3
)

fem.run()

err = L2Error(
    fem, u_true
)
print(f">>> L2 Error: {err.error: .8e}")

err = L2FError(
    fem, f
)
print(f">>> L2 Error: {err.error: .8e}")

# plots
# -----

fig, ax = plt.subplots(1, 2)

ax[0].plot(fem.mesh.points.squeeze(), fem.mesh.values)
ax[0].plot(np.linspace(0, 1, 100), u_true(np.linspace(0, 1, 100)), linestyle='--')

ax[1].plot(
    fem.mesh.points.squeeze(),
    fem.mesh.values - u_true(fem.mesh.points.squeeze())
)

plt.show()
