#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/16 22:14
# @file    : main.py
# @project : fepy
# software : PyCharm

import numpy as np

import matplotlib.pyplot as plt

from fepy.fem.fem1d import LinearFEM1D
from fepy.mesh.mesh1d import UniformIntervalMesh
from fepy.boundary.boundary import Dirichlet
from fepy.mesh.mesh1d import Adaptive1D
from fepy.error.priori import L2Error
from fepy.error.posteriori import L2FError
from fepy.vision.vision1d import plot_interval, plot_interval_err


# Functions
# ---------

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
        box=[0, 1], n=8 + 1
    ),
    boundary=Dirichlet(
        np.array([0., 0.])
    ),
    gaussian_n=3
)

adp = Adaptive1D(
    err=L2FError, fem=fem, f=f
)
adp.fem.run()
print(f"{fem.mesh.points.size=}")
plot_interval(fem, u_true)
plt.show()

adp.adaptive()
adp.fem.run()
print(f"{fem.mesh.points.size=}")
plot_interval(fem, u_true)
plt.show()

adp.adaptive()
adp.fem.run()
print(f"{fem.mesh.points.size=}")
plot_interval(fem, u_true)
plt.show()
