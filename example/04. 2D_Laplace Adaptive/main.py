#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/16 23:04
# @file    : main.py
# @project : fepy
# software : PyCharm

import numpy as np
import matplotlib.pyplot as plt

from fepy.fem.fem2d import TriLinearFEM2D
from fepy.mesh.mesh2d import UniformSquareTriMesh, RandomSquareTriMesh
from fepy.boundary.boundary import Dirichlet
from fepy.mesh.mesh2d import AdaptiveTri2D
from fepy.error.priori import L2Error
from fepy.error.posteriori import L2FError
from fepy.vision.vision2d.triangle import plot_tri


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
        [[0, 1], [0, 1]], [4 + 1, 4 + 1], 'linspace'
    ),
    # mesh=RandomSquareTriMesh(
    #     box=[[0, 1], [0, 1]],
    #     n_points=1024
    # ),
    boundary=Dirichlet(0.),
    gaussian_n=3
)

adp = AdaptiveTri2D(
    err=L2FError, fem=fem, f=f
)
adp.fem.run()
print(f"{fem.mesh.points.size=}")
plot_tri(fem)
plt.show()

adp.adaptive()
adp.fem.run()
print(f"{fem.mesh.points.size=}")
plot_tri(fem)
plt.show()

adp.adaptive()
adp.fem.run()
print(f"{fem.mesh.points.size=}")
plot_tri(fem)
plt.show()
