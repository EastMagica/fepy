#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/16 12:41
# @file    : circle.py
# @project : fepy
# software : PyCharm

import numpy as np
import matplotlib.pyplot as plt

from fepy.fem.fem2d import TriLinearFEM2D
from fepy.mesh.mesh2d import UniformCircleTriMesh
from fepy.boundary.boundary import Dirichlet
from fepy.vision.vision2d.triangle import plot_tri


# Init condition
# --------------

def f(x, y):
    return (
        4 * np.pi * np.sin(np.pi * (x**2 + y**2))
        + 4 * np.pi ** 2 * (x**2 + y**2) * np.cos(np.pi * (x ** 2 + y ** 2))
    )


def u_true(x, y):
    return np.sin(np.pi * (x**2 + y**2))


def variation(basis_v, basis_g, gauss_p, gauss_w):
    f_v = f(*gauss_p.T)
    f_elem = np.dot(f_v * gauss_w, basis_v)
    a_elem = np.dot(basis_g.T, basis_g) * np.sum(gauss_w)
    return a_elem, f_elem


# compute numerical solution
# --------------------------

fem = TriLinearFEM2D(
    variation=variation,
    mesh=UniformCircleTriMesh(
        # radian=[32, 24, 24, 16, 12, 8],
        # radius=[1., 0.9, 0.8, 0.7, 0.6, 0.3],
        radian={8: 64},
        radius=1,
        center_point=[0, 0],
    ),
    boundary=Dirichlet(0.),
    gaussian_n=3
)

fem.run()


# Plots
# -----

plot_tri(fem)

plt.show()
