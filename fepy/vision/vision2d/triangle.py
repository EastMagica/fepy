#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/16 13:22
# @file    : triangle.py
# @project : fepy
# software : PyCharm

import numpy as np
import matplotlib.pyplot as plt


# Functions
# ---------

def plot_tri(fem):
    fig0, ax0 = init_3d_figure()
    plot_trisurf(ax0, fem)

    fig1, ax1 = init_2d_figure()
    plot_tripcolor(ax1, fem)

    fig2, ax2 = init_2d_figure()
    plot_triplot(ax2, fem)

    plt.show()


def init_2d_figure():
    fig, ax = plt.subplots(
        subplot_kw={
            'aspect': True
        }
    )
    return fig, ax


def init_3d_figure():
    fig, ax = plt.subplots(
        subplot_kw={
            'proj_type': 'ortho',
            'projection': '3d'
        }
    )
    return fig, ax


def plot_trisurf(ax, fem):
    ax.plot_trisurf(
        fem.mesh.mtri,
        fem.mesh.values,
        cmap='viridis',
        linewidth=0.2,
        antialiased=True
    )


def plot_tripcolor(ax, fem):
    ax.tripcolor(
        fem.mesh.mtri,
        fem.mesh.values,
        edgecolors='black',
    )
    ax.set_xlim(
        np.min(fem.mesh.points[:, 0]),
        np.max(fem.mesh.points[:, 0])
    )
    ax.set_ylim(
        np.min(fem.mesh.points[:, 1]),
        np.max(fem.mesh.points[:, 1])
    )


def plot_triplot(ax, fem):
    ax.triplot(
        fem.mesh.mtri,
        color='tab:blue',
        marker='.'
    )
