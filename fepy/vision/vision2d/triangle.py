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

def init_adaptive_tri(step):
    fig1, ax1 = init_2d_figure(
        2, step,
        figsize=(2.5 * step, 5),
        sharex='all',
        sharey='all',
    )
    return fig1, ax1


def show_adaptive_tri(figs, fem, err, pointer=0):
    fig1, ax1 = figs

    plot_triplot(
        ax1[0, pointer], fem,
    )
    plot_tripcolor(
        ax1[1, pointer], fem
    )

    title = 'Origin' if pointer == 0 else f"Adaptive {pointer}"

    ax1[0, pointer].set_title(
        f'{title}\nUse Points: {fem.mesh.points.size}'
    )
    ax1[1, pointer].set_title(f'L2 Error: {err.error: .2e}', y=1.03)


def plot_tri(fem, points=False):
    fig0, ax0 = init_3d_figure()
    plot_trisurf(ax0, fem)

    fig1, ax1 = init_2d_figure(
        1, 2,
        sharex='all',
        sharey='all',
        figsize=(8, 4)
    )
    plot_tripcolor(
        ax1[0], fem
    )
    plot_triplot(
        ax1[1], fem,
    )

    if points is True:
        ax1[0].set_xlable(f'Use Points: {fem.mesh.points.size}')

    plt.show()


def init_2d_figure(*args, **kwargs):
    fig, ax = plt.subplots(
        *args, **kwargs,
        subplot_kw={
            'aspect': True
        }
    )
    return fig, ax


def init_3d_figure(*args, **kwargs):
    fig, ax = plt.subplots(
        *args, **kwargs,
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


def plot_triplot(ax, fem, title=""):
    ax.set_title(title)
    ax.triplot(
        fem.mesh.mtri,
        color='tab:blue',
        marker='.'
    )
