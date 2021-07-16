#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/16 21:36
# @file    : interval.py
# @project : fepy
# software : PyCharm

import numpy as np

import matplotlib.pyplot as plt


# Functions
# ---------

def plot_interval(fem, u_true=None):
    fig, ax = init_2d_figure()
    plot_uh(fem, ax)
    if u_true is not None:
        plot_u_true(fem, u_true, ax)
    ax.legend(loc='upper right')


def plot_interval_err(err, title="Error", color="tab:blue"):
    fig, ax = init_2d_figure()
    ax.set_title(title)
    ax.set_xlabel("Simplices")
    ax.set_ylabel("Error")
    ax.bar(
        np.arange(len(err.error_simplices)),
        err.error_simplices,
        color=color
    )


def init_2d_figure():
    fig, ax = plt.subplots(
        subplot_kw={
            # 'aspect': True
        }
    )
    return fig, ax


def plot_uh(fem, ax):
    ax.set_title("$u_h$")
    ax.plot(
        fem.mesh.points,
        fem.mesh.values,
        marker='.',
        color='tab:blue',
        label='uh',
        zorder=3
    )


def plot_u_true(fem, u_true, ax):
    ax.set_title("$u$")
    x = np.linspace(
        *fem.mesh.option['box'][0],
        100
    )
    ax.plot(
        x,
        u_true(x),
        linestyle='--',
        label="u true",
        color='tab:orange',
        zorder=1
    )
    ax.scatter(
        fem.mesh.points,
        u_true(fem.mesh.points),
        color='tab:orange',
        zorder=1
    )
