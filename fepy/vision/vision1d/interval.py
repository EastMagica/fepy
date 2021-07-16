#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/16 21:36
# @file    : interval.py
# @project : fepy
# software : PyCharm

import matplotlib.pyplot as plt


# Functions
# ---------

def plot_interval(fem):
    fig, ax = init_2d_figure()
    plot_line(fem, ax)


def init_2d_figure():
    fig, ax = plt.subplots(
        subplot_kw={
            'aspect': True
        }
    )
    return fig, ax


def plot_line(fem, ax):
    ax.plot(
        fem.mesh.points,
        fem.mesh.values,
        marker='.'
    )
