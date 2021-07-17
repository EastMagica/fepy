#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/16 22:43
# @file    : tri_adaptive.py
# @project : fepy
# software : PyCharm

import numpy as np

from fepy.basic.time import run_time
from fepy.adaptive.basic import MetaAdaptive
from fepy.vision.vision2d.triangle import init_adaptive_tri, show_adaptive_tri


# Constant
# --------

edge_point_index = np.array([[0, 1], [1, 2], [2, 0]], dtype=int)


# Classes
# -------

class AdaptiveTri2D(MetaAdaptive):
    def __init__(self, fem, f, step=3, err_class=None):
        super().__init__(fem, f, step, err_class)
        self.figure = init_adaptive_tri(self.step)

    @run_time("Adaptive 2D Tri Mesh")
    def adaptive(self):
        error = self.err.error_simplices
        index, = np.where(error > np.average(error))

        insert_points_index = list()

        for k, v in enumerate(self.fem.mesh.simplices[index]):
            insert_points_index.extend(
                tri_add_point(v, self.fem.mesh.points[v])
            )

        insert_points_index = unique_point_index(insert_points_index)
        insert_points = np.asarray([
            (self.fem.mesh.points[item[0]] + self.fem.mesh.points[item[1]]) / 2
            for item in insert_points_index
        ])
        new_points = np.vstack([
            self.fem.mesh.points,
            insert_points
        ])
        self.fem.mesh.create(new_points)
        self.fem.init_values()

    def vision(self, pointer):
        show_adaptive_tri(
            self.figure, self.fem, self.err, pointer
        )


# Functions
# ---------

def unique_point_index(point_index):
    point_index_set = set()
    for item in point_index:
        sort_item = sorted(item)
        point_index_set.add(
            f"{sort_item[0]}+{sort_item[1]}"
        )
    point_index_list = [
        [int(n_item) for n_item in item.split('+')]
        for item in point_index_set
    ]
    return point_index_list


def tri_add_point(unit_v, unit_p):
    side = tri_side(unit_p)
    side_index = np.argsort(side)
    split_edge = list()
    split_edge.append(
        edge_point_index[side_index[2]]
    )
    if side[side_index[1]] > side[side_index[0]] * 2:
        split_edge.append(
            edge_point_index[side_index[1]]
        )
    insert_point_index = [
        [unit_v[n0], unit_v[n1]]
        for n0, n1 in split_edge
    ]
    return insert_point_index


def tri_side(unit_p):
    side_lens = np.array([
        len_side(*unit_p[edge_point_index[0]]),
        len_side(*unit_p[edge_point_index[1]]),
        len_side(*unit_p[edge_point_index[2]]),
    ])
    return side_lens


def len_side(p0, p1):
    return np.sqrt(np.sum((p0 - p1) ** 2))
