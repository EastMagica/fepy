#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/2/5 13:16
# @file    : basic.py
# @project : fepy
# software : PyCharm

import h5py
import numpy as np


# HDF5
# ----

def _save_hdf5(filename, **kwargs):
    h5_file = h5py.File(filename, 'w')
    for key, value in kwargs.items():
        h5_file[key] = value
    h5_file.close()


def _load_hdf5(filename):
    pass


# Numpy
# -----

def _save_numpy():
    pass


def _load_numpy():
    pass


# JSON
# ----

def _save_json():
    pass


def _load_json():
    pass






