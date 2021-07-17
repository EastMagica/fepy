#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2021/7/17 14:24
# @file    : basic.py
# @project : fepy
# software : PyCharm

import abc

from fepy.error.posteriori import L2FError


# Meta Class
# ----------

class MetaAdaptive(metaclass=abc.ABCMeta):
    def __init__(self, fem, f, step=3, err_class=None):
        if err_class is None:
            self.err_class = L2FError
        else:
            self.err_class = err_class

        self.f = f
        self.fem = fem
        self.err = None
        self.step = step

    def run(self):
        self.fem.run()
        self.err = self.err_class(self.fem, self.f)
        self.vision(0)
        for i in range(1, self.step):
            print("----------------")
            self.adaptive()
            self.fem.run()
            self.err = self.err_class(self.fem, self.f)
            self.vision(i)

    @abc.abstractmethod
    def adaptive(self):
        raise NotImplementedError

    @abc.abstractmethod
    def vision(self, pointer):
        raise NotImplementedError
