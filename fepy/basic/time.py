#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/2/6 16:12
# @file    : time.py
# @project : fepy
# software : PyCharm

import datetime


def get_now(fmt=None):
    fmt = fmt if fmt else '%Y%m%d%%H%M%S'
    now_time = datetime.datetime.now()
    return now_time.strftime(fmt)


def get_now_name(fms: str, fmt: str = None):
    return fms.format(get_now(fmt))
