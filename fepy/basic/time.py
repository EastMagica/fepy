#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/2/6 16:12
# @file    : time.py
# @project : fepy
# software : PyCharm

import time
import datetime


# Functions
# ---------

def run_time(module_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            solution = func(*args, **kwargs)
            t1 = time.perf_counter()
            time_used = t1 - t0
            print(f">>> {module_name}: {time_used:.2f}s")
            return solution
        return wrapper
    return decorator


def get_now(fmt=None):
    fmt = fmt if fmt else '%Y%m%d%%H%M%S'
    now_time = datetime.datetime.now()
    return now_time.strftime(fmt)


def get_now_name(fms: str, fmt: str = None):
    return fms.format(get_now(fmt))
