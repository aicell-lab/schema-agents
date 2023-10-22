#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/1 11:59
@Author  : alexanderwu
@File    : const.py
"""
import os

PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.getcwd())
DATA_PATH = PROJECT_ROOT / 'data'

MEM_TTL = 24 * 30 * 3600
