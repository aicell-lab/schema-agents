#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

from loguru import logger as _logger

from schema_agents.const import PROJECT_ROOT


def define_log_level(print_level="INFO", logfile_level="DEBUG"):
    """Define log level."""
    _logger.remove()
    _logger.add(sys.stderr, level=print_level)
    _logger.add(PROJECT_ROOT / 'logs/log.txt', level=logfile_level)
    return _logger


logger = define_log_level()
