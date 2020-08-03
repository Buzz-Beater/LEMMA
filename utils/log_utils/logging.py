#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Logging."""

import builtins
import decimal
import logging
import sys
import simplejson

import utils.model_utils.distributed as du


def _suppress_print():
    """
    Suppresses printing from the current process.
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


def setup_logging():
    """
    Sets up the log_utils for multiple processes. Only enable the log_utils for the
    master process, and suppress log_utils for the non-master processes.
    """
    # Set up log_utils format.
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"

    if du.is_master_proc():
        # Enable log_utils for the master process.
        logging.root.handlers = []
        logging.basicConfig(
            level=logging.INFO, format=_FORMAT, stream=sys.stdout
        )
    else:
        # Suppress log_utils for non-master processes.
        _suppress_print()


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)


def log_json_stats(stats):
    """
    Logs json stats.
    Args:
        stats (dict): a dictionary of statistical information to log.
    """
    stats = {
        k: decimal.Decimal("{:.5f}".format(v)) if isinstance(v, float) else v
        for k, v in stats.items()
    }
    json_stats = simplejson.dumps(stats, sort_keys=False, use_decimal=True)
    logger = get_logger(__name__)
    logger.info("{:s}".format(json_stats))
