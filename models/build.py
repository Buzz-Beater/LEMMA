#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Model construction functions."""

import torch
import torch.nn as nn
import models.losses as losses
from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video model_utils.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


def build_model(cfg):
    """
    Builds the video model_utils.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
    """
    assert (
        cfg.NUM_GPUS <= torch.cuda.device_count()
    ), "Cannot use more GPU devices than available"

    # Construct the model_utils
    name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(name)(cfg)
    # Determine the GPU used by the current process
    cur_device = torch.cuda.current_device()
    # Transfer the model_utils to the current GPU device
    model = model.cuda(device=cur_device)
    # Use multi-process data parallel model_utils in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model_utils replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, find_unused_parameters=True, device_ids=[cur_device], output_device=cur_device
        )
    return model

_LOSSES = {
                "cross_entropy": nn.CrossEntropyLoss, "bce": nn.BCELoss, "bce_withlogits":nn.BCEWithLogitsLoss,
                "custom_loss": losses.FusedLoss
           }

def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
