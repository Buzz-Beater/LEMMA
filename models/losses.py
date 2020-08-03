#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""
import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from datasets.metadata import Metadata

class FusedLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(FusedLoss, self).__init__(size_average, reduce, reduction)
        self.reduction = reduction

    def forward(self, input, labels, meta, cfg):
        loss = 0
        label_logits, act_logits, object_logits, task_logits, _ = input
        mapping = {'verb': 'action', 'noun': 'object', 'hoi': 'hoi'}
        pos_weight = {'verb': 1, 'task': 1, 'noun': 1, 'hoi': 5}
        loss += F.binary_cross_entropy_with_logits(label_logits, labels, reduction=self.reduction,
                                                   pos_weight=torch.tensor(
                                                       [cfg.TRAIN.BCE_POS_WEIGHT]
                                                   ).cuda(non_blocking=True))

        if cfg.EXP.SUPERVISION != 'none':
            num_actions = meta['num_actions']
            task_labels = meta['task_labels']
            for b_idx in range(task_logits.size(0)):
                num_action = int(num_actions[b_idx].item())
                loss += F.binary_cross_entropy_with_logits(task_logits[b_idx][:num_action],
                                                           task_labels[b_idx][:num_action],
                                                           reduction=self.reduction, pos_weight=torch.tensor(
                        [pos_weight['task']]
                    ).cuda(non_blocking=True))

        if cfg.EXP.MODEL_TYPE != 'plain':
            action_labels = meta['action_labels']
            object_labels = meta['obj_labels']
            num_objects = meta['num_objects']
            num_actions = meta['num_actions']
            loss += F.cross_entropy(act_logits.view(-1, act_logits.size(-1)), action_labels.view(-1, action_labels.size(-1)).squeeze(-1),
                                    ignore_index=-1,
                                    reduction=self.reduction)
            for b_idx in range(act_logits.size(0)):
                num_action = int(num_actions[b_idx].item())
                for a_idx in range(num_action):
                    obj_pos = int(num_objects[b_idx][a_idx].item())
                    if obj_pos == 0:
                        continue
                    loss += F.binary_cross_entropy_with_logits(object_logits[b_idx][a_idx][:obj_pos],
                                                               object_labels[b_idx][a_idx][:obj_pos],
                                                               reduction=self.reduction,
                                                               pos_weight=torch.tensor([
                                                                   pos_weight['noun']
                                                               ]).cuda(non_blocking=True))
        return loss
