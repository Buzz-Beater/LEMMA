"""
  
    Created on 2/22/20

    @author: Baoxiong Jia

    Description:

"""

import torch
from datasets.metadata import Metadata

@torch.no_grad()
def gen_pred_labels(composed_labels):
    act_num = 3
    obj_num = 3
    batch_size = composed_labels.size(0)
    act_pred_labels = torch.zeros(batch_size, len(Metadata.action)).cuda(non_blocking=True)
    obj_pred_labels = torch.zeros(batch_size, act_num, obj_num, len(Metadata.object)).cuda(non_blocking=True)
    batch_dict = dict()
    _, hoi_indices = torch.topk(composed_labels, 3, dim=-1)
    for batch_idx in range(hoi_indices.size(0)):
        for class_idx in hoi_indices[batch_idx]:
            if batch_idx not in batch_dict.keys():
                batch_dict[batch_idx] = []
            hoi_str = Metadata.hoi[class_idx]
            act_str, obj_str = hoi_str.split('$')
            pos_strs = obj_str.split(':')
            objs = []
            for pos_idx, pos_objs in enumerate(pos_strs):
                if pos_objs != '':
                    pos_obj_ids = [Metadata.object_index[x] for x in pos_objs.split('@')[1].split('|')]
                    objs.append(pos_obj_ids)
            batch_dict[batch_idx].append([Metadata.action_index[act_str], objs])

    for batch_idx in batch_dict.keys():
        batch_dict[batch_idx] = sorted(batch_dict[batch_idx], key=lambda x : x[0])
        for a_idx, a_info in enumerate(batch_dict[batch_idx]):
            act_id, objs = a_info
            act_pred_labels[batch_idx][act_id] = 1.0
            for pos_idx, o_info in enumerate(objs):
                for obj in o_info:
                    obj_pred_labels[batch_idx][a_idx][pos_idx][obj] = 1.0

    return act_pred_labels, obj_pred_labels

@torch.no_grad()
def gen_labels(composed_labels):
    act_num = 3
    obj_num = 3
    batch_size = composed_labels.size(0)
    act_labels = torch.zeros(batch_size, len(Metadata.action)).cuda(non_blocking=True)
    obj_labels = torch.zeros(batch_size, act_num, obj_num, len(Metadata.object)).cuda(non_blocking=True)
    batch_dict = dict()
    for batch_idx in range(composed_labels.size(0)):
        for class_idx, val in enumerate(composed_labels[batch_idx]):
            if val == 0:
                continue
            if batch_idx not in batch_dict.keys():
                batch_dict[batch_idx] = []
            hoi_str = Metadata.hoi[class_idx]
            act_str, obj_str = hoi_str.split('$')
            pos_strs = obj_str.split(':')
            objs = []
            for pos_idx, pos_objs in enumerate(pos_strs):
                if pos_objs != '':
                    pos_obj_ids = [Metadata.object_index[x] for x in pos_objs.split('@')[1].split('|')]
                    objs.append(pos_obj_ids)
            batch_dict[batch_idx].append([Metadata.action_index[act_str], objs])

    for batch_idx in batch_dict.keys():
        batch_dict[batch_idx] = sorted(batch_dict[batch_idx], key=lambda x : x[0])
        for a_idx, a_info in enumerate(batch_dict[batch_idx]):
            act_id, objs = a_info
            act_labels[batch_idx][act_id] = 1.0
            for pos_idx, o_info in enumerate(objs):
                for obj in o_info:
                    obj_labels[batch_idx][a_idx][pos_idx][obj] = 1.0

    return act_labels, obj_labels

@torch.no_grad()
def eval_prediction_results(preds, labels):
    pred_labels = (torch.sigmoid(preds) >= 0.5).type(torch.float32).cuda(non_blocking=True)
    tp, tn, fp, fn = binary_correct(pred_labels, labels)
    return pred_labels, tp, tn, fp, fn


@torch.no_grad()
def binary_correct(pred, labels, meta=None):
    pred_ = (pred >= 0.5).type(torch.FloatTensor).cuda()
    truth = (labels >= 0.5).type(torch.FloatTensor).cuda()
    tp = pred_.mul(truth).sum()
    tn = (1 - pred_).mul(1 - truth).sum()
    fp = pred_.mul(1 - truth).sum()
    fn = (1 - pred_).mul(truth).sum()
    return tp, tn, fp, fn


@torch.no_grad()
def eval_pred(pred, labels, meta, cfg):
    label_logits, act_logits, object_logits, task_logits, label_features = pred
    pred_labels = (torch.sigmoid(label_logits) >= 0.5).type(torch.float32).cuda()
    tp, tn, fp, fn = binary_correct(pred_labels, labels)

    # Plain label, generate predicted labels
    if isinstance(act_logits, list) or isinstance(object_logits, list):
        act_pred_labels, obj_pred_labels = gen_pred_labels(pred_labels)
    else:
        act_num = act_logits.size(-2)
        _, act_pred_labels = torch.max(act_logits.view(-1, act_logits.size(-1)), dim=-1)
        act_pred_labels = act_pred_labels.view(-1, act_num, 1)
        act_pred_labels = (torch.sum(torch.zeros(act_pred_labels.size(0), act_pred_labels.size(1), len(Metadata.action)).cuda().scatter_(-1, act_pred_labels, 1), dim=1) > 0).type(torch.float)
        obj_pred_labels = (torch.sigmoid(object_logits) >= 0.5).type(torch.float32).cuda()
    gt_act_labels = meta['action_labels']
    gt_obj_labels = meta['obj_labels']
    return tp, tn, fp, fn, pred_labels, act_pred_labels, obj_pred_labels, gt_act_labels, gt_obj_labels

    # if cfg.EXP.MODEL_TYPE == 'plain':
    #     pred_labels = torch.sigmoid(label_logits)
    #     tp, tn, fp, fn = binary_correct(pred_labels, labels)
    # else:
    #     act_logits = torch.softmax(act_logits, )
    # return tp, tn, fp, fn

    # if cfg.EXP.MODEL_TYPE == 'plain':
    #     pred_labels = torch.sigmoid(label_logits)
    #     tp, tn, fp, fn = binary_correct(pred_labels, labels)
    #     return tp, tn, fp, fn
    # else:
    #     pass