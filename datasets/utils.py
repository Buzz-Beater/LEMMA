"""
  
    Created on 2/16/20

    @author: Baoxiong Jia

    Description:

"""
import logging
import random
import json
import pickle
import cv2
import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from datasets.metadata import Metadata
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.structures.instances import Instances

logger = logging.getLogger(__name__)

def temporal_sample_offset(length, target_length):
    start = random.randint(0, max(length - target_length - 1, 0))
    return start


def get_frames(frame_list, src='rgb'):
    if src == 'rgb':
        imgs = [cv2.imread(img_path[0]) for img_path in frame_list]
        if all(img is not None for img in imgs):
            imgs = torch.as_tensor(np.stack(imgs))
        else:
            logger.error('Failed to load {} image'.format(src))
            raise Exception("Failed to load images {}".format(frame_list))
    else:
        img_xs = [cv2.imread(img_path[0], cv2.IMREAD_GRAYSCALE) for img_path in frame_list]
        img_ys = [cv2.imread(img_path[1], cv2.IMREAD_GRAYSCALE) for img_path in frame_list]
        if all(img is not None for img in img_xs) and all(img is not None for img in img_ys):
            # stack flow images
            img_xs = torch.as_tensor(np.stack(img_xs))
            img_ys = torch.as_tensor(np.stack(img_ys))
            print(img_xs)
            imgs = torch.cat((img_xs, img_ys), dim=0)
        else:
            logger.error('Failed to load {} image'.format(src))
            raise Exception('Failed to load {} images {}'.format(src, frame_list))
    return imgs


def to_tensor(x):
    return torch.FloatTensor(x)


def parse_gt_result(path, cfg):
    symbols = ['vid_name', 'frame_id', 'pid', 'flow', 'rgb', 'bbox', 'action', 'task', 'hoi']
    with (path).open('rb') as f:
        frame_infos = pickle.load(f)
    frame_paths = {}
    gt_bboxes_labels = {}
    gt_tasks = {}
    vid_name_to_idx = {}
    vid_idx_to_name = []
    for frame_info in frame_infos:
        vid_name = frame_info[symbols.index('vid_name')]
        frame_id = int(frame_info[symbols.index('frame_id')])
        bbox = frame_info[symbols.index('bbox')]
        task = frame_info[symbols.index('task')]
        pid = frame_info[symbols.index('pid')]
        label = frame_info[symbols.index('action')]
        hoi = frame_info[symbols.index('hoi')]
        if vid_name not in vid_name_to_idx.keys():
            idx = len(vid_name_to_idx)
            vid_name_to_idx[vid_name] = idx
            vid_idx_to_name.append(vid_name)
            gt_bboxes_labels[vid_name] = {}
            frame_paths[vid_name] = {}
            gt_tasks[vid_name] = {}
        if frame_id not in gt_bboxes_labels[vid_name].keys():
            gt_bboxes_labels[vid_name][frame_id] = [[bbox, label, task, hoi, pid]]
        else:
            gt_bboxes_labels[vid_name][frame_id].append([bbox, label, task, hoi, pid])
        if frame_id not in frame_paths[vid_name].keys():
            frame_paths[vid_name][frame_id] = frame_info[symbols.index(cfg.EXP.IMG_TYPE)]
    return frame_paths, gt_bboxes_labels, vid_idx_to_name, vid_name_to_idx


def parse_detection_result(path, cfg):
    pred_symbols = ['vid_name', 'frame_id', 'bbox', 'score']
    with path.open('rb') as f:
        pred_infos = pickle.load(f)

    bboxes_labels = {}
    for pred_info in pred_infos:
        vid_name = pred_info[pred_symbols.index('vid_name')]
        frame_id = pred_info[pred_symbols.index('frame_id')]
        bbox = pred_info[pred_symbols.index('bbox')]
        score = pred_info[pred_symbols.index('score')]
        if score < cfg.DATA.DETECTION_SCORE_THRESH:
            continue
        if vid_name not in bboxes_labels.keys():
            bboxes_labels[vid_name] = {}
        if frame_id not in bboxes_labels[vid_name].keys():
            bboxes_labels[vid_name][frame_id] = [[bbox, np.array([-1])]]
        else:
            bboxes_labels[vid_name][frame_id].append([bbox, np.array([-1])])
    return bboxes_labels


def get_selected_indices(frame_paths, bboxes_labels, vid_name_to_idx, cfg, mode, anno_rel_path, split_file=None):
    if split_file is not None:
        with split_file.open('r') as f:
            split = json.load(f)
    else:
        split = list()
    selected_indices = list()
    split += [vid_name + '|{}'.format(x) for x in ['P1', 'P2'] for vid_name in split]

    if mode == 'train':
        with (
                Path(cfg.DATA.GT_LIST_DIR) / anno_rel_path / '{}_fpv_segments_indices.p'.format(cfg.EXP.TASK, cfg.EXP.VIEW_TYPE)
            ).open('rb') as f:
            indices = pickle.load(f)
        for pair in indices:
            video_name, seg_start, length = pair
            if len(split) != 0 and video_name not in split:
                continue
            flag = False
            for (bbox, label, task, hoi, pid) in bboxes_labels[video_name][seg_start]:
                if Metadata.hoi_index["null$"] not in hoi:
                    flag = True
            if flag:
                for _ in range(cfg.DATA.TRAIN_SAMPLE_SEGMENT):
                    selected_indices.append((vid_name_to_idx[video_name], seg_start, length))
    else:
        if mode == 'train':
            target_fps = cfg.DATA.TRAIN_CLIPS_PER_SECOND
        elif mode == 'all':
            target_fps = cfg.DATA.EXT_CLIPS_PER_SECOND
        else:
            target_fps = cfg.DATA.TEST_CLIPS_PER_SECOND
        sample_freq = cfg.DATA.FPS // target_fps
        test_count = set()
        for video_name, frames in frame_paths.items():
            if len(split) != 0 and video_name not in split:
                continue
            test_count.add(video_name)
            frame_paths[video_name] = [x for _, x in frame_paths[video_name].items()]
            num_frames = len(frames)
            for i in range(num_frames):
                if (i + 1) % sample_freq == 0:

                    # Generating Non-null actions
                    flag = False
                    for (bbox, label, task, hoi, pid) in bboxes_labels[video_name][i]:
                        if mode == 'train' or mode == 'val':
                            if Metadata.hoi_index["null$"] not in hoi:
                                flag = True
                        else:
                            flag = True
                    if flag:
                        selected_indices.append((vid_name_to_idx[video_name], i, -1))
        print(len(test_count))
    return selected_indices


def visualize_sequence(img, bboxes, labels, cfg):
    mapping = {'verb': 'action', 'noun': 'object', 'hoi': 'hoi'}
    MetadataCatalog.get('vis').set(thing_classes=getattr(Metadata, mapping[cfg.EXP.LABEL_TYPE]))
    metadata = MetadataCatalog.get('vis')
    classes = list()
    boxes = list()
    for box, label in zip(bboxes, labels):
        for idx, x in enumerate(label):
            if x == 1:
                classes.append(idx)
                boxes.append(box)
    outputs = {"instances": Instances((img.shape[0], img.shape[1]), pred_boxes=boxes, pred_classes=classes)}
    v = Visualizer(img,
                    metadata=metadata,
                    scale=0.8,
                    instance_mode=ColorMode.IMAGE
                )
    vis = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
    plt.imshow(vis)
    plt.show()