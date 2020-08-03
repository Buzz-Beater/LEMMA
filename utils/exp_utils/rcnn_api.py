"""
  
    Created on 2/15/20

    @author: Baoxiong Jia

    Description:

"""
import argparse
import json
import glob
import sys
import pickle
import random

from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from datasets.utils import parse_gt_result
from config.config import Config

sys.path.append('/home/baoxiong/Projects/DOME/experiments/src')

def get_data_dicts(file, data_path):
    dataset_dicts = {}
    symbols = ['vid_name', 'frame_id', 'pid', 'flow', 'rgb', 'bbox', 'action', 'task', 'object']
    with Path(file).open('rb') as f:
        frame_infos = pickle.load(f)
    for frame_info in frame_infos:
        vid_name = frame_info[symbols.index('vid_name')]
        frame_id = frame_info[symbols.index('frame_id')]
        pid = frame_info[symbols.index('pid')]
        img_path = frame_info[symbols.index('rgb')][0]
        bbox = frame_info[symbols.index('bbox')]
        # bbox = np.true_divide(bbox, 2)
        height = 540
        width = 960
        obj = {
            "bbox": [max(0, bbox[0]), max(0, bbox[1]),
                     min(bbox[2], width), min(height, bbox[3])],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [],
            "category_id": 0,
            "iscrowd": 0
        }

        id = vid_name + '$' + str(frame_id)
        if id not in dataset_dicts.keys():
            dataset_dicts[id] = {}
            dataset_dicts[id]['file_name'] = str(data_path / img_path)
            dataset_dicts[id]['image_id'] = id
            dataset_dicts[id]['height'] = height
            dataset_dicts[id]['width'] = width
            dataset_dicts[id]['annotations'] = []
        dataset_dicts[id]['annotations'].append(obj)

    dataset_dicts = [val for _, val in dataset_dicts.items()]
    return dataset_dicts

def main(args):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))

    dataset_base_path = Path('/home/baoxiong/Datasets')
    data_path = dataset_base_path / 'data'
    anno_path = dataset_base_path / 'annotations' / 'full'
    detection_path = dataset_base_path / 'detections'
    rcnn_cache_path = detection_path / 'cache'

    dataset_list = ['rec_tpv_frames']

    for d in dataset_list:
        DatasetCatalog.register(d, lambda d=d: get_data_dicts(anno_path / '{}.p'.format(d), data_path))
        MetadataCatalog.get(d).set(thing_classes=['person'])
    if args.vis:
        dataset_dicts = get_data_dicts(anno_path / 'rec_tpv_frames.p', data_path)
        metadata = MetadataCatalog.get("rec_tpv_frames")
        for d in random.sample(dataset_dicts, 5):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
            vis = visualizer.draw_dataset_dict(d).get_image()
            plt.imshow(vis)
            plt.show()

    cfg.OUTPUT_DIR = args.output
    cfg.DATASETS.TRAIN = ("rec_tpv_frames_full",)
    cfg.DATASETS.TEST = ("rec_tpv_frames_full",)
    cfg.DATALOADER.NUM_WORKERS = 8

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 210000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    if not Path(cfg.OUTPUT_DIR).exists():
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if args.mode == 'train':
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = str(Path(cfg.OUTPUT_DIR) / 'model_0067999.pth')
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=True)
        trainer.train()
    else:
        cfg.MODEL.WEIGHTS = str(Path(cfg.OUTPUT_DIR) / 'model_0007099.pth')
        predictor = DefaultPredictor(cfg)
        for dataset in dataset_list:
            dataset_anno_path = anno_path / '{}.p'.format(dataset)
            dataset_dicts = get_data_dicts(dataset_anno_path, data_path)
            pred_infos = []
            for idx, d in enumerate(dataset_dicts):
                if (idx + 1) % 1000 == 0:
                    print('Finished detection for {}/{} images'.format(idx + 1, len(dataset_dicts)))
                img = cv2.imread(d["file_name"])
                vid_name, frame_id = d["image_id"].split('$')
                cache_file_path = rcnn_cache_path / '{}_{}.p'.format(vid_name, frame_id)
                if cache_file_path.exists() and args.cache:
                    with cache_file_path.open('rb') as f:
                        outputs = pickle.load(f)
                else:
                    outputs = predictor(img)
                    pred_boxes = [pred_box.cpu().numpy() for pred_box in outputs["instances"].pred_boxes]
                    outputs["instances"].pred_boxes = pred_boxes
                    pred_scores = [pred_score.cpu().numpy() for pred_score in outputs["instances"].scores]
                    outputs["instances"].scores = pred_scores

                    if not cache_file_path.parent.exists():
                        cache_file_path.parent.mkdir(parents=True)
                    with cache_file_path.open('wb') as f:
                        pickle.dump(outputs, f)

                if len(outputs['instances'].pred_boxes) == 0 and len(d['annotations']) != 0:
                    print(vid_name, frame_id)

                pred_boxes = outputs['instances'].pred_boxes
                pred_scores = outputs["instances"].scores

                if args.vis:
                    v = Visualizer(img[:, :, ::-1],
                                   metadata=metadata,
                                   scale=0.8,
                                   instance_mode=ColorMode.IMAGE
                                   )
                    vis = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
                    plt.imshow(vis)
                    plt.show()

                for pred_box, pred_score in zip(pred_boxes, pred_scores):
                    pred_infos.append([vid_name, int(frame_id), pred_box, pred_score])

            if not detection_path.exists():
                detection_path.mkdir()
            with (detection_path / '{}.p'.format(dataset)).open('wb') as f:
                pickle.dump(pred_infos, f)
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        from detectron2.data import build_detection_test_loader
        evaluator = COCOEvaluator('rec_tpv_frames_full', cfg, False, output_dir=cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(cfg, 'rec_tpv_frames_full')
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=True)
        inference_on_dataset(trainer.model, val_loader, evaluator)


if __name__ == '__main__':
    paths = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='code mode (train / inference)')
    parser.add_argument('--output', type=str, default= str(paths.tmp_path / 'rcnn_checkpoints'), help='rcnn save')
    parser.add_argument('--cache', type=bool, default=True, help='whether or not cache inference results for images')
    parser.add_argument('--vis', type=bool, default=False, help='whether or not cache inference results for images')
    args = parser.parse_args()
    args.paths = paths
    main(args)