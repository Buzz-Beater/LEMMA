"""
  
    Created on 2/27/20

    @author: Baoxiong Jia

    Description:

"""

import json
import sys
sys.path.append('/home/baoxiong/Projects/DOME/experiments/src')
import argparse
import time
import pickle
import numpy as np
import pprint
import torch
from tqdm import tqdm
from pathlib import Path

import utils.model_utils.checkpoint as cu
import utils.eval_utils.metrics as metrics
import utils.model_utils.misc as misc
import utils.log_utils.logging as logging

import models.optimizer as optim
from models.pred_models import FusedLSTM, MLP, FeatureBank

from datasets import loader
from utils.log_utils.meters import PredMeter

from config.default import get_cfg, map_cfg
from config.config import Config
from datasets.metadata import Metadata


logger = logging.get_logger(__name__)


def train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg):
    model.train()
    train_meter.iter_tic()

    for cur_iter, (inputs, labels, _, meta) in enumerate(tqdm(train_loader, desc='Batch Loop Training')):
        lr = cfg.lr
        optim.set_lr(optimizer, lr)

        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        for key, val in meta.items():
            meta[key] = val.cuda(non_blocking=True)

        preds, task_preds = model(inputs, meta)

        task_labels = meta['task_labels']
        task_labels = task_labels.view(-1, task_labels.size(-1))
        task_preds = task_preds.view(-1, task_preds.size(-1))
        preds = preds.view(-1, preds.size(-1))
        labels = labels.view(-1, labels.size(-1))

        label_loss_fun = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor([20]).cuda(non_blocking=True))
        task_loss_fun = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor([5]).cuda(non_blocking=True))

        if cfg.task:
            loss = task_loss_fun(task_preds, task_labels)
        else:
            loss = label_loss_fun(preds, labels)

        misc.check_nan_losses(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_labels, tp, tn, fp, fn = metrics.eval_prediction_results(preds, labels)
        pred_labels, labels, tp, fp, fn = (pred_labels.cpu(), labels.cpu(), tp.item(), fp.item(), fn.item())
        task_preds = (torch.sigmoid(task_preds) >= 0.5).type(torch.float32).cuda()
        task_preds, task_labels = (task_preds.cpu(), task_labels.cpu())

        train_meter.iter_toc()

        train_meter.update_stats(tp, fp, fn, pred_labels, labels, task_preds, task_labels, loss.item(), lr, preds.size(0))

        train_meter.iter_tic()

    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loaders, model, val_meters, cur_epoch, cfg, save=False):
    model.eval()
    label_dict = dict()
    for key, val_loader in val_loaders.items():
        val_meter = val_meters[key]
        val_meter.iter_tic()
        with Path('/home/baoxiong/Datasets/annotations/splits/{}_val.json'.format(key)).open('r') as f:
            vid_names = json.load(f)

        for cur_iter, (inputs, labels, index, meta) in enumerate(val_loader):
            vid_name = vid_names[index]

            if vid_name not in label_dict.keys():
                for pid in range(inputs.size(2)):
                    pid_name = 'P{}'.format(pid + 1)
                    label_dict[vid_name + '|' + pid_name] = dict()

            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            for key, val in meta.items():
                meta[key] = val.cuda(non_blocking=True)

            preds, task_preds = model(inputs, meta)
            task_preds = (torch.sigmoid(task_preds) >= 0.5).type(torch.float32).cuda()
            task_labels = meta['task_labels']

            eval_task_labels = task_labels.view(-1, task_labels.size(-1))
            eval_preds = preds.view(-1, preds.size(-1))
            eval_labels = labels.view(-1, labels.size(-1))
            eval_task_preds = task_preds.view(-1, task_preds.size(-1))

            eval_pred_labels, tp, tn, fp, fn = metrics.eval_prediction_results(eval_preds, eval_labels)
            eval_pred_labels, eval_labels, tp, fp, fn = (eval_pred_labels.cpu(), eval_labels.cpu(), tp.item(),
                                                         fp.item(), fn.item())
            eval_task_preds, eval_task_labels = (eval_task_preds.cpu(), eval_task_labels.cpu())

            preds = eval_pred_labels.view(preds.size(0), preds.size(1), preds.size(2), preds.size(3))
            task_preds = eval_task_preds.view(
                task_preds.size(0), task_preds.size(1),
                task_preds.size(2), task_preds.size(3)
            )

            val_meter.iter_toc()
            val_meter.update_stats(
                tp, fp, fn, eval_pred_labels, eval_labels, eval_task_preds,
                eval_task_labels, None, None, eval_preds.size(0)
            )

            val_meter.iter_tic()

            labels = labels.unsqueeze(0)
            for pid in range(inputs.size(2)):
                pid_name = 'P{}'.format(pid + 1)
                for frame_id in range(inputs.size(1)):
                    label_dict[vid_name + '|' + pid_name][frame_id] = [
                            preds[0, frame_id, pid, :], labels[0, frame_id, pid, :],
                            task_preds[0, frame_id, pid, :], task_labels[0, frame_id, pid, :]
                        ]
        val_meter.log_epoch_stats(cur_epoch)
        val_meter.reset()

    if save:
        save_dir = Path(cfg.OUTPUT_DIR) / 'ant_preds'
        results = []
        for vid_id, vid_meta in label_dict.items():
            for frame_id, frame_info in vid_meta.items():
                pred_labels = frame_info[0].cpu().nonzero().flatten().numpy().tolist()
                gt_labels = frame_info[1].cpu().nonzero().flatten().numpy().tolist()
                task_pred_labels = frame_info[2].cpu().nonzero().flatten().numpy().tolist()
                task_gt_labels = frame_info[3].cpu().nonzero().flatten().numpy().tolist()
                results.append([vid_id, frame_id, pred_labels, gt_labels, task_pred_labels, task_gt_labels])
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        with (save_dir / 'ant_preds.p').open('wb') as f:
            pickle.dump(results, f)




def train(cfg):
    """
        Train a video model_utils for many epochs on train set and evaluate it on val set.
        Args:
            cfg (CfgNode): settings. Details can be found in
                slowfast/config/defaults.py
        """
    # Set random seed from settings.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup log_utils format.
    logging.setup_logging()

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # FPV using feature slowfast no_embed
    fpv_size = (cfg.RESNET.WIDTH_PER_GROUP * 32 + cfg.RESNET.WIDTH_PER_GROUP * 32 // cfg.SLOWFAST.BETA_INV) * 2

    # TPV using feature slowfast plain
    if cfg.use_extra and cfg.extra == 'tpv':
        extra_size = cfg.RESNET.WIDTH_PER_GROUP * 32 + cfg.RESNET.WIDTH_PER_GROUP * 32 // cfg.SLOWFAST.BETA_INV
    else:
        extra_size = fpv_size

    if  cfg.model == 'lstm':
        hidden_size = fpv_size
        model = FusedLSTM(fpv_size, extra_size, hidden_size, cfg)
    elif cfg.model == 'featurebank':
        model = FeatureBank(fpv_size, extra_size, cfg)
    elif cfg.model == 'mlp':
        model = MLP(fpv_size, extra_size, cfg)
    else:
        print('no model listed')
        exit()

    cur_device = torch.cuda.current_device()
    # Transfer the model_utils to the current GPU device
    model = model.cuda(device=cur_device)

    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info('Load from last checkpoint.')
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        checkpoint_epoch = cu.load_checkpoint(
            last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
        )
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != '':
        logger.info('Load from given checkpoint file {}.'.format(cfg.TRAIN.CHECKPOINT_FILE_PATH))
        checkpoint_epoch = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == 'caffe2',
        )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0

    # Create the video train and val loaders.
    train_loader = loader.construct_pred_loader(cfg, 'train', subclass='pred')
    types = ['1x1', '1x2', '2x1', '2x2']
    val_loaders = {key : loader.construct_pred_loader(cfg, 'val', subclass=key) for key in types}

    train_meter = PredMeter(len(train_loader), cfg, 'train')
    val_meters = {key : PredMeter(len(val_loader), cfg, 'val', subclass=key) for key, val_loader in val_loaders.items()}

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    if cfg.save:
        eval_epoch(val_loaders, model, val_meters, 0, cfg, vis=cfg.save)
    else:
        for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
            # Shuffle the dataset.
            loader.shuffle_dataset(train_loader, cur_epoch)
            # Train for one epoch.
            train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg)

            # Save a checkpoint.
            if cu.is_checkpoint_epoch(cur_epoch, cfg.TRAIN.CHECKPOINT_PERIOD):
                cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)

            logger.info('Checkpoint saved at {}'.format(cur_epoch))

            # Evaluate the model_utils on validation set.
            if misc.is_eval_epoch(cfg, cur_epoch):
                eval_epoch(val_loaders, model, val_meters, cur_epoch, cfg)
                logger.info('Evaluation finished')


