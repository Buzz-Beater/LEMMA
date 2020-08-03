"""
  
    Created on 2/16/20

    @author: Baoxiong Jia

    Description:

"""
import numpy as np
import pprint
import torch

import utils.model_utils.checkpoint as cu
import utils.model_utils.distributed as du
from utils.model_utils.precise_bn import get_bn_modules, update_bn_stats
import utils.eval_utils.metrics as metrics
import utils.model_utils.misc as misc
import utils.log_utils.logging as logging

from models.build import get_loss_func
import models.optimizer as optim
from models import build_model

from datasets import loader

from utils.log_utils.meters import RecMeter

logger = logging.get_logger(__name__)

def train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model_utils): the video model_utils to train.
        optimizer (optim): the optimizer to perform optimization on the model_utils's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): settings. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda()

        for key, val in meta.items():
            if isinstance(val, (list,)):
                for i in range(len(val)):
                    val[i] = val[i].cuda(non_blocking=True)
            else:
                meta[key] = val.cuda(non_blocking=True)


        preds = model(inputs, meta)

        # Explicitly declare reduction to mean.
        loss_fun = get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

        # Compute the loss.
        loss = loss_fun(preds, labels, meta, cfg)

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters.
        optimizer.step()

        (
            tp, tn, fp, fn,
            pred_labels, act_pred_labels, obj_pred_labels,
            gt_act_labels, gt_obj_labels
        ) = metrics.eval_pred(preds, labels, meta, cfg)

        if cfg.NUM_GPUS > 1:
            loss = du.all_reduce([loss])[0]
            tp = sum(du.all_gather_unaligned(tp))
            fp = sum(du.all_gather_unaligned(fp))
            fn = sum(du.all_gather_unaligned(fn))
            pred_labels = torch.cat(du.all_gather_unaligned(pred_labels), dim=0)
            act_pred_labels = torch.cat(du.all_gather_unaligned(act_pred_labels), dim=0)
            obj_pred_labels = torch.cat(du.all_gather_unaligned(obj_pred_labels), dim=0)
            labels = torch.cat(du.all_gather_unaligned(labels), dim=0)
            gt_act_labels = torch.cat(du.all_gather_unaligned(gt_act_labels), dim=0)
            gt_obj_labels = torch.cat(du.all_gather_unaligned(gt_obj_labels), dim=0)

        loss, tp, fp, fn = (loss.item(), tp.item(), fp.item(), fn.item())

        (
            pred_labels, act_pred_labels, obj_pred_labels,
            gt_act_labels, gt_obj_labels, labels
        ) = (
            pred_labels.cpu(), act_pred_labels.cpu(), obj_pred_labels.cpu(),
            gt_act_labels.cpu(), gt_obj_labels.cpu(), labels.cpu()
        )

        train_meter.iter_toc()
        # Update and log stats.
        train_meter.update_stats(
            tp, fp, fn, pred_labels, act_pred_labels, obj_pred_labels, labels, gt_act_labels, gt_obj_labels, loss, lr,
            inputs[0].size(0) * cfg.NUM_GPUS
        )
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    """
    Evaluate the model_utils on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model_utils): model_utils to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): settings. Details can be found in
            slowfast/config/defaults.py
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        # Transferthe data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda()
        for key, val in meta.items():
            if isinstance(val, (list,)):
                for i in range(len(val)):
                    val[i] = val[i].cuda(non_blocking=True)
            else:
                meta[key] = val.cuda(non_blocking=True)

        preds = model(inputs, meta)

        (
            tp, tn, fp, fn,
            pred_labels, act_pred_labels, obj_pred_labels,
            gt_act_labels, gt_obj_labels
        ) = metrics.eval_pred(preds, labels, meta, cfg)

        if cfg.NUM_GPUS > 1:
            if cfg.NUM_GPUS > 1:
                tp = sum(du.all_gather_unaligned(tp))
                fp = sum(du.all_gather_unaligned(fp))
                fn = sum(du.all_gather_unaligned(fn))
                pred_labels = torch.cat(du.all_gather_unaligned(pred_labels), dim=0)
                act_pred_labels = torch.cat(du.all_gather_unaligned(act_pred_labels), dim=0)
                obj_pred_labels = torch.cat(du.all_gather_unaligned(obj_pred_labels), dim=0)
                labels = torch.cat(du.all_gather_unaligned(labels), dim=0)
                gt_act_labels = torch.cat(du.all_gather_unaligned(gt_act_labels), dim=0)
                gt_obj_labels = torch.cat(du.all_gather_unaligned(gt_obj_labels), dim=0)
        (
            pred_labels, act_pred_labels, obj_pred_labels,
            gt_act_labels, gt_obj_labels, labels
        ) = (
            pred_labels.cpu(), act_pred_labels.cpu(), obj_pred_labels.cpu(),
            gt_act_labels.cpu(), gt_obj_labels.cpu(), labels.cpu()
        )

        tp, fp, fn = (tp.item(), fp.item(), fn.item())

        val_meter.iter_toc()
        val_meter.update_stats(
            tp, fp, fn, pred_labels, act_pred_labels, obj_pred_labels, labels, gt_act_labels, gt_obj_labels, None, None,
            inputs[0].size(0) * cfg.NUM_GPUS
        )
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model_utils): model_utils to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
    """

    def _gen_loader():
        for inputs, _, _, meta in loader:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
            yield inputs, meta

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


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

    # Build the video model_utils and print model_utils statistics.
    model = build_model(cfg)
    # if du.is_master_proc():
    #     misc.log_model_info(model, cfg, is_train=True)

    # Construct the optimizer.
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
    train_loader = loader.construct_loader(cfg, 'train')
    val_loader = loader.construct_loader(cfg, 'val')

    train_meter = RecMeter(len(train_loader), cfg, 'train')
    val_meter = RecMeter(len(val_loader), cfg, 'val')

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)
        # Train for one epoch.
        train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg)

        # Compute precise BN stats.
        if cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0:
            logger.info('Updating precise BN stats')
            calculate_and_update_precise_bn(
                train_loader, model, cfg.BN.NUM_BATCHES_PRECISE
            )
            logger.info('Precise BN update finished')

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(cur_epoch, cfg.TRAIN.CHECKPOINT_PERIOD):
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)

        logger.info('Checkpoint saved at {}'.format(cur_epoch))

        # Evaluate the model_utils on validation set.
        if misc.is_eval_epoch(cfg, cur_epoch):
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg)
            logger.info('Evaluation finished')