"""
  
    Created on 2/17/20

    @author: Baoxiong Jia

    Description:

"""
import datetime
import numpy as np
import json
from pathlib import Path
import sklearn.metrics as sk_metrics

import torch
from fvcore.common.timer import Timer
from collections import deque
import utils.log_utils.logging as logging
from datasets.metadata import Metadata

class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class RecMeter(object):
    def __init__(self, epoch_iters, cfg, mode):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): settings.
        """
        self._cfg = cfg
        self._mode = mode
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window).
        self.avg_pre = ScalarMeter(cfg.LOG_PERIOD)
        self.avg_rec = ScalarMeter(cfg.LOG_PERIOD)
        self.avg_f1 = ScalarMeter(cfg.LOG_PERIOD)

        self.num_samples = 0
        self.all_label_preds = []
        self.all_act_preds = []
        self.all_obj_preds = []
        self.all_labels = []
        self.all_act_labels = []
        self.all_obj_labels = []
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.avg_pre.reset()
        self.avg_rec.reset()
        self.avg_f1.reset()

        self.num_samples = 0
        self.all_label_preds = []
        self.all_act_preds = []
        self.all_obj_preds = []
        self.all_labels = []
        self.all_act_labels = []
        self.all_obj_labels = []
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, tp, fp, fn, label_preds, act_preds, obj_preds, labels, gt_act_labels, gt_obj_labels, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        # Current minibatch stats
        if tp == 0:
            avg_pre = 0
            avg_rec = 0
            avg_f1 = 0
        else:
            avg_pre = tp / (tp + fp)
            avg_rec = tp / (tp + fn)
            avg_f1 = (2.0 * tp) / (2.0 * tp + fn + fp)
        self.avg_pre.add_value(avg_pre)
        self.avg_rec.add_value(avg_rec)
        self.avg_f1.add_value(avg_f1)
        self.tp += tp
        self.fn += fn
        self.fp += fp
        self.num_samples += mb_size

        self.all_label_preds.append(label_preds)
        self.all_act_preds.append(act_preds)
        self.all_obj_preds.append(obj_preds)
        self.all_labels.append(labels)
        self.all_act_labels.append(gt_act_labels)
        self.all_obj_labels.append(gt_obj_labels)

        if self._mode == 'train':
            self.loss.add_value(loss)
            self.loss_total += loss * mb_size
            self.lr = lr

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
                self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "R": "[{}/{}|{}/{}]".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH, cur_iter + 1, self.epoch_iters),
            "eta": eta,
            "pre": self.avg_pre.get_win_median(),
            "rec": self.avg_rec.get_win_median(),
            "f1": self.avg_f1.get_win_median(),
        }
        if self._mode == 'train':
            stats["id"] = "train_iter"
            stats["loss"] = self.loss.get_win_median()
            stats["lr"] = self.lr
        else:
            stats["id"] = "val_iter"
        logging.log_json_stats(stats)


    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
                self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))

        results = self.finalize_metrics()
        stats = {
            "R": "[{}/{}]".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "eta": eta,
        }
        if self._mode == 'train':
            stats["loss"] = self.loss_total / self.num_samples
            stats["lr"] = self.lr
            stats["id"] = "train"
        else:
            stats["id"] = "val"
        log_list = ['prec', 'rec', 'f1']
        for key in results.keys():
            # for log_idx, val in enumerate(log_list):
            stats['{}'.format(key)] = results[key][:-1]

        logging.log_json_stats(stats)
        save_path = Path(self._cfg.OUTPUT_DIR) / 'log'
        if not save_path.exists():
            save_path.mkdir(parents=True)
        with (save_path / '{}_{}.json'.format(self._mode, cur_epoch)).open('w') as f:
            json.dump(stats, f)

    def finalize_metrics(self):
        results = dict()
        all_preds = torch.cat(self.all_label_preds, dim=0).numpy()
        all_labels = torch.cat(self.all_labels, dim=0).numpy()
        all_act_preds = torch.cat(self.all_act_preds, dim=0).numpy()
        all_act_labels = (torch.cat(self.all_act_labels, dim=0) + 1)
        # Add one additional dimension for
        all_act_labels = (torch.sum(torch.zeros(all_act_labels.size(0), all_act_labels.size(1), (len(Metadata.action) + 1)).scatter_(-1, all_act_labels, 1)[:, :, 1:], dim=1) > 0).type(torch.float).numpy()
        all_obj_preds = torch.cat(self.all_obj_preds, dim=0)
        all_obj_preds = all_obj_preds.view(-1, all_obj_preds.size(-1)).numpy()
        all_obj_labels = torch.cat(self.all_obj_labels, dim=0)
        all_obj_labels = all_obj_labels.view(-1, all_obj_labels.size(-1)).numpy()

        # binary cross entropy
        results['hois'] = sk_metrics.precision_recall_fscore_support(all_labels, all_preds, labels=list(range(len(Metadata.hoi))), average='micro')
        results['actions'] = sk_metrics.precision_recall_fscore_support(all_act_labels, all_act_preds, labels=list(range(len(Metadata.action))), average='micro')
        results['objects'] = sk_metrics.precision_recall_fscore_support(all_obj_labels, all_obj_preds, labels=list(range(len(Metadata.object))), average='micro')
        return results


class PredMeter(object):
    def __init__(self, epoch_iters, cfg, mode, subclass='all'):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): settings.
        """
        self._cfg = cfg
        self._mode = mode
        self._subclass= subclass
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window).
        self.avg_pre = ScalarMeter(cfg.LOG_PERIOD)
        self.avg_rec = ScalarMeter(cfg.LOG_PERIOD)
        self.avg_f1 = ScalarMeter(cfg.LOG_PERIOD)

        self.num_samples = 0
        self.all_label_preds = []
        self.all_labels = []
        self.all_task_preds = []
        self.all_task_labels = []
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.avg_pre.reset()
        self.avg_rec.reset()
        self.avg_f1.reset()

        self.num_samples = 0
        self.all_label_preds = []
        self.all_labels = []
        self.all_task_preds = []
        self.all_task_labels = []
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, tp, fp, fn, label_preds, labels, task_preds, task_labels, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        # Current minibatch stats
        if tp == 0:
            avg_pre = 0
            avg_rec = 0
            avg_f1 = 0
        else:
            avg_pre = tp / (tp + fp)
            avg_rec = tp / (tp + fn)
            avg_f1 = (2.0 * tp) / (2.0 * tp + fn + fp)
        self.avg_pre.add_value(avg_pre)
        self.avg_rec.add_value(avg_rec)
        self.avg_f1.add_value(avg_f1)
        self.tp += tp
        self.fn += fn
        self.fp += fp
        self.num_samples += mb_size

        self.all_label_preds.append(label_preds)
        self.all_labels.append(labels)
        self.all_task_preds.append(task_preds)
        self.all_task_labels.append(task_labels)

        if self._mode == 'train':
            self.loss.add_value(loss)
            self.loss_total += loss * mb_size
            self.lr = lr

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
                self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "R": "[{}/{}|{}/{}]".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH, cur_iter + 1, self.epoch_iters),
            "eta": eta,
            "pre": self.avg_pre.get_win_median(),
            "rec": self.avg_rec.get_win_median(),
            "f1": self.avg_f1.get_win_median(),
        }
        if self._mode == 'train':
            stats["id"] = "train_iter"
            stats["loss"] = self.loss.get_win_median()
            stats["lr"] = self.lr
        else:
            stats["id"] = "val_iter"
        logging.log_json_stats(stats)


    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
                self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))

        results = self.finalize_metrics()
        stats = {
            "R": "[{}/{}]".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "eta": eta,
        }
        if self._mode == 'train':
            stats["loss"] = self.loss_total / self.num_samples
            stats["lr"] = self.lr
            stats["id"] = "train"
        else:
            stats["id"] = "val"
            stats['subclass'] = self._subclass
        for key in results.keys():
            # for log_idx, val in enumerate(log_list):
            stats['{}'.format(key)] = results[key][:-1]

        logging.log_json_stats(stats)
        save_path = Path(self._cfg.OUTPUT_DIR) / 'log'
        if not save_path.exists():
            save_path.mkdir(parents=True)
        with (save_path / '{}_{}.json'.format(self._mode, cur_epoch)).open('w') as f:
            json.dump(stats, f)

    @torch.no_grad()
    def finalize_metrics(self):
        results = dict()
        all_preds = torch.cat(self.all_label_preds, dim=0).numpy()
        all_labels = torch.cat(self.all_labels, dim=0).numpy()
        all_task_preds = torch.cat(self.all_task_preds, dim=0).numpy()
        all_task_labels = torch.cat(self.all_task_labels, dim=0).numpy()

        # binary cross entropy
        results['hois'] = sk_metrics.precision_recall_fscore_support(all_labels, all_preds, labels=list(range(len(Metadata.hoi))), average='micro')
        results['task'] = sk_metrics.precision_recall_fscore_support(all_task_labels, all_task_preds, labels=list(range(len(Metadata.task))), average='micro')
        return results