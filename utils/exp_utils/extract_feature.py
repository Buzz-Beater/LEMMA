"""
  
    Created on 2/16/20

    @author: Baoxiong Jia

    Description:

"""
import sys
sys.path.append('/home/baoxiong/Projects/DOME/experiments/src')
import argparse
from config.default import get_cfg, map_cfg
from config.config import Config
import pickle
import numpy as np
import torch
from pathlib import Path

from datasets.metadata import Metadata
import utils.model_utils.checkpoint as cu
import utils.model_utils.multiprocessing as mpu
import utils.model_utils.distributed as du
import utils.eval_utils.metrics as metrics
from datasets import loader
from models import build_model
from utils.log_utils.meters import RecMeter
import utils.log_utils.logging as logging

logger = logging.get_logger(__name__)


@torch.no_grad()
def extract_feature(val_loader, model, test_meter, cfg, save=True, vis=True):
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
    test_meter.iter_tic()
    feature_dict = dict()
    preds_dict = dict()
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
        label_features = preds[4]
        metadatas = meta['metadata']

        index_file = Path(cfg.DATA.SAVE_PATH) / '{}_{}_{}_{}'.format(cfg.EXP.TASK, cfg.EXP.VIEW_TYPE,
                                                        cfg.EXP.IMG_TYPE, cfg.EXP.VIDEO_TYPE) / 'vid_idx_to_name.p'
        if not index_file.exists():
            vid_idx_to_name = []
            vid_name_to_idx = {}
            with Path(cfg.DATA.GT_LIST_DIR) / 'full' / '{}_{}_frames.p'.format(cfg.EXP.task, cfg.EXP.VIEW_TYPE).open('rb') as f:
                frame_infos = pickle.load(f)
            for frame_info in frame_infos:
                vid_name = frame_info[0]
                if vid_name not in vid_name_to_idx.keys():
                    idx = len(vid_name_to_idx)
                    vid_name_to_idx[vid_name] = idx
                    vid_idx_to_name.append(vid_name)
        else:
            with index_file.open('rb') as f:
                vid_idx_to_name = pickle.load(f)

        (
            tp, tn, fp, fn,
            pred_labels, act_pred_labels, obj_pred_labels,
            gt_act_labels, gt_obj_labels
        ) = metrics.eval_pred(preds, labels, meta, cfg)

        if cfg.NUM_GPUS > 1:
            tp = sum(du.all_gather_unaligned(tp))
            fp = sum(du.all_gather_unaligned(fp))
            fn = sum(du.all_gather_unaligned(fn))
            label_features = torch.cat(du.all_gather_unaligned(label_features), dim=0)
            metadatas = torch.cat(du.all_gather_unaligned(metadatas), dim=0)
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
        label_features, metadatas = label_features.cpu(), metadatas.cpu()

        if vis:
            for pred_label, label, metadata in zip(pred_labels, labels, metadatas):
                vid_idx, frame_id, pid = metadata
                vid_name = vid_idx_to_name[vid_idx]
                if vid_name not in preds_dict.keys():
                    preds_dict[vid_name] = dict()
                if frame_id not in preds_dict[vid_name].keys():
                    preds_dict[vid_name][frame_id] = [pred_label, label]

        if save:
            for label_feature, metadata in zip(label_features, metadatas):
                vid_idx, frame_id, pid = metadata
                pid = 'P{}'.format(pid + 1)
                vid_name = vid_idx_to_name[vid_idx].split('|')[0]
                if vid_name not in feature_dict.keys():
                    feature_dict[vid_name] = dict()
                if pid not in feature_dict[vid_name].keys():
                    feature_dict[vid_name][pid] = dict()
                if frame_id not in feature_dict[vid_name][pid].keys():
                    feature_dict[vid_name][pid][frame_id] = label_feature

        tp, fp, fn = (tp.item(), fp.item(), fn.item())

        test_meter.iter_toc()
        test_meter.update_stats(
            tp, fp, fn, pred_labels, act_pred_labels, obj_pred_labels, labels, gt_act_labels, gt_obj_labels, None, None,
            inputs[0].size(0) * cfg.NUM_GPUS
        )
        test_meter.log_iter_stats(0, cur_iter)
        test_meter.iter_tic()

    if save:
        save_dir = Path(cfg.OUTPUT_DIR) / 'features'
        for vid_name, vid_meta in feature_dict.items():
            for pid, features in vid_meta.items():
                vid_path = save_dir / vid_name / pid / cfg.EXP.VIEW_TYPE
                if not vid_path.exists():
                    vid_path.mkdir(parents=True)
                save_features = np.array([val.numpy() for _, val in sorted(features.items(), key=lambda x: x[0].item())])
                frame_ids = np.array([x.numpy() for x, _ in sorted(features.items(), key=lambda x: x[0].item())])
                np.save(str(vid_path / 'features'), save_features)
                np.save(str(vid_path / 'frame_ids'), frame_ids)

    if vis:
        results = []
        save_dir = Path(cfg.OUTPUT_DIR) / 'preds'
        for vid_name, vid_meta in preds_dict.items():
            for frame_id, frame_info in vid_meta.items():
                pred_labels = frame_info[0]
                gt_labels = frame_info[1]
                preds = pred_labels.nonzero().flatten().numpy().tolist()
                gts = gt_labels.nonzero().flatten().numpy().tolist()
                results.append([vid_name, frame_id, preds, gts])
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        with (save_dir / 'vis_preds.p').open('wb') as f:
            pickle.dump(results, f)

    test_meter.log_epoch_stats(0)
    test_meter.reset()


def extract(cfg):
    """
    Perform multi-view testing on the pretrained video model_utils.
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
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model_utils and print model_utils statistics.
    model = build_model(cfg)
    # if du.is_master_proc():
    #     misc.log_model_info(model, cfg, is_train=False)

    # Load a checkpoint to test if applicable.
    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        cu.load_checkpoint(
            cfg.TEST.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TEST.CHECKPOINT_TYPE == "caffe2",
        )
    elif cu.has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.ture
        cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
    else:
        # raise NotImplementedError("Unknown way to load checkpoint.")
        logger.info("Testing with random initialization. Only for debugging.")

    # Create video testing loaders.
    if cfg.EXTRACT_FEATURE:
        test_loader = loader.construct_loader(cfg, "all")
    else:
        test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model_utils for {} iterations".format(len(test_loader)))

    test_meter = RecMeter(len(test_loader), cfg, mode='test')

    extract_feature(test_loader, model, test_meter, cfg, save=cfg.EXTRACT_FEATURE and not cfg.VIS, vis=cfg.VIS)


if __name__ == '__main__':
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            return argparse.ArgumentTypeError('Unsupported value encountered')

    parser = argparse.ArgumentParser()
    parser.add_argument('--view', default='fpv', type=str, help='view for evaluation for extraction')
    parser.add_argument('--model_type', default='fused', type=str, help='model type for evaluation')
    parser.add_argument('--model_name', default='slowfast', type=str, help='model name for evaluation')
    parser.add_argument('--extract', default=False, type=str2bool, help='whether or not extract feature')
    parser.add_argument('--preds', default=True, type=str2bool, help='whether extract preds only without storing features')
    parser.add_argument('--num_gpus', default=8, type=int, help='number of GPU to use')
    parser.add_argument('--workers', default=4, type=int, help='number of workers')
    parser.add_argument('--batch_size', default=96, type=int, help='batch size for eval')
    parser.add_argument('--cache', default=True, type=str2bool, help='caching dataset construction results')
    parser.add_argument('--debug', default=False, type=str2bool, help='initialization of debugging mode')
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    args = parser.parse_args()
    cfg = get_cfg()
    view_type = args.view
    model_type = args.model_type
    model_name = args.model_name

    model_cfg = '/home/baoxiong/Projects/LEMMA/experiments/src/settings/{}_fpv_plain_hoi.yaml'.format(model_name)
    cfg.merge_from_file(model_cfg)
    cfg.EXP.VIEW_TYPE = view_type
    cfg.EXP.MODEL_TYPE = model_type
    cfg = map_cfg(cfg)
    cfg.OUTPUT_DIR = '/home/baoxiong/HDD/features'
    cfg.TRAIN.CHECKPOINT_FILE_PATH = '/home/baoxiong/models/{}_{}_{}.pyth'.format(model_name, model_type, view_type)

    cfg.MODEL.NUM_CLASSES = len(Metadata.hoi)
    cfg.EXTRACT_FEATURE = args.extract
    cfg.VIS = args.preds
    cfg.EMBED_PATH = str(Path(Config().intermediate_path) / 'embeddings' / 'embedding.p')
    cfg.TRAIN.CHECKPOINT_TYPE = 'pytorch'
    cfg.DEBUG = args.debug
    cfg.TEST.BATCH_SIZE = args.batch_size
    cfg.DATA_LOADER.NUM_WORKERS = args.workers
    cfg.NUM_GPUS = args.num_gpus
    cfg.TRAIN.CACHE = args.cache
    torch.multiprocessing.spawn(
        mpu.run,
        nprocs=cfg.NUM_GPUS,
        args=(
            cfg.NUM_GPUS,
            extract,
            args.init_method,
            cfg.SHARD_ID,
            cfg.NUM_SHARDS,
            cfg.DIST_BACKEND,
            cfg,
        ),
        daemon=False,
    )
