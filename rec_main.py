"""
  
    Created on 2/16/20

    @author: Baoxiong Jia

    Description:

"""

import argparse
import torch
from pathlib import Path

import utils.model_utils.checkpoint as cu
import utils.model_utils.multiprocessing as mpu
from config.default import get_cfg, map_cfg
from config.config import Config
from datasets.metadata import Metadata

from utils.exp_utils.recognition import train
from utils.exp_utils.extract_feature import extract

def parse_args(proj_cfg):
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            return argparse.ArgumentTypeError('Unsupported value encountered')

    parser = argparse.ArgumentParser(
        description="Baseline Settings"
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="settings/i3d.yaml",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="Output dir",
        default=str(proj_cfg.intermediate_path),
        type=str
    )
    parser.add_argument(
        "--rng_seed",
        help="Random seed",
        default=0,
        type=int
    )
    parser.add_argument(
        "opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        '--debug',
        help='Debugging mode',
        default=False,
        type=str2bool
    )
    return parser.parse_args()

def load_config(args):
    """
    Given the arguemnts, load and initialize the settings.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed

    cfg = map_cfg(cfg)
    backbone = Path(args.cfg_file).stem.split('_')[0]
    cfg.OUTPUT_DIR = str(Path(args.output_dir) / '{}_{}x{}_e{}b{}_{}_{}_{}_{}_{}_{}'.format(
                                                    backbone, cfg.DATA.NUM_FRAMES, cfg.DATA.SAMPLING_RATE,
                                                    cfg.SOLVER.MAX_EPOCH, cfg.TRAIN.BATCH_SIZE, cfg.EXP.TASK,
                                                    cfg.EXP.LABEL_TYPE, cfg.EXP.VIEW_TYPE, cfg.EXP.IMG_TYPE,
                                                    cfg.EXP.SUPERVISION, cfg.EXP.MODEL_TYPE
                                                ))
    cfg.EMBED_PATH =  str(Path(args.output_dir) / 'embeddings' / 'embedding.p')
    if cfg.EXP.LABEL_TYPE == 'verb':
        cfg.MODEL.NUM_CLASSES = len(Metadata.action)
    elif cfg.EXP.LABEL_TYPE == 'noun':
        cfg.MODEL.NUM_CLASSES = len(Metadata.object)
    else:
        cfg.MODEL.NUM_CLASSES = len(Metadata.hoi)

    cfg.DEBUG = args.debug

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg

def main():
    proj_cfg = Config()
    args = parse_args(proj_cfg)
    cfg = load_config(args)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        if cfg.NUM_GPUS > 1:
            torch.multiprocessing.spawn(
                mpu.run,
                nprocs=cfg.NUM_GPUS,
                args=(
                    cfg.NUM_GPUS,
                    train,
                    args.init_method,
                    cfg.SHARD_ID,
                    cfg.NUM_SHARDS,
                    cfg.DIST_BACKEND,
                    cfg,
                ),
                daemon=False,
            )
        else:
            train(cfg=cfg)

    if cfg.EXTRACT_FEATURE:
        extract(cfg=cfg)


if __name__ == '__main__':
    main()
