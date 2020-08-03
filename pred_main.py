"""
  
    Created on 2/16/20

    @author: Baoxiong Jia

    Description:

"""

import argparse

import utils.model_utils.checkpoint as cu
from config.default import get_cfg, map_cfg
from datasets.metadata import Metadata
from utils.exp_utils.prediction import train


def parse_args():
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
        '--model',
        default='featurebank',
        type=str,
        help='default model to use in prediction'
    )
    parser.add_argument(
        '--use_extra',
        default=True,
        type=str2bool,
        help="whether or not use the other person's view as extra feature"
    )
    parser.add_argument(
        '--epochs',
        default=100,
        type=int,
        help='number of epochs for training'
    )
    parser.add_argument(
        '--task',
        default=False,
        type=str2bool,
        help='Evaluating task anticipation (True) or action anticipation (False)'
    )
    parser.add_argument(
        '--extra',
        default='tpv',
        type=str,
        help='Extra view feature from the other person used (tpv/fpv)'
    )
    parser.add_argument(
        '--lr',
        default=0.001,
        type=float,
        help='Learning rate'
    )
    parser.add_argument(
        '--save',
        default=False,
        type=str2bool,
        help='Save predictions option'
    )
    parser.add_argument(
        '--feat_path',
        default='/home/baoxiong/HDD/features',
        type=str,
        help='Default recognition feature path'
    )
    parser.add_argument(
        '--output_base_path',
        default='/home/baoxiong/HDD/temp/prediction',
        type=str,
        help='Default folder for storing trained models'
    )
    parser.add_argument(
        '--use_tpv',
        help='whether or not use tpv',
        type=str2bool,
        default=True
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
    cfg = map_cfg(cfg)
    cfg.MODEL.NUM_CLASSES = len(Metadata.hoi)
    cfg.DEBUG = args.debug
    cfg.SOLVER.MAX_EPOCH = args.epochs
    cfg.model = args.model
    cfg.use_extra = args.use_extra
    cfg.extra = args.extra
    cfg.task = args.task
    cfg.lr = args.lr
    cfg.save = args.save
    cfg.feat_path = args.feat_path
    cfg.OUTPUT_DIR = '{}/{}_{}_{}_{}'.format(args.output_base_path, cfg.model,
                                             'w' if cfg.use_extra else 'w-o',
                                             args.extra, 't' if cfg.task else 'n-t')
    cfg.TRAIN.DATASET = 'Lemma_seq'
    cfg.TEST.DATASET = 'Lemma_seq'
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg

def main():
    args = parse_args()
    cfg = load_config(args)
    train(cfg)
    print('loader went through')


if __name__ == '__main__':
    main()
