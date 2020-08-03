"""

    Created on 2/20/20

    @author: Baoxiong Jia

    Description:

"""

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Configs."""
from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

_C.EXTRACT_FEATURE = False

# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

# BN epsilon.
_C.BN.EPSILON = 1e-5

# BN momentum.
_C.BN.MOMENTUM = 0.1

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0


# ---------------------------------------------------------------------------- #
# Experiment options
# ---------------------------------------------------------------------------- #
_C.EXP = CfgNode()

# Type of image for inference (rgb | flow)
_C.EXP.IMG_TYPE = "rgb"

# Type of label for inference (verb | noun | hoi)
_C.EXP.LABEL_TYPE = "verb"

# Type of view source for inference (tpv | fpv)
_C.EXP.VIEW_TYPE = "fpv"

# Type of supervision added (none | task)
_C.EXP.SUPERVISION = 'none'

# Type of model, whether using new model (plain | composed)
_C.EXP.MODEL_TYPE = 'plain'

# Type of task performed on the data (pred | rec)
_C.EXP.TASK = "rec"

# Type of activities in the video (1x1 | 1x2 | 2x1 | 2x2)
_C.EXP.VIDEO_TYPE = '1x1'

_C.EXP.USE_BOTH_VIEW = False

# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model_utils, else skip training.
_C.TRAIN.ENABLE = True

_C.TRAIN.BCE_POS_WEIGHT = 20

# Dataset.
_C.TRAIN.DATASET = "kinetics"

# Caching ground truth files
_C.TRAIN.CACHE = False

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 64

# Evaluate model_utils on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 1

# Save model_utils checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 1

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# Checkpoint types include `caffe2` or `pytorch`.
_C.TRAIN.CHECKPOINT_TYPE = "pytorch"

# If True, perform inflation when loading checkpoint.
_C.TRAIN.CHECKPOINT_INFLATE = False

# Training augmentation parameters
# Whether to use color augmentation method.
_C.TRAIN.USE_COLOR_AUGMENTATION = False

# Whether to only use PCA jitter augmentation when using color augmentation
# method (otherwise combine with color jitter method).
_C.TRAIN.PCA_JITTER_ONLY = True

# Eigenvalues for PCA jittering. Note PCA is RGB based.
_C.TRAIN.PCA_EIGVAL = [0.225, 0.224, 0.229]

# Eigenvectors for PCA jittering.
_C.TRAIN.PCA_EIGVEC = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# Segment for sampling
_C.TEST.SEGMENT_LENGTH = 64

# If True test the model_utils, else skip the testing.
_C.TEST.ENABLE = True

# Dataset for testing.
_C.TEST.DATASET = "kinetics"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 8

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""

# Numb_er of clips to sample from a video uniformly for aggregating the
# prediction results.
_C.TEST.NUM_ENSEMBLE_VIEWS = 10

# Number of crops to sample from a frame spatially for aggregating the
# prediction results.
_C.TEST.NUM_SPATIAL_CROPS = 3

# Checkpoint types include `caffe2` or `pytorch`.
_C.TEST.CHECKPOINT_TYPE = "pytorch"

# Whether to do horizontal flipping during test.
_C.TEST.FORCE_FLIP = False

# -----------------------------------------------------------------------------
# ResNet options
# -----------------------------------------------------------------------------
_C.RESNET = CfgNode()

# Transformation function.
_C.RESNET.TRANS_FUNC = "bottleneck_transform"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt).
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C.RESNET.STRIDE_1X1 = False

#  If true, initialize the gamma of the final BN of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_BN = False

# Number of weight layers.
_C.RESNET.DEPTH = 50

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]

# Size of stride on different res stages.
_C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]

# Size of dilation on different res stages.
_C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]


# -----------------------------------------------------------------------------
# Nonlocal options
# -----------------------------------------------------------------------------
_C.NONLOCAL = CfgNode()

# Index of each stage and block to add nonlocal layers.
_C.NONLOCAL.LOCATION = [[[]], [[]], [[]], [[]]]

# Number of group for nonlocal for each stage.
_C.NONLOCAL.GROUP = [[1], [1], [1], [1]]

# Instatiation to use for non-local layer.
_C.NONLOCAL.INSTANTIATION = "dot_product"


# Size of pooling layers used in Non-Local.
_C.NONLOCAL.POOL = [
    # Res2
    [[1, 2, 2], [1, 2, 2]],
    # Res3
    [[1, 2, 2], [1, 2, 2]],
    # Res4
    [[1, 2, 2], [1, 2, 2]],
    # Res5
    [[1, 2, 2], [1, 2, 2]],
]

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model architecture.
_C.MODEL.ARCH = "slowfast"

# BackBone name
_C.MODEL.MODEL_NAME = "FusedModel"

_C.MODEL.PRETRAIN_FUSED = False

# BackBone name
_C.MODEL.BACKBONE_NAME = "SlowFast"

# The number of classes to predict for the model_utils.
_C.MODEL.NUM_CLASSES = 400

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"

# Model architectures that has one single pathway.
_C.MODEL.SINGLE_PATHWAY_ARCH = ["c2d", "i3d", "slowonly"]

# Model architectures that has multiple pathways.
_C.MODEL.MULTI_PATHWAY_ARCH = ["slowfast"]

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01


# -----------------------------------------------------------------------------
# Slowfast options
# -----------------------------------------------------------------------------
_C.SLOWFAST = CfgNode()

# Corresponds to the inverse of the channel reduction ratio, $\beta$ between
# the Slow and Fast pathways.
_C.SLOWFAST.BETA_INV = 8

# Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
# Fast pathways.
_C.SLOWFAST.ALPHA = 8

# Ratio of channel dimensions between the Slow and Fast pathways.
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2

# Kernel dimension used for fusing information from Fast pathway to Slow
# pathway.
_C.SLOWFAST.FUSION_KERNEL_SZ = 5


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()


# Paths for train list file
_C.DATA.GT_LIST_DIR = "/home/baoxiong/Datasets/annotations"
_C.DATA.PRED_LIST_DIR = "/home/baoxiong/Datasets/detections"

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = "/home/baoxiong/Datasets/data"

# Training metadata save path
_C.DATA.SAVE_PATH = "/home/baoxiong/Datasets/intermediate"

# The fps of the data
_C.DATA.FPS = 24

# Video path prefix if any.
_C.DATA.PATH_PREFIX = ""

# The spatial crop size of the input clip.
_C.DATA.CROP_SIZE = 224

# The number of frames of the input clip.
_C.DATA.NUM_FRAMES = 8

# The video sampling rate of the input clip.
_C.DATA.SAMPLING_RATE = 8

# Whether use BGR or not
_C.DATA.BGR = False

# The mean value of the video raw pixels across the R G B channels.
_C.DATA.DATA_MEAN = [0.45, 0.45, 0.45]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.DATA_STD = [0.225, 0.225, 0.225]

# The mean value of the video raw pixels across the R G B channels.
_C.DATA.RGB_MEAN = [0.45, 0.45, 0.45]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.RGB_STD = [0.225, 0.225, 0.225]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.FLOW_MEAN = [0.45, 0.45]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.FLOW_STD = [0.225, 0.225]

# List of input frame channel dimensions.
_C.DATA.INPUT_CHANNEL_NUM = [3, 3]

# list of input rgb frame channel dimensions
_C.DATA.RGB_CHANNEL_NUM = [3, 3]

# list of input flow frame channel dimensions
_C.DATA.FLOW_CHANNEL_NUM = [2, 2]

# Sampling frames for training
_C.DATA.TRAIN_SAMPLE_SEGMENT = 3

# Sampling clips per second for training
_C.DATA.TRAIN_CLIPS_PER_SECOND = 4

# Sampling clips per second for testing and evaluation
_C.DATA.TEST_CLIPS_PER_SECOND = 5

_C.DATA.EXT_CLIPS_PER_SECOND = 5

# The spatial augmentation jitter scales for training.
_C.DATA.TRAIN_JITTER_SCALES = [256, 320]

# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224

# The spatial crop size for testing.
_C.DATA.TEST_CROP_SIZE = 256

# This option controls the score threshold for the predicted boxes to use.
_C.DATA.DETECTION_SCORE_THRESH = 0.9


# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# The index of the current machine.
_C.SHARD_ID = 0

# Output basedir.
_C.OUTPUT_DIR = "./tmp"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# Log period in iters.
_C.LOG_PERIOD = 10

# Distributed backend.
_C.DIST_BACKEND = "nccl"


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False


# ---------------------------------------------------------------------------- #
# Detection options.
# ---------------------------------------------------------------------------- #
_C.DETECTION = CfgNode()

# Whether enable video detection.
_C.DETECTION.ENABLE = False

# Aligned version of RoI. More details can be found at slowfast/models/head_helper.py
_C.DETECTION.ALIGNED = True

# Spatial scale factor.
_C.DETECTION.SPATIAL_SCALE_FACTOR = 16

# RoI tranformation resolution.
_C.DETECTION.ROI_XFORM_RESOLUTION = 7


def _assert_and_infer_cfg(cfg):
    # BN assertions.
    if cfg.BN.USE_PRECISE_STATS:
        assert cfg.BN.NUM_BATCHES_PRECISE >= 0
    # TRAIN assertions.
    assert cfg.TRAIN.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # TEST assertions.
    assert cfg.TEST.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0
    assert cfg.TEST.NUM_SPATIAL_CROPS == 3

    # RESNET assertions.
    assert cfg.RESNET.NUM_GROUPS > 0
    assert cfg.RESNET.WIDTH_PER_GROUP > 0
    assert cfg.RESNET.WIDTH_PER_GROUP % cfg.RESNET.NUM_GROUPS == 0

    # General assertions.
    assert cfg.SHARD_ID < cfg.NUM_SHARDS

    return cfg

def map_cfg(cfg):
    # Constraints added from view type
    if cfg.EXP.VIEW_TYPE == 'tpv':
        cfg.DETECTION.ENABLE = True
        cfg.BN.USE_PRECISE_STATS = True
    else:
        cfg.DETECTION.ENABLE = False

    if cfg.EXP.IMG_TYPE == 'flow':
        cfg.TRAIN.USE_COLOR_AUGMENTATION = False
        cfg.DATA.INPUT_CHANNEL_NUM = cfg.DATA.FLOW_CHANNEL_NUM
        cfg.DATA.DATA_MEAN = cfg.DATA.FLOW_MEAN
        cfg.DATA.DATA_STD = cfg.DATA.FLOW_STD

    if cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        cfg.TRAIN.AUTO_RESUME = False

    cfg.TEST.DATASET = cfg.TRAIN.DATASET

    if cfg.EXP.MODEL_TYPE != 'pain' or cfg.EXP.SUPERVISION != 'task':
        cfg.MODEL.LOSS_FUNC = 'custom_loss'

    if cfg.EXTRACT_FEATURE:
        cfg.TRAIN.ENABLE = False

    return _assert_and_infer_cfg(cfg)


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())