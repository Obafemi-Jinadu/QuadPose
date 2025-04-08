
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Hanbin Dai (daihanbin.ac@gmail.com) and Feng Zhang (zhangfengwcy@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'pose_hrnet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.TARGET_TYPE = 'gaussian'
_C.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
_C.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
_C.MODEL.SIGMA = 2
_C.MODEL.EXTRA = CN(new_allowed=True)

_C.LOSS = CN()
_C.LOSS.USE_OHKM = False
_C.LOSS.TOPK = 8
_C.LOSS.USE_TARGET_WEIGHT = True
_C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'mpii'
_C.DATASET.OTHERS = True
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.PSEUDO_TRAIN_SET = 'pseudos'
_C.DATASET.TEST_SET = 'valid'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.HYBRID_JOINTS_TYPE = ''
_C.DATASET.SELECT_DATA = False

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 30
_C.DATASET.PROB_HALF_BODY = 0.0
_C.DATASET.NUM_JOINTS_HALF_BODY = 8
_C.DATASET.COLOR_RGB = False

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001
_C.TRAIN.EXTRA_EPOCH = 230
_C.TRAIN.EXTRA_EPOCH = 230

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32
# Test Model Epoch
_C.TEST.FLIP_TEST = False
_C.TEST.POST_PROCESS = False
_C.TEST.SHIFT_HEATMAP = False
_C.TEST.BLUR_KERNEL = 11
_C.TEST.DECODE_MODE = "DAEC"
_C.TEST.DAEC = CN(new_allowed=True)
_C.TEST.DAEC.USE_EMPIRICAL_FORMULA = True
_C.TEST.DAEC.EXPAND_EDGE = 7
_C.TEST.DAEC.DELTA = 2

_C.TEST.USE_GT_BBOX = False

# nms
_C.TEST.IMAGE_THRE = 0.1
_C.TEST.NMS_THRE = 0.6
_C.TEST.SOFT_NMS = False
_C.TEST.OKS_THRE = 0.5
_C.TEST.IN_VIS_THRE = 0.0
_C.TEST.COCO_BBOX_FILE = ''
_C.TEST.BBOX_THRE = 1.0
_C.TEST.MODEL_FILE = ''

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    if args.dataDir:
        cfg.DATA_DIR = args.dataDir

    cfg.DATASET.ROOT = os.path.join(
        cfg.DATA_DIR, cfg.DATASET.ROOT
    )

    cfg.MODEL.PRETRAINED = os.path.join(
        cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    )

    if cfg.TEST.MODEL_FILE:
        cfg.TEST.MODEL_FILE = os.path.join(
            cfg.DATA_DIR, cfg.TEST.MODEL_FILE
        )

    cfg.freeze()
    
    
_C2 = CN()

_C2.OUTPUT_DIR = ''
_C2.LOG_DIR = ''
_C2.DATA_DIR = ''
_C2.GPUS = (0,)
_C2.WORKERS = 4
_C2.PRINT_FREQ = 20
_C2.AUTO_RESUME = False
_C2.PIN_MEMORY = True
_C2.RANK = 0

# Cudnn related params
_C2.CUDNN = CN()
_C2.CUDNN.BENCHMARK = True
_C2.CUDNN.DETERMINISTIC = False
_C2.CUDNN.ENABLED = True

# common params for NETWORK
_C2.MODEL = CN()
_C2.MODEL.NAME = 'pose_hrnet'
_C2.MODEL.INIT_WEIGHTS = True
_C2.MODEL.PRETRAINED = ''
_C2.MODEL.NUM_JOINTS = 17
_C2.MODEL.TAG_PER_JOINT = True
_C2.MODEL.TARGET_TYPE = 'gaussian'
_C2.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
_C2.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
_C2.MODEL.SIGMA = 2
_C2.MODEL.EXTRA = CN(new_allowed=True)

_C2.LOSS = CN()
_C2.LOSS.USE_OHKM = False
_C2.LOSS.TOPK = 8
_C2.LOSS.USE_TARGET_WEIGHT = True
_C2.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

# DATASET related params
_C2.DATASET = CN()
_C2.DATASET.ROOT = ''
_C2.DATASET.DATASET = 'mpii'
_C2.DATASET.OTHERS = True
_C2.DATASET.TRAIN_SET = 'train'
_C2.DATASET.PSEUDO_TRAIN_SET = 'pseudos'
_C2.DATASET.TEST_SET = 'valid'
_C2.DATASET.DATA_FORMAT = 'jpg'
_C2.DATASET.HYBRID_JOINTS_TYPE = ''
_C2.DATASET.SELECT_DATA = False

# training data augmentation
_C2.DATASET.FLIP = True
_C2.DATASET.SCALE_FACTOR = 0.25
_C2.DATASET.ROT_FACTOR = 30
_C2.DATASET.PROB_HALF_BODY = 0.0
_C2.DATASET.NUM_JOINTS_HALF_BODY = 8
_C2.DATASET.COLOR_RGB = False

# train
_C2.TRAIN = CN()

_C2.TRAIN.LR_FACTOR = 0.1
_C2.TRAIN.LR_STEP = [90, 110]
_C2.TRAIN.LR = 0.001
_C2.TRAIN.EXTRA_EPOCH = 230
_C2.TRAIN.EXTRA_EPOCH = 230

_C2.TRAIN.OPTIMIZER = 'adam'
_C2.TRAIN.MOMENTUM = 0.9
_C2.TRAIN.WD = 0.0001
_C2.TRAIN.NESTEROV = False
_C2.TRAIN.GAMMA1 = 0.99
_C2.TRAIN.GAMMA2 = 0.0

_C2.TRAIN.BEGIN_EPOCH = 0
_C2.TRAIN.END_EPOCH = 140

_C2.TRAIN.RESUME = False
_C2.TRAIN.CHECKPOINT = ''

_C2.TRAIN.BATCH_SIZE_PER_GPU = 32
_C2.TRAIN.SHUFFLE = True

# testing
_C2.TEST = CN()

# size of images for each device
_C2.TEST.BATCH_SIZE_PER_GPU = 32
# Test Model Epoch
_C2.TEST.FLIP_TEST = False
_C2.TEST.POST_PROCESS = False
_C2.TEST.SHIFT_HEATMAP = False
_C2.TEST.BLUR_KERNEL = 11
_C2.TEST.DECODE_MODE = "DAEC"
_C2.TEST.DAEC = CN(new_allowed=True)
_C2.TEST.DAEC.USE_EMPIRICAL_FORMULA = True
_C2.TEST.DAEC.EXPAND_EDGE = 7
_C2.TEST.DAEC.DELTA = 2

_C2.TEST.USE_GT_BBOX = False

# nms
_C2.TEST.IMAGE_THRE = 0.1
_C2.TEST.NMS_THRE = 0.6
_C2.TEST.SOFT_NMS = False
_C2.TEST.OKS_THRE = 0.5
_C2.TEST.IN_VIS_THRE = 0.0
_C2.TEST.COCO_BBOX_FILE = ''
_C2.TEST.BBOX_THRE = 1.0
_C2.TEST.MODEL_FILE = ''

# debug
_C2.DEBUG = CN()
_C2.DEBUG.DEBUG = False
_C2.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C2.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C2.DEBUG.SAVE_HEATMAPS_GT = False
_C2.DEBUG.SAVE_HEATMAPS_PRED = False

    
def update_config2(cfg2, args):
    cfg2.defrost()
    cfg2.merge_from_file(args.cfg2)
    cfg2.merge_from_list(args.opts)

    if args.modelDir:
        cfg2.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg2.LOG_DIR = args.logDir

    if args.dataDir:
        cfg2.DATA_DIR = args.dataDir

    cfg2.DATASET.ROOT = os.path.join(
        cfg2.DATA_DIR, cfg2.DATASET.ROOT
    )

    cfg2.MODEL.PRETRAINED = os.path.join(
        cfg2.DATA_DIR, cfg2.MODEL.PRETRAINED
    )

    if cfg2.TEST.MODEL_FILE:
        cfg2.TEST.MODEL_FILE = os.path.join(
            cfg2.DATA_DIR, cfg2.TEST.MODEL_FILE
        )

    cfg2.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
        
    with open(sys.argv[2], 'w') as f:
        print(_C2, file=f)


