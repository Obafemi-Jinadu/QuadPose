# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Sen Yang (yangsenius@seu.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
#print(sys.path)
#sys.path.append('./TransPose/TransPose-main/lib/nms')
import argparse
import os
import pprint
import shutil

import numpy as np
import random

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import train
from core.function import train_with_pseudo
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import models
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    seed = 22
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
   
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,"animal","real",
        transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    )
    
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,"animal","real",
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    
 
    if cfg.DATASET.OTHERS:  
    
        extra_train_dataset_ap10k = eval('dataset.'+cfg.DATASET.DATASET)(
            cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
            "ap10k", 'real',
            transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
        )
        valid_dataset_ap10k = eval('dataset.'+cfg.DATASET.DATASET)(
            cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
            "ap10k", 'real',
            transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
        )


        extra_train_dataset_elephant = eval('dataset.'+'coco_elephant')(
            cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
            "elephant", 'real',
            transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
        )
        valid_dataset_elephant = eval('dataset.'+'coco_elephant')(
            cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
            "elephant", 'real',
            transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
        )


        train_loader_ap10k = torch.utils.data.DataLoader(
            extra_train_dataset_ap10k,
            batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
            shuffle=cfg.TRAIN.SHUFFLE,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY
        )

        valid_loader_ap10k = torch.utils.data.DataLoader(
        valid_dataset_ap10k,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
        )



        train_loader_elephant = torch.utils.data.DataLoader(
            extra_train_dataset_elephant,
            batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
            shuffle=cfg.TRAIN.SHUFFLE,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY
        )
        valid_loader_elephant = torch.utils.data.DataLoader(
        valid_dataset_elephant,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
        )

    best_perf = 0.0
    best_model = False
    last_epoch = -1

    optimizer = get_optimizer(cfg, model)

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']

        writer_dict['train_global_steps'] = checkpoint['train_global_steps']
        writer_dict['valid_global_steps'] = checkpoint['valid_global_steps']

        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
         optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
         last_epoch=last_epoch
     )

    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
     #   optimizer, cfg.TRAIN.END_EPOCH, eta_min=cfg.TRAIN.LR_END, last_epoch=last_epoch)

    model.cuda()

    data_list = [train_loader,train_loader_ap10k,train_loader_elephant]
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):

        logger.info("=> current learning rate is {:.6f}".format(lr_scheduler.get_last_lr()[0]))
        # train for one epoch
        if epoch<100:
            data_list = [train_loader]
            train_with_pseudo(cfg,data_list , model, criterion, optimizer, epoch,
                final_output_dir, tb_log_dir, writer_dict, False)
                
        elif epoch>=100 and epoch<150:
            data_list = [train_loader, train_loader_ap10k]
            train_with_pseudo(cfg,data_list , model, criterion, optimizer, epoch,
                final_output_dir, tb_log_dir, writer_dict, False)
        else:
            data_list = [train_loader,train_loader_ap10k,train_loader_elephant]
            train_with_pseudo(cfg,data_list , model, criterion, optimizer, epoch,
                final_output_dir, tb_log_dir, writer_dict, False)

        # evaluate on validation set 
        if epoch %10==0 or epoch ==349:
            perf_indicator = validate(
                cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tb_log_dir, writer_dict
            )
            
            
            
            if cfg.DATASET.OTHERS:
                    perf_indicator_ap10k = validate(
                            cfg, valid_loader_ap10k, valid_dataset_ap10k, model, criterion,
                            final_output_dir, tb_log_dir, writer_dict
                        )
                
                
                    perf_indicator_elephant = validate(
                            cfg, valid_loader_elephant, valid_dataset_elephant, model, criterion,
                            final_output_dir, tb_log_dir, writer_dict
                        )      
                    perf_indicator = perf_indicator   +  0.5*perf_indicator_elephant  +0.5*perf_indicator_ap10k
		
        lr_scheduler.step()

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
            'train_global_steps': writer_dict['train_global_steps'],
            'valid_global_steps': writer_dict['valid_global_steps'],
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()

