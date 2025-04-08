
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Victor Oludare
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

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
from config import cfg2
from config import update_config
from config import update_config2
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
    #parser.add_argument('--cfg2',
     #                   help='experiment configure file name',
      #                  required=True,
       #                 type=str)


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
    #update_config2(cfg2, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

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
    #writer_dict['writer'].add_graph(model, (dump_input, ))

    #logger.info(get_model_summary(model, dump_input))

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
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        "animal", 'real',
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    
    
     
    
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        "animal", 'real',
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
        
        
        extra_train_dataset_ap10k_1 = eval('dataset.'+cfg.DATASET.DATASET)(
                cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
                "ap10k_1", 'real',
                transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])
                )
                
        extra_train_dataset_ap10k_2 = eval('dataset.'+cfg.DATASET.DATASET)(
                cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
                "ap10k_2", 'real',
                transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])
            )
            
        extra_train_dataset_ap10k_3 = eval('dataset.'+cfg.DATASET.DATASET)(
                cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
                "ap10k_3", 'real',
                transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])
            )
        extra_train_dataset_ap10k_4 = eval('dataset.'+cfg.DATASET.DATASET)(
                cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
                "ap10k_4", 'real',
                transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])
            )
            
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
       
       
        
        train_loader_ap10k_1 = torch.utils.data.DataLoader(
                extra_train_dataset_ap10k_1,
                batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
                shuffle=cfg.TRAIN.SHUFFLE,
                num_workers=cfg.WORKERS,
                pin_memory=cfg.PIN_MEMORY
            )
        train_loader_ap10k = torch.utils.data.DataLoader(
                extra_train_dataset_ap10k,
                batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
                shuffle=cfg.TRAIN.SHUFFLE,
                num_workers=cfg.WORKERS,
                pin_memory=cfg.PIN_MEMORY
            )
        train_loader_ap10k_2 = torch.utils.data.DataLoader(
                extra_train_dataset_ap10k_2,
                batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
                shuffle=cfg.TRAIN.SHUFFLE,
                num_workers=cfg.WORKERS,
                pin_memory=cfg.PIN_MEMORY
            )
        train_loader_ap10k_3 = torch.utils.data.DataLoader(
                extra_train_dataset_ap10k_2,
                batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
                shuffle=cfg.TRAIN.SHUFFLE,
                num_workers=cfg.WORKERS,
                pin_memory=cfg.PIN_MEMORY
            )
        train_loader_ap10k_4 = torch.utils.data.DataLoader(
                extra_train_dataset_ap10k_2,
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
        
   
        
         #################################################
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


        ################################################
        
        
        
     

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
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )
    end_epoch = cfg.TRAIN.END_EPOCH + cfg.TRAIN.EXTRA_EPOCH
    semi_end_epoch = cfg.TRAIN.END_EPOCH//2 + end_epoch//2
    dataloader_list = [train_loader]
    
    for epoch in range(begin_epoch, end_epoch):
        #lr_scheduler.step()
        

        #if epoch >= semi_end_epoch: #>250
        if epoch ==350:
            print("*****Training phase 3*******")
            #dataloader_list.append(train_loader_tiger)
            dataloader_list =[train_loader_elephant,train_loader, train_loader_ap10k_1,train_loader_ap10k_2,train_loader_ap10k_3, train_loader_ap10k_4]
            train_with_pseudo(cfg, dataloader_list, model, criterion, optimizer, epoch,
                final_output_dir, tb_log_dir, writer_dict, False)
            #train_with_pseudo(cfg, train_loader_ap10k, model, criterion, optimizer, epoch,
                #final_output_dir, tb_log_dir, writer_dict,False)
            #train_with_pseudo(cfg, train_loader_tiger, model, criterion, optimizer, epoch,
                #final_output_dir, tb_log_dir, writer_dict, False)
        elif (epoch>=130) and (epoch<350):
        #elif (epoch >= (cfg.TRAIN.END_EPOCH)) and (epoch < semi_end_epoch): #>150 <250
            print("*****Training phase 2*******")
            #dataloader_list.append(train_loader_ap10k)
            dataloader_list = [train_loader_elephant,train_loader, train_loader_ap10k_1,train_loader_ap10k_2,train_loader_ap10k_3,train_loader_ap10k_4]
            train_with_pseudo(cfg, dataloader_list, model, criterion, optimizer, epoch,
                final_output_dir, tb_log_dir, writer_dict, False)
        
        elif (epoch>=110) and (epoch<130):
        #elif (epoch >= (cfg.TRAIN.END_EPOCH)) and (epoch < semi_end_epoch): #>150 <250
            print("*****Training phase 2*******")
            #dataloader_list.append(train_loader_ap10k)
            dataloader_list = [train_loader_elephant,train_loader,train_loader_ap10k_1,train_loader_ap10k_2,train_loader_ap10k_3 ]
            train_with_pseudo(cfg, dataloader_list, model, criterion, optimizer, epoch,
                final_output_dir, tb_log_dir, writer_dict, False)
        elif (epoch>=90) and (epoch<110):
        #elif (epoch >= (cfg.TRAIN.END_EPOCH)) and (epoch < semi_end_epoch): #>150 <250
            print("*****Training phase 2*******")
            #dataloader_list.append(train_loader_ap10k)
            dataloader_list = [train_loader_elephant,train_loader,train_loader_ap10k_1,train_loader_ap10k_2 ]
            train_with_pseudo(cfg, dataloader_list, model, criterion, optimizer, epoch,
                final_output_dir, tb_log_dir, writer_dict, False)
        elif (epoch>=60) and (epoch<90):
        #elif (epoch >= (cfg.TRAIN.END_EPOCH)) and (epoch < semi_end_epoch): #>150 <250
            print("*****Training phase 2*******")
            #dataloader_list.append(train_loader_ap10k)
            dataloader_list = [train_loader_elephant,train_loader,train_loader_ap10k_1]
            train_with_pseudo(cfg, dataloader_list, model, criterion, optimizer, epoch,
                final_output_dir, tb_log_dir, writer_dict, False)
        elif (epoch>=30) and (epoch<60):
        #elif (epoch >= (cfg.TRAIN.END_EPOCH)) and (epoch < semi_end_epoch): #>150 <250
            print("*****Training phase 2*******")
            #dataloader_list.append(train_loader_ap10k)
            dataloader_list = [train_loader_elephant,train_loader]
            train_with_pseudo(cfg, dataloader_list, model, criterion, optimizer, epoch,
                final_output_dir, tb_log_dir, writer_dict, False)
            #train_with_pseudo(cfg, train_loader_ap10k, model, criterion, optimizer, epoch,
               # final_output_dir, tb_log_dir, writer_dict, False)
        else:
            dataloader_list = [train_loader_elephant]
            # train for one epoch
            print("*****Training phase 1*******")
            train_with_pseudo(cfg, dataloader_list, model, criterion, optimizer, epoch,
                final_output_dir, tb_log_dir, writer_dict, False)

        # evaluate on validation set
        lr_scheduler.step()
        #print('lr details: ', lr_scheduler)
        print('lr details, current lr, last lr: ', lr_scheduler.get_lr(),lr_scheduler.get_last_lr())
        if epoch %10==0:
        #if epoch>=350:
            perf_indicator = validate(
                cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tb_log_dir, writer_dict
            )
            if cfg.DATASET.OTHERS:
                perf_indicator_ap10k = validate(
                        cfg, valid_loader_ap10k, valid_dataset_ap10k, model, criterion,
                        final_output_dir, tb_log_dir, writer_dict
                    )
                #perf_indicator_tiger = validate(
                 #       cfg, valid_loader_tiger, valid_dataset_tiger, model, criterion,
                  #      final_output_dir, tb_log_dir, writer_dict
                  #  )   
            
                perf_indicator_elephant = validate(
                        cfg, valid_loader_elephant, valid_dataset_elephant, model, criterion,
                        final_output_dir, tb_log_dir, writer_dict
                    )      
                perf_indicator = perf_indicator + 0.5*perf_indicator_ap10k  +  0.5*perf_indicator_elephant
            

           

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
            }, best_model, final_output_dir)

            #if (epoch > 0) and (epoch % 10 == 0):
                
                # train for one epoch on pseudo generated dataset
                #dataloader_list.append(pseudo_train_loader)
                #train_with_pseudo(cfg, dataloader_list, model, criterion, optimizer, epoch,
                    #final_output_dir, tb_log_dir, writer_dict, True)
                #dataloader_list.pop()


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

