
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Hanbin Dai (daihanbin.ac@gmail.com) and Feng Zhang (zhangfengwcy@gmail.com)
# Modified by Victor Oludare
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from copy import deepcopy
from torch.autograd import Variable
import time
import logging
import os

import numpy as np
import torch
from torch import nn
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
import pickle


logger = logging.getLogger(__name__)



def train_with_pseudo(config, train_loader_list, model, criterion, optimizer, epoch, output_dir, tb_log_dir, writer_dict,pseudos ):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

   

   
    model.train()
    end = time.time()
    
    for train_loader in train_loader_list:
        if train_loader != 'pseudo_train_loader':
    
            for i, (input, target, target_weight, meta) in enumerate(train_loader):
                # measure data loading time
                
                #cat_id = 
                data_time.update(time.time() - end)
                #optimizer.zero_grad()
            
                # compute output
                outputs1, outputs2 = model(input, meta['cat_id'])
                outputs = outputs1 if torch.numel(outputs2)==0 else outputs2
                

                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)
                
                

                if isinstance(outputs, list):
                    loss = criterion(outputs[0], target, target_weight)
                    for output in outputs[1:]:
                        loss += criterion(output, target, target_weight)
                else:
                    output = outputs
                    loss = criterion(output, target, target_weight)
                    ### IN USE
                    

                # compute gradient and do update step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # measure accuracy and record loss
                losses.update(loss.item(), input.size(0))

                _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(), target.detach().cpu().numpy())
                acc.update(avg_acc, cnt)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % config.PRINT_FREQ == 0:
                    msg = (
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t"
                        "Speed {speed:.1f} samples/s\t"
                        "Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t"
                        "Loss {loss.val:.5f} ({loss.avg:.5f})\t"
                        "Accuracy {acc.val:.3f} ({acc.avg:.3f})".format(
                            epoch,
                            i,
                            len(train_loader),
                            batch_time=batch_time,
                            speed=input.size(0) / batch_time.val,
                            data_time=data_time,
                            loss=losses,
                            acc=acc,
                        )
                    )
                    logger.info(msg)

                    writer = writer_dict["writer"]
                    global_steps = writer_dict["train_global_steps"]
                    writer.add_scalar("train_loss", losses.val, global_steps)
                    writer.add_scalar("train_acc", acc.val, global_steps)
                    writer_dict["train_global_steps"] = global_steps + 1

                    prefix = "{}_{}".format(os.path.join(output_dir, "train"), i)
                    save_debug_images(config, input, meta, target, pred * 4, output, prefix)
            
        else:
            for i, (input, target, target_weight, meta) in enumerate(train_loader):
                # measure data loading time
                data_time.update(time.time() - end)
                # compute output
                outputs = model(input)

                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)

                if isinstance(outputs, list):
                    loss = criterion(outputs[0], target, target_weight)
                    for output in outputs[1:]:
                        loss += alpha*criterion(output, target, target_weight)
                else:
                    output = outputs
                    loss = alpha*criterion(output, target, target_weight)

                # loss = criterion(output, target, target_weight)

                # compute gradient and do update step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # measure accuracy and record loss
                losses.update(loss.item(), input.size(0))

                _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(), target.detach().cpu().numpy())
                acc.update(avg_acc, cnt)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % config.PRINT_FREQ == 0:
                    msg = (
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t"
                        "Speed {speed:.1f} samples/s\t"
                        "Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t"
                        "Loss {loss.val:.5f} ({loss.avg:.5f})\t"
                        "Accuracy {acc.val:.3f} ({acc.avg:.3f})".format(
                            epoch,
                            i,
                            len(train_loader),
                            batch_time=batch_time,
                            speed=input.size(0) / batch_time.val,
                            data_time=data_time,
                            loss=losses,
                            acc=acc,
                        )
                    )
                    logger.info(msg)

                    writer = writer_dict["writer"]
                    global_steps = writer_dict["train_global_steps"]
                    writer.add_scalar("train_loss", losses.val, global_steps)
                    writer.add_scalar("train_acc", acc.val, global_steps)
                    writer_dict["train_global_steps"] = global_steps + 1

                    prefix = "{}_{}".format(os.path.join(output_dir, "train"), i)
                    save_debug_images(config, input, meta, target, pred * 4, output, prefix)
                



def train(config, train_loader, model, criterion, optimizer, epoch, output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(), target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = (
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t"
                "Speed {speed:.1f} samples/s\t"
                "Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t"
                "Loss {loss.val:.5f} ({loss.avg:.5f})\t"
                "Accuracy {acc.val:.3f} ({acc.avg:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    speed=input.size(0) / batch_time.val,
                    data_time=data_time,
                    loss=losses,
                    acc=acc,
                )
            )
            logger.info(msg)

            writer = writer_dict["writer"]
            global_steps = writer_dict["train_global_steps"]
            writer.add_scalar("train_loss", losses.val, global_steps)
            writer.add_scalar("train_acc", acc.val, global_steps)
            writer_dict["train_global_steps"] = global_steps + 1

            prefix = "{}_{}".format(os.path.join(output_dir, "train"), i)
            save_debug_images(config, input, meta, target, pred * 4, output, prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir, tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    #config2 = config
    #config2.DATASET.DATASET = 'coco_elephant'
    

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    input_1, target_1, target_weight_1, meta_1 = next(iter(val_loader))
    tmp_ids = meta_1['cat_id'][0]
    if tmp_ids ==0:
    	all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
    elif tmp_ids==1:
    	all_preds = np.zeros((num_samples, 20, 3), dtype=np.float32)
    
    
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    counter =0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):

            # compute output
            outputs1, outputs2 = model(input,meta['cat_id'])
            outputs = outputs1 if torch.numel(outputs2)==0 else outputs2
         
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped1, outputs_flipped2 = model(input_flipped,meta['cat_id'])
                outputs_flipped = outputs_flipped1 if torch.numel(outputs_flipped2)==0 else outputs_flipped2

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(), val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
                #print('outputs_from_source ',output_flipped)

                output = (output + output_flipped) * 0.5
                #print('outputs_from_source ',output)

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(), target.cpu().numpy())
            
            
            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta["center"].numpy()
            s = meta["scale"].numpy()
            score = meta["score"].numpy()
          
            preds, maxvals = get_final_preds(config, output.clone().cpu().numpy(), c, s)
            
           
            #if preds.shape[1]==20:
            #	all_preds = np.zeros((num_samples, 20, 3), dtype=np.float32)
            #	print('all_predddsss ',all_preds.shape, num_samples)
            #	counter+=1

            all_preds[idx : idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx : idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx : idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx : idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx : idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx : idx + num_images, 5] = score
            image_path.extend(meta["image"])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = (
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {acc.val:.3f} ({acc.avg:.3f})".format(
                        i, len(val_loader), batch_time=batch_time, loss=losses, acc=acc
                    )
                )
                logger.info(msg)

                prefix = "{}_{}".format(os.path.join(output_dir, "val"), i)
                save_debug_images(config, input, meta, target, pred * 4, output, prefix)
        
        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path, filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict["writer"]
            global_steps = writer_dict["valid_global_steps"]
            writer.add_scalar("valid_loss", losses.avg, global_steps)
            writer.add_scalar("valid_acc", acc.avg, global_steps)
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars("valid", dict(name_value), global_steps)
            else:
                writer.add_scalars("valid", dict(name_values), global_steps)
            writer_dict["valid_global_steps"] = global_steps + 1

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info("| Arch " + " ".join(["| {}".format(name) for name in names]) + " |")
    logger.info("|---" * (num_values + 1) + "|")

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + "..."

    # yang added
    values_100_based = []
    for value in values:
        if value < 1:
            values_100_based.append(100 * value)
        else:
            values_100_based.append(value)

    logger.info("| " + full_arch_name + " " + " ".join(["| {:.3f}".format(value) for value in values_100_based]) + " |")
    #


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

