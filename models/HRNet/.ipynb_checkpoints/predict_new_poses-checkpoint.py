from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import os
import time

import torch
import numpy as np

from visualize import update_config, add_path

lib_path = osp.join('lib')
add_path(lib_path)

import dataset as dataset
from config import cfg
import models

import cv2
import torchvision.transforms as T
from config import cfg

from lib.utils import transforms, vis
from models_yolo.experimental import attempt_load
# import utils_yolo
import utils_yolo.torch_utils as torch_utils
import utils_yolo.general as general
import utils_yolo.datasets as datasets
from lib.core.inference import get_final_preds
# from utils_yolo.inference import get_final_preds
# from utils.torch_utils import select_device, time_synchronized
# from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
# from utils.datasets import letterbox

class Predict_pseudos:
    """
    Generate pseudo labels
    """
    
    def __init__(self,
                 model_name,
                 filename,
                 nof_joints=33,
                 resolution=(256,192),
                 interpolation=cv2.INTER_CUBIC,
                 return_bounding_boxes=32,
                 max_batch_size=32,
                 detect_conf_thres=0.8,
                 detect_iou_thres=0.5,
                 device=torch.device("cuda"),
                 class_name = "dog",
                 yolo_weights='yolov5x.pt',
                 decode=False):

        self.nof_joints = nof_joints
        self.interpolation = interpolation
        self.return_bounding_boxes = return_bounding_boxes
        self.conf_thres = detect_conf_thres
        self.max_batch_size = max_batch_size
        self.iou_thres = detect_iou_thres
        self.decode = decode
        self.class_name = class_name
        self.yolo_weights = yolo_weights
        self.device = device
        self.resolution = resolution
        self.model_name = model_name
        self.filename = filename
        self.flip_pairs = [[3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],[15,16], [17,18],
                           [21,22],[23,24],[25,26],[27,28],[29,30],[31,32]]
                      

        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        f = open(filename, "r")
        update_config(cfg, filename)

        assert self.model_name in ['T-R', 'T-H', 'T-H-L', 'T-R-A4', 'T-H-A6', 'T-H-A5', 'T-H-A4', 'T-R-A4-DirectAttention']

        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.resolution[0], self.resolution[1])),
            T.ToTensor(),
            self.normalize
        ])

        self.model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
                        cfg, is_train=True
                    )

        if cfg.TEST.MODEL_FILE:
            print(" loading model from {}".format(cfg.TEST.MODEL_FILE))
            self.model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
        else:
            raise ValueError("please choose one ckpt in cfg.TEST.MODEL_FILE")

        self.model.to(device)
        
        # print("model params:{:.3f}M".format(sum([p.numel() for p in self.model.parameters()])/1000**2))

        

    def get_preds(self, inputs):
            
            with torch.no_grad():
                self.model.eval()
                tmp = []
                tmp2 = []
                
                inputs.to(self.device)
                outputs = self.model(inputs)
                if isinstance(outputs, list):
                    output = outputs[-1]
                else:
                    output = outputs
                    
                if cfg.TEST.FLIP_TEST: 
                    input_flipped = np.flip(inputs.cpu().numpy(), 3).copy()
                    input_flipped = torch.from_numpy(input_flipped).cuda()
                    outputs_flipped = self.model(input_flipped)

                    if isinstance(outputs_flipped, list):
                        output_flipped = outputs_flipped[-1]
                    else:
                        output_flipped = outputs_flipped

                    output_flipped = transforms.flip_back(output_flipped.cpu().numpy(),
                                            self.flip_pairs)
                    output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                    output = (output + output_flipped) * 0.5
                    
                preds, maxvals = get_final_preds(
                        cfg, output.clone().cpu().numpy(), None, None, transform_back=False)
            
            return outputs, preds, maxvals
        
    def predict(self, image):
        """
        Predicts the animal pose on a single image
        """
        if len(image.shape) == 3:
            return self._predict_single(image)
        else:
            raise ValueError("Wrong image format.")

    def _predict_single(self, image):
        self.bbox = self.get_bboxes(image, weights=self.yolo_weights)
        nof_animals = len(self.bbox)
        boxes = np.empty((nof_animals, 4), dtype=np.int32)
        images = torch.empty((nof_animals, 3, self.resolution[0], self.resolution[1]))

        if self.bbox is not None:
            for i, bb in enumerate(self.bbox):

                (x1, y1, x2, y2) = bb[0], bb[1], bb[2], bb[3]
                x1 = int(round(x1))
                x2 = int(round(x2))
                y1 = int(round(y1))
                y2 = int(round(y2))

                # Adapt detections to match HRNet input aspect ratio
                correction_factor = self.resolution[0] / self.resolution[1] * (x2 - x1) / (y2 - y1)
                if correction_factor > 1:
                    # increase y side
                    center = y1 + (y2 - y1) // 2
                    length = int(round((y2 - y1) * correction_factor))
                    y1 = max(0, center - length//2)
                    y2 = min(image.shape[0], center + length // 2)
                elif correction_factor < 1:
                    center = x1 + (x2 - x1) // 2
                    length = int(round((x2 - x1) * 1 / correction_factor))
                    x1 = max(0, center - length // 2)
                    x2 = min(image.shape[1], center + length // 2)

                boxes[i] = [x1, y1, x2, y2]
                images[i] = self.transform(image[y1:y2, x1:x2, ::-1])

        if images.shape[0] > 0:
            images = images.to(self.device)

            with torch.no_grad():
                if len(images) <= self.max_batch_size:
                    
                    # out = self.model(images)
                    out, preds, maxvals = self.get_preds(images)#get_final_preds(cfg, out.clone().cpu().numpy(), None, None, transform_back=False) #, mode=cfg.TEST.DECODE_MODE
                    # from heatmap_coord to original_image_coord
                    preds = np.array([p*4+0.5 for p in preds[0]])
                else:
                    out = torch.empty(
                        (images.shape[0], self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
                        device=self.device
                    )

                    for i in range(0, len(images, self.max_batch_size)):
                        out[i:i + self.max_batch_size] = self.model(images[i:i + self.max_batch_size])

            out = out.detach().cpu().numpy()
            pts = np.empty((out.shape[0], out.shape[1], 3), dtype=np.float32)

            # For each animal, for each joint: y, x, confidence
            if self.decode == False:
                for i, animal in enumerate(out):
                    for j, joint in enumerate(animal):
                        pt = np.unravel_index(np.argmax(joint), (self.resolution[0] // 4, self.resolution[1] // 4))
                        # 0: pt_y / (height // 4) * (bb_y2 - bb_y1) + bb_y1
                        # 1: pt_x / (width // 4) * (bb_x2 - bb_x1) + bb_x1
                        # 2: confidences
                        pts[i, j, 0] = pt[0] * 1. / (self.resolution[0] // 4) * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
                        pts[i, j, 1] = pt[1] * 1. / (self.resolution[1] // 4) * (boxes[i][2] - boxes[i][0]) + boxes[i][0]
                        pts[i, j, 2] = joint[pt]
            else:

                for i, animal in enumerate(preds):
                    for j in range(len(animal)):
                        pt = animal[j]
                        print("pt", pt)

                        pts[i,j,0] = pt[1] * 1. / (self.resolution[0] // 4) * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
                        pts[i,j,1] = pt[0] * 1. / (self.resolution[1] // 4) * (boxes[i][2] - boxes[i][0]) + boxes[i][0]
                        pts[i,j,2] = maxvals[i,j]
        else:
            pts = np.empty((0,0,3), dtype=np.float32)


        if self.return_bounding_boxes:
            return boxes, pts
        else:
            return pts


    # Retrieve bounding boxes of the animal specified
    def get_bboxes(self, img0, weights='yolov5x.pt', classes=list(range(0, 80))):

        imgsz = 640
        device = '0'

        # Initialize
        device = torch_utils.select_device(device)  # make new output folder

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = general.check_img_size(imgsz, s=model.stride.max())  # check img_size

        half = False
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        class_map = names.index(self.class_name)

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None
        bboxes = []  # run once

        img = datasets.letterbox(img0, new_shape=imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

            # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=True)[0]

        # Apply NMS
        pred = general.non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=classes,
                                   agnostic=True)  # opt.classes
        t2 = torch_utils.time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s, im0 = '', img0
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = general.scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Append results
                for *xyxy, conf, cls in det:
                    if cls == class_map:
                        bboxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])

        return bboxes


