import cv2
import numpy as np
from numpy import random
import torch
from torchvision.transforms import transforms
import time

from models_yolo.hrnet import HRNet
from models_yolo.poseresnet import PoseResNet

from models_yolo.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device, time_synchronized
from utils.inference import *
from config import cfg
from config import update_config
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Get keypoints network configuration')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
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
    return parser.parse_args("")

class HRNet_pseudos_with_bbox:
    """
    The class provides a simple and customizable method to load the HRNet network, load the official pre-trained
    weights, and predict the animal pose on single images.

    """

    def __init__(self,
                 c,
                 nof_joints,
                 checkpoint_path,
                 model_name='HRNet',
                 resolution=(256, 192),
                 interpolation=cv2.INTER_CUBIC,
                 return_bounding_boxes=True,
                 max_batch_size=32,
                 detect_conf_thres=0.8,
                 detect_iou_thres=0.5,
                 device=torch.device("cuda"), class_name="dog",
                 decode=False):
        """
        Initializes a new HRNet_pseudos object.
        HRNet (and YOLOv5) are initialized on the torch.device("device") and
        its (their) pre-trained weights will be loaded from disk.

        Args:
            c (int): number of channels (when using HRNet model) or resnet size (when using PoseResNet model).
            nof_joints (int): number of joints.
            checkpoint_path (str): path to an official hrnet checkpoint or a checkpoint obtained with `train_coco.py`.
            model_name (str): model name (HRNet or PoseResNet).
                Valid names for HRNet are: `HRNet`, `hrnet`
                Default: "HRNet"
            resolution (tuple): hrnet input resolution - format: (height, width).
                Default: (384, 288)
            interpolation (int): opencv interpolation algorithm.
                Default: cv2.INTER_CUBIC
            return_bounding_boxes (bool): if True, bounding boxes will be returned along with poses by self.predict.
                Default: False
            max_batch_size (int): maximum batch size used in hrnet inference.
                Useless without multiperson=True.
                Default: 32
            device (:class:`torch.device`): the hrnet (and yolo) inference will be run on this device.
                Default: torch.device("cpu")
            detect_conf_thres (float): confidence threshold for object detector
                Default: 0.8
            detect_iou_thres (float): iou threshold for object detector
                Default: 0.5
            yolo_weights (str): path to the yolov5 weight
                Default: yolov5x.pt
            decode (bool): Specify whether to use a coordinate decoding strategy
                Default: False
        """

        self.c = c
        self.nof_joints = nof_joints
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.resolution = resolution  # in the form (height, width) as in the original implementation
        self.interpolation = interpolation
        self.return_bounding_boxes = return_bounding_boxes
        self.max_batch_size = max_batch_size
        self.device = device
        self.class_name = class_name
        self.conf_thres = detect_conf_thres
        self.iou_thres = detect_iou_thres
        self.decode = decode
        self.args = parse_args()
        update_config(cfg, self.args)

        if model_name in ('HRNet', 'hrnet'):
            self.model = HRNet(c=c, nof_joints=nof_joints)
        elif model_name in ('PoseResNet', 'poseresnet', 'ResNet', 'resnet'):
            self.model = PoseResNet(resnet_size=c, nof_joints=nof_joints)
        else:
            raise ValueError('Wrong model name.')

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)

        if 'cuda' in str(self.device):
            print("device: 'cuda' - ", end="")

            if 'cuda' == str(self.device):
                # if device is set to 'cuda', all available GPUs will be used
                print("%d GPU(s) will be used" % torch.cuda.device_count())
                device_ids = None
            else:
                # if device is set to 'cuda:IDS', only that/those device(s) will be used
                print("GPU(s) '%s' will be used" % str(self.device))
                device_ids = [int(x) for x in str(self.device)[5:].split(',')]

            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        elif 'cpu' == str(self.device):
            print("device: 'cpu'")
        else:
            raise ValueError('Wrong device name.')

        self.model = self.model.to(device)
        self.model.eval()

        # Transform inputs to HRNet model resolution
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.resolution[0], self.resolution[1])),  # (height, width)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image):
        """
        Predicts the animal pose on a single image

        Args:
            image (:class:`np.ndarray`):
                the image(s) on which the human pose will be estimated.

                image is expected to be in the opencv format.
                image can be:
                    - a single image with shape=(height, width, BGR color channel)

        Returns:
            :class:`np.ndarray`:
                a numpy array containing human joints for each (detected) person.

                Format:
                    if image is a single image:
                        shape=(# of animals, # of joints (nof_joints), 3);  dtype=(np.float32).

                Each joint has 3 values: (y position, x position, joint confidence).
                If self.return_bounding_boxes, the class returns a list with (bounding boxes, animal joints)
        """
        if len(image.shape) == 3:
            return self._predict_single(image)
        else:
            raise ValueError('Wrong image format.')

    def _predict_single(self, image, bboxes):
        self.bbox = bboxes
        nof_animals = len(self.bbox)
        boxes = np.empty((nof_animals, 4), dtype=np.int32)
        images = torch.empty((nof_animals, 3, self.resolution[0], self.resolution[1]))  # (height, width)

        if self.bbox is not None:
            for i, bb in enumerate(self.bbox):

                (x1, y1, x2, y2) = bb[0], bb[1], bb[2], bb[3]
                x1 = int(round(x1))
                x2 = int(round(x2))
                y1 = int(round(y1))
                y2 = int(round(y2))

                # Adapt detections to match HRNet input aspect ratio (as suggested by xtyDoge in issue #14)
                correction_factor = self.resolution[0] / self.resolution[1] * (x2 - x1) / (y2 - y1)
                if correction_factor > 1:
                    # increase y side
                    center = y1 + (y2 - y1) // 2
                    length = int(round((y2 - y1) * correction_factor))
                    y1 = max(0, center - length // 2)
                    y2 = min(image.shape[0], center + length // 2)
                elif correction_factor < 1:
                    # increase x side
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
                    out = self.model(images)
                    # Decode output coordinates of each animal keypoint
                    preds, maxvals = get_final_preds(cfg, out.detach().cpu().numpy(), mode=cfg.TEST.DECODE_MODE)
                else:
                    out = torch.empty(
                        (images.shape[0], self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
                        device=self.device
                    )
                    for i in range(0, len(images), self.max_batch_size):
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

                        pts[i,j,1] = pt[0] * 1. / (self.resolution[1] // 4) * (boxes[i][2] - boxes[i][0]) + boxes[i][0]
                        pts[i,j,0] = pt[1] * 1. / (self.resolution[0] // 4) * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
                        pts[i,j,2] = maxvals[i,j]
        else:
            pts = np.empty((0, 0, 3), dtype=np.float32)

        if self.return_bounding_boxes:
            return boxes, pts
        else:
            return pts

    # Retrieve bounding boxes of the animal specified
    def get_bboxes(self, img0, weights='yolov5x.pt', classes=list(range(0, 80))):

        imgsz = 640
        device = '0'

        # Initialize
        device = select_device(device)  # make new output folder

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

        half = False
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        class_map = names.index(self.class_name)

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None
        bboxes = []  # run once

        img = letterbox(img0, new_shape=imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

            # Inference
        t1 = time_synchronized()
        pred = model(img, augment=True)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=classes, agnostic=True)  # opt.classes
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s, im0 = '', img0
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Append results
                for *xyxy, conf, cls in det:
                    if cls == class_map:
                        bboxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])

        return bboxes
