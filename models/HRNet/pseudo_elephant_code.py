from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as osp
import sys
import os
import torch
from visualize import update_config, add_path
lib_path = osp.join('lib')
add_path(lib_path)
import dataset as dataset
from config import cfg
import models
import torchvision.transforms as T
import numpy as np 
from lib.core.inference import get_final_preds
from lib.utils import transforms, vis
import cv2
import numpy as np
from numpy import random
import torch
import time
from models_yolo.hrnet import HRNet
from models_yolo.poseresnet import PoseResNet
from models_yolo.experimental import attempt_load
import matplotlib.pyplot as plt
from mmdet.apis import inference_detector, init_detector
local_runtime = False
import json
import pickle
import shutil

def process_mmdet_results_modified(mmdet_results):
    """Process mmdet results, and return a list of bboxes.

    Args:
        mmdet_results (list|tuple): mmdet results.
        cat_id (int): category id (default: 1 for human)

    Returns:
        person_results (list): a list of detected bounding boxes
    """
    if isinstance(mmdet_results, tuple):
        det_results = mmdet_results[0]
    else:
        det_results = mmdet_results
    
    det_results = det_results[20]
    #det_results.pop(0)
    
    bboxes = np.zeros((1,5))
    
    for i in range(len(det_results)):
        bboxes = np.vstack((bboxes,det_results[i]))
        
    #bboxes = det_results[0]
    bboxes = bboxes[1:,:]
    person_results = []
    for bbox in bboxes:
        person = {}
        person['bbox'] = bbox
        person_results.append(person)

    return person_results
    
    
os.environ['CUDA_VISIBLE_DEVICES'] ='0'

file_name =  './experiments/coco/hrnet/w32_384x288_adam_lr1e-3.yaml' # ch

f = open(file_name, 'r')
update_config(cfg, file_name)

model_name = 'T-H-A4'
assert model_name in ['T-R', 'T-H','T-H-L','T-R-A4', 'T-H-A6', 'T-H-A5', 'T-H-A4' ,'T-R-A4-DirectAttention']

normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
transform_pipeline = T.Compose([
            T.ToPILImage(),
            T.Resize((384, 288)),  # (height, width)
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

device = torch.device('cuda')
model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
    cfg, is_train=True
)

checkpoint = torch.load('') # model weights


state_dict = checkpoint#['state_dict']
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict, strict=True)


model.to(device)
print("model params:{:.3f}M".format(sum([p.numel() for p in model.parameters()])/1000**2))


def pred_single(image,bbox, resolution=(384, 288), animal_type='elephant', decode=False, transform = transform_pipeline, model=model):
    if animal_type=='elephant':
        nof_joints = 33
        cat_id = torch.tensor(0)
    else:
        nof_joints =20
        cat_id = torch.tensor(1)
    nof_animals = len(bbox)
    
    boxes = np.zeros((nof_animals, 4), dtype=np.int32)
    images = torch.zeros((nof_animals, 3, resolution[0], resolution[1]))  # (height, width)

    if bbox is not None:
        for i, bb in enumerate(bbox):

            (x1, y1, x2, y2) = bb[0], bb[1], bb[2], bb[3]
            x1 = int(round(x1))
            x2 = int(round(x2))
            y1 = int(round(y1))
            y2 = int(round(y2))
            #boxes[i] = [x1, y1, x2, y2]
            # Adapt detections to match HRNet input aspect ratio (as suggested by xtyDoge in issue #14)
            correction_factor = resolution[0] / resolution[1] * (x2 - x1) / (y2 - y1)
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
            images[i] = transform(image[y1:y2, x1:x2, ::-1])

    if images.shape[0] > 0:
        images = images.to('cuda')

        with torch.no_grad():
            if len(images) <= 0:
                out = model(images)
                print('hereeee')
                # Decode output coordinates of each animal keypoint
                preds, maxvals = get_final_preds(cfg, out.detach().cpu().numpy(), mode=cfg.TEST.DECODE_MODE)
            else:
                out = torch.empty(
                    (images.shape[0], nof_joints, resolution[0] // 4, resolution[1] // 4),
                    device='cuda'
                )
                for i in range(len(images)):
                    out[i:i +1],_ = model(images[i:i +1],cat_id)

        out = out.detach().cpu().numpy()
        pts = np.empty((out.shape[0], out.shape[1], 3), dtype=np.float32)

        # For each animal, for each joint: y, x, confidence
        if decode == False:
            for i, animal in enumerate(out):
                for j, joint in enumerate(animal):
                    pt = np.unravel_index(np.argmax(joint), (resolution[0] // 4, resolution[1] // 4))
                    # 0: pt_y / (height // 4) * (bb_y2 - bb_y1) + bb_y1
                    # 1: pt_x / (width // 4) * (bb_x2 - bb_x1) + bb_x1
                    # 2: confidences
                    pts[i, j, 0] = pt[0] * 1. / (resolution[0] // 4) * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
                    pts[i, j, 1] = pt[1] * 1. / (resolution[1] // 4) * (boxes[i][2] - boxes[i][0]) + boxes[i][0]
                    pts[i, j, 2] = joint[pt]
        else:

            for i, animal in enumerate(preds):
                for j in range(len(animal)):
                    pt = animal[j]

                    pts[i,j,1] = pt[0] * 1. / (resolution[1] // 4) * (boxes[i][2] - boxes[i][0]) + boxes[i][0]
                    pts[i,j,0] = pt[1] * 1. / (resolution[0] // 4) * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
                    pts[i,j,2] = maxvals[i,j]
    else:
        pts = np.empty((0, 0, 3), dtype=np.float32)

    return boxes, pts
 

category_id = 0
is_crowd = 0
categories= [{"supercategory": "animal", "id": 0, "name": "animal", "keypoints": ["bottom_trunk", "mid_trunk", 
                                                                      "top_trunk", "bottom_right_tusk",
                                                                      "bottom_left_tusk", "top_right_tusk",
                                                                      "top_left_tusk", "right_eye",
                                                                      "left_eye", "right_bottom_ear", 
                                                                      "left_bottom_ear", 
                                                                      "right_bottom_tip_ear",
                                                                      "left_bottom_tip_ear", 
                                                                      "right_side_tip_ear",
                                                                      "left_side_tip_ear", "top_right_ear", 
                                                                      "top_left_ear", "top_right_tip_ear",
                                                                      "top_left_tip_ear", "hoof", "tail",
                                                                      "right_front_elbow", "left_front_elbow",
                                                                      "right_back_elbow", "left_back_elbow",
                                                                      "right_front_knee", "left_front_knee", 
                                                                      "right_back_knee", "left_back_knee",
                                                                      "right_front_foot", "left_front_foot",
                                                                      "right_back_foot", "left_back_foot"],
  "skeleton": [[1, 2], [2, 3], [3, 8], [3, 9], [8, 9], [8, 16], [9, 17], [3, 10], [3, 11], [10, 20], 
               [11, 20], [10, 22], [11, 22], [10, 23], [11, 23], [22, 26], [23, 27], [26, 30],
               [27, 31], [20, 21], [21, 24], [21, 25], [24, 28], [25, 29], [28, 32], [29, 33]]}]

info = {"description": "Animal Pose Dataset", 'url':"https://sites.google.com/view/animal-pose/",
                   "version": 1.0, "year": 2019, "contributor": "Jinkun Cao", "date_created": "2019"}

licenses = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/","id": 1,
                          "name": "Attribution-NonCommercial-ShareAlike License"}]



det_config = 'faster_rcnn_r50_fpn_coco.py'
det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
det_model = init_detector(det_config, det_checkpoint)
pseudo = '' #pseudo data path

pseudo_image_list = []
pseudo_annotations = []

M = 0
imgs_in = os.listdir(pseudo)
images_list_pets=[]
annotations_pets=[]

boxes = []


   
for i, im_name in enumerate(imgs_in):
    image_name_ = imgs_in[i]
    image_name = os.path.join(pseudo, image_name_)

    img = cv2.imread(image_name)

    h, w, _ = img.shape
    mmdet_results = inference_detector(det_model, img)
    person_results = process_mmdet_results_modified(mmdet_results)
    bboxes = [i['bbox']     for i in person_results if i['bbox'][4]>0.3]
    if len(bboxes)==0:
       
        continue
   
    bbox_coords, joints = pred_single(img,bboxes,model=model, transform=transform_pipeline)
   
    image_id = str(i)
    folder_ = 'pseudos_hrnet'


    complete_path =os.path.join("data/coco/images", folder_)
    new_name = im_name
    img_dict = {"license": 1, "file_name": new_name, "coco_url": os.path.join(complete_path, new_name),
                "height": h, "width": w, "date_captured": "2019",
                "flickr_url": None, "id": int(im_name.split('.')[0])
                }


    boxes.append(bbox_coords)
    bbox_first = bbox_coords[0]
    
    
    bbox = [int(bbox_first[0]), int(bbox_first[1]),int(bbox_first[2]), int(bbox_first[3])]
  
    height = int(bbox_first[3] - bbox_first[1])
    width = int(bbox_first[2] - bbox_first[0])
    coords = joints[0]



    id_ = "00000" + str(i)
    i += 1
    init_kp = [0] * 99
    num_keypoints = 0


    for pts_idx, coord in enumerate(coords):
        if coord[-1]>=0.9:
            init_kp[pts_idx * 3] = int(coord[1])
            init_kp[(pts_idx * 3) + 1] = int(coord[0])
            init_kp[(pts_idx * 3) + 2] = 2
            num_keypoints += 1

    

    if num_keypoints==0 :
        continue
        
   

    images_list_pets.append(img_dict)

    keypoints_annotations = {"num_keypoints": num_keypoints, "area": int(0.75 * height * width),
                        "iscrowd": is_crowd, "keypoints": init_kp, "image_id": int(im_name.split('.')[0]), "bbox": bbox,
                        "category_id": category_id, "id": int(image_id)+M}
    annotations_pets.append(keypoints_annotations)

    M+=1


for image_list in [(images_list_pets, annotations_pets)]:
 
    root = {"info": info, "licenses": licenses, "images": image_list[0], "annotations": image_list[1], "categories": categories}
    print("img1", len(image_list[0]))
    print("ann1", len(image_list[1]))
    
    with open("pseudo_elephant32_keypoints_train2017" + ".json", 'w') as fp:
     
        json.dump(root, fp)
    fp.close()

ann_path = "" # annotation file
move = True
if not os.path.exists(ann_path):
    os.makedirs(ann_path)
if move == True:
    shutil.move("pseudo_elephant32_keypoints_train2017.json", ann_path + "/pseudo_elephant32_keypoints_train2017.json")

    
