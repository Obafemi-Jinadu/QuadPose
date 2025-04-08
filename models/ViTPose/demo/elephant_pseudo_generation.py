import sys
import argparse
import cv2
import shutil
import pickle
import json
import glob
from scipy.io import loadmat
import numpy as np
import os
import tqdm

import argparse
import torch, torchvision
print('torch version:', torch.__version__, torch.cuda.is_available())
print('torchvision version:', torchvision.__version__)

# Check MMPose installation
import mmpose
print('mmpose version:', mmpose.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print('cuda version:', get_compiling_cuda_version())
print('compiler information:', get_compiler_version())


from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result, process_mmdet_results,inference_top_down_pose_model_modified
                        ,vis_pose_result_modified)
from mmdet.apis import inference_detector, init_detector
local_runtime = False


from torchvision.transforms import transforms
import time

import numpy as np
from IPython.display import Image, display
import tempfile
import os.path as osp

parser = argparse.ArgumentParser(description='Get arguments')
parser.add_argument('--difficulty', type=str,
                    help='easy or hard')
args = parser.parse_args()
difficulty = args.difficulty
def GenPseudo(difficulty=difficulty):
  
    categories = [{'supercategory': 'animal',
    'id': 1,
    'name': 'animal',
    'keypoints': ['bottom_trunk',
    'mid_trunk',
    'top_trunk',
    'bottom_right_tusk',
    'bottom_left_tusk',
    'top_right_tusk',
    'top_left_tusk',
    'right_eye',
    'left_eye',
    'right_bottom_ear',
    'left_bottom_ear',
    'right_bottom_tip_ear',
    'left_bottom_tip_ear',
    'right_side_tip_ear',
    'left_side_tip_ear',
    'top_right_ear',
    'top_left_ear',
    'top_right_tip_ear',
    'top_left_tip_ear',
    'hoof',
    'tail',
    'right_front_elbow',
    'left_front_elbow',
    'right_back_elbow',
    'left_back_elbow',
    'right_front_knee',
    'left_front_knee',
    'right_back_knee',
    'left_back_knee',
    'right_front_foot',
    'left_front_foot',
    'right_back_foot',
    'left_back_foot'],
    'skeleton': [[1, 2],
    [2, 3],
    [3, 8],
    [3, 9],
    [8, 9],
    [8, 16],
    [9, 17],
    [3, 10],
    [3, 11],
    [10, 20],
    [11, 20],
    [10, 22],
    [11, 22],
    [10, 23],
    [11, 23],
    [22, 26],
    [23, 27],
    [26, 30],
    [27, 31],
    [20, 21],
    [21, 24],
    [21, 25],
    [24, 28],
    [25, 29],
    [28, 32],
    [29, 33]]}]
    
    
    
    
    # initialize pose model

    #pose_config = '/media/obafemi/New Volume/ViTPose/vitpose_b_elephant.py'
    #pose_config = '/media/obafemi/New Volume/ViTPose/vitpose_b_femi_animalpose.py'
    #pose_checkpoint = '/media/obafemi/New Volume/ViTPose/work_dirs/vitpose_b_femi_animalpose/best_AP_epoch_200.pth'


    pose_config = '/media/obafemi/New Volume/ViTPose/vitpose_b_elephant_teacher.py'
    pose_checkpoint = '/media/obafemi/New Volume/ViTPose/work_dirs/vitpose_b_elephant_teacher_lr_reduced earlier/best_AP_epoch_330.pth'
    det_config = 'mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
    det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'




    pose_model = init_pose_model(pose_config, pose_checkpoint)
    # initialize detector

    det_model = init_detector(det_config, det_checkpoint)
    pseudo = '/media/obafemi/New Volume/ViTPose/data/elephant_pseudo/data/train2017'
    #pseudo = '/media/obafemi/New Volume/ViTPose/demo/rhino'

    #classes = os.listdir('/media/obafemi/New Volume/ViTPose/unlabeled_dataset')
    #classes= ['elephants']

    category_id = 0
    is_crowd = 0


    pseudo_image_list = []
    pseudo_annotations = []

    M = 0

    imgs_in = os.listdir(pseudo)

    for i, im_name in enumerate(imgs_in):
    
        image_name_ = imgs_in[i]
        image_name = os.path.join(pseudo, image_name_)

        img = cv2.imread(image_name)
        if img is None:
            print(image_name)
        h, w, _ = img.shape
        #if h * w < 256 * 256:
        #   continue
        mmdet_results = inference_detector(det_model, img)
        
        final_det = []
    
        person_results = process_mmdet_results(mmdet_results, cat_id=21)

        pose_results, returned_outputs = inference_top_down_pose_model_modified(pose_model,
                                                                    img,
                                                                    person_results,
                                                                    bbox_thr=0.3,
                                                                    format='xyxy',
                                                                        cat_id=0
                                                                            )#dataset=pose_model.cfg.data.test.type)

        if len(pose_results)==0:
            continue
        pseudo_flag = {"pseudo_flag": True, "pseudo_difficulty":difficulty}
        for k in range(len(pose_results)):
        
            bbox_coords, joints = pose_results[k]['bbox'], pose_results[k]['keypoints']
            if len(bbox_coords)==0:
                continue
            image_id = str(i)    
            folder_ = 'pseudos'
            complete_path = os.path.join("data/coco/images", folder_)
            new_name = im_name
            img_dict = {"license": 1, "file_name": new_name, "coco_url": os.path.join(complete_path, new_name),
                                        "height": h, "width": w, "date_captured": "2019",
                                        "flickr_url": None, "id": int(image_id)
                                        }
            pseudo_image_list.append(img_dict)
            bbox_first = bbox_coords

            bbox = [int(bbox_first[0]), int(bbox_first[1]), int(bbox_first[2]),int(bbox_first[3])]

            height = int(bbox_first[3] - bbox_first[1])
            width = int(bbox_first[2] - bbox_first[0])

            coords = joints

            init_kp = [0]*99
            num_keypoints= 0
            
            if difficulty=='easy':
                for pts_idx, coord in enumerate(coords):
                    
                    if coord[-1] >= 0.9:  #iniitally 0.4, changed to 0.7
                        init_kp[pts_idx * 3] = int(coord[0])
                        init_kp[(pts_idx * 3) + 1] = int(coord[1])
                        init_kp[(pts_idx * 3) + 2] = 2
                        num_keypoints += 1
            elif difficulty=='hard':
                for pts_idx, coord in enumerate(coords):
                    if coord[-1] < 0.9: 
                        init_kp[pts_idx * 3] = int(coord[0])
                        init_kp[(pts_idx * 3) + 1] = int(coord[1])
                        init_kp[(pts_idx * 3) + 2] = 2
                        num_keypoints += 1

            
            keypoints_annotations = {"num_keypoints": num_keypoints, "area": int(0.75 * height * width),
                                    "iscrowd": is_crowd, "keypoints": init_kp, "image_id": int(image_id),
                                    "bbox": bbox,
                                    "category_id": category_id, "id": int(image_id)+M, "pseudo_details":pseudo_flag}

            pseudo_annotations.append(keypoints_annotations)
            M+=1
            #print('done')
        
        
        
    info = {"description": "Animal Pose Dataset", 'url': "https://sites.google.com/view/animal-pose/",
                    "version": 1.0, "year": 2019, "contributor": "Jinkun Cao", "date_created": "2019"}

    licenses = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"}]
   

    root = {"info": info, "licenses": licenses, "images": pseudo_image_list, "annotations": pseudo_annotations,
                    "categories": categories}
    print("number of images", len(pseudo_image_list))
    print("number of annotations", len(pseudo_annotations))

    with open("pseudo_elephant_keypoints_raw_"+difficulty+ ".json", 'w') as fp:
        json.dump(root, fp)
    fp.close()

    ann_path = "/media/obafemi/New Volume/ViTPose/data/elephant_pseudo/annotations"
    if not os.path.exists(ann_path):
        os.makedirs(ann_path)
    shutil.move("pseudo_elephant_keypoints_raw_"+difficulty+".json", ann_path + "/pseudo_elephant_keypoints_raw_"+difficulty+".json")


if __name__ == "__main__":
    GenPseudo(difficulty)

