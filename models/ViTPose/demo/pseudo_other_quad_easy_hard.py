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

# initialize pose model
import numpy as np
import os
import tqdm
import json
import shutil


pose_config = '/media/obafemi/New Volume/ViTPose/vitpose_b_elephant_teacher.py'
pose_checkpoint = '/media/obafemi/New Volume/ViTPose/work_dirs/vitpose_b_elephant_teacher_lr_reduced earlier/best_AP_epoch_330.pth'
det_config = 'mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'




pose_model = init_pose_model(pose_config, pose_checkpoint)
# initialize detector

det_model = init_detector(det_config, det_checkpoint)
pseudo = '/media/obafemi/New Volume/ViTPose/data/ap-60k_pseudos'


category_id = 1
is_crowd = 0
categories= [{"supercategory": "animal",
          "id": 1, "name": "animal", 
          "keypoints": ["L_eye", "R_eye", "L_ear", "R_ear", "Nose", 
                        "Throat", "Tail", "withers", "L_F_elbow", "R_F_elbow",
                        "L_B_elbow", "R_B_elbow", "L_F_knee", "R_F_knee", "L_B_knee", 
                        "R_B_knee", "L_F_paw", "R_F_paw", "L_B_paw", "R_B_paw"], 
          "skeleton": [[20, 16], [16, 12], [19, 15], [15, 11], [18, 14], [14, 10],
                       [17, 13], [13, 9], [12, 7], [11, 7], [7, 8], [9, 6], 
                       [10, 6], [6, 8], [5, 6], [4, 2], [3, 1], [1, 2], [1, 5], [2, 5]]}]


pseudo_image_list_easy = []
pseudo_annotations_easy = []
pseudo_image_list_hard = []
pseudo_annotations_hard = []

M = 0
imgs_in = os.listdir(pseudo)

for i, im_name in enumerate(imgs_in):

    image_name_ = imgs_in[i]
    image_name = os.path.join(pseudo, image_name_)

    img = cv2.imread(image_name)

    h, w, _ = img.shape
    #if h * w < 256 * 256:
    # continue
    mmdet_results = inference_detector(det_model, img)
    person_results = process_mmdet_results_modified(mmdet_results)
  
    if len(person_results)==0:
        continue
    pose_results, returned_outputs = inference_top_down_pose_model_modified(pose_model,
                                                                img,
                                                                person_results,
                                                                bbox_thr=0.3,
                                                                format='xyxy',
                                                                    cat_id=1
                                                                        )


    if len(pose_results)==0:
        continue

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

        bbox_first = bbox_coords

        bbox = [int(bbox_first[0]), int(bbox_first[1]), int(bbox_first[2]),int(bbox_first[3])]

        height = int(bbox_first[3] - bbox_first[1])
        width = int(bbox_first[2] - bbox_first[0])

        coords = joints

        init_kp = [0]*60
        num_keypoints = 0
        for pts_idx, coord in enumerate(coords):
            if coord[-1] >= 0.9:  #iniitally 0.4, changed to 0.7
                init_kp[pts_idx * 3] = int(coord[0])
                init_kp[(pts_idx * 3) + 1] = int(coord[1])
                init_kp[(pts_idx * 3) + 2] = 2
                num_keypoints += 1


        keypoints_annotations = {"num_keypoints": num_keypoints, "area": int(0.75 * height * width),
                                "iscrowd": is_crowd, "keypoints": init_kp, "image_id": int(image_id),
                                "bbox": bbox,
                                "category_id": category_id, "id": int(image_id)+M}
        if num_keypoints>=17:
            pseudo_annotations_easy.append(keypoints_annotations)
            pseudo_image_list_easy.append(img_dict)
        else:
            pseudo_annotations_hard.append(keypoints_annotations)
            pseudo_image_list_hard.append(img_dict)

        M+=1


info = {"description": "Animal Pose Dataset", 'url': "https://sites.google.com/view/animal-pose/",
                "version": 1.0, "year": 2019, "contributor": "Jinkun Cao", "date_created": "2019"}

licenses = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License"}]


root_easy = {"info": info, "licenses": licenses, "images": pseudo_image_list_easy, "annotations": pseudo_annotations_easy,
                "categories": categories}
root_hard = {"info": info, "licenses": licenses, "images": pseudo_image_list_hard, "annotations": pseudo_annotations_hard,
                "categories": categories}
print("number of images", len(pseudo_image_list_easy),len(pseudo_image_list_hard))
print("number of annotations", len(pseudo_annotations_easy), len(pseudo_annotations_hard))

with open("pseudo2_other_keypoints_raw_pt_easy_other_pt9" + ".json", 'w') as fp:
    json.dump(root_easy, fp)
fp.close()
with open("pseudo2_other_keypoints_raw_pt_hard_other_pt9" + ".json", 'w') as fp:
    json.dump(root_hard, fp)
fp.close()

ann_path = "data/coco/annotations"
if not os.path.exists(ann_path):
    os.makedirs(ann_path)
shutil.move("pseudo2_other_keypoints_raw_pt_easy_other_pt9.json", ann_path + "/pseudo2_other_keypoints_raw_easy_pt_9.json")
shutil.move("pseudo2_other_keypoints_raw_pt_hard_other_pt9.json", ann_path + "/pseudo2_other_keypoints_raw_hard_pt_9.json")

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
    
    det_results.pop(20)
    det_results.pop(0)
    
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
    
    
 



