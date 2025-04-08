import sys
import os
sys.path.append(os.getcwd() + "simple-HRNet")

from models.detectors.YOLOv3 import YOLOv3
import json
import matplotlib.pyplot as plt
from misc.visualization import *

import glob
import cv2
from tqdm import tqdm
import imageio

from SimpleHRNet import SimpleHRNet


color = (255,255,255)
thickness = 2
radius = 2
link_pairs = [
            [19,15], [15,11],[18,14],[14,10],
            [17,13],[13,9],[16,12],[12,8],
            [11,6],[10,6],[6,7],[8,5],
            [9,5],[5,7],[4,5],[3,1],[2,0],
            [0,1],[0,4],[1,4]
          ]

link_colors = [(0,2,255),(0,2,255),(0,2,255),
    (0,2,255), (0,2,255), (0,2,255),
    (0,2,255),(0,2,255),(255,2,0), (255,2,0), (255,0,0),
    (255,0,0), (255,0,0),(255,0,0),
    (50,205,50),(100,255,0), (100,255,0),
    (0,255,100), (0,255,0), (0,255,0)]


def visualize_keypoints(image, joints, link_pairs=link_pairs, link_colors=link_colors, threshold=0.5, thickness=2):
    for i in range(len(joints)):
        coords = joints[i]
        for coord in coords:
            if coord[-1] >= threshold:
                image = cv2.circle(image, (int(coord[1]), int(coord[0])), radius, color, thickness)

        for i in range(0, len(link_pairs)):
            d_joints = link_pairs[i]
            coord1 = d_joints[0]
            coord2 = d_joints[1]
            point1 = coords[coord1]
            point2 = coords[coord2]
            if point1[-1] >= threshold and point2[-1] >= threshold:
                cv2.line(image, (int(point1[1]), int(point1[0])), (int(point2[1]), int(point2[0])), link_colors[i], \
                         thickness=thickness)

def get_imgs_keypoints(imgs_folder, weight_path, animal_name, output_folder):
    imgs_in = os.listdir(imgs_folder)
    model = SimpleHRNet(48, 20, weight_path, class_name=animal_name)
    output_path = os.path.join(output_folder, animal_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i in tqdm(range(8079, len(imgs_in))):
        img_name = os.path.join(imgs_folder, imgs_in[i])
        image = cv2.imread(img_name, cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        if h * w < 256 * 256:
            continue
        bbox, joints = model.predict(image)
        visualize_keypoints(image, joints)

        cv2.imwrite(os.path.join(output_path, imgs_in[i]), image)

imgs_path = "dogs-vs-cats/dogs"
output_folder = "dogs-vs-cats/pose_plots"
weigth_path = "all_animals/output_human384/coco/pose_hrnet/w48_384x288_adam_lr1e-3/model_best.pth"
animal_name = "dog"

import os
import pickle
import shutil

dataset_path = "dogs-vs-cats"
i = 1

for data_path in ["dogs", "cats"]:
    image_files = os.listdir(os.path.join(dataset_path, data_path))
    name_keys = {}
    print(len(image_files))
    for img_nm in image_files:
        im = os.path.join(dataset_path,data_path,img_nm)
        out_name = "0"*(12-len(str(i))) + str(i) + ".jpg"
        complete_path = os.path.join(os.getcwd(), "data/coco/images",data_path)
        if not os.path.exists(complete_path):
            os.makedirs(complete_path)
        out = os.path.join(complete_path, out_name)
        name_keys[img_nm[:-4]] = str(i)
        i += 1
        shutil.copy(im, out)

    output_dict = open(data_path + "_mapping_pseudos.pkl", "wb")
    #print(name_keys)
    pickle.dump(name_keys, output_dict)
    output_dict.close()

dog_pkl_file = open("dogs_mapping_pseudos.pkl", "rb")
cat_pkl_file = open("cats_mapping_pseudos.pkl", "rb")

dog_mapping = pickle.load(dog_pkl_file)
cat_mapping = pickle.load(cat_pkl_file)
dog_pkl_file.close()
cat_pkl_file.close()

import matplotlib.pyplot as plt

info = {"description": "Animal Pose Dataset", 'url':"https://sites.google.com/view/animal-pose/",
                   "version": 1.0, "year": 2019, "contributor": "Jinkun Cao", "date_created": "2019"}

licenses = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/","id": 1,
                          "name": "Attribution-NonCommercial-ShareAlike License"}]
#
image_folder = "dogs-vs-cats"
class_names = ["cats", "dogs"]
dataset_url = "https://sites.google.com/view/animal-pose/"
images_list_pets = []

import xmltodict

categories = [{"supercategory": "animal", "id": 1, "name": "animal", "keypoints":["L_eye", "R_eye",
                "L_ear", "R_ear", "Nose", "Throat", "Tail", "withers", "L_F_elbow", "R_F_elbow", "L_B_elbow",
                "R_B_elbow", "L_F_knee", "R_F_knee", "L_B_knee", "R_B_knee", "L_F_paw", "R_F_paw", "L_B_paw", "R_B_paw"],
               "skeleton": [[20,16], [16,12],[19,15],[15,11],[18,14],[14,10],[17,13],[13,9],[12,7],[11,7],[7,8],[9,6],
                [10,6],[6,8],[5,6],[4,2],[3,1],[1,2],[1,5],[2,5]]
               }]

i = 1
category_id = 1
is_crowd = 0

lookup_idx = {
            "l_eye":[0,1,2],"r_eye":[3,4,5],"l_ear":[6,7,8],"l_earbase":[6,7,8],"r_ear":[9,10,11],"r_earbase":[9,10,11],
            "nose":[12,13,14],"throat":[15,16,17],"tail":[18,19,20],"tailbase":[18,19,20],"withers":[21,22,23],"l_f_elbow":[24,25,26],
            "r_f_elbow":[27,28,29], "l_b_elbow":[30,31,32],"r_b_elbow":[33,34,35],"l_f_knee":[36,37,38],"r_f_knee":[39,40,41],
            "l_b_knee":[42,43,44],"r_b_knee":[45,46,47],"l_f_paw":[48,49,50],"r_f_paw":[51,52,53],"l_b_paw":[54,55,56],
            "r_b_paw":[57,58,59]
        }

annotations_pets = []
weigth_path = "all_animals/output_human384/coco/pose_hrnet/w48_384x288_adam_lr1e-3/model_best.pth"
threshold = 0.6

for animal_name in ["dog", "cat"]:
    imgs_folder = os.path.join(image_folder, animal_name + "s")
    imgs_in = os.listdir(imgs_folder)
    model = SimpleHRNet(48, 20, weigth_path, class_name=animal_name)

    for i in tqdm(range(len(imgs_in))):
        image_name = imgs_in[i]
        img_name = os.path.join(imgs_folder, image_name)
        image = cv2.imread(img_name, cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        if h * w < 256 * 256:
            continue
        bbox_coords, joints = model.predict(image)
        if len(bbox_coords) == 0 or len(bbox_coords) > 1:
            continue
        key_name = image_name[:-4]

        if key_name in dog_mapping:
            folder_ = "dogs"
            image_id = dog_mapping[key_name]
            new_id = dog_mapping[key_name]
        elif key_name in cat_mapping:
            folder_ = "cats"
            image_id = cat_mapping[key_name]
            new_id = cat_mapping[key_name]

        complete_path = os.path.join(os.getcwd(), "data/coco/images", folder_)
        new_name = '0' * (12 - len(new_id)) + new_id + ".jpg"
        img_dict = {"license": 1, "file_name": new_name, "coco_url": os.path.join(complete_path, new_name),
                    "height": h, "width": w, "date_captured": "2019",
                    "flickr_url": None, "id": int(new_id)
                    }

        images_list_pets.append(img_dict)

        bbox_first = bbox_coords[0]
        bbox = [int(bbox_first[0]), int(bbox_first[1]), int(bbox_first[2] - bbox_first[0]),
                int(bbox_first[3] - bbox_first[1])]
        height = int(bbox_first[3] - bbox_first[1])
        width = int(bbox_first[2] - bbox_first[0])
        coords = joints[0]

        id_ = "00000" + str(i)
        i += 1
        init_kp = [0] * 60

        num_keypoints = 0
        for pts_idx, coord in enumerate(coords):
            if coord[-1] > threshold:
                init_kp[pts_idx * 3] = int(coord[1])
                init_kp[(pts_idx * 3) + 1] = int(coord[0])
                init_kp[(pts_idx * 3) + 2] = 2
                num_keypoints += 1

        keypoints_annotations = {"num_keypoints": num_keypoints, "area": int(0.75 * height * width),
                                 "iscrowd": is_crowd, "keypoints": init_kp, "image_id": int(image_id), "bbox": bbox,
                                 "category_id": category_id, "id": int(image_id)}

        annotations_pets.append(keypoints_annotations)

import json
for image_list in [(images_list_pets, annotations_pets)]:
    root = {"info": info, "licenses": licenses, "images": image_list[0], "annotations": image_list[1], "categories": categories}
    print("img1", len(image_list[0]))
    print("ann1", len(image_list[1]))

    with open("pseudo_animal_keypoints" + ".json", 'w') as fp:
        json.dump(root, fp)
    fp.close()

ann_path = "data/coco/annotations"
move = True
if not os.path.exists(ann_path):
    os.makedirs(ann_path)
if move == True:
    shutil.move("pseudo_animal_keypoints.json", ann_path + "/pseudo_animal_keypoints.json")
