import os
import matplotlib.pyplot as plt
from helper_functions import *
from misc.visualization import *
from HRNet_pseudos import HRNet_pseudos
from tqdm import tqdm

categories = [{"supercategory": "animal", "id": 1, "name": "animal", "keypoints":["L_eye", "R_eye",
                "L_ear", "R_ear", "Nose", "Throat", "Tail", "withers", "L_F_elbow", "R_F_elbow", "L_B_elbow",
                "R_B_elbow", "L_F_knee", "R_F_knee", "L_B_knee", "R_B_knee", "L_F_paw", "R_F_paw", "L_B_paw", "R_B_paw"],
               "skeleton": [[20,16], [16,12],[19,15],[15,11],[18,14],[14,10],[17,13],[13,9],[12,7],[11,7],[7,8],[9,6],
                [10,6],[6,8],[5,6],[4,2],[3,1],[1,2],[1,5],[2,5]]
               }]

skeleton = categories[0]["skeleton"]


color = (255,255,255) 
thickness = 2
radius = 2
#connecting pairs
link_pairs = [
            [19,15], [15,11],[18,14],[14,10],
            [17,13],[13,9],[16,12],[12,8],
            [11,6],[10,6],[6,7],[8,5],
            [9,5],[5,7],[4,5],[3,1],[2,0],
            [0,1],[0,4],[1,4]
          ]

#colors of joints
link_colors = [(0,2,255),(0,2,255),(0,2,255),
    (0,2,255), (0,2,255), (0,2,255),
    (0,2,255),(0,2,255),(255,2,0), (255,2,0), (255,0,0), 
    (255,0,0), (255,0,0),(255,0,0),
    (50,205,50),(100,255,0), (100,255,0), 
    (0,255,100), (0,255,0), (0,255,0)]

thresh = 0.3