# QuadPose: A Unified Approach to Pose Estimation in Elephants and Other Quadrupeds using Noisy Labels <a href="https://www.researchsquare.com/article/rs-6397651/latest"><img src="https://img.shields.io/badge/In_Review-Paper_Preprint-blue" ></a> </h1> 

## Highlights
- The Standardization of quadruped mammal pose estimation into two tasks: one for the African elephant species and another encompassing all other quadruped mammal species.
-  The development of neural networks following a binary classifier that dynamically routes input data through one of two specialized prediction heads based on dataset standardization type (i.e., African elephant or other quadrupeds).
- A pseudo-label generation strategy for improving model generalizability to unseen animal species using shared anatomical similarities.
   
![image](https://github.com/Obafemi-Jinadu/QuadPose/blob/141368c384cbcfc77d9232ead3867afe064d74d2/images/overall.png)

## JumboPose
- A novel manually annotated dataset for African elephant pose estimation. Instructions on how to download JumboPose can be found [here](https://tufts.box.com/s/w3btcqfc5pdsjbaw607o3cjp23v704nh)
<p align="center">
  <img src="https://github.com/Obafemi-Jinadu/QuadPose/blob/7938734cb6ef9cc581f591565fec7d9f17358f6f/images/eleph.png?raw=true" width="800"/>
  <br>
  <em>JumboPose</em>
</p>



 ## Results
<p align="center">
  <img src="https://github.com/Obafemi-Jinadu/QuadPose/blob/7a910ec7e0b46ae161b7161ab12dff2ab463d4ff/images/final_overall.png?raw=true" width="800"/>
  <br>
  <em></em>
</p>

Model weights can be downloaded [here](https://tufts.box.com/s/2f2tndlahxn2n0kynvpmqdem72qpke8p)

## Acknowledgments
This work is heavily inspired by the following repositories: [ViTPose](https://github.com/ViTAE-Transformer/ViTPose/tree/main), [HRNet](https://github.com/HRNet/HRNet-Human-Pose-Estimation/tree/master), [TransPose](https://github.com/yangsenius/TransPose/tree/main). We thank you.

For setting up, each model kindly follow the respective links. HRNet and TransPose have identical requirements.

## Todo
- [x] Include model run instructions
- [x] Demo script

 
## Citatons
```
Obafemi Jinadu, Karen Panetta, Jamie Heller et al. A Unified Approach to Pose Estimation in Elephants and Other Quadrupeds using Noisy Labels, 09 April 2025, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-6397651/v1]
```
