dataset_info_eleph = dict(
    dataset_name='elephant',
    paper_info=dict(
        author='Cao, Jinkun and Tang, Hongyang and Fang, Hao-Shu and '
        'Shen, Xiaoyong and Lu, Cewu and Tai, Yu-Wing',
        title='Cross-Domain Adaptation for Animal Pose Estimation',
        container='The IEEE International Conference on '
        'Computer Vision (ICCV)',
        year='2019',
        homepage='https://sites.google.com/view/animal-pose/',
    ),
    keypoint_info= {
        0:
        dict(
            name='bottom_trunk', id=0, color=[204, 255, 255], type='upper', swap=''),
        1:
        dict(
            name='mid_trunk',
            id=1,
            color=[204, 255, 255],
            type='upper',
            swap=''),
       
        2:
        dict(
            name='top_trunk',
            id=2,
            color=[204, 255, 255],
            type='upper',
            swap=''),
        3:
        dict(name='bottom_right_tusk', id=3, color=[225, 145, 65], type='upper', swap='bottom_left_tusk'),
        4:
        dict(name='bottom_left_tusk', id=4, color=[185, 176, 0], type='upper', swap='bottom_right_tusk'),
        5:
        dict(
            name='top_right_tusk', id=5, color=[225, 145, 65], type='upper',
            swap='top_left_tusk'),
        6:
        dict(
            name='top_left_tusk', id=6, color=[185, 176, 0], type='upper', swap='top_right_tusk'),
        7:
        dict(
            name='right_eye',
            id=7,
            color=[225, 145, 65],
            type='upper',
            swap='left_eye'),
        8:
        dict(
            name='left_eye',
            id=8,
            color=[185, 176, 0],
            type='upper',
            swap='right_eye'),
        9:
        dict(
            name='right_bottom_ear',
            id=9,
            color=[225, 145, 65],
            type='upper',
            swap='left_bottom_ear'),
        10:
        dict(
            name='left_bottom_ear',
            id=10,
            color=[185, 176, 0],
            type='upper',
            swap='right_bottom_ear'),
        11:
        dict(
            name='right_bottom_tip_ear',
            id=11,
            color=[225, 145, 65],
            type='upper',
            swap='left_bottom_tip_ear'),
        12:
        dict(
            name='left_bottom_tip_ear',
            id=12,
            color=[185, 176, 0],
            type='upper',
            swap='right_bottom_tip_ear'),
        13:
        dict(
            name='right_side_tip_ear',
            id=13,
            color=[225, 145, 65],
            type='upper',
            swap='left_side_tip_ear'),
        14:
        dict(
            name='left_side_tip_ear',
            id=14,
            color=[185, 176, 0],
            type='upper',
            swap='right_side_tip_ear'),
        15:
        dict(
            name='top_right_ear',
            id=15,
            color=[225, 145, 65],
            type='upper',
            swap='top_left_ear'),
        16:
        dict(
            name='top_left_ear',
            id=16,
            color=[185, 176, 0],
            type='upper',
            swap='top_right_ear'),
        17:
        dict(
            name='top_right_tip_ear',
            id=17,
            color=[225, 145, 65],
            type='upper',
            swap='top_left_tip_ear'),
        18:
        dict(
            name='top_left_tip_ear',
            id=18,
            color=[185, 176, 0],
            type='upper',
            swap='top_right_tip_ear'),
            
      	19:
        dict(
            name='hoof',
            id=19,
            color=[204, 255, 255],
            type='upper',
            swap=''),
            
     	20:
        dict(
            name='tail',
            id=20,
            color=[204, 255, 255],
            type='lower',
            swap=''),
            
       21:
        dict(
            name='right_front_elbow',
            id=21,
            color=[225, 145, 65],
            type='upper',
            swap='left_front_elbow'),
            
            
       22:
        dict(
            name='left_front_elbow',
            id=22,
            color=[185, 176, 0],
            type='upper',
            swap='right_front_elbow'),
            
            
       23:
        dict(
            name='right_back_elbow',
            id=23,
            color=[225, 145, 65],
            type='lower',
            swap='left_back_elbow'),
            
       24:
        dict(
            name='left_back_elbow',
            id=24,
            color=[185, 176, 0],
            type='lower',
            swap='right_back_elbow'),
            
       25:
        dict(
            name='right_front_knee',
            id=25,
            color=[225, 145, 65],
            type='upper',
            swap='left_front_knee'),
            
            
       26:
        dict(
            name='left_front_knee',
            id=26,
            color=[185, 176, 0],
            type='upper',
            swap='right_front_knee'),
            
      27:
        dict(
            name='right_back_knee',
            id=27,
            color=[225, 145, 65],
            type='lower',
            swap='left_back_knee'),
            
            
      28:
        dict(
            name='left_back_knee',
            id=28,
            color=[185, 176, 0],
            type='lower',
            swap='right_back_knee'),
            
      29:
        dict(
            name='right_front_foot',
            id=29,
            color=[225, 145, 65],
            type='upper',
            swap='left_front_foot'),
      
      30:
        dict(
            name='left_front_foot',
            id=30,
            color=[185, 176, 0],
            type='upper',
            swap='right_front_foot'),
            
       31:
        dict(
            name='right_back_foot',
            id=31,
            color=[225, 145, 65],
            type='lower',
            swap='left_back_foot'),
            
      32:
        dict(
            name='left_back_foot',
            id=32,
            color=[185, 176, 0],
            type='lower',
            swap='right_back_foot')
    },
    skeleton_info={
        0: dict(link=('bottom_trunk', 'mid_trunk'), id=0, color=[204, 255, 255]),
        1: dict(link=('mid_trunk', 'top_trunk'), id=1, color=[204, 255, 255]),
        2: dict(link=('top_trunk', 'right_eye'), id=2, color=[204, 255, 255]),
        3: dict(link=('top_trunk', 'left_eye'), id=3, color=[204, 255, 255]),
        4: dict(link=('left_eye', 'right_eye'), id=4, color=[204, 255, 255]),
        5: dict(link=('right_eye', 'top_right_ear'), id=5, color=[204, 255, 255]),
        6: dict(link=('left_eye', 'top_left_ear'), id=6, color=[204, 255, 255]),
        7: dict(link=('top_left_ear', 'hoof'), id=7, color=[204, 255, 255]),
        8: dict(link=('top_right_ear', 'hoof'), id=8, color=[204, 255, 255]),
        9: dict(link=('top_right_ear',"top_right_tip_ear" ), id=9, color=[204, 255, 255]),
        
        10: dict(link=("top_right_tip_ear", "right_side_tip_ear"), id=10, color=[204, 255, 255]),
        11: dict(link=("right_side_tip_ear", "right_bottom_tip_ear"), id=11, color=[204, 255, 255]),
        12: dict(link=('right_bottom_ear', "right_bottom_tip_ear"), id=12, color=[204, 255, 255]),
        
        13: dict(link=('top_left_ear',"top_left_tip_ear" ), id=13, color=[204, 255, 255]),
        #13: dict(link=('top_right_ear', "right_side_tip_ear"), id=13, color=[255, 128, 0]),
        14: dict(link=("left_side_tip_ear", "left_bottom_tip_ear"), id=14, color=[204, 255, 255]),
        15: dict(link=('left_bottom_ear', "left_bottom_tip_ear"), id=15, color=[204, 255, 255]),
        
        
        
        
        
        16: dict(link=("top_left_tip_ear", "left_side_tip_ear"), id=16, color=[204, 255, 255]),
       
        
        #18: dict(link=('left_bottom_ear','left_front_elbow'), id=18, color=[255, 128, 0]),
        17: dict(link=('right_front_elbow', 'right_front_knee'), id=17, color=[204, 255, 255]),
        18: dict(link=('left_front_elbow', 'left_front_knee'), id=18, color=[204, 255, 255]),
        19: dict(link=('right_front_knee', 'right_front_foot'), id=19, color=[204, 255, 255]),
        20: dict(link=('left_front_knee', 'left_front_foot'), id=20, color=[204, 255, 255]),
        21: dict(link=('hoof', 'tail'), id=21, color=[204, 255, 255]),
        22: dict(link=('tail', 'right_back_elbow'), id=22, color=[204, 255, 255]),
        23: dict(link=('tail', 'left_back_elbow'), id=23, color=[204, 255, 255]),
        24: dict(link=('right_back_elbow', 'right_back_knee'), id=24, color=[204, 255, 255]),
        25: dict(link=('left_back_elbow', 'left_back_knee'), id=25, color=[204, 255, 255]),
        26: dict(link=('right_back_knee', 'right_back_foot'), id=26, color=[204, 255, 255]),
        27: dict(link=('left_back_knee', 'left_back_foot'), id=27, color=[204, 255, 255]),
       
        30: dict(link=('top_right_ear', 'top_right_ear'), id=30, color=[204, 255, 255])
       
    },
    
     joint_weights = [1., 1., 1., 1.2, 1.2, 1.2, 1.2,1.2, 1.2, 1.2, 1.2, 1, 1.2, 1.2, 1.2, 1.2, 1., 1.2, 1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.5,1.5,1.5,1.5,1.5,1.5],
    # Note: The original paper did not provide enough information about
    # the sigmas. We modified from 'https://github.com/cocodataset/'
    # 'cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py#L523'
    #default sigmas commented (author provided sigma values and joint weights)
    #sigmas=[
        #0.025, 0.025, 0.026, 0.035, 0.035, 0.10, 0.10, 0.10, 0.107, 0.107,
       # 0.107, 0.107, 0.087, 0.087, 0.087, 0.087, 0.089, 0.089, 0.089, 0.089
    #])
    
    
    sigmas = [.045, .045, .045, .045, .035, .082, .045, .087, .087, .082, .107, .082, 0.127, .087, .082, 0.127, .087, .072, .062,
             .107, 0.025, 0.025, 0.026, 0.035, 0.035, 0.10, 0.10, 0.10, 0.107, 0.107, 0.089, 0.089, 0.089
             ])
             
 
dataset_info1 = dict(
    dataset_name='animalpose',
    paper_info=dict(
        author='Cao, Jinkun and Tang, Hongyang and Fang, Hao-Shu and '
        'Shen, Xiaoyong and Lu, Cewu and Tai, Yu-Wing',
        title='Cross-Domain Adaptation for Animal Pose Estimation',
        container='The IEEE International Conference on '
        'Computer Vision (ICCV)',
        year='2019',
        homepage='https://sites.google.com/view/animal-pose/',
    ),
    keypoint_info={
        0:
        dict(
            name='L_eye', id=0, color=[0, 255, 0], type='upper', swap='R_eye'),
        1:
        dict(
            name='R_eye',
            id=1,
            color=[255, 128, 0],
            type='upper',
            swap='L_eye'),
        2:
        dict(
            name='L_ear',
            id=2,
            color=[0, 255, 0],
            type='upper',
            swap='R_ear'),
        3:
        dict(
            name='R_ear',
            id=3,
            color=[255, 128, 0],
            type='upper',
            swap='L_ear'),
        4:
        dict(name='Nose', id=4, color=[225, 145, 65], type='upper', swap=''),
        5:
        dict(name='Throat', id=5, color=[225, 145, 65], type='upper', swap=''),
        6:
        dict(
            name='Tail', id=6, color=[225, 145, 65], type='lower',
            swap=''),
        7:
        dict(
            name='Withers', id=7, color=[225, 145, 65], type='upper', swap=''),
        8:
        dict(
            name='L_F_elbow',
            id=8,
            color=[0, 255, 0],
            type='upper',
            swap='R_F_elbow'),
        9:
        dict(
            name='R_F_elbow',
            id=9,
            color=[255, 128, 0],
            type='upper',
            swap='L_F_elbow'),
        10:
        dict(
            name='L_B_elbow',
            id=10,
            color=[0, 255, 0],
            type='lower',
            swap='R_B_elbow'),
        11:
        dict(
            name='R_B_elbow',
            id=11,
            color=[255, 128, 0],
            type='lower',
            swap='L_B_elbow'),
        12:
        dict(
            name='L_F_knee',
            id=12,
            color=[0, 255, 0],
            type='upper',
            swap='R_F_knee'),
        13:
        dict(
            name='R_F_knee',
            id=13,
            color=[255, 128, 0],
            type='upper',
            swap='L_F_knee'),
        14:
        dict(
            name='L_B_knee',
            id=14,
            color=[0, 255, 0],
            type='lower',
            swap='R_B_knee'),
        15:
        dict(
            name='R_B_knee',
            id=15,
            color=[255, 128, 0],
            type='lower',
            swap='L_B_knee'),
        16:
        dict(
            name='L_F_paw',
            id=16,
            color=[0, 255, 0],
            type='upper',
            swap='R_F_paw'),
        17:
        dict(
            name='R_F_paw',
            id=17,
            color=[255, 128, 0],
            type='upper',
            swap='L_F_paw'),
        18:
        dict(
            name='L_B_paw',
            id=18,
            color=[0, 255, 0],
            type='lower',
            swap='R_B_paw'),
        19:
        dict(
            name='R_B_paw',
            id=19,
            color=[255, 128, 0],
            type='lower',
            swap='L_B_paw')
    },
    skeleton_info={
        0: dict(link=('L_eye', 'R_eye'), id=0, color=[255, 0, 128]),
        1: dict(link=('L_eye', 'L_ear'), id=1, color=[12, 12, 250]),
        2: dict(link=('R_eye', 'R_ear'), id=2, color=[12, 12, 250]),
        3: dict(link=('L_eye', 'Nose'), id=3, color=[12, 12, 250]),
        4: dict(link=('R_eye', 'Nose'), id=4, color=[12, 12, 250]),
        5: dict(link=('Nose', 'Throat'), id=5, color=[0,200,255]),
        6: dict(link=('Throat', 'Withers'), id=6, color=[0, 215, 255]),
        7: dict(link=('Tail', 'Withers'), id=7, color=[0, 215, 255]),
        8: dict(link=('Throat', 'L_F_elbow'), id=8, color=[0, 215, 255]),
        9: dict(link=('L_F_elbow', 'L_F_knee'), id=9, color=[0, 215, 255]),
        10: dict(link=('L_F_knee', 'L_F_paw'), id=10, color=[255, 0, 128]),
        11: dict(link=('Throat', 'R_F_elbow'), id=11, color=[0, 215, 255]),
        12: dict(link=('R_F_elbow', 'R_F_knee'), id=12, color=[0, 215, 255]),
        13: dict(link=('R_F_knee', 'R_F_paw'), id=13, color=[255, 0, 128]),
        14: dict(link=('Tail', 'L_B_elbow'), id=14, color=[0, 215, 255]),
        15: dict(link=('L_B_elbow', 'L_B_knee'), id=15, color=[0, 215, 255]),
        16: dict(link=('L_B_knee', 'L_B_paw'), id=16, color=[255, 0, 128]),
        17: dict(link=('Tail', 'R_B_elbow'), id=17, color=[0, 215, 255]),
        18: dict(link=('R_B_elbow', 'R_B_knee'), id=18, color=[0, 215, 255]),
        19: dict(link=('R_B_knee', 'R_B_paw'), id=19, color=[255, 0, 128])
    },
    #joint_weights=[
     #   1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.2, 1.2,
      #  1.5, 1.5, 1.5, 1.5
    #],
     joint_weights = [1., 1., 1., 1.2, 1.2, 1.2, 1.2,1.2, 1, 1.2, 1.2, 1, 1.2, 1.2, 1.2, 1.2, 1., 1.2, 1.2,1.2],
    # Note: The original paper did not provide enough information about
    # the sigmas. We modified from 'https://github.com/cocodataset/'
    # 'cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py#L523'
    #default sigmas commented (author provided sigma values and joint weights)
    #sigmas=[
        #0.025, 0.025, 0.026, 0.035, 0.035, 0.10, 0.10, 0.10, 0.107, 0.107,
       # 0.107, 0.107, 0.087, 0.087, 0.087, 0.087, 0.089, 0.089, 0.089, 0.089
    #])
    
    
    sigmas = [.045, .045, .045, .045, .035, .082, .045, .087, .107, .082, .107, .082, 0.127, .087, .082, 0.127, .087, .072, .062,
             .107]) 
             
