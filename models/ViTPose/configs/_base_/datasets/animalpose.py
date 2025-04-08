dataset_info = dict(
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
        dict(name='Nose', id=4, color=[51, 153, 255], type='upper', swap=''),
        5:
        dict(name='Throat', id=5, color=[51, 153, 255], type='upper', swap=''),
        6:
        dict(
            name='Tail', id=6, color=[51, 153, 255], type='lower',
            swap=''),
        7:
        dict(
            name='Withers', id=7, color=[51, 153, 255], type='upper', swap=''),
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
        0: dict(link=('L_eye', 'R_eye'), id=0, color=[51, 153, 255]),
        1: dict(link=('L_eye', 'L_ear'), id=1, color=[0, 255, 0]),
        2: dict(link=('R_eye', 'R_ear'), id=2, color=[255, 128, 0]),
        3: dict(link=('L_eye', 'Nose'), id=3, color=[0, 255, 0]),
        4: dict(link=('R_eye', 'Nose'), id=4, color=[255, 128, 0]),
        5: dict(link=('Nose', 'Throat'), id=5, color=[51, 153, 255]),
        6: dict(link=('Throat', 'Withers'), id=6, color=[51, 153, 255]),
        7: dict(link=('Tail', 'Withers'), id=7, color=[51, 153, 255]),
        8: dict(link=('Throat', 'L_F_elbow'), id=8, color=[0, 255, 0]),
        9: dict(link=('L_F_elbow', 'L_F_knee'), id=9, color=[0, 255, 0]),
        10: dict(link=('L_F_knee', 'L_F_paw'), id=10, color=[0, 255, 0]),
        11: dict(link=('Throat', 'R_F_elbow'), id=11, color=[255, 128, 0]),
        12: dict(link=('R_F_elbow', 'R_F_knee'), id=12, color=[255, 128, 0]),
        13: dict(link=('R_F_knee', 'R_F_paw'), id=13, color=[255, 128, 0]),
        14: dict(link=('Tail', 'L_B_elbow'), id=14, color=[0, 255, 0]),
        15: dict(link=('L_B_elbow', 'L_B_knee'), id=15, color=[0, 255, 0]),
        16: dict(link=('L_B_knee', 'L_B_paw'), id=16, color=[0, 255, 0]),
        17: dict(link=('Tail', 'R_B_elbow'), id=17, color=[255, 128, 0]),
        18: dict(link=('R_B_elbow', 'R_B_knee'), id=18, color=[255, 128, 0]),
        19: dict(link=('R_B_knee', 'R_B_paw'), id=19, color=[255, 128, 0])
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
             
             

