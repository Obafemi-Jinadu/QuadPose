from data_info import dataset_info1 
_base_ = [
    './configs/_base_/default_runtime.py',
    './configs/_base_/datasets/elephant.py',     
]

evaluation = dict(interval=10, metric='mAP', save_best='AP')

optimizer = dict(
    type='Adam',
    lr=5e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[18, 150])
total_epochs = 350
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=33,
    dataset_joints=33,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32
    ])


channel_cfg2 = dict(
    num_output_channels=20,
    dataset_joints=20,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17,18,19],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17,18,19
    ])


# model settings
model = dict(
    type='TopDown',
    #pretrained=None,
    pretrained = 'mae_pretrain_vit_base.pth',
    backbone=dict(
        type='ViT',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=768,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1),
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)
        ,),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11),
    
        )

data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
    meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score','flip_pairs','category_id'],
)


data_cfg2 = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg2['num_output_channels'],
    num_joints=channel_cfg2['dataset_joints'],
    dataset_channel=channel_cfg2['dataset_channel'],
    inference_channel=channel_cfg2['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
    meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score','flip_pairs','category_id'],
)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score','flip_pairs','category_id'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score','flip_pairs','category_id'
            
        ]),
]

test_pipeline1 = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score','flip_pairs','category_id'
            
        ]),
]




test_pipeline = val_pipeline
data_root = 'data'
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train1=[
        
    	dict(
        type='ElephantDataset',
        ann_file=f'./{data_root}/elephants/annotations/elephant_keypoints_train2017.json',
        img_prefix=f'./{data_root}/elephants/data/train2017/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
        
        ],
        
        train2=[dict(
        type='AnimalPoseDataset',
        ann_file=f'./{data_root}/animalpose/annotations/animal_keypoints_train2017.json',
        img_prefix=f'./{data_root}/animalpose/data/train2017/',
        data_cfg=data_cfg2,
        pipeline=train_pipeline,
        dataset_info=dataset_info1),
        
    	dict(
        type='AnimalPoseDataset',
        ann_file=f'./{data_root}/ap-10k/annotations/ap10k_keypoints_train2017.json',
        img_prefix=f'./{data_root}/ap-10k/data/train2017/',
        data_cfg=data_cfg2,
        pipeline=train_pipeline,
        dataset_info=dataset_info1),
        
        
        ],
        
       val0=
        
        dict(
        type='AnimalPoseDataset',
        ann_file=f'./{data_root}/animalpose/annotations/animal_keypoints_val2017.json',
        img_prefix=f'./{data_root}/animalpose/data/val2017/',
        data_cfg=data_cfg2,
        pipeline=val_pipeline,
        dataset_info=dataset_info1),
      
        
        
        
        val1=
        
        dict(
        type='AnimalPoseDataset',
        ann_file=f'./{data_root}/ap-10k/annotations/ap10k_keypoints_val2017.json',
        img_prefix=f'./{data_root}/ap-10k/data/val2017/',
        data_cfg=data_cfg2,
        pipeline=val_pipeline,
        dataset_info=dataset_info1),
      
         val2 = dict(
        type='ElephantDataset',
        ann_file=f'./{data_root}/elephants/annotations/elephant_keypoints_val2017.json',
        img_prefix=f'./{data_root}/elephants/data/val2017/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
        
        
        
        
)
