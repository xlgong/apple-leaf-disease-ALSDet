default_scope = 'mmyolo'
class_name = ('frog_eye_leaf_spot', 'powdery_mildew', 'rust', 'scab', 'mosaic')
metainfo = dict(
    classes=('frog_eye_leaf_spot', 'powdery_mildew', 'rust', 'scab', 'mosaic'),
    palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
             (106, 0, 228)])
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='cosine',
        lr_factor=0.01,
        max_epochs=100),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_param_scheduler=False,
        save_best='auto',
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='mmdet.DetVisualizationHook',
        draw=True,
        test_out_dir='9_20_yolov7_tiny_syncbn_fast_8x16b-300e',
        show=False,
        wait_time=2))
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
file_client_args = dict(backend='disk')
data_root = 'data2/coco/'
dataset_type = 'YOLOv5CocoDataset'
img_scale = (640, 640)
max_epochs = 100
save_epoch_intervals = 10
train_batch_size_per_gpu = 8
train_num_workers = 8
persistent_workers = True
val_batch_size_per_gpu = 1
val_num_workers = 2
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=1,
    img_size=640,
    size_divisor=32,
    extra_pad_ratio=0.5)
anchors = [[(12, 16), (19, 36), (40, 28)], [(36, 75), (76, 55), (72, 146)],
           [(142, 110), (192, 243), (459, 401)]]
strides = [8, 16, 32]
num_det_layers = 3
num_classes = 5
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv7Backbone',
        arch='Tiny',
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='LeakyReLU', inplace=True, negative_slope=0.1)),
    neck=dict(
        type='YOLOv7PAFPN',
        block_cfg=dict(type='TinyDownSampleBlock', middle_ratio=0.25),
        upsample_feats_cat_first=False,
        in_channels=[128, 256, 512],
        out_channels=[64, 128, 256],
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='LeakyReLU', inplace=True, negative_slope=0.1),
        is_tiny_version=True,
        use_repconv_outs=False),
    bbox_head=dict(
        type='YOLOv7Head',
        head_module=dict(
            type='YOLOv7HeadModule',
            num_classes=5,
            in_channels=[128, 256, 512],
            featmap_strides=[8, 16, 32],
            num_base_priors=3),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=[[(12, 16), (19, 36), (40, 28)],
                        [(36, 75), (76, 55), (72, 146)],
                        [(142, 110), (192, 243), (459, 401)]],
            strides=[8, 16, 32]),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.5),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            reduction='mean',
            loss_weight=0.05,
            return_iou=True),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=1.0),
        obj_level_weights=[4.0, 1.0, 0.4],
        prior_match_thr=4.0,
        simota_candidate_topk=10,
        simota_iou_weight=3.0,
        simota_cls_weight=1.0),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300))
pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True)
]
mosiac4_pipeline = [
    dict(
        type='Mosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        pre_transform=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True)
        ]),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_translate_ratio=0.1,
        scaling_ratio_range=(0.5, 1.6),
        border=(-320, -320),
        border_val=(114, 114, 114))
]
mosiac9_pipeline = [
    dict(
        type='Mosaic9',
        img_scale=(640, 640),
        pad_val=114.0,
        pre_transform=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True)
        ]),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_translate_ratio=0.1,
        scaling_ratio_range=(0.5, 1.6),
        border=(-320, -320),
        border_val=(114, 114, 114))
]
randchoice_mosaic_pipeline = dict(
    type='RandomChoice',
    transforms=[[{
        'type':
        'Mosaic',
        'img_scale': (640, 640),
        'pad_val':
        114.0,
        'pre_transform': [{
            'type': 'LoadImageFromFile',
            'file_client_args': {
                'backend': 'disk'
            }
        }, {
            'type': 'LoadAnnotations',
            'with_bbox': True
        }]
    }, {
        'type': 'YOLOv5RandomAffine',
        'max_rotate_degree': 0.0,
        'max_shear_degree': 0.0,
        'max_translate_ratio': 0.1,
        'scaling_ratio_range': (0.5, 1.6),
        'border': (-320, -320),
        'border_val': (114, 114, 114)
    }],
                [{
                    'type':
                    'Mosaic9',
                    'img_scale': (640, 640),
                    'pad_val':
                    114.0,
                    'pre_transform': [{
                        'type': 'LoadImageFromFile',
                        'file_client_args': {
                            'backend': 'disk'
                        }
                    }, {
                        'type': 'LoadAnnotations',
                        'with_bbox': True
                    }]
                }, {
                    'type': 'YOLOv5RandomAffine',
                    'max_rotate_degree': 0.0,
                    'max_shear_degree': 0.0,
                    'max_translate_ratio': 0.1,
                    'scaling_ratio_range': (0.5, 1.6),
                    'border': (-320, -320),
                    'border_val': (114, 114, 114)
                }]],
    prob=[0.8, 0.2])
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoice',
        transforms=[[{
            'type':
            'Mosaic',
            'img_scale': (640, 640),
            'pad_val':
            114.0,
            'pre_transform': [{
                'type': 'LoadImageFromFile',
                'file_client_args': {
                    'backend': 'disk'
                }
            }, {
                'type': 'LoadAnnotations',
                'with_bbox': True
            }]
        }, {
            'type': 'YOLOv5RandomAffine',
            'max_rotate_degree': 0.0,
            'max_shear_degree': 0.0,
            'max_translate_ratio': 0.1,
            'scaling_ratio_range': (0.5, 1.6),
            'border': (-320, -320),
            'border_val': (114, 114, 114)
        }],
                    [{
                        'type':
                        'Mosaic9',
                        'img_scale': (640, 640),
                        'pad_val':
                        114.0,
                        'pre_transform': [{
                            'type': 'LoadImageFromFile',
                            'file_client_args': {
                                'backend': 'disk'
                            }
                        }, {
                            'type': 'LoadAnnotations',
                            'with_bbox': True
                        }]
                    }, {
                        'type': 'YOLOv5RandomAffine',
                        'max_rotate_degree': 0.0,
                        'max_shear_degree': 0.0,
                        'max_translate_ratio': 0.1,
                        'scaling_ratio_range': (0.5, 1.6),
                        'border': (-320, -320),
                        'border_val': (114, 114, 114)
                    }]],
        prob=[0.8, 0.2]),
    dict(
        type='YOLOv5MixUp',
        alpha=8.0,
        beta=8.0,
        prob=0.05,
        pre_transform=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='RandomChoice',
                transforms=[[{
                    'type':
                    'Mosaic',
                    'img_scale': (640, 640),
                    'pad_val':
                    114.0,
                    'pre_transform': [{
                        'type': 'LoadImageFromFile',
                        'file_client_args': {
                            'backend': 'disk'
                        }
                    }, {
                        'type': 'LoadAnnotations',
                        'with_bbox': True
                    }]
                }, {
                    'type': 'YOLOv5RandomAffine',
                    'max_rotate_degree': 0.0,
                    'max_shear_degree': 0.0,
                    'max_translate_ratio': 0.1,
                    'scaling_ratio_range': (0.5, 1.6),
                    'border': (-320, -320),
                    'border_val': (114, 114, 114)
                }],
                            [{
                                'type':
                                'Mosaic9',
                                'img_scale': (640, 640),
                                'pad_val':
                                114.0,
                                'pre_transform': [{
                                    'type': 'LoadImageFromFile',
                                    'file_client_args': {
                                        'backend': 'disk'
                                    }
                                }, {
                                    'type': 'LoadAnnotations',
                                    'with_bbox': True
                                }]
                            }, {
                                'type': 'YOLOv5RandomAffine',
                                'max_rotate_degree': 0.0,
                                'max_shear_degree': 0.0,
                                'max_translate_ratio': 0.1,
                                'scaling_ratio_range': (0.5, 1.6),
                                'border': (-320, -320),
                                'border_val': (114, 114, 114)
                            }]],
                prob=[0.8, 0.2])
        ]),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]
train_dataloader = dict(
    batch_size=18,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='data2/coco/',
        metainfo=dict(
            classes=('frog_eye_leaf_spot', 'powdery_mildew', 'rust', 'scab',
                     'mosaic'),
            palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                     (106, 0, 228)]),
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='RandomChoice',
                transforms=[[{
                    'type':
                    'Mosaic',
                    'img_scale': (640, 640),
                    'pad_val':
                    114.0,
                    'pre_transform': [{
                        'type': 'LoadImageFromFile',
                        'file_client_args': {
                            'backend': 'disk'
                        }
                    }, {
                        'type': 'LoadAnnotations',
                        'with_bbox': True
                    }]
                }, {
                    'type': 'YOLOv5RandomAffine',
                    'max_rotate_degree': 0.0,
                    'max_shear_degree': 0.0,
                    'max_translate_ratio': 0.1,
                    'scaling_ratio_range': (0.5, 1.6),
                    'border': (-320, -320),
                    'border_val': (114, 114, 114)
                }],
                            [{
                                'type':
                                'Mosaic9',
                                'img_scale': (640, 640),
                                'pad_val':
                                114.0,
                                'pre_transform': [{
                                    'type': 'LoadImageFromFile',
                                    'file_client_args': {
                                        'backend': 'disk'
                                    }
                                }, {
                                    'type': 'LoadAnnotations',
                                    'with_bbox': True
                                }]
                            }, {
                                'type': 'YOLOv5RandomAffine',
                                'max_rotate_degree': 0.0,
                                'max_shear_degree': 0.0,
                                'max_translate_ratio': 0.1,
                                'scaling_ratio_range': (0.5, 1.6),
                                'border': (-320, -320),
                                'border_val': (114, 114, 114)
                            }]],
                prob=[0.8, 0.2]),
            dict(
                type='YOLOv5MixUp',
                alpha=8.0,
                beta=8.0,
                prob=0.05,
                pre_transform=[
                    dict(
                        type='LoadImageFromFile',
                        file_client_args=dict(backend='disk')),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        type='RandomChoice',
                        transforms=[[{
                            'type':
                            'Mosaic',
                            'img_scale': (640, 640),
                            'pad_val':
                            114.0,
                            'pre_transform': [{
                                'type': 'LoadImageFromFile',
                                'file_client_args': {
                                    'backend': 'disk'
                                }
                            }, {
                                'type': 'LoadAnnotations',
                                'with_bbox': True
                            }]
                        }, {
                            'type': 'YOLOv5RandomAffine',
                            'max_rotate_degree': 0.0,
                            'max_shear_degree': 0.0,
                            'max_translate_ratio': 0.1,
                            'scaling_ratio_range': (0.5, 1.6),
                            'border': (-320, -320),
                            'border_val': (114, 114, 114)
                        }],
                                    [{
                                        'type':
                                        'Mosaic9',
                                        'img_scale': (640, 640),
                                        'pad_val':
                                        114.0,
                                        'pre_transform': [{
                                            'type': 'LoadImageFromFile',
                                            'file_client_args': {
                                                'backend': 'disk'
                                            }
                                        }, {
                                            'type': 'LoadAnnotations',
                                            'with_bbox': True
                                        }]
                                    }, {
                                        'type': 'YOLOv5RandomAffine',
                                        'max_rotate_degree': 0.0,
                                        'max_shear_degree': 0.0,
                                        'max_translate_ratio': 0.1,
                                        'scaling_ratio_range': (0.5, 1.6),
                                        'border': (-320, -320),
                                        'border_val': (114, 114, 114)
                                    }]],
                        prob=[0.8, 0.2])
                ]),
            dict(type='YOLOv5HSVRandomAug'),
            dict(type='mmdet.RandomFlip', prob=0.5),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'flip', 'flip_direction'))
        ]))
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
    dict(
        type='LetterResize',
        scale=(640, 640),
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='data2/coco/',
        metainfo=dict(
            classes=('frog_eye_leaf_spot', 'powdery_mildew', 'rust', 'scab',
                     'mosaic'),
            palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                     (106, 0, 228)]),
        test_mode=True,
        data_prefix=dict(img='val2017/'),
        ann_file='annotations/instances_val2017.json',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
            dict(
                type='LetterResize',
                scale=(640, 640),
                allow_scale_up=False,
                pad_val=dict(img=114)),
            dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'pad_param'))
        ],
        batch_shapes_cfg=dict(
            type='BatchShapePolicy',
            batch_size=1,
            img_size=640,
            size_divisor=32,
            extra_pad_ratio=0.5)))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='data2/coco/',
        metainfo=dict(
            classes=('frog_eye_leaf_spot', 'powdery_mildew', 'rust', 'scab',
                     'mosaic'),
            palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                     (106, 0, 228)]),
        test_mode=True,
        data_prefix=dict(img='test2017/'),
        ann_file='annotations/instances_test2017.json',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
            dict(
                type='LetterResize',
                scale=(640, 640),
                allow_scale_up=False,
                pad_val=dict(img=114)),
            dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'pad_param'))
        ],
        batch_shapes_cfg=dict(
            type='BatchShapePolicy',
            batch_size=1,
            img_size=640,
            size_divisor=32,
            extra_pad_ratio=0.5)))
param_scheduler = None
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
        batch_size_per_gpu=8),
    constructor='YOLOv7OptimWrapperConstructor')
val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='data2/coco/annotations/instances_val2017.json',
    metric='bbox')
test_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='data2/coco/annotations/instances_test2017.json',
    metric='bbox',
    classwise=True)
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,
    val_interval=10,
    dynamic_intervals=[(270, 1)])
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49)
]
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
launcher = 'none'
work_dir = 'Experiment_2023_3_27/9_20_yolov7_tiny_syncbn_fast_8x16b-300e'
