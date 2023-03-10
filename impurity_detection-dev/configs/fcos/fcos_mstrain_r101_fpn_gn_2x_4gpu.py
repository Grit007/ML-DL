# model settings
model = dict(
    type='FCOS',
    pretrained='open-mmlab://resnet101_caffe',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        style='caffe'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.4,
        neg_iou_thr=0.25,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.15),
    max_per_img=100)
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/root/project/pytorch/highspeed.v3/data/road_disease_det.v1.1/'
#
img_norm_cfg = dict(mean=[59.19478056508471, 58.40534126673214, 57.407552487498634],
                    std=[10.395, 10.12, 10.375], to_rgb=False)
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'road_disease_train/',
        img_scale=[(1836, 768)],
        multiscale_mode='value',
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.0,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    val=dict(type=dataset_type,
             ann_file=data_root + 'annotations/val.json',
             img_prefix=data_root + 'road_disease_val/',
             img_scale=(1836, 768),
             img_norm_cfg=img_norm_cfg,
             size_divisor=32,
             flip_ratio=0,
             with_mask=False,
             with_crowd=False,
             with_label=False,
             test_mode=True),

    test=dict(type=dataset_type,
              ann_file=data_root + 'annotations/val.json',
              img_prefix=data_root + 'road_disease_val/',
              img_scale=(1836, 768),
              img_norm_cfg=img_norm_cfg,
              size_divisor=32,
              flip_ratio=0,
              with_mask=False,
              with_crowd=False,
              with_label=False,
              test_mode=True))
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 10,
    step=[16, 22])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 24
device_ids = range(4)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './ckpt_dirs/fcos_r101_fpn'
load_from = None
resume_from = None
workflow = [('train', 1)]
