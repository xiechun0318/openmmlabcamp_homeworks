_base_ = [
    '../_base_/models/efficientnet_b4.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/default_runtime.py',
]

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=380,
        efficientnet_style=True,
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandomErasing'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='CenterCrop',
        crop_size=380,
        efficientnet_style=True,
        interpolation='bicubic'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

model = dict(
    head=dict(
        num_classes=5,
        topk=(1,),
    ))

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        data_prefix='data/flower/train',
        ann_file='data/flower/train.txt',
        classes='data/flower/classes.txt',
        pipeline=train_pipeline
    ),
    val=dict(
        data_prefix='data/flower/val',
        ann_file='data/flower/val.txt',
        classes='data/flower/classes.txt',
        pipeline=test_pipeline
    ),
    test=dict(
        data_prefix='data/flower/val',
        ann_file='data/flower/val.txt',
        classes='data/flower/classes.txt',
        pipeline=test_pipeline
    ))

optimizer =dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
# optimizer =dict(type='AdamW', lr=0.001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# lr_config = dict(policy='step', step=[15])
lr_config = dict(policy='step', step=[30, 60, 90])

evaluation = dict(interval=1, save_best='auto', metric='accuracy', metric_options={'topk': (1,)})

runner = dict(type='EpochBasedRunner', max_epochs=100)

log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

load_from='checkpoints/efficientnet-b4_3rdparty_8xb32-aa-advprop_in1k_20220119-38c2238c.pth'