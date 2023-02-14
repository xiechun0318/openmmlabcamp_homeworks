_base_ = [
    './configs/_base_/models/fast_scnn.py', './configs/_base_/datasets/pascal_voc12.py',
    './configs/_base_/default_runtime.py', './configs/_base_/schedules/schedule_80k.py'
]

# norm_cfg = dict(type='BN', requires_grad=True, momentum=0.01)
norm_cfg = dict(type='BN', requires_grad=True)

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor, 
             decode_head=dict(num_classes=21), 
             auxiliary_head=[
                 dict(type='FCNHead',
                      in_channels=128,
                      channels=32,
                      num_convs=1,
                      num_classes=21,
                      in_index=-2,
                      norm_cfg=norm_cfg,
                      concat_input=False,
                      align_corners=False,
                      loss_decode=dict(type='CrossEntropyLoss', 
                                       use_sigmoid=True, 
                                       loss_weight=0.4)),
                dict(
                    type='FCNHead',
                    in_channels=64,
                    channels=32,
                    num_convs=1,
                    num_classes=21,
                    in_index=-3,
                    norm_cfg=norm_cfg,
                    concat_input=False,
                    align_corners=False,
                    loss_decode=dict(
                        type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),])


# Re-config the data sampler.
train_dataloader = dict(batch_size=8, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

# Re-config the optimizer.
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=4e-5)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

load_from = './fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853-0cec9937.pth'
