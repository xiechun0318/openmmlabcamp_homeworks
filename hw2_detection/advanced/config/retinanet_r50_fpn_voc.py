_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


model = dict(
    bbox_head=dict(
        num_classes=20))

data = dict(
    samples_per_gpu=8)
evaluation=dict(save_best='auto')

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)


# learning policy
# lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=8)
log_config = dict(interval=200)

load_from = 'checkpoints/retinanet/retinanet_r50_fpn_mstrain_3x_coco_20210718_220633-88476508.pth'