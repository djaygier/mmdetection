_base_ = './co_dino_5scale_swin_l_16xb1_16e_o365tococo.py'

# 1. Classes setting
metainfo = {
    'classes': ('Ant', ),
    'palette': [
        (220, 20, 60),
    ]
}
num_classes = 1

# 2. Model setting
# We need to update num_classes in all heads
model = dict(
    query_head=dict(num_classes=num_classes),
    roi_head=[
        dict(
            type='CoStandardRoIHead',
            bbox_head=dict(num_classes=num_classes))
    ],
    bbox_head=[
        dict(
            type='CoATSSHead',
            num_classes=num_classes)
    ]
)

# 3. Dataset setting
data_root = 'dataset/'

train_dataloader = dict(
    batch_size=1,  # Adjust based on your GPU memory
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train.json',
        data_prefix=dict(img='')))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='valid.json',
        data_prefix=dict(img='')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'valid.json')
test_evaluator = val_evaluator

# 4. Learning rate setting
# Co-DINO is sensitive to LR. 1e-4 is the default for 16 GPUs.
# If training on 1 GPU, you might want to scale it down,
# but AdamW usually handles it better.
optim_wrapper = dict(optimizer=dict(lr=1e-4))

# 5. Checkpoint setting
default_hooks = dict(
    checkpoint=dict(by_epoch=True, interval=1, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50))

# 6. Load Pretrained Weights
# load_from is already set in the base config to the o365tococo checkpoint.
# If you want to change it, uncomment the line below:
# load_from = 'https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'
