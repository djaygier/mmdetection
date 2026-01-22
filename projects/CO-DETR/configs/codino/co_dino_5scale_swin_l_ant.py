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
# Update num_classes in all heads using nested dict override syntax.
# This ensures we do NOT delete the bbox_roi_extractor and other
# required fields during inheritance.
model = dict(
    query_head=dict(num_classes=num_classes),
    # Override just the num_classes inside the nested bbox_head of roi_head[0]
    roi_head=[
        dict(
            bbox_head=dict(
                num_classes=num_classes))
    ],
    # Override just the num_classes inside bbox_head[0]
    bbox_head=[
        dict(num_classes=num_classes)
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
optim_wrapper = dict(optimizer=dict(lr=1e-4))

# 5. Checkpoint setting
default_hooks = dict(
    checkpoint=dict(by_epoch=True, interval=1, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50))

# 6. Load Pretrained Weights
# load_from is already set in the base config to the o365tococo checkpoint.
