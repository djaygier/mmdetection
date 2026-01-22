_base_ = './co_dino_5scale_swin_l_16xb1_16e_o365tococo.py'

# Custom imports for ViT and SFP
custom_imports = dict(
    imports=['projects.CO-DETR.codetr'], allow_failed_imports=False)

# 1. Classes setting
metainfo = {
    'classes': ('Ant', ),
    'palette': [
        (220, 20, 60),
    ]
}
num_classes = 1

# 2. ViT-B Backbone (DINOv3 Base) + SFP Neck
# Window block indexes for ViT-B (12 layers)
window_block_indexes = (
    list(range(0, 3)) + list(range(4, 7)) + list(range(8, 11)))
residual_block_indexes = []

num_dec_layer = 6
loss_lambda = 2.0

model = dict(
    _delete_=True,  # Delete entire parent model and redefine
    type='CoDETR',
    use_lsj=False,  # Disable LSJ since we use fixed 640x640 resolution
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ViT',
        img_size=640,  # Your image size
        pretrain_img_size=518,  # DINOv3 pretrain size
        patch_size=16,
        embed_dim=768,  # ViT-B
        depth=12,  # ViT-B
        num_heads=12,  # ViT-B
        mlp_ratio=4.0,  # Standard MLP ratio
        drop_path_rate=0.1,
        window_size=16,
        window_block_indexes=window_block_indexes,
        qkv_bias=True,
        use_act_checkpoint=False,
        xattn=False,  # Set True if you have xformers installed
        use_rope=True,  # DINOv3 uses RoPE
        init_values=1e-5,  # LayerScale init
        num_register_tokens=4,  # DINOv3 uses 4 register tokens
        frozen_stages=12,  # Freeze all 12 layers of ViT-B
        init_cfg=dict(type='Pretrained', checkpoint='models/dinov3_vitb14_pretrain.pth')),
    neck=dict(
        type='SFP',
        in_channels=[768],  # ViT-B embed_dim
        out_channels=256,
        num_outs=5,
        use_p2=True,
        use_act_checkpoint=False),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, 
            loss_weight=1.0 * num_dec_layer * loss_lambda),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0 * num_dec_layer * loss_lambda)),
    query_head=dict(
        type='CoDINOHead',
        num_query=900,
        num_classes=num_classes,
        in_channels=2048,
        as_two_stage=True,
        dn_cfg=dict(
            label_noise_scale=0.5,
            box_noise_scale=0.4,
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=300)),
        transformer=dict(
            type='CoDinoTransformer',
            with_coord_feat=False,
            num_co_heads=2,
            num_feature_levels=5,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                with_cp=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=5,
                        dropout=0.0),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DinoTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=5,
                            dropout=0.0),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            temperature=20,
            normalize=True),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    roi_head=[
        dict(
            type='CoStandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32, 64],
                finest_scale=56),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0 * num_dec_layer * loss_lambda),
                loss_bbox=dict(
                    type='GIoULoss',
                    loss_weight=10.0 * num_dec_layer * loss_lambda)))
    ],
    bbox_head=[
        dict(
            type='CoATSSHead',
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[4, 8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0 * num_dec_layer * loss_lambda),
            loss_bbox=dict(
                type='GIoULoss',
                loss_weight=2.0 * num_dec_layer * loss_lambda),
            loss_centerness=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0 * num_dec_layer * loss_lambda)),
    ],
    train_cfg=[
        dict(
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='FocalLossCost', weight=2.0),
                    dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                    dict(type='IoUCost', iou_mode='giou', weight=2.0)
                ])),
        dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=4000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)
    ],
    test_cfg=[
        dict(
            max_per_img=300,
            nms=dict(type='soft_nms', iou_threshold=0.8)),
        dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.0,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100)),
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.0,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=100),
    ])

# 3. Dataset setting
data_root = 'dataset/'
image_size = (640, 640)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_shape', 'batch_input_shape'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_shape', 'batch_input_shape'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=20,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train.json',
        data_prefix=dict(img=''),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=8,
    num_workers=10,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='valid.json',
        data_prefix=dict(img=''),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'valid.json')
test_evaluator = val_evaluator

# 4. Optimizer with Layer-wise LR Decay (important for ViT)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    constructor='ViTLearningRateDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, decay_rate=0.8, decay_type='layer_wise'),  # 12 layers for ViT-B
    loss_scale='dynamic')

# 5. Schedule
max_epochs = 12
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[7],
        gamma=0.1)
]

# 6. Checkpoint setting
default_hooks = dict(
    checkpoint=dict(by_epoch=True, interval=1, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50))

# No pretrained Co-DINO weights for ViT - train from scratch with DINOv2 backbone
load_from = None
