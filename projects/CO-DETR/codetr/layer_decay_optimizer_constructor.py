# Copyright (c) OpenMMLab. All rights reserved.
# Ported from original CO-DETR repo for MMDetection v3
import json
import warnings
import torch.distributed as dist

from mmengine.optim import DefaultOptimWrapperConstructor
from mmengine.logging import print_log
from mmengine.dist import get_dist_info
from mmdet.registry import OPTIM_WRAPPER_CONSTRUCTORS


def get_layer_id_for_vit(var_name, max_layer_id):
    """Get the layer id to set the different learning rates for ViT."""
    if var_name in ('backbone.cls_token', 'backbone.mask_token', 'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.patch_embed'):
        return 0
    elif var_name.startswith('backbone.blocks'):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return max_layer_id - 1


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class ViTLearningRateDecayOptimizerConstructor(DefaultOptimWrapperConstructor):
    """Optimizer constructor with layer-wise learning rate decay for ViT."""

    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list with layer decay."""
        parameter_groups = {}
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        decay_rate = self.paramwise_cfg.get('decay_rate')
        decay_type = self.paramwise_cfg.get('decay_type', 'layer_wise')

        print_log(f'Build ViTLearningRateDecayOptimizerConstructor {decay_type} {decay_rate} - {num_layers}')

        weight_decay = self.base_wd
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue

            if len(param.shape) == 1 or name.endswith('.bias') or name in (
                    'pos_embed', 'cls_token') or 'relative_position_bias_table' in name or 'rel_pos' in name:
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay

            if 'layer_wise' in decay_type:
                if 'ViT' in module.backbone.__class__.__name__:
                    layer_id = get_layer_id_for_vit(name, num_layers)
                else:
                    layer_id = num_layers - 1
            else:
                layer_id = num_layers - 1

            group_name = f'layer_{layer_id}_{group_name}'

            if group_name not in parameter_groups:
                scale = decay_rate ** (num_layers - layer_id - 1)
                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * self.base_lr,
                }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)

        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
            print_log(f'Param groups = {json.dumps(to_display, indent=2)}')

        params.extend(parameter_groups.values())


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class ViTLayerDecayOptimizerConstructor(ViTLearningRateDecayOptimizerConstructor):
    """Alias for ViTLearningRateDecayOptimizerConstructor with layer decay config translation."""

    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):
        if paramwise_cfg is not None:
            paramwise_cfg.setdefault('decay_type', 'layer_wise')
            if 'layer_decay_rate' in paramwise_cfg:
                paramwise_cfg['decay_rate'] = paramwise_cfg.pop('layer_decay_rate')
        super(ViTLayerDecayOptimizerConstructor, self).__init__(optim_wrapper_cfg, paramwise_cfg)
