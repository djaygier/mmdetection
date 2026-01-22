# Copyright (c) OpenMMLab. All rights reserved.
"""Custom hook to apply FP8 (via torchao) and torch.compile to the backbone."""
import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class FP8CompileBackboneHook(Hook):
    """Hook to apply FP8 quantization and torch.compile to the backbone.
    
    This hook runs after the model is built but before training starts.
    It applies torchao's float8 training to the backbone and compiles it
    with torch.compile for maximum performance.
    
    Args:
        compile_mode (str): The torch.compile mode. Defaults to 'default'.
        enable_fp8 (bool): Whether to enable FP8. Defaults to True.
    """
    
    priority = 'VERY_HIGH'  # Run early to ensure compilation happens before training
    
    def __init__(self, compile_mode: str = 'default', enable_fp8: bool = True):
        self.compile_mode = compile_mode
        self.enable_fp8 = enable_fp8
        self._compiled = False
    
    def before_train(self, runner) -> None:
        """Apply FP8 and compile to backbone before training starts."""
        if self._compiled:
            return
            
        model = runner.model
        # Handle DistributedDataParallel wrapper
        if hasattr(model, 'module'):
            model = model.module
        
        backbone = model.backbone
        
        # Apply FP8 via torchao
        if self.enable_fp8:
            try:
                from torchao.float8 import convert_to_float8_training
                runner.logger.info('Applying FP8 (torchao) to backbone...')
                convert_to_float8_training(backbone)
                runner.logger.info('FP8 applied successfully.')
            except ImportError:
                runner.logger.warning(
                    'torchao not installed. Skipping FP8. '
                    'Install with: pip install torchao')
            except Exception as e:
                runner.logger.warning(f'Failed to apply FP8: {e}')
        
        # Apply torch.compile to backbone
        try:
            runner.logger.info(
                f'Compiling backbone with torch.compile(mode={self.compile_mode!r})...')
            compiled_backbone = torch.compile(backbone, mode=self.compile_mode)
            
            # Replace the backbone in the model
            if hasattr(runner.model, 'module'):
                runner.model.module.backbone = compiled_backbone
            else:
                runner.model.backbone = compiled_backbone
                
            runner.logger.info('Backbone compiled successfully.')
        except Exception as e:
            runner.logger.warning(f'Failed to compile backbone: {e}')
        
        self._compiled = True
