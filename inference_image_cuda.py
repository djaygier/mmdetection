# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import sys
import importlib.util
import mmcv
import torch
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS

# =============================================================================
# CO-DETR IMPORT HANDLING
# =============================================================================
def import_codetr():
    # Helper to import the custom CO-DETR project which has a hyphen in the path
    path = os.path.join('projects', 'CO-DETR', 'codetr', '__init__.py')
    if os.path.exists(path):
        spec = importlib.util.spec_from_file_location("projects.CO-DETR.codetr", path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["projects.CO-DETR.codetr"] = module
            spec.loader.exec_module(module)
            return True
    return False

if not import_codetr():
    print("Warning: Could not import projects.CO-DETR.codetr. Ensure you are running from the mmdetection root.")

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection image demo (CUDA)')
    parser.add_argument('--img', default='img.jpeg', help='Image file')
    parser.add_argument('--config', default='projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_ant.py', help='Config file')
    parser.add_argument('--checkpoint', default='work_dirs/co_dino_5scale_swin_l_ant/epoch_1.pth', help='Checkpoint file')
    parser.add_argument('--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument('--out', type=str, default='img_output_cuda.jpg', help='Output image file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device used for inference')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Fix for PyTorch 2.6+ where weights_only=True by default
    try:
        import mmengine.logging.history_buffer
        if hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals([mmengine.logging.history_buffer.HistoryBuffer])
    except (ImportError, AttributeError):
        pass

    if not torch.cuda.is_available() and args.device != 'cpu':
        print("Warning: CUDA is not available on this machine. Inference will likely fail or require --device cpu.")

    if not os.path.exists(args.img):
        print(f"Error: Image file {args.img} not found. Please ensure the image is in the current directory.")
        return

    print(f"Initializing model on {args.device}...")
    if not os.path.exists(args.checkpoint):
        print(f"Warning: Checkpoint {args.checkpoint} not found. You might need to train the model first or provide a valid path.")
    
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # Init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    print(f"Starting inference on {args.img}...")
    result = inference_detector(model, args.img)
    
    img = mmcv.imread(args.img)
    visualizer.add_datasample(
        name='result',
        image=img,
        data_sample=result,
        draw_gt=False,
        show=False,
        pred_score_thr=args.score_thr)
    
    vis_img = visualizer.get_image()

    if args.out:
        mmcv.imwrite(vis_img, args.out)
        print(f"Result saved to {args.out}")
    else:
        mmcv.imshow(vis_img, 'result')

if __name__ == '__main__':
    main()
