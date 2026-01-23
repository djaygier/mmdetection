# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import sys
import importlib.util
import cv2
import mmcv
import torch
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
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
    parser = argparse.ArgumentParser(description='MMDetection video demo (CUDA)')
    parser.add_argument('--video', default='video.mp4', help='Video file')
    parser.add_argument('--config', default='projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_ant.py', help='Config file')
    parser.add_argument('--checkpoint', default='/workspace/mmdetection/work_dirs/co_dino_5scale_swin_l_ant/epoch_21.pth', help='Checkpoint file')
    parser.add_argument('--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument('--out', type=str, default='video_output_cuda.mp4', help='Output video file')
    parser.add_argument('--show', action='store_true', help='Show video')
    parser.add_argument('--wait-time', type=float, default=1, help='The interval of show (s), 0 is block')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device used for inference')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Fix for PyTorch 2.6+ where weights_only=True by default
    # The checkpoint contains mmengine.HistoryBuffer which needs to be allowed
    try:
        import mmengine.logging.history_buffer
        if hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals([mmengine.logging.history_buffer.HistoryBuffer])
    except (ImportError, AttributeError):
        pass

    if not torch.cuda.is_available() and args.device != 'cpu':
        print("Warning: CUDA is not available on this machine. Inference will likely fail or require --device cpu.")

    print(f"Initializing model on {args.device}...")
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # Build test pipeline
    model.cfg.test_dataloader.dataset.pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    # Init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    if not os.path.exists(args.video):
        print(f"Error: Video file {args.video} not found.")
        return

    video_reader = mmcv.VideoReader(args.video)
    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))
        print(f"Output will be saved to {args.out}")

    print("Starting inference...")
    for frame in track_iter_progress((video_reader, len(video_reader))):
        result = inference_detector(model, frame, test_pipeline=test_pipeline)
        
        visualizer.add_datasample(
            name='video',
            image=frame,
            data_sample=result,
            draw_gt=False,
            show=False,
            pred_score_thr=args.score_thr)
        
        vis_frame = visualizer.get_image()

        if args.show:
            cv2.namedWindow('video', 0)
            mmcv.imshow(vis_frame, 'video', args.wait_time)
        
        if video_writer:
            video_writer.write(vis_frame)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Output saved to {args.out}")

if __name__ == '__main__':
    main()
