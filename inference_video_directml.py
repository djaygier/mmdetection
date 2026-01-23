# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import sys
import importlib.util

# =============================================================================
# MMCV PATCHING (Necessary for DirectML/Windows without CUDA)
# =============================================================================
def patch_mmcv():
    import mmcv
    import sys
    from types import ModuleType
    import importlib.machinery

    # Try to import _ext, if it fails, mock it safely
    try:
        import mmcv._ext
    except ImportError:
        # Create a real module object that supports __getattr__
        # Create a real module object that supports __getattr__
        class MockModule(ModuleType):
            def __getattr__(self, name):
                # Provide functional mocks for NMS to allow CPU inference
                if name == 'nms' or name == 'soft_nms':
                    def functional_nms(boxes, scores, **kwargs):
                        import torchvision
                        # torchvision.ops.nms returns indices of kept boxes
                        # It doesn't support soft_nms, so we fallback to standard NMS
                        iou_threshold = kwargs.get('iou_threshold', 0.5)
                        if 'iou_threshold' not in kwargs and 'iou_thr' in kwargs:
                            iou_threshold = kwargs['iou_thr']
                            
                        keep = torchvision.ops.nms(boxes, scores, iou_threshold)
                        
                        # soft_nms returns (dets, keep) where dets has modified scores
                        # standard nms returns (dets, keep) in mmcv format
                        # We construct the return value to match mmcv signature
                        dets = torch.cat((boxes[keep], scores[keep][:, None]), dim=1)
                        return dets, keep
                    return functional_nms
                
                # For other ops, return dummy that returns None (might still crash if result is used)
                def dummy_func(*args, **kwargs):
                    return None
                return dummy_func

        mock_name = 'mmcv._ext'
        mock_module = MockModule(mock_name)
        
        # Set essential attributes to make it look like a real module
        mock_module.__file__ = os.path.abspath('mock_mmcv_ext.py')
        mock_module.__path__ = []
        
        # Create a dummy spec
        loader = importlib.machinery.SourceFileLoader(mock_name, mock_module.__file__)
        mock_module.__spec__ = importlib.machinery.ModuleSpec(mock_name, loader)
        
        sys.modules[mock_name] = mock_module
        
        # Also patch mmcv.ops directly because it might have already imported from a failed _ext attempt
        # or it might use the python wrapper that we want to override
        try:
             import mmcv.ops.nms
             # This is tricky because we need to override the function object in the module
             # We will handle this by letting the mock be used when mmcv modules import it
             pass
        except:
             pass

        print("Successfully mocked mmcv._ext with functional NMS")

# Apply patch before importing mmdet
patch_mmcv()

# =============================================================================
# CO-DETR IMPORT HANDLING
# =============================================================================
def import_codetr():
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
    print("Warning: Could not import projects.CO-DETR.codetr.")

# =============================================================================
# DETECTRON2 / MMDET IMPORTS
# =============================================================================
import cv2
import mmcv
import torch
import torch_directml
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video demo with DirectML')
    parser.add_argument('--video', default='video.mp4', help='Video file')
    parser.add_argument('--config', default='projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_ant.py', help='Config file')
    parser.add_argument('--checkpoint', default='SWINL-Co-DETR/epoch_12.pth', help='Checkpoint file')
    parser.add_argument('--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument('--out', type=str, default='video_output.mp4', help='Output video file')
    parser.add_argument('--show', action='store_true', help='Show video')
    parser.add_argument('--wait-time', type=float, default=1, help='The interval of show (s), 0 is block')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: "cpu", "dml", or "auto"')
    return parser.parse_args()

def main():
    args = parse_args()
    
    dml_device = None
    if torch_directml.is_available():
        dml_device = torch_directml.device()
        print(f"DirectML available at: {dml_device}")
    
    # helper to decide device
    target_device = 'cpu'
    if args.device == 'dml':
        if dml_device:
            target_device = dml_device
        else:
            print("Warning: DirectML not found, falling back to CPU")
    elif args.device == 'auto':
        if dml_device:
             target_device = dml_device
    
    print(f"Selected inference device: {target_device}")

    # Load on CPU first
    print(f"Loading model from {args.config} on CPU...")
    model = init_detector(args.config, args.checkpoint, device='cpu')
    
    if str(target_device) != 'cpu':
        print(f"\nAttempting to move model to {target_device}...")
        print("Note: Large models like Swin-L may crash DirectML backend due to broadcast limitations.")
        try:
            model = model.to(target_device)
            print(f"Successfully moved model to {target_device}.")
        except Exception as e:
            print(f"Failed to move model: {e}")
            print("Falling back to CPU.")
            target_device = 'cpu' # fallback

    model.cfg.test_dataloader.dataset.pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

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
            name='video', image=frame, data_sample=result,
            draw_gt=False, show=False, pred_score_thr=args.score_thr)
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
