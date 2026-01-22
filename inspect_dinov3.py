import torch
import sys
import os

def inspect_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Error: File {checkpoint_path} not found.")
        return

    print(f"Inspecting checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    # Handle both wrapped (state_dict) and unwrapped checkpoints
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    print("\nRegister and Token Analysis:")
    keys_of_interest = ['pos_embed', 'cls_token', 'register_tokens', 'mask_token']
    
    for key in state_dict.keys():
        for k_idx in keys_of_interest:
            if k_idx in key:
                print(f"{key}: {state_dict[key].shape}")

    # Inspect first few layers to see name structure
    print("\nLayer Name Structure (first 10 keys):")
    for i, key in enumerate(state_dict.keys()):
        if i < 10:
            print(f"  {key}")
        else:
            break

    # Specifically check ViT block structure
    print("\nViT Specifics:")
    blocks = [k for k in state_dict.keys() if 'blocks.0.' in k]
    if blocks:
        print(f"Found {len([k for k in state_dict.keys() if '.blocks.' in k and '.weight' in k]) // (len(blocks)//2 if len(blocks)>2 else 1)}? layers roughly.")
        print(f"Example block 0 key: {blocks[0]}")

if __name__ == "__main__":
    path = "models/dinov3_vitb14_pretrain.pth"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    inspect_checkpoint(path)
