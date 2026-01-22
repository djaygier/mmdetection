import torch
import sys
import os
import re

def inspect_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Error: File {checkpoint_path} not found.")
        return

    print(f"Inspecting checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
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

    print(f"\nTotal keys in checkpoint: {len(state_dict)}")

    # Important tokens and embeddings
    print("\n=== Special Tokens & Embeddings ===")
    special_keys = ['cls_token', 'pos_embed', 'register_tokens', 'storage_tokens', 
                    'mask_token', 'patch_embed', 'rope']
    
    for key in state_dict.keys():
        for special in special_keys:
            if special in key:
                print(f"  {key}: {state_dict[key].shape}")
                break

    # Count transformer blocks
    block_nums = set()
    for key in state_dict.keys():
        match = re.match(r'blocks\.(\d+)\.', key)
        if match:
            block_nums.add(int(match.group(1)))
    
    num_blocks = len(block_nums)
    print(f"\n=== Architecture ===")
    print(f"  Number of transformer blocks: {num_blocks}")
    
    # Get embed dim from a known layer
    if 'blocks.0.norm1.weight' in state_dict:
        embed_dim = state_dict['blocks.0.norm1.weight'].shape[0]
        print(f"  Embed dimension: {embed_dim}")
    
    # Check attention structure
    if 'blocks.0.attn.qkv.weight' in state_dict:
        qkv_shape = state_dict['blocks.0.attn.qkv.weight'].shape
        print(f"  Attention QKV weight: {qkv_shape} (fused qkv)")
    elif 'blocks.0.attn.q_proj.weight' in state_dict:
        q_shape = state_dict['blocks.0.attn.q_proj.weight'].shape
        print(f"  Attention Q proj: {q_shape} (separate q/k/v)")
    
    # Check MLP structure
    if 'blocks.0.mlp.w1.weight' in state_dict:
        mlp_shape = state_dict['blocks.0.mlp.w1.weight'].shape
        print(f"  MLP w1 weight: {mlp_shape} (SwiGLU style)")
    elif 'blocks.0.mlp.fc1.weight' in state_dict:
        mlp_shape = state_dict['blocks.0.mlp.fc1.weight'].shape
        print(f"  MLP fc1 weight: {mlp_shape} (standard MLP)")

    # Key name mapping hints
    print("\n=== Key Mapping for vit.py ===")
    if 'storage_tokens' in state_dict:
        shape = state_dict['storage_tokens'].shape
        print(f"  ⚠️  Checkpoint uses 'storage_tokens' {shape}")
        print(f"     Your vit.py uses 'register_tokens' - RENAME NEEDED in load logic!")
    if 'register_tokens' in state_dict:
        print(f"  ✓  'register_tokens' matches vit.py")
        
    # First 15 keys for reference
    print("\n=== First 15 keys ===")
    for i, key in enumerate(list(state_dict.keys())[:15]):
        print(f"  {key}")

if __name__ == "__main__":
    path = "models/dinov3_vitb14_pretrain.pth"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    inspect_checkpoint(path)
