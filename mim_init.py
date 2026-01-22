import os
import os.path as osp
import shutil
import platform
import mmdet

def manual_mim_init():
    """Manually create the .mim directory and link/copy necessary folders.
    This version detects the ACTUAL installed location of mmdet.
    """
    filenames = [
        'tools', 'configs', 'demo', 'model-index.yml', 'dataset-index.yml'
    ]
    
    # 1. Detect where mmdet is actually installed
    mmdet_path = osp.dirname(mmdet.__file__)
    mim_path = osp.join(mmdet_path, '.mim')
    
    # 2. Find the current repository root (where tools/configs are)
    repo_path = os.getcwd()
    
    print(f"Active mmdet package found at: {mmdet_path}")
    print(f"Initializing MIM at: {mim_path} using content from: {repo_path}")
    
    os.makedirs(mim_path, exist_ok=True)

    for filename in filenames:
        src_path = osp.join(repo_path, filename)
        tar_path = osp.join(mim_path, filename)
        
        if not osp.exists(src_path):
            print(f"Warning: {filename} not found in {repo_path}. Skipping.")
            continue

        # Remove existing target if it exists
        if osp.isfile(tar_path) or osp.islink(tar_path):
            os.remove(tar_path)
        elif osp.isdir(tar_path):
            shutil.rmtree(tar_path)

        # Create link or copy
        if platform.system() != 'Windows':
            try:
                os.symlink(src_path, tar_path)
                print(f"Linked {filename} -> {tar_path}")
            except OSError:
                if osp.isdir(src_path):
                    shutil.copytree(src_path, tar_path)
                else:
                    shutil.copyfile(src_path, tar_path)
                print(f"Copied {filename} (symlink failed)")
        else:
            if osp.isdir(src_path):
                shutil.copytree(src_path, tar_path)
            else:
                shutil.copyfile(src_path, tar_path)
            print(f"Copied {filename}")

    print("\nMIM Initialization Complete! You can now run the training command.")

if __name__ == '__main__':
    manual_mim_init()
