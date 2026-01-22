import os
import os.path as osp
import shutil
import platform
import mmdet
import sys

def manual_mim_init():
    """Aggressive MIM initialization.
    Targets both the active mmdet path and the site-packages path to ensure
    no 'FileNotFoundError' can occur.
    """
    filenames = [
        'tools', 'configs', 'demo', 'model-index.yml', 'dataset-index.yml'
    ]
    
    # 1. Identify all possible target locations
    targets = []
    
    # Active import location
    mmdet_path = osp.dirname(mmdet.__file__)
    targets.append(osp.join(mmdet_path, '.mim'))
    
    # Potential site-packages location from the error message
    site_pkg_path = "/venv/main/lib/python3.12/site-packages/mmdet/.mim"
    if site_pkg_path not in targets:
        targets.append(site_pkg_path)

    # 2. Find the current repository root (where tools/configs are)
    repo_path = os.getcwd()
    
    print(f"Repository Root: {repo_path}")
    print(f"Active Package Path: {mmdet_path}")

    for mim_path in targets:
        print(f"\n--- Initializing MIM at: {mim_path} ---")
        
        # Ensure parent exists
        parent_dir = osp.dirname(mim_path)
        if not osp.exists(parent_dir):
            print(f"Warning: Package directory {parent_dir} does not exist. Skipping.")
            continue
            
        os.makedirs(mim_path, exist_ok=True)

        for filename in filenames:
            src_path = osp.join(repo_path, filename)
            tar_path = osp.join(mim_path, filename)
            
            if not osp.exists(src_path):
                print(f"  Warning: {filename} not found in {repo_path}. Skipping.")
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
                    print(f"  Linked {filename} -> {tar_path}")
                except OSError:
                    if osp.isdir(src_path):
                        shutil.copytree(src_path, tar_path)
                    else:
                        shutil.copyfile(src_path, tar_path)
                    print(f"  Copied {filename} (symlink failed)")
            else:
                if osp.isdir(src_path):
                    shutil.copytree(src_path, tar_path)
                else:
                    shutil.copyfile(src_path, tar_path)
                print(f"  Copied {filename}")

    print("\nInitialization Complete! Please run the training command again.")

if __name__ == '__main__':
    manual_mim_init()
