import os
import os.path as osp
import shutil
import platform

def manual_mim_init():
    """Manually create the .mim directory and link/copy necessary folders.
    This bypasses the need for 'python setup.py develop' which often fails
    due to build isolation or Python 3.12 compatibility issues.
    """
    filenames = [
        'tools', 'configs', 'demo', 'model-index.yml', 'dataset-index.yml'
    ]
    # Root of the repository
    repo_path = os.getcwd()
    # The .mim directory inside the package
    mim_path = osp.join(repo_path, 'mmdet', '.mim')
    
    print(f"Initializing MIM at: {mim_path}")
    os.makedirs(mim_path, exist_ok=True)

    for filename in filenames:
        src_path = osp.join(repo_path, filename)
        tar_path = osp.join(mim_path, filename)
        
        if not osp.exists(src_path):
            print(f"Warning: {filename} not found in repository root. Skipping.")
            continue

        # Remove existing target if it exists
        if osp.isfile(tar_path) or osp.islink(tar_path):
            os.remove(tar_path)
        elif osp.isdir(tar_path):
            shutil.rmtree(tar_path)

        # On Linux/macOS, we use symlinks for better development experience
        # On Windows, we copy if symlink creation fails
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
            # Windows copy mode
            if osp.isdir(src_path):
                shutil.copytree(src_path, tar_path)
            else:
                shutil.copyfile(src_path, tar_path)
            print(f"Copied {filename}")

    print("\nMIM Initialization Complete!")

if __name__ == '__main__':
    manual_mim_init()
