import json
import os

def fix_paths(json_file):
    if not os.path.exists(json_file):
        print(f"File {json_file} not found. Skipping.")
        return
        
    print(f"Fixing paths in {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    for img in data.get('images', []):
        img['file_name'] = img['file_name'].replace('\\', '/')
        
    with open(json_file, 'w') as f:
        json.dump(data, f)
    print(f"Done fixing {json_file}")

if __name__ == "__main__":
    for split in ['train', 'valid', 'test']:
        fix_paths(f'dataset/{split}.json')
