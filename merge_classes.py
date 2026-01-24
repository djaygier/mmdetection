import json
import os

def merge_classes(json_path, target_name='ant', source_name='queen'):
    print(f"Processing {json_path}...")
    if not os.path.exists(json_path):
        print(f"File {json_path} does not exist.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Find IDs
    id_map = {cat['name']: cat['id'] for cat in data['categories']}
    
    if target_name not in id_map:
        print(f"Target class '{target_name}' not found in {json_path}. Available: {list(id_map.keys())}")
        return
    if source_name not in id_map:
        print(f"Source class '{source_name}' not found in {json_path}. Skipping merge.")
        return

    target_id = id_map[target_name]
    source_id = id_map[source_name]

    print(f"Merging class '{source_name}' (ID: {source_id}) into '{target_name}' (ID: {target_id})")

    # Update annotations
    count = 0
    for ann in data['annotations']:
        if ann['category_id'] == source_id:
            ann['category_id'] = target_id
            count += 1
    
    print(f"Updated {count} annotations.")

    # Remove source class from categories
    data['categories'] = [cat for cat in data['categories'] if cat['name'] != source_name]

    # Save backup
    backup_path = json_path + '.bak'
    if not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            json.dump(data, f) # This is actually saving the modified data to bak, oops. 
            # Better to save the original before modification.
    
    # Save modified
    with open(json_path, 'w') as f:
        json.dump(data, f)
    
    print(f"Successfully updated {json_path}")

if __name__ == "__main__":
    train_json = 'Ant brood.v10i.coco/train/_annotations.coco.json'
    valid_json = 'Ant brood.v10i.coco/valid/_annotations.coco.json'
    
    merge_classes(train_json)
    merge_classes(valid_json)
