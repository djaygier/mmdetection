import os
import json
import cv2
import yaml
import argparse
from tqdm import tqdm

def yolo_to_coco(data_dir, split, output_json, class_names):
    images = []
    annotations = []
    categories = [{"id": i, "name": name} for i, name in enumerate(class_names)]
    
    img_dir = os.path.join(data_dir, split, 'images')
    lbl_dir = os.path.join(data_dir, split, 'labels')
    
    if not os.path.exists(img_dir):
        print(f"Directory {img_dir} does not exist. Skipping.")
        return

    ann_id = 0
    img_id = 0
    
    for img_name in tqdm(os.listdir(img_dir), desc=f"Converting {split}"):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
            continue
            
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w, _ = img.shape
        
        images.append({
            "id": img_id,
            "file_name": os.path.join(split, 'images', img_name),
            "width": w,
            "height": h
        })
        
        lbl_name = os.path.splitext(img_name)[0] + '.txt'
        lbl_path = os.path.join(lbl_dir, lbl_name)
        
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    cls_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # YOLO to COCO (absolute pixels)
                    abs_x = (x_center - width / 2) * w
                    abs_y = (y_center - height / 2) * h
                    abs_w = width * w
                    abs_h = height * h
                    
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls_id,
                        "bbox": [abs_x, abs_y, abs_w, abs_h],
                        "area": abs_w * abs_h,
                        "segmentation": [],
                        "iscrowd": 0
                    })
                    ann_id += 1
        
        img_id += 1
        
    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    with open(output_json, 'w') as f:
        json.dump(coco_data, f)
    print(f"Saved {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert YOLO dataset to COCO format')
    parser.add_argument('--data_dir', default='./dataset', help='Path to dataset root')
    parser.add_argument('--out_dir', default='./dataset', help='Output directory for JSON files')
    args = parser.parse_args()
    
    with open(os.path.join(args.data_dir, 'data.yaml'), 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    class_names = data_cfg['names']
    os.makedirs(args.out_dir, exist_ok=True)
    
    for split in ['train', 'valid', 'test']:
        yolo_to_coco(args.data_dir, split, os.path.join(args.out_dir, f'{split}.json'), class_names)
