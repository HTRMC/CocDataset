import json
import os
import shutil
from pathlib import Path
import random
from PIL import Image

def labelme_to_yolo(json_path, img_path, class_mapping):
    """Convert LabelMe format to YOLO format"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Get image dimensions
    img = Image.open(img_path)
    img_width, img_height = img.size

    yolo_annotations = []
    label_counts = {}

    for shape in data.get('shapes', []):
        label = shape['label']

        # Get class id from mapping (e.g., air_defense_12 -> class_id based on level)
        class_id = class_mapping.get(label, -1)

        if class_id == -1:
            print(f"Warning: Unknown label '{label}' in {json_path}")
            continue

        # Count labels
        label_counts[label] = label_counts.get(label, 0) + 1

        # Get bounding box points
        points = shape['points']
        x1, y1 = points[0]
        x2, y2 = points[1]

        # Convert to YOLO format (center_x, center_y, width, height) normalized
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = abs(x2 - x1) / img_width
        height = abs(y2 - y1) / img_height

        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return yolo_annotations, label_counts

def find_labeled_images(base_dir):
    """Find all images with corresponding JSON annotations"""
    labeled_pairs = []

    # Check TH11, TH12, TH13, th14 directories
    for th_dir in ['TH11', 'TH12', 'TH13', 'th14']:
        th_path = Path(base_dir) / th_dir / 'Labeled'

        if not th_path.exists():
            print(f"Skipping {th_dir} - Labeled folder not found")
            continue

        # Find all JSON files
        json_files = list(th_path.glob('*.json'))

        for json_file in json_files:
            # Find corresponding image (try .jpg first, then .png)
            img_name = json_file.stem
            img_jpg = th_path.parent / f"{img_name}.jpg"
            img_png = th_path.parent / f"{img_name}.png"

            if img_jpg.exists():
                labeled_pairs.append((str(img_jpg), str(json_file)))
            elif img_png.exists():
                labeled_pairs.append((str(img_png), str(json_file)))

    return labeled_pairs

def prepare_yolo_dataset(base_dir, output_dir, train_ratio=0.8):
    """Prepare YOLO dataset from LabelMe annotations"""

    # Class mapping for different air defense levels
    class_mapping = {
        'air_defense_9': 0,
        'air_defense_10': 1,
        'air_defense_11': 2,
        'air_defense_12': 3
    }

    # Find all labeled image-json pairs
    print("Finding labeled images...")
    labeled_pairs = find_labeled_images(base_dir)
    print(f"Found {len(labeled_pairs)} labeled images")

    if len(labeled_pairs) == 0:
        print("No labeled images found!")
        return

    # Shuffle and split
    random.seed(42)
    random.shuffle(labeled_pairs)

    split_idx = int(len(labeled_pairs) * train_ratio)
    train_pairs = labeled_pairs[:split_idx]
    val_pairs = labeled_pairs[split_idx:]

    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Statistics for class distribution
    class_counts = {name: 0 for name in class_mapping.keys()}

    # Process training set
    print("\nProcessing training set...")
    for img_path, json_path in train_pairs:
        # Convert annotations
        yolo_annotations, label_counts = labelme_to_yolo(json_path, img_path, class_mapping)

        # Update class counts
        for label, count in label_counts.items():
            class_counts[label] = class_counts.get(label, 0) + count

        # Copy image
        img_name = Path(img_path).name
        dest_img = Path(output_dir) / 'images' / 'train' / img_name
        shutil.copy(img_path, dest_img)

        # Save YOLO format label
        label_name = Path(img_path).stem + '.txt'
        dest_label = Path(output_dir) / 'labels' / 'train' / label_name
        with open(dest_label, 'w') as f:
            f.write('\n'.join(yolo_annotations))

    # Process validation set
    print("Processing validation set...")
    for img_path, json_path in val_pairs:
        # Convert annotations
        yolo_annotations, label_counts = labelme_to_yolo(json_path, img_path, class_mapping)

        # Update class counts
        for label, count in label_counts.items():
            class_counts[label] = class_counts.get(label, 0) + count

        # Copy image
        img_name = Path(img_path).name
        dest_img = Path(output_dir) / 'images' / 'val' / img_name
        shutil.copy(img_path, dest_img)

        # Save YOLO format label
        label_name = Path(img_path).stem + '.txt'
        dest_label = Path(output_dir) / 'labels' / 'val' / label_name
        with open(dest_label, 'w') as f:
            f.write('\n'.join(yolo_annotations))

    print(f"\nDataset prepared successfully!")
    print(f"Training images: {len(train_pairs)}")
    print(f"Validation images: {len(val_pairs)}")
    print(f"\nClass distribution:")
    for label in sorted(class_counts.keys()):
        print(f"  {label}: {class_counts[label]} instances")

if __name__ == '__main__':
    base_dir = r'C:\Users\rafma\OneDrive\Desktop\Python\CocDataset'
    output_dir = r'C:\Users\rafma\OneDrive\Desktop\Python\CocDataset\yolo_dataset'

    # Using 95/5 split to maximize training data (314 train, 17 val)
    prepare_yolo_dataset(base_dir, output_dir, train_ratio=0.95)
