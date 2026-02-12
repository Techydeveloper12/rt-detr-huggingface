"""
Dataset Module for RT-DETR with Hugging Face Transformers
Supports COCO format annotations for object detection
"""

import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from typing import Dict, List, Any
import numpy as np
from transformers import RTDetrImageProcessor


class CocoDetectionDataset(Dataset):
    """
    COCO Format Dataset for Object Detection with RT-DETR
    
    Expected COCO JSON format:
    {
        "images": [
            {
                "id": 1,
                "file_name": "image1.jpg",
                "width": 640,
                "height": 480
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [x, y, width, height],  # COCO format
                "area": 12345,
                "iscrowd": 0
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "class_name"
            }
        ]
    }
    """
    
    def __init__(
        self,
        images_dir: str,
        annotations_file: str,
        image_processor: RTDetrImageProcessor,
        train: bool = True
    ):
        """
        Args:
            images_dir: Directory containing images
            annotations_file: Path to COCO format JSON annotations
            image_processor: Hugging Face RTDetrImageProcessor
            train: Whether this is training set (for augmentation)
        """
        self.images_dir = images_dir
        self.image_processor = image_processor
        self.train = train
        
        # Load COCO annotations
        print(f"Loading annotations from: {annotations_file}")
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build image and category mappings
        self.images = {img['id']: img for img in coco_data['images']}
        self.categories = {cat['id']: cat for cat in coco_data['categories']}
        
        # Group annotations by image
        self.img_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # Filter to images with annotations
        self.image_ids = [img_id for img_id in self.img_to_anns.keys() 
                         if len(self.img_to_anns[img_id]) > 0]
        
        print(f"Loaded {len(self.image_ids)} images with annotations")
        print(f"Total categories: {len(self.categories)}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get image and annotations
        
        Returns:
            Dict with 'pixel_values', 'labels' (containing 'class_labels' and 'boxes')
        """
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations
        annotations = self.img_to_anns[img_id]
        
        # Prepare annotations in COCO format for the image processor
        # The HuggingFace image processor expects a dict with 'image_id' and 'annotations'
        coco_annotations = []
        for ann in annotations:
            coco_ann = {
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],  # COCO format: [x, y, width, height]
                'area': ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
                'iscrowd': ann.get('iscrowd', 0)
            }
            coco_annotations.append(coco_ann)
        
        # Create the annotation structure expected by HuggingFace
        annotation_dict = {
            'image_id': img_id,
            'annotations': coco_annotations
        }
        
        # Apply image processor (handles resizing, normalization, and format conversion)
        # The processor will convert COCO bbox format to the format expected by RT-DETR
        encoding = self.image_processor(
            images=image,
            annotations=annotation_dict,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        pixel_values = encoding['pixel_values'].squeeze(0)
        target = encoding['labels'][0]
        
        return {
            'pixel_values': pixel_values,
            'labels': target
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function for batching
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        Batched data
    """
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = [item['labels'] for item in batch]
    
    return {
        'pixel_values': pixel_values,
        'labels': labels
    }


def create_dataloaders(config, image_processor):
    """
    Create training and validation dataloaders
    
    Args:
        config: RTDETRConfig object
        image_processor: RTDetrImageProcessor instance
    
    Returns:
        train_dataset, val_dataset
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = CocoDetectionDataset(
        images_dir=config.dataset.train_images_dir,
        annotations_file=config.dataset.train_annotations,
        image_processor=image_processor,
        train=True
    )
    
    val_dataset = CocoDetectionDataset(
        images_dir=config.dataset.val_images_dir,
        annotations_file=config.dataset.val_annotations,
        image_processor=image_processor,
        train=False
    )
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def verify_coco_format(annotations_file: str) -> bool:
    """
    Verify COCO format annotations
    
    Args:
        annotations_file: Path to COCO JSON file
    
    Returns:
        bool: True if valid
    """
    try:
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        # Check required fields
        required_keys = ['images', 'annotations', 'categories']
        for key in required_keys:
            if key not in data:
                print(f"❌ Missing required field: '{key}'")
                return False
        
        # Validate structure
        if not isinstance(data['images'], list):
            print("❌ 'images' must be a list")
            return False
        
        if not isinstance(data['annotations'], list):
            print("❌ 'annotations' must be a list")
            return False
        
        if not isinstance(data['categories'], list):
            print("❌ 'categories' must be a list")
            return False
        
        # Check image fields
        for img in data['images']:
            required_img_keys = ['id', 'file_name', 'width', 'height']
            if not all(k in img for k in required_img_keys):
                print(f"❌ Image missing required fields: {required_img_keys}")
                return False
        
        # Check annotation fields
        for ann in data['annotations']:
            required_ann_keys = ['id', 'image_id', 'category_id', 'bbox']
            if not all(k in ann for k in required_ann_keys):
                print(f"❌ Annotation missing required fields: {required_ann_keys}")
                return False
            
            # Validate bbox format
            if len(ann['bbox']) != 4:
                print(f"❌ Invalid bbox format (must be [x, y, w, h]): {ann['bbox']}")
                return False
        
        # Check category fields
        for cat in data['categories']:
            if 'id' not in cat or 'name' not in cat:
                print(f"❌ Category missing 'id' or 'name'")
                return False
        
        print(f"✅ COCO format validation passed!")
        print(f"   Images: {len(data['images'])}")
        print(f"   Annotations: {len(data['annotations'])}")
        print(f"   Categories: {len(data['categories'])}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON format: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def create_coco_template(
    images_dir: str,
    output_file: str,
    class_names: List[str]
):
    """
    Create a COCO format template for annotation
    
    Args:
        images_dir: Directory containing images
        output_file: Output JSON file path
        class_names: List of class names
    """
    from pathlib import Path
    
    images_dir = Path(images_dir)
    
    # Supported image extensions
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in images_dir.iterdir() 
                   if f.suffix.lower() in image_exts]
    
    # Create COCO structure
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add images
    for idx, img_path in enumerate(sorted(image_files), start=1):
        try:
            img = Image.open(img_path)
            width, height = img.size
            
            coco_format["images"].append({
                "id": idx,
                "file_name": img_path.name,
                "width": width,
                "height": height
            })
        except Exception as e:
            print(f"Warning: Could not process {img_path}: {e}")
    
    # Add categories (starting from ID 1 as per COCO convention)
    for idx, class_name in enumerate(class_names, start=1):
        coco_format["categories"].append({
            "id": idx,
            "name": class_name
        })
    
    # Save template
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=2)
    
    print(f"✅ Created COCO template with {len(image_files)} images")
    print(f"   Saved to: {output_file}")
    print(f"   Categories: {len(class_names)}")
    print(f"\n⚠️  You need to add annotations manually or use a labeling tool like:")
    print(f"   - CVAT (https://www.cvat.ai/)")
    print(f"   - LabelImg")
    print(f"   - VGG Image Annotator (VIA)")
