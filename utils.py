"""
Utility Functions for RT-DETR Training and Inference
"""

import torch
import random
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Any


def set_seed(seed: int):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🎲 Random seed set to: {seed}")


def check_cuda():
    """Check CUDA availability and print GPU info"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        device_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        
        print(f"\n💻 CUDA Information:")
        print(f"   Available: Yes")
        print(f"   Devices: {device_count}")
        print(f"   Current device: {current_device}")
        print(f"   Device name: {device_name}")
        print(f"   Device memory: {device_memory:.2f} GB")
        
        return True
    else:
        print(f"\n💻 CUDA Information:")
        print(f"   Available: No (using CPU)")
        return False


def estimate_gpu_memory(
    batch_size: int,
    image_size: int = 640,
    num_classes: int = 80
) -> float:
    """
    Estimate GPU memory requirements (rough estimate)
    
    Args:
        batch_size: Training batch size
        image_size: Input image size
        num_classes: Number of classes
    
    Returns:
        Estimated memory in GB
    """
    # Rough estimation formula
    # Base model: ~2GB
    # Per image: ~0.5GB for 640x640
    
    base_memory = 2.0
    per_image_memory = (image_size / 640) ** 2 * 0.5
    total_memory = base_memory + (batch_size * per_image_memory)
    
    print(f"\n📊 Estimated GPU Memory:")
    print(f"   Batch size: {batch_size}")
    print(f"   Image size: {image_size}")
    print(f"   Estimated memory: ~{total_memory:.2f} GB")
    
    if total_memory > 8:
        print(f"   ⚠️  Warning: May exceed typical GPU memory (8GB)")
        print(f"   Consider reducing batch size or image size")
    
    return total_memory


def count_parameters(model) -> Dict[str, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    info = {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }
    
    print(f"\n🔢 Model Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    print(f"   Frozen: {frozen_params:,}")
    
    return info


def verify_dataset_paths(config) -> bool:
    """
    Verify that dataset paths exist
    
    Args:
        config: RTDETRConfig object
    
    Returns:
        bool: True if all paths exist
    """
    print(f"\n📁 Verifying dataset paths...")
    
    all_exist = True
    
    # Check training paths
    if not os.path.exists(config.dataset.train_images_dir):
        print(f"   ❌ Training images not found: {config.dataset.train_images_dir}")
        all_exist = False
    else:
        print(f"   ✅ Training images: {config.dataset.train_images_dir}")
    
    if not os.path.exists(config.dataset.train_annotations):
        print(f"   ❌ Training annotations not found: {config.dataset.train_annotations}")
        all_exist = False
    else:
        print(f"   ✅ Training annotations: {config.dataset.train_annotations}")
    
    # Check validation paths
    if not os.path.exists(config.dataset.val_images_dir):
        print(f"   ❌ Validation images not found: {config.dataset.val_images_dir}")
        all_exist = False
    else:
        print(f"   ✅ Validation images: {config.dataset.val_images_dir}")
    
    if not os.path.exists(config.dataset.val_annotations):
        print(f"   ❌ Validation annotations not found: {config.dataset.val_annotations}")
        all_exist = False
    else:
        print(f"   ✅ Validation annotations: {config.dataset.val_annotations}")
    
    return all_exist


def print_training_summary(
    train_dataset,
    val_dataset,
    config
):
    """
    Print training summary
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: RTDETRConfig object
    """
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    print(f"\n📊 Dataset:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Classes: {config.dataset.num_classes}")
    print(f"   Class names: {', '.join(config.dataset.class_names[:5])}{'...' if len(config.dataset.class_names) > 5 else ''}")
    
    print(f"\n🤖 Model:")
    print(f"   Architecture: {config.model.model_name}")
    print(f"   Pre-trained: {config.model.from_pretrained}")
    print(f"   Queries: {config.model.num_queries}")
    
    print(f"\n🎯 Training:")
    print(f"   Epochs: {config.training.num_epochs}")
    print(f"   Batch size: {config.training.batch_size}")
    print(f"   Learning rate: {config.training.learning_rate}")
    print(f"   Optimizer: AdamW")
    print(f"   Scheduler: {config.training.lr_scheduler_type}")
    print(f"   FP16: {config.training.fp16}")
    
    # Calculate training steps
    steps_per_epoch = len(train_dataset) // config.training.batch_size
    total_steps = steps_per_epoch * config.training.num_epochs
    
    print(f"\n⏱️  Training Steps:")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Total steps: {total_steps}")
    print(f"   Gradient accumulation: {config.training.gradient_accumulation_steps}")
    
    print(f"\n💾 Output:")
    print(f"   Output directory: {config.training.output_dir}")
    print(f"   Logging: {config.training.report_to}")
    
    print("="*70 + "\n")


def save_results_summary(
    results: List[Dict],
    output_file: str
):
    """
    Save inference results summary
    
    Args:
        results: List of prediction results
        output_file: Path to save summary
    """
    summary = {
        'total_images': len(results),
        'successful': sum(1 for r in results if r.get('success', False)),
        'failed': sum(1 for r in results if not r.get('success', False)),
        'results': []
    }
    
    total_detections = 0
    for r in results:
        if r.get('success', False):
            num_det = len(r['predictions']['boxes'])
            total_detections += num_det
            
            summary['results'].append({
                'image': str(r['image_path']),
                'num_detections': num_det,
                'classes': r['predictions']['class_names']
            })
        else:
            summary['results'].append({
                'image': str(r['image_path']),
                'error': r.get('error', 'Unknown error')
            })
    
    summary['total_detections'] = total_detections
    summary['avg_detections_per_image'] = total_detections / max(summary['successful'], 1)
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Results Summary:")
    print(f"   Total images: {summary['total_images']}")
    print(f"   Successful: {summary['successful']}")
    print(f"   Failed: {summary['failed']}")
    print(f"   Total detections: {summary['total_detections']}")
    print(f"   Avg detections/image: {summary['avg_detections_per_image']:.2f}")
    print(f"   Summary saved to: {output_file}")


def get_image_files(directory: str, recursive: bool = False) -> List[str]:
    """
    Get all image files from a directory
    
    Args:
        directory: Directory path
        recursive: Whether to search recursively
    
    Returns:
        List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    directory = Path(directory)
    
    if recursive:
        pattern = '**/*'
    else:
        pattern = '*'
    
    image_files = [
        str(f) for f in directory.glob(pattern)
        if f.suffix.lower() in image_extensions
    ]
    
    return sorted(image_files)


def create_output_directory(base_dir: str, experiment_name: str = None) -> str:
    """
    Create output directory with timestamp
    
    Args:
        base_dir: Base directory
        experiment_name: Optional experiment name
    
    Returns:
        Created directory path
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        dir_name = f"{experiment_name}_{timestamp}"
    else:
        dir_name = timestamp
    
    output_dir = os.path.join(base_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Created output directory: {output_dir}")
    
    return output_dir


def log_system_info():
    """Log system information"""
    import platform
    import sys
    
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    
    print(f"\n💻 System:")
    print(f"   Platform: {platform.system()} {platform.release()}")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   PyTorch: {torch.__version__}")
    
    try:
        import transformers
        print(f"   Transformers: {transformers.__version__}")
    except:
        print(f"   Transformers: Not installed")
    
    check_cuda()
    
    print("="*70 + "\n")


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f} (current: {self.val:.4f})"
