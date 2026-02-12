"""
RT-DETR v2 Object Detection with Hugging Face Transformers
Main entry point and usage guide

This is a professional implementation of RT-DETR for object detection
using the Hugging Face Transformers library.
"""

import sys
import os


def print_header():
    """Print welcome header"""
    print("\n" + "="*80)
    print("RT-DETR V2 - OBJECT DETECTION WITH HUGGING FACE TRANSFORMERS")
    print("="*80)
    print("\nProfessional object detection training and inference system")
    print("Using official Hugging Face transformers library\n")


def print_quick_start():
    """Print quick start guide"""
    print("="*80)
    print("QUICK START GUIDE")
    print("="*80)
    
    print("\n1️⃣  INSTALLATION")
    print("-" * 80)
    print("   pip install -r requirements.txt")
    
    print("\n2️⃣  PREPARE DATASET (COCO Format)")
    print("-" * 80)
    print("""   dataset/
   ├── train/
   │   ├── images/
   │   │   ├── img1.jpg
   │   │   └── ...
   │   └── annotations.json  (COCO format)
   └── val/
       ├── images/
       └── annotations.json
    """)
    
    print("\n3️⃣  TRAIN MODEL")
    print("-" * 80)
    print("""   python train.py \\
     --train-images data/train/images \\
     --train-ann data/train/annotations.json \\
     --val-images data/val/images \\
     --val-ann data/val/annotations.json \\
     --num-classes 3 \\
     --class-names car truck bus \\
     --epochs 50 \\
     --batch-size 8
    """)
    
    print("\n4️⃣  RUN INFERENCE")
    print("-" * 80)
    print("""   python predict.py \\
     --model outputs \\
     --image test.jpg \\
     --conf-thresh 0.5
    """)


def print_commands():
    """Print available commands"""
    print("\n" + "="*80)
    print("AVAILABLE COMMANDS")
    print("="*80)
    
    commands = [
        ("Training", "python train.py --help", "Train RT-DETR model on custom dataset"),
        ("Inference", "python predict.py --help", "Run predictions on images"),
        ("Verify Dataset", "python -c \"from dataset import verify_coco_format; verify_coco_format('annotations.json')\"", "Validate COCO format"),
        ("Create Template", "python -c \"from dataset import create_coco_template; create_coco_template('images/', 'out.json', ['car'])\"", "Generate annotation template"),
        ("Check GPU", "python -c \"from utils import check_cuda; check_cuda()\"", "Check CUDA availability"),
    ]
    
    for name, cmd, desc in commands:
        print(f"\n📌 {name}")
        print(f"   {desc}")
        print(f"   {cmd}")


def print_important_params():
    """Print important configuration parameters"""
    print("\n" + "="*80)
    print("IMPORTANT PARAMETERS")
    print("="*80)
    
    print("\n🔴 ALWAYS CONFIGURE (Required for each dataset):")
    print("-" * 80)
    params = [
        ("--num-classes", "Number of object classes in your dataset"),
        ("--class-names", "List of class names (e.g., car truck bus)"),
        ("--train-images", "Path to training images directory"),
        ("--train-ann", "Path to training COCO JSON annotations"),
        ("--val-images", "Path to validation images directory"),
        ("--val-ann", "Path to validation COCO JSON annotations"),
    ]
    
    for param, desc in params:
        print(f"   {param:20s} : {desc}")
    
    print("\n🟡 FREQUENTLY CONFIGURE (Tune for performance):")
    print("-" * 80)
    params = [
        ("--epochs", "Number of training epochs (default: 50)"),
        ("--batch-size", "Training batch size (default: 8)"),
        ("--lr", "Learning rate (default: 1e-5)"),
        ("--model-name", "HuggingFace model (default: PekingU/rtdetr_v2_r50vd)"),
        ("--fp16", "Enable mixed precision training (faster)"),
    ]
    
    for param, desc in params:
        print(f"   {param:20s} : {desc}")
    
    print("\n🟢 OPTIONAL (Usually keep defaults):")
    print("-" * 80)
    params = [
        ("--output-dir", "Output directory (default: outputs)"),
        ("--num-workers", "Data loading workers (default: 4)"),
        ("--device", "Device: cuda or cpu (default: cuda)"),
    ]
    
    for param, desc in params:
        print(f"   {param:20s} : {desc}")


def print_examples():
    """Print usage examples"""
    print("\n" + "="*80)
    print("USAGE EXAMPLES")
    print("="*80)
    
    print("\n📋 Example 1: Train Vehicle Detector")
    print("-" * 80)
    print("""python train.py \\
  --train-images datasets/vehicles/train/images \\
  --train-ann datasets/vehicles/train/annotations.json \\
  --val-images datasets/vehicles/val/images \\
  --val-ann datasets/vehicles/val/annotations.json \\
  --num-classes 3 \\
  --class-names car truck bus \\
  --epochs 50 \\
  --batch-size 8 \\
  --output-dir vehicle_detector
    """)
    
    print("\n📋 Example 2: Train with FP16 and Gradient Accumulation")
    print("-" * 80)
    print("""python train.py \\
  --train-images data/train/images \\
  --train-ann data/train/annotations.json \\
  --val-images data/val/images \\
  --val-ann data/val/annotations.json \\
  --num-classes 5 \\
  --class-names person car dog cat bird \\
  --epochs 100 \\
  --batch-size 4 \\
  --gradient-accumulation 4 \\
  --fp16 \\
  --lr 2e-5
    """)
    
    print("\n📋 Example 3: Batch Inference")
    print("-" * 80)
    print("""python predict.py \\
  --model outputs \\
  --image-dir test_images/ \\
  --conf-thresh 0.6 \\
  --output-dir predictions \\
  --save-summary
    """)


def print_file_structure():
    """Print project file structure"""
    print("\n" + "="*80)
    print("PROJECT FILES")
    print("="*80)
    
    files = [
        ("config.py", "Configuration module - Easy parameter setup"),
        ("dataset.py", "COCO dataset loading and validation"),
        ("trainer.py", "Training with Hugging Face Trainer API"),
        ("inference.py", "Prediction and visualization"),
        ("utils.py", "Utility functions"),
        ("train.py", "Main training script ⭐"),
        ("predict.py", "Main inference script ⭐"),
        ("requirements.txt", "Python dependencies"),
        ("README.md", "Comprehensive documentation"),
    ]
    
    for filename, description in files:
        print(f"   {filename:20s} - {description}")


def main():
    """Main function"""
    print_header()
    print_quick_start()
    print_commands()
    print_important_params()
    print_examples()
    print_file_structure()
    
    print("\n" + "="*80)
    print("📚 DOCUMENTATION")
    print("="*80)
    print("\n   For complete documentation, see README.md")
    print("   For help with commands:")
    print("      python train.py --help")
    print("      python predict.py --help")
    
    print("\n" + "="*80)
    print("🎯 KEY FEATURES")
    print("="*80)
    print("""
   ✅ Uses official Hugging Face Transformers
   ✅ Pre-trained RT-DETR models from PekingU
   ✅ COCO format dataset support
   ✅ Easy configuration per dataset
   ✅ Professional training pipeline
   ✅ Mixed precision (FP16) training
   ✅ Inference with visualization
   ✅ Production-ready code
    """)
    
    print("="*80)
    print("\nReady to start? Run: python train.py --help")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
