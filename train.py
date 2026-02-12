"""
Main Training Script for RT-DETR v2 with Hugging Face Transformers
Run this script to train object detection model on your custom dataset
"""

import argparse
import sys
import os

from config import RTDETRConfig, create_custom_config, print_config_summary
from dataset import create_dataloaders, verify_coco_format
from trainer import load_model_and_processor, train_model, evaluate_model
from utils import (
    set_seed,
    check_cuda,
    verify_dataset_paths,
    print_training_summary,
    log_system_info
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train RT-DETR v2 Object Detection with Hugging Face Transformers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with custom dataset
  python train.py \\
    --train-images data/train/images \\
    --train-ann data/train/annotations.json \\
    --val-images data/val/images \\
    --val-ann data/val/annotations.json \\
    --num-classes 3 \\
    --class-names car truck bus \\
    --epochs 50 \\
    --batch-size 8

  # Resume training
  python train.py --config outputs/config.json --resume outputs/checkpoint-1000
        """
    )
    
    # Configuration file
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration JSON file')
    
    # Dataset arguments (REQUIRED if not using config file)
    parser.add_argument('--train-images', type=str,
                       help='Path to training images directory')
    parser.add_argument('--train-ann', type=str,
                       help='Path to training annotations (COCO JSON)')
    parser.add_argument('--val-images', type=str,
                       help='Path to validation images directory')
    parser.add_argument('--val-ann', type=str,
                       help='Path to validation annotations (COCO JSON)')
    
    # Class configuration (REQUIRED)
    parser.add_argument('--num-classes', type=int,
                       help='Number of object classes')
    parser.add_argument('--class-names', type=str, nargs='+',
                       help='List of class names (e.g., --class-names car truck bus)')
    
    # Model arguments
    parser.add_argument('--model-name', type=str, default='PekingU/rtdetr_v2_r50vd',
                       help='Hugging Face model name (default: PekingU/rtdetr_v2_r50vd)')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Do not use pre-trained weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=None,
                       help='Weight decay')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for checkpoints')
    
    # Advanced training options
    parser.add_argument('--fp16', action='store_true',
                       help='Use mixed precision training (FP16)')
    parser.add_argument('--gradient-accumulation', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint path')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run validation, no training')
    parser.add_argument('--verify-dataset', action='store_true',
                       help='Verify COCO format before training')
    
    return parser.parse_args()


def main():
    # Print header
    print("\n" + "="*70)
    print("RT-DETR V2 TRAINING - HUGGING FACE TRANSFORMERS")
    print("="*70 + "\n")
    
    # Parse arguments
    args = parse_args()
    
    # Log system information
    log_system_info()
    
    # Load or create configuration
    if args.config:
        print(f"📝 Loading configuration from: {args.config}")
        config = RTDETRConfig.load(args.config)
    else:
        # Check required arguments
        required_args = ['train_images', 'train_ann', 'val_images', 'val_ann', 
                        'num_classes', 'class_names']
        missing_args = [arg for arg in required_args if getattr(args, arg) is None]
        
        if missing_args:
            print(f"❌ Error: Missing required arguments: {', '.join(missing_args)}")
            print(f"\nPlease provide either:")
            print(f"  1. --config <path_to_config.json>")
            print(f"  2. All required dataset arguments")
            print(f"\nUse --help for more information")
            sys.exit(1)
        
        # Create configuration
        print(f"📝 Creating configuration from command line arguments")
        config = create_custom_config(
            num_classes=args.num_classes,
            class_names=args.class_names,
            train_images_dir=args.train_images,
            train_annotations=args.train_ann,
            val_images_dir=args.val_images,
            val_annotations=args.val_ann
        )
    
    # Update config with command line arguments
    if args.model_name:
        config.model.model_name = args.model_name
    if args.no_pretrained:
        config.model.from_pretrained = False
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.weight_decay:
        config.training.weight_decay = args.weight_decay
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.fp16:
        config.training.fp16 = True
    if args.gradient_accumulation:
        config.training.gradient_accumulation_steps = args.gradient_accumulation
    if args.num_workers:
        config.training.num_workers = args.num_workers
    if args.resume:
        config.training.resume_from_checkpoint = args.resume
    if args.seed:
        config.seed = args.seed
    if args.device:
        config.device = args.device
    
    # Print configuration
    print_config_summary(config)
    
    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
    
    # Verify dataset paths
    if not verify_dataset_paths(config):
        print(f"\n❌ Some dataset paths are missing!")
        print(f"Please check your dataset configuration.")
        sys.exit(1)
    
    # Verify COCO format if requested
    if args.verify_dataset:
        print(f"\n🔍 Verifying COCO format...")
        if not verify_coco_format(config.dataset.train_annotations):
            print(f"❌ Training annotations verification failed!")
            sys.exit(1)
        if not verify_coco_format(config.dataset.val_annotations):
            print(f"❌ Validation annotations verification failed!")
            sys.exit(1)
    
    # Set random seed
    set_seed(config.seed)
    
    # Check CUDA
    check_cuda()
    
    # Save configuration
    config_save_path = os.path.join(config.training.output_dir, 'config.json')
    os.makedirs(config.training.output_dir, exist_ok=True)
    config.save(config_save_path)
    
    # Load model and processor
    model, image_processor = load_model_and_processor(config)
    
    # Create datasets
    print(f"\n📂 Loading datasets...")
    train_dataset, val_dataset = create_dataloaders(config, image_processor)
    
    # Print training summary
    print_training_summary(train_dataset, val_dataset, config)
    
    # Validate-only mode
    if args.validate_only:
        print(f"\n🔍 Running validation only (no training)...")
        from trainer import RTDETRTrainer, create_training_arguments
        from dataset import collate_fn
        
        training_args = create_training_arguments(config)
        trainer = RTDETRTrainer(
            model=model,
            args=training_args,
            eval_dataset=val_dataset,
            data_collator=collate_fn,
        )
        
        metrics = evaluate_model(trainer)
        print(f"\n✅ Validation completed")
        sys.exit(0)
    
    # Train model
    try:
        trainer = train_model(model, train_dataset, val_dataset, config, image_processor)
        
        # Final evaluation
        print(f"\n📊 Running final evaluation...")
        final_metrics = evaluate_model(trainer)
        
        print(f"\n" + "="*70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\n📁 Model saved to: {config.training.output_dir}")
        print(f"📊 Final eval loss: {final_metrics.get('eval_loss', 'N/A')}")
        print(f"\n💡 To run inference:")
        print(f"   python predict.py --model {config.training.output_dir} --image <image_path>")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
