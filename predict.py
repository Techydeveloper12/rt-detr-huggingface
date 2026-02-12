"""
Main Inference Script for RT-DETR v2 with Hugging Face Transformers
Run predictions on images using trained model
"""

import argparse
import sys
import os
from pathlib import Path

from config import RTDETRConfig
from inference import load_predictor
from utils import get_image_files, save_results_summary, log_system_info


def parse_args():
    parser = argparse.ArgumentParser(
        description='RT-DETR v2 Object Detection Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on single image
  python predict.py --model outputs --image test.jpg

  # Predict on directory of images
  python predict.py --model outputs --image-dir test_images/

  # Predict with custom thresholds
  python predict.py --model outputs --image test.jpg --conf-thresh 0.7

  # Save only JSON predictions (no visualizations)
  python predict.py --model outputs --image test.jpg --no-vis
        """
    )
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint directory')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (defaults to model/config.json)')
    
    # Input arguments (one of these required)
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image')
    parser.add_argument('--image-dir', type=str, default=None,
                       help='Path to directory of images')
    parser.add_argument('--image-list', type=str, default=None,
                       help='Path to text file with list of image paths')
    
    # Inference parameters
    parser.add_argument('--conf-thresh', type=float, default=0.1,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                       help='IoU threshold for NMS (default: 0.5)')
    parser.add_argument('--max-det', type=int, default=100,
                       help='Maximum detections per image (default: 100)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='predictions',
                       help='Output directory for predictions')
    parser.add_argument('--no-vis', action='store_true',
                       help='Do not save visualizations')
    parser.add_argument('--no-json', action='store_true',
                       help='Do not save JSON predictions')
    parser.add_argument('--save-summary', action='store_true',
                       help='Save results summary JSON')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for inference')
    
    # Display options
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed predictions')
    
    return parser.parse_args()


def get_input_images(args) -> list:
    """Get list of input images from arguments"""
    image_paths = []
    
    # Single image
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            sys.exit(1)
        image_paths.append(args.image)
    
    # Directory of images
    if args.image_dir:
        if not os.path.exists(args.image_dir):
            print(f"❌ Directory not found: {args.image_dir}")
            sys.exit(1)
        
        found_images = get_image_files(args.image_dir, recursive=False)
        if not found_images:
            print(f"❌ No images found in: {args.image_dir}")
            sys.exit(1)
        
        image_paths.extend(found_images)
        print(f"📂 Found {len(found_images)} images in directory")
    
    # List file
    if args.image_list:
        if not os.path.exists(args.image_list):
            print(f"❌ Image list not found: {args.image_list}")
            sys.exit(1)
        
        with open(args.image_list, 'r') as f:
            for line in f:
                path = line.strip()
                if path and os.path.exists(path):
                    image_paths.append(path)
                elif path:
                    print(f"⚠️  Warning: Image not found: {path}")
    
    if not image_paths:
        print(f"❌ No input images specified!")
        print(f"   Use --image, --image-dir, or --image-list")
        sys.exit(1)
    
    return image_paths


def main():
    # Print header
    print("\n" + "="*70)
    print("RT-DETR V2 INFERENCE - HUGGING FACE TRANSFORMERS")
    print("="*70 + "\n")
    
    # Parse arguments
    args = parse_args()
    
    # Get input images
    image_paths = get_input_images(args)
    print(f"📸 Total images to process: {len(image_paths)}\n")
    
    # Load configuration
    if args.config:
        config_path = args.config
    else:
        # Try to find config in model directory
        config_path = os.path.join(args.model, 'config.json')
        if not os.path.exists(config_path):
            # Try parent directory
            config_path = os.path.join(os.path.dirname(args.model), 'config.json')
    
    if os.path.exists(config_path):
        print(f"📝 Loading config from: {config_path}")
        config = RTDETRConfig.load(config_path)
    else:
        print(f"⚠️  Config file not found, using default configuration")
        config = RTDETRConfig()
    
    # Update inference config
    config.inference.checkpoint_path = args.model
    config.inference.confidence_threshold = args.conf_thresh
    config.inference.iou_threshold = args.iou_thresh
    config.inference.max_detections = args.max_det
    config.inference.output_dir = args.output_dir
    config.inference.save_visualizations = not args.no_vis
    config.inference.save_predictions_json = not args.no_json
    config.device = args.device
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load predictor
    print(f"🔮 Loading predictor...")
    try:
        predictor = load_predictor(args.model, config)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print inference settings
    print(f"\n⚙️  Inference Settings:")
    print(f"   Confidence threshold: {config.inference.confidence_threshold}")
    print(f"   IoU threshold: {config.inference.iou_threshold}")
    print(f"   Max detections: {config.inference.max_detections}")
    print(f"   Save visualizations: {config.inference.save_visualizations}")
    print(f"   Save JSON: {config.inference.save_predictions_json}")
    print(f"   Output directory: {config.inference.output_dir}")
    
    # Run inference
    print(f"\n" + "="*70)
    print("RUNNING INFERENCE")
    print("="*70)
    
    all_results = []
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] {Path(image_path).name}")
        
        try:
            predictions = predictor.predict_and_save(image_path)
            
            # Print predictions if verbose
            if args.verbose:
                predictor.print_predictions(predictions)
            
            all_results.append({
                'image_path': image_path,
                'predictions': predictions,
                'success': True
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            all_results.append({
                'image_path': image_path,
                'error': str(e),
                'success': False
            })
    
    # Print summary
    print(f"\n" + "="*70)
    print("INFERENCE COMPLETED")
    print("="*70)
    
    successful = sum(1 for r in all_results if r.get('success', False))
    failed = len(all_results) - successful
    
    print(f"\n📊 Summary:")
    print(f"   Total images: {len(all_results)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    
    if successful > 0:
        total_detections = sum(
            len(r['predictions']['boxes']) 
            for r in all_results 
            if r.get('success', False)
        )
        avg_detections = total_detections / successful
        
        print(f"   Total detections: {total_detections}")
        print(f"   Avg detections/image: {avg_detections:.2f}")
    
    print(f"\n📁 Output saved to: {args.output_dir}")
    
    # Save results summary if requested
    if args.save_summary:
        summary_path = os.path.join(args.output_dir, 'results_summary.json')
        save_results_summary(all_results, summary_path)
    
    print("="*70 + "\n")
    
    # Exit with error code if any failed
    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
