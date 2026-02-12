"""
Inference Module for RT-DETR with Hugging Face Transformers
Handles prediction, visualization, and result saving
"""

import torch
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Union
from pathlib import Path


class RTDETRPredictor:
    """
    RT-DETR Predictor using Hugging Face model
    """
    
    def __init__(
        self,
        model_path: str,
        config,
        device: str = None
    ):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model checkpoint
            config: RTDETRConfig object
            device: Device to run on ('cuda' or 'cpu')
        """
        self.config = config
        self.device = device or config.device
        
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA not available, using CPU")
            self.device = 'cpu'
        
        print(f"\n🔮 Loading model from: {model_path}")
        
        # Load RT-DETR v2 model
        self.model = RTDetrV2ForObjectDetection.from_pretrained(model_path)
        
        # Try to load image processor from checkpoint, fallback to original model
        try:
            self.image_processor = RTDetrImageProcessor.from_pretrained(model_path)
        except (OSError, ValueError) as e:
            print(f"   ℹ️  Image processor not found in checkpoint, loading from original model...")
            # Load from the original model name stored in config
            original_model = self.model.config._name_or_path
            if original_model and original_model.startswith("PekingU/"):
                self.image_processor = RTDetrImageProcessor.from_pretrained(original_model)
            else:
                # Default to standard RT-DETR v2 processor
                self.image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r50vd")
            print(f"   ✅ Loaded image processor from: {original_model or 'PekingU/rtdetr_v2_r50vd'}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Get class names
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        self.num_classes = self.model.config.num_labels
        
        print(f"✅ Model loaded successfully")
        print(f"   Device: {self.device}")
        print(f"   Classes: {self.num_classes}")
        
        # Generate colors for visualization
        np.random.seed(42)
        self.colors = [
            tuple(np.random.randint(0, 255, 3).tolist())
            for _ in range(self.num_classes)
        ]
        
        # Create output directory
        if config.inference.save_visualizations or config.inference.save_predictions_json:
            os.makedirs(config.inference.output_dir, exist_ok=True)
    
    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        confidence_threshold: float = None,
        return_tensors: bool = False
    ) -> Dict:
        """
        Predict objects in an image
        
        Args:
            image_path: Path to input image
            confidence_threshold: Confidence threshold (uses config if None)
            return_tensors: Whether to return tensors or numpy arrays
        
        Returns:
            Dictionary with predictions:
                - boxes: Bounding boxes
                - scores: Confidence scores
                - labels: Class labels
                - class_names: Class names
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Preprocess
        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        outputs = self.model(**inputs)
        
        # Get target image size for post-processing
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        
        # Post-process outputs
        results = self.image_processor.post_process_object_detection(
            outputs,
            threshold=confidence_threshold or self.config.inference.confidence_threshold,
            target_sizes=target_sizes
        )[0]
        
        # Convert to numpy if requested
        if not return_tensors:
            boxes = results['boxes'].cpu().numpy()
            scores = results['scores'].cpu().numpy()
            labels = results['labels'].cpu().numpy()
        else:
            boxes = results['boxes']
            scores = results['scores']
            labels = results['labels']
        
        # Limit to max detections
        max_det = self.config.inference.max_detections
        if len(boxes) > max_det:
            top_indices = np.argsort(scores)[-max_det:]
            boxes = boxes[top_indices]
            scores = scores[top_indices]
            labels = labels[top_indices]
        
        # Get class names
        if not return_tensors:
            class_names = [self.id2label.get(int(label), f"class_{int(label)}") 
                          for label in labels]
        else:
            class_names = [self.id2label.get(label.item(), f"class_{label.item()}") 
                          for label in labels]
        
        predictions = {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'class_names': class_names
        }
        
        return predictions
    
    def visualize(
        self,
        image_path: str,
        predictions: Dict,
        save_path: str = None,
        show_confidence: bool = True
    ) -> Image.Image:
        """
        Visualize predictions on image
        
        Args:
            image_path: Path to input image
            predictions: Predictions from predict()
            save_path: Path to save visualization (optional)
            show_confidence: Whether to show confidence scores
        
        Returns:
            PIL Image with visualizations
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except:
                font = ImageFont.load_default()
        
        # Draw each detection
        boxes = predictions['boxes']
        scores = predictions['scores']
        labels = predictions['labels']
        class_names = predictions['class_names']
        
        for box, score, label, class_name in zip(boxes, scores, labels, class_names):
            # Get coordinates
            x1, y1, x2, y2 = box if isinstance(box, (list, tuple)) else box.tolist()
            
            # Get color for this class
            label_idx = int(label) if not isinstance(label, int) else label
            color = self.colors[label_idx % len(self.colors)]
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Prepare label text
            if show_confidence:
                score_val = float(score) if not isinstance(score, float) else score
                label_text = f"{class_name}: {score_val:.2f}"
            else:
                label_text = class_name
            
            # Get text bounding box
            bbox = draw.textbbox((x1, y1), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw background for text
            draw.rectangle(
                [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
                fill=color
            )
            
            # Draw text
            draw.text((x1 + 2, y1 - text_height - 2), label_text, fill='white', font=font)
        
        # Save if path provided
        if save_path:
            image.save(save_path)
            print(f"   💾 Visualization saved: {save_path}")
        
        return image
    
    def predict_and_save(
        self,
        image_path: str,
        output_name: str = None
    ) -> Dict:
        """
        Predict on image and save results
        
        Args:
            image_path: Path to input image
            output_name: Custom output name (uses input name if None)
        
        Returns:
            Predictions dictionary
        """
        # Get predictions
        predictions = self.predict(image_path)
        
        # Generate output name
        if output_name is None:
            output_name = Path(image_path).stem
        
        num_detections = len(predictions['boxes'])
        print(f"   🎯 Detected {num_detections} objects")
        
        # Save visualization
        if self.config.inference.save_visualizations:
            vis_path = os.path.join(
                self.config.inference.output_dir,
                f"{output_name}_prediction.jpg"
            )
            self.visualize(image_path, predictions, vis_path)
        
        # Save predictions as JSON
        if self.config.inference.save_predictions_json:
            json_path = os.path.join(
                self.config.inference.output_dir,
                f"{output_name}_predictions.json"
            )
            
            # Convert numpy arrays to lists for JSON serialization
            json_predictions = {
                'image': str(image_path),
                'num_detections': num_detections,
                'boxes': predictions['boxes'].tolist() if hasattr(predictions['boxes'], 'tolist') else predictions['boxes'],
                'scores': predictions['scores'].tolist() if hasattr(predictions['scores'], 'tolist') else predictions['scores'],
                'labels': predictions['labels'].tolist() if hasattr(predictions['labels'], 'tolist') else predictions['labels'],
                'class_names': predictions['class_names']
            }
            
            with open(json_path, 'w') as f:
                json.dump(json_predictions, f, indent=2)
            
            print(f"   💾 Predictions saved: {json_path}")
        
        return predictions
    
    def predict_batch(
        self,
        image_paths: List[str],
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Predict on multiple images
        
        Args:
            image_paths: List of image paths
            show_progress: Whether to show progress
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i, image_path in enumerate(image_paths, 1):
            if show_progress:
                print(f"\n[{i}/{len(image_paths)}] Processing: {Path(image_path).name}")
            
            try:
                predictions = self.predict_and_save(image_path)
                results.append({
                    'image_path': image_path,
                    'predictions': predictions,
                    'success': True
                })
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'success': False
                })
        
        return results
    
    def print_predictions(self, predictions: Dict, max_show: int = 10):
        """
        Print predictions in readable format
        
        Args:
            predictions: Predictions dictionary
            max_show: Maximum number of predictions to show
        """
        num_detections = len(predictions['boxes'])
        
        if num_detections == 0:
            print("   No objects detected")
            return
        
        print(f"   Detections:")
        for i, (box, score, class_name) in enumerate(
            zip(
                predictions['boxes'][:max_show],
                predictions['scores'][:max_show],
                predictions['class_names'][:max_show]
            ),
            1
        ):
            score_val = float(score) if not isinstance(score, float) else score
            print(f"      {i}. {class_name}: {score_val:.3f}")
        
        if num_detections > max_show:
            print(f"      ... and {num_detections - max_show} more")


def load_predictor(model_path: str, config) -> RTDETRPredictor:
    """
    Load predictor from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        config: RTDETRConfig object
    
    Returns:
        RTDETRPredictor instance
    """
    return RTDETRPredictor(model_path, config)
