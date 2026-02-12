"""
Configuration Module for RT-DETR v2 with Hugging Face Transformers
Easy-to-configure parameters for object detection training and inference
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import json
import os


@dataclass
class DatasetConfig:
    """
    Dataset Configuration - CONFIGURE THIS FOR EACH DATASET
    """
    # Dataset paths (COCO format)
    train_images_dir: str = "data/train/images"
    train_annotations: str = "data/train/annotations.json"  # COCO JSON
    val_images_dir: str = "data/val/images"
    val_annotations: str = "data/val/annotations.json"  # COCO JSON
    
    # MUST CONFIGURE: Class configuration
    num_classes: int = 80  # Number of object classes (excluding background)
    class_names: List[str] = field(default_factory=lambda: [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane'
        # Add all your class names here
    ])
    
    # ID to label mapping (COCO categories start from 1)
    id2label: Optional[Dict[int, str]] = None
    label2id: Optional[Dict[str, int]] = None
    
    # Image preprocessing
    image_size: int = 640
    
    def __post_init__(self):
        """Build label mappings"""
        if self.id2label is None:
            self.id2label = {i: name for i, name in enumerate(self.class_names)}
        if self.label2id is None:
            self.label2id = {name: i for i, name in enumerate(self.class_names)}


@dataclass
class ModelConfig:
    """
    Model Configuration - Hugging Face RT-DETR
    """
    # Model selection
    model_name: str = "PekingU/rtdetr_v2_r50vd"
    # Available RT-DETR v2 models on HuggingFace:
    # - "PekingU/rtdetr_v2_r50vd" (RT-DETR v2, ResNet-50 backbone)
    # - "PekingU/rtdetr_v2_r101vd" (RT-DETR v2, ResNet-101 backbone, more accurate)
    # 
    # RT-DETR v1 models (legacy):
    # - "PekingU/rtdetr_r50vd" (RT-DETR v1, ResNet-50)
    # - "PekingU/rtdetr_r101vd" (RT-DETR v1, ResNet-101)
    
    # Use pre-trained weights
    from_pretrained: bool = True
    
    # Model parameters
    num_queries: int = 300  # Number of object queries
    
    # Cache directory for downloaded models
    cache_dir: Optional[str] = "./model_cache"


@dataclass
class TrainingConfig:
    """
    Training Configuration - CONFIGURE THESE FOR YOUR TRAINING
    """
    # Basic training settings
    output_dir: str = "outputs"
    num_epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    
    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"  # Options: linear, cosine, polynomial
    warmup_steps: int = 500
    warmup_ratio: float = 0.1
    
    # Gradient settings
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 0.1
    fp16: bool = False  # Mixed precision training
    
    # Evaluation and saving
    evaluation_strategy: str = "epoch"  # Options: "no", "steps", "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 3  # Keep only last 3 checkpoints
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    # Logging
    logging_dir: str = "logs"
    logging_steps: int = 50
    report_to: str = "tensorboard"  # Options: "tensorboard", "wandb", "none"
    
    # Data loading
    num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.0


@dataclass
class InferenceConfig:
    """
    Inference Configuration
    """
    # Model checkpoint
    checkpoint_path: str = "outputs/checkpoint-best"
    
    # Post-processing thresholds
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.5  # For NMS
    max_detections: int = 100
    
    # Input/Output settings
    input_size: int = 640
    output_dir: str = "predictions"
    save_visualizations: bool = True
    save_predictions_json: bool = True
    
    # Device
    device: str = "cuda"  # "cuda" or "cpu"


@dataclass
class RTDETRConfig:
    """
    Main Configuration Class for RT-DETR v2
    """
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Global settings
    seed: int = 42
    device: str = "cuda"
    
    def validate(self):
        """Validate configuration"""
        # Check if class names match num_classes
        if len(self.dataset.class_names) != self.dataset.num_classes:
            raise ValueError(
                f"Number of class names ({len(self.dataset.class_names)}) "
                f"doesn't match num_classes ({self.dataset.num_classes})"
            )
        
        # Check if dataset paths exist
        if not os.path.exists(self.dataset.train_annotations):
            print(f"Warning: Training annotations not found: {self.dataset.train_annotations}")
        
        return True
    
    def save(self, path: str):
        """Save configuration to JSON file"""
        from dataclasses import asdict
        
        config_dict = asdict(self)
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Configuration saved to: {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct nested dataclasses
        dataset_cfg = DatasetConfig(**config_dict.get('dataset', {}))
        model_cfg = ModelConfig(**config_dict.get('model', {}))
        training_cfg = TrainingConfig(**config_dict.get('training', {}))
        inference_cfg = InferenceConfig(**config_dict.get('inference', {}))
        
        return cls(
            dataset=dataset_cfg,
            model=model_cfg,
            training=training_cfg,
            inference=inference_cfg,
            seed=config_dict.get('seed', 42),
            device=config_dict.get('device', 'cuda')
        )


def create_custom_config(
    num_classes: int,
    class_names: List[str],
    train_images_dir: str,
    train_annotations: str,
    val_images_dir: str,
    val_annotations: str,
    **kwargs
) -> RTDETRConfig:
    """
    Quick helper to create configuration for custom dataset
    
    Args:
        num_classes: Number of object classes
        class_names: List of class names
        train_images_dir: Path to training images
        train_annotations: Path to training COCO JSON
        val_images_dir: Path to validation images
        val_annotations: Path to validation COCO JSON
        **kwargs: Additional config overrides
    
    Returns:
        RTDETRConfig object
    """
    config = RTDETRConfig()
    
    # Dataset configuration
    config.dataset.num_classes = num_classes
    config.dataset.class_names = class_names
    config.dataset.train_images_dir = train_images_dir
    config.dataset.train_annotations = train_annotations
    config.dataset.val_images_dir = val_images_dir
    config.dataset.val_annotations = val_annotations
    
    # Apply any additional overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config.model, key):
            setattr(config.model, key, value)
    
    return config


def print_config_summary(config: RTDETRConfig):
    """Print configuration summary"""
    print("\n" + "="*70)
    print("RT-DETR V2 CONFIGURATION SUMMARY")
    print("="*70)
    
    print("\n📁 DATASET:")
    print(f"  Classes: {config.dataset.num_classes}")
    print(f"  Class names: {config.dataset.class_names[:5]}{'...' if len(config.dataset.class_names) > 5 else ''}")
    print(f"  Train images: {config.dataset.train_images_dir}")
    print(f"  Val images: {config.dataset.val_images_dir}")
    
    print("\n🤖 MODEL:")
    print(f"  Model: {config.model.model_name}")
    print(f"  Pretrained: {config.model.from_pretrained}")
    print(f"  Queries: {config.model.num_queries}")
    
    print("\n🎯 TRAINING:")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Output dir: {config.training.output_dir}")
    print(f"  FP16: {config.training.fp16}")
    
    print("\n💻 DEVICE:")
    print(f"  Device: {config.device}")
    print(f"  Seed: {config.seed}")
    
    print("="*70 + "\n")
