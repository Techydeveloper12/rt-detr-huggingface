"""
Training Module for RT-DETR with Hugging Face Transformers
Handles model training, evaluation, and checkpointing
"""

import torch
from transformers import (
    RTDetrV2ForObjectDetection,
    RTDetrImageProcessor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)
from typing import Dict, List, Any
import numpy as np
from dataset import collate_fn


class SaveImageProcessorCallback(TrainerCallback):
    """
    Callback to save image processor with each checkpoint
    """
    def __init__(self, image_processor):
        self.image_processor = image_processor
    
    def on_save(self, args, state, control, **kwargs):
        """Save image processor whenever a checkpoint is saved"""
        if state.is_world_process_zero:
            # Save to the checkpoint directory
            output_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
            self.image_processor.save_pretrained(output_dir)
        return control


def load_model_and_processor(config):
    """
    Load RT-DETR model and image processor from Hugging Face
    
    Args:
        config: RTDETRConfig object
    
    Returns:
        model, image_processor
    """
    print(f"\n🤖 Loading model: {config.model.model_name}")
    
    # Load image processor (shared between v1 and v2)
    image_processor = RTDetrImageProcessor.from_pretrained(
        config.model.model_name,
        cache_dir=config.model.cache_dir
    )
    
    # Load model
    if config.model.from_pretrained:
        print("   Loading pre-trained weights...")
        model = RTDetrV2ForObjectDetection.from_pretrained(
            config.model.model_name,
            num_labels=config.dataset.num_classes,
            id2label=config.dataset.id2label,
            label2id=config.dataset.label2id,
            ignore_mismatched_sizes=True,  # Allow different number of classes
            cache_dir=config.model.cache_dir
        )
    else:
        print("   Initializing model from scratch...")
        from transformers import RTDetrV2Config as RTDetrModelConfig
        
        model_config = RTDetrModelConfig.from_pretrained(
            config.model.model_name,
            num_labels=config.dataset.num_classes,
            id2label=config.dataset.id2label,
            label2id=config.dataset.label2id,
        )
        model = RTDetrV2ForObjectDetection(model_config)
    
    print(f"   ✅ Model loaded successfully")
    print(f"   Classes: {config.dataset.num_classes}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return model, image_processor


def create_training_arguments(config) -> TrainingArguments:
    """
    Create Hugging Face TrainingArguments from config
    
    Args:
        config: RTDETRConfig object
    
    Returns:
        TrainingArguments
    """
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        
        # Learning rate scheduler
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_steps=config.training.warmup_steps,
        warmup_ratio=config.training.warmup_ratio,
        
        # Gradient settings
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_grad_norm=config.training.max_grad_norm,
        fp16=config.training.fp16,
        
        # Evaluation and saving
        eval_strategy=config.training.evaluation_strategy,
        save_strategy=config.training.save_strategy,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=False,  # For loss, lower is better
        
        # Logging
        logging_dir=config.training.logging_dir,
        logging_steps=config.training.logging_steps,
        report_to=config.training.report_to,
        
        # Data loading
        dataloader_num_workers=config.training.num_workers,
        dataloader_pin_memory=config.training.dataloader_pin_memory,
        
        # Other
        remove_unused_columns=False,  # Important for object detection
        push_to_hub=False,
        seed=config.seed,
    )
    
    return training_args


class RTDETRTrainer(Trainer):
    """
    Custom Trainer for RT-DETR with proper loss computation
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss for RT-DETR
        
        Args:
            model: The model
            inputs: Input batch
            return_outputs: Whether to return outputs
            num_items_in_batch: Number of items in the batch (for compatibility with Transformers 5.x)
        
        Returns:
            loss or (loss, outputs)
        """
        # Forward pass - RT-DETR expects both pixel_values and labels
        # Don't pop labels, the model needs them to compute loss
        outputs = model(**inputs)
        
        # RT-DETR outputs contain loss in the outputs
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics
    
    Note: For proper object detection metrics (mAP), you would need
    to implement COCO evaluation. This is a simplified version.
    
    Args:
        eval_pred: EvalPrediction object
    
    Returns:
        Dict of metrics
    """
    # This is a placeholder - for production use,
    # implement proper COCO mAP evaluation
    predictions, labels = eval_pred
    
    # RT-DETR returns loss directly, so we can use that
    # For proper evaluation, you'd compute mAP, mAP@50, etc.
    
    return {
        "eval_metric": 0.0  # Placeholder
    }


def train_model(model, train_dataset, val_dataset, config, image_processor):
    """
    Train RT-DETR model using Hugging Face Trainer
    
    Args:
        model: RT-DETR model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: RTDETRConfig object
        image_processor: RT-DETR image processor
    
    Returns:
        Trainer object after training
    """
    print("\n🎯 Setting up training...")
    
    # Create training arguments
    training_args = create_training_arguments(config)
    
    # Create callbacks
    callbacks = []
    if config.training.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.training.early_stopping_patience,
                early_stopping_threshold=config.training.early_stopping_threshold
            )
        )
    
    # Add callback to save image processor with checkpoints
    callbacks.append(SaveImageProcessorCallback(image_processor))
    
    # Create trainer
    trainer = RTDETRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=callbacks,
    )
    
    # Save image processor to output directory before training starts
    image_processor.save_pretrained(config.training.output_dir)
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"Output directory: {config.training.output_dir}")
    print(f"Total epochs: {config.training.num_epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print("="*70 + "\n")
    
    # Resume from checkpoint if specified
    checkpoint = config.training.resume_from_checkpoint
    
    # Train
    try:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        # Save final model
        trainer.save_model()
        
        # Save image processor with the model
        image_processor.save_pretrained(config.training.output_dir)
        
        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Save training state
        trainer.save_state()
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Final model saved to: {config.training.output_dir}")
        print(f"Training metrics saved")
        print("="*70 + "\n")
        
        return trainer
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Saving checkpoint...")
        trainer.save_model(f"{config.training.output_dir}/interrupted")
        print(f"Checkpoint saved to: {config.training.output_dir}/interrupted")
        raise
    
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        raise


def evaluate_model(trainer, eval_dataset=None):
    """
    Evaluate trained model
    
    Args:
        trainer: Trained Trainer object
        eval_dataset: Evaluation dataset (uses trainer's if None)
    
    Returns:
        Dict of evaluation metrics
    """
    print("\n📊 Evaluating model...")
    
    if eval_dataset is not None:
        metrics = trainer.evaluate(eval_dataset)
    else:
        metrics = trainer.evaluate()
    
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
    print(f"\n✅ Evaluation completed")
    print(f"   Eval loss: {metrics.get('eval_loss', 'N/A')}")
    
    return metrics


def save_model_for_inference(trainer, output_path: str):
    """
    Save model and processor for inference
    
    Args:
        trainer: Trained Trainer object
        output_path: Path to save model
    """
    print(f"\n💾 Saving model for inference to: {output_path}")
    
    # Save model
    trainer.save_model(output_path)
    
    print(f"✅ Model saved successfully")
    print(f"   You can now use this model for inference")
    print(f"   Path: {output_path}")
