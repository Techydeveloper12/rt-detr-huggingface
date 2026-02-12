# RT-DETR v2 Object Detection with Hugging Face Transformers

Professional, modular implementation of RT-DETR (Real-Time Detection Transformer) v2 for object detection using Hugging Face Transformers library. This provides an easy-to-use, production-ready solution for training and deploying object detection models on custom datasets.

## 🌟 Features

- ✅ **Hugging Face Integration**: Uses official `transformers` library for RT-DETR
- ✅ **Easy Configuration**: Simple parameter setup for different datasets
- ✅ **COCO Format Support**: Standard COCO JSON annotation format
- ✅ **Pre-trained Models**: Leverage official RT-DETR pre-trained weights
- ✅ **Professional Training**: Built on Hugging Face `Trainer` API
- ✅ **Mixed Precision**: FP16 training support for faster training
- ✅ **Production Ready**: Complete training and inference pipeline

## 📁 Project Structure

```
rt-detr-huggingface/
├── config.py           # Configuration module (CONFIGURE THIS!)
├── dataset.py          # COCO dataset loading and validation
├── trainer.py          # Training with HF Trainer
├── inference.py        # Prediction and visualization
├── utils.py            # Utility functions
├── train.py            # Main training script
├── predict.py          # Main inference script
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Dataset (COCO Format)

Your dataset must follow the COCO JSON annotation format:

```
dataset/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── annotations.json
└── val/
    ├── images/
    │   └── ...
    └── annotations.json
```

#### COCO Annotation Format

```json
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
      "bbox": [x, y, width, height],
      "area": 12345,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "car"
    }
  ]
}
```

**Important Notes:**
- `bbox` format: `[x_min, y_min, width, height]` (COCO standard)
- `category_id` starts from 1 (not 0)
- Coordinates are absolute pixel values

### 3. Training

#### Basic Training Command

```bash
python train.py \
  --train-images data/train/images \
  --train-ann data/train/annotations.json \
  --val-images data/val/images \
  --val-ann data/val/annotations.json \
  --num-classes 3 \
  --class-names car truck bus \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-5
```

#### Training with Configuration File

```bash
# Save your configuration
python -c "
from config import create_custom_config
config = create_custom_config(
    num_classes=3,
    class_names=['car', 'truck', 'bus'],
    train_images_dir='data/train/images',
    train_annotations='data/train/annotations.json',
    val_images_dir='data/val/images',
    val_annotations='data/val/annotations.json'
)
config.save('my_config.json')
"

# Train with config
python train.py --config my_config.json
```

#### Advanced Training Options

```bash
python train.py \
  --train-images data/train/images \
  --train-ann data/train/annotations.json \
  --val-images data/val/images \
  --val-ann data/val/annotations.json \
  --num-classes 5 \
  --class-names person car dog cat bird \
  --epochs 100 \
  --batch-size 16 \
  --lr 2e-5 \
  --fp16 \
  --gradient-accumulation 2 \
  --output-dir my_detector \
  --verify-dataset
```

### 4. Inference

#### Single Image Prediction

```bash
python predict.py \
  --model outputs \
  --image test.jpg \
  --conf-thresh 0.5
```

#### Batch Prediction

```bash
# Directory of images
python predict.py \
  --model outputs \
  --image-dir test_images/ \
  --conf-thresh 0.6 \
  --output-dir predictions

# Image list file
python predict.py \
  --model outputs \
  --image-list images.txt \
  --save-summary
```

## ⚙️ Configuration Guide

### Essential Parameters (MUST Configure for Each Dataset)

These parameters **must** be configured for each new dataset:

```python
from config import create_custom_config

config = create_custom_config(
    num_classes=3,                          # Number of your classes
    class_names=['car', 'truck', 'bus'],   # Your class names
    train_images_dir='data/train/images',
    train_annotations='data/train/annotations.json',
    val_images_dir='data/val/images',
    val_annotations='data/val/annotations.json'
)
```

### Common Training Parameters

Parameters you'll frequently adjust:

```python
# Training duration
config.training.num_epochs = 50              # 50-100 typical
config.training.batch_size = 8               # Adjust based on GPU memory

# Learning
config.training.learning_rate = 1e-5         # 1e-5 to 5e-5 typical
config.training.weight_decay = 1e-4

# Performance
config.training.fp16 = True                  # Enable mixed precision
config.training.gradient_accumulation_steps = 2  # Effective larger batch
```

### Model Selection

Choose the right RT-DETR model:

```python
# Faster, less accurate
config.model.model_name = "PekingU/rtdetr_r50vd"

# More accurate, slower
config.model.model_name = "PekingU/rtdetr_r101vd"

# Pre-trained on COCO + Objects365 (better transfer learning)
config.model.model_name = "PekingU/rtdetr_r50vd_coco_o365"
```

### Inference Parameters

```python
config.inference.confidence_threshold = 0.5  # Detection confidence
config.inference.iou_threshold = 0.5         # NMS threshold
config.inference.max_detections = 100        # Max objects per image
```

## 📊 Dataset Tools

### Verify COCO Format

Check if your annotations are correctly formatted:

```bash
python -c "from dataset import verify_coco_format; verify_coco_format('data/train/annotations.json')"
```

### Create Annotation Template

Generate a template for your images:

```python
from dataset import create_coco_template

create_coco_template(
    images_dir='data/images',
    output_file='annotations_template.json',
    class_names=['car', 'truck', 'bus']
)
```

Then annotate using tools like:
- [CVAT](https://www.cvat.ai/) (Recommended)
- [LabelImg](https://github.com/tzutalin/labelImg)
- [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/)

## 🎯 Available Models

RT-DETR models available on Hugging Face:

| Model | Backbone | Speed | Accuracy | Use Case |
|-------|----------|-------|----------|----------|
| `PekingU/rtdetr_r50vd` | ResNet-50 | Fast | Good | Real-time applications |
| `PekingU/rtdetr_r101vd` | ResNet-101 | Medium | Better | Balanced |
| `PekingU/rtdetr_r50vd_coco_o365` | ResNet-50 | Fast | Best (transfer) | Custom datasets |

## 💡 Examples

### Example 1: Vehicle Detection

```bash
# Train vehicle detector
python train.py \
  --train-images datasets/vehicles/train/images \
  --train-ann datasets/vehicles/train/annotations.json \
  --val-images datasets/vehicles/val/images \
  --val-ann datasets/vehicles/val/annotations.json \
  --num-classes 3 \
  --class-names car truck bus \
  --epochs 50 \
  --batch-size 8 \
  --output-dir vehicle_detector

# Run inference
python predict.py \
  --model vehicle_detector \
  --image-dir test_images/ \
  --conf-thresh 0.6
```

### Example 2: Multi-Class Object Detection

```bash
python train.py \
  --train-images data/train/images \
  --train-ann data/train/annotations.json \
  --val-images data/val/images \
  --val-ann data/val/annotations.json \
  --num-classes 10 \
  --class-names person bicycle car motorcycle airplane bus train truck boat "traffic light" \
  --epochs 100 \
  --batch-size 16 \
  --lr 2e-5 \
  --fp16 \
  --model-name PekingU/rtdetr_r50vd_coco_o365
```

### Example 3: Resume Training

```bash
python train.py \
  --config outputs/config.json \
  --resume outputs/checkpoint-1000
```

## 📈 Training Tips

### GPU Memory Optimization

If you encounter out-of-memory errors:

1. **Reduce batch size**: `--batch-size 4`
2. **Enable gradient accumulation**: `--gradient-accumulation 2`
3. **Use mixed precision**: `--fp16`
4. **Choose smaller model**: `--model-name PekingU/rtdetr_r50vd`

```bash
# Low memory configuration
python train.py \
  --batch-size 4 \
  --gradient-accumulation 4 \
  --fp16 \
  ... # other args
```

### Learning Rate Guidelines

- **Small datasets (<1000 images)**: `1e-5`
- **Medium datasets (1000-10000)**: `2e-5` to `5e-5`
- **Large datasets (>10000)**: `5e-5` to `1e-4`

### Training Duration

- **Quick test**: 10-20 epochs
- **Standard training**: 50 epochs
- **High accuracy**: 100+ epochs

## 📝 Output Files

### Training Outputs

```
outputs/
├── checkpoint-100/          # Periodic checkpoints
├── checkpoint-200/
├── checkpoint-best/         # Best model
├── config.json             # Saved configuration
├── trainer_state.json      # Training state
├── training_args.bin       # Training arguments
└── runs/                   # TensorBoard logs
```

### Inference Outputs

```
predictions/
├── image1_prediction.jpg   # Visualization
├── image1_predictions.json # Predictions data
├── image2_prediction.jpg
├── image2_predictions.json
└── results_summary.json    # Overall summary
```

### Prediction JSON Format

```json
{
  "image": "test.jpg",
  "num_detections": 3,
  "boxes": [[x1, y1, x2, y2], ...],
  "scores": [0.95, 0.87, 0.76],
  "labels": [0, 1, 2],
  "class_names": ["car", "truck", "bus"]
}
```

## 🔧 Advanced Usage

### Custom Training Pipeline

```python
from config import RTDETRConfig, create_custom_config
from dataset import create_dataloaders
from trainer import load_model_and_processor, train_model
from utils import set_seed

# Create config
config = create_custom_config(
    num_classes=3,
    class_names=['car', 'truck', 'bus'],
    train_images_dir='data/train/images',
    train_annotations='data/train/annotations.json',
    val_images_dir='data/val/images',
    val_annotations='data/val/annotations.json'
)

# Customize
config.training.num_epochs = 100
config.training.batch_size = 16
config.training.fp16 = True

# Set seed
set_seed(config.seed)

# Load model
model, image_processor = load_model_and_processor(config)

# Create datasets
train_dataset, val_dataset = create_dataloaders(config, image_processor)

# Train
trainer = train_model(model, train_dataset, val_dataset, config)
```

### Custom Inference Pipeline

```python
from config import RTDETRConfig
from inference import load_predictor

# Load config
config = RTDETRConfig.load('outputs/config.json')
config.inference.confidence_threshold = 0.7

# Load predictor
predictor = load_predictor('outputs', config)

# Predict
predictions = predictor.predict('test.jpg')

# Visualize
image = predictor.visualize('test.jpg', predictions, 'output.jpg')

# Print results
predictor.print_predictions(predictions)
```

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Solution: Reduce batch size and use gradient accumulation
python train.py ... --batch-size 4 --gradient-accumulation 4 --fp16
```

**2. Dataset Not Found**
```bash
# Solution: Verify paths
python -c "from utils import verify_dataset_paths; from config import RTDETRConfig; config = RTDETRConfig(); verify_dataset_paths(config)"
```

**3. Invalid COCO Format**
```bash
# Solution: Verify annotations
python -c "from dataset import verify_coco_format; verify_coco_format('annotations.json')"
```

**4. Model Not Loading**
```bash
# Solution: Check internet connection for downloading pre-trained weights
# Or specify cache directory
python train.py ... --model-name PekingU/rtdetr_r50vd
```

## 📦 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- CUDA 11.0+ (for GPU training)

See `requirements.txt` for complete list.

## 🎓 Understanding RT-DETR

RT-DETR (Real-Time Detection Transformer) is a state-of-the-art object detector that:

- Uses **transformer architecture** for object detection
- Achieves **real-time performance** (unlike DETR)
- Provides **end-to-end** detection (no NMS during training)
- Works with various **backbone networks** (ResNet, etc.)

### Architecture Components

1. **Backbone**: Feature extraction (ResNet-50/101)
2. **Encoder**: Hybrid encoder for efficient feature processing
3. **Decoder**: IoU-aware query selection mechanism
4. **Detection Head**: Classification and bounding box regression

## 📚 Additional Resources

- [RT-DETR Paper](https://arxiv.org/abs/2304.08069)
- [Hugging Face RT-DETR Docs](https://huggingface.co/docs/transformers/model_doc/rt_detr)
- [COCO Dataset Format](https://cocodataset.org/#format-data)

## 🤝 Contributing

This is a professional template for RT-DETR training and inference. Feel free to customize for your specific needs.

## 📄 License

This project uses Hugging Face Transformers and follows their licensing terms.

---

**Need Help?** 
- Check the configuration in `config.py` for all available parameters
- Run `python train.py --help` or `python predict.py --help` for command-line options
- Verify your dataset format with the validation tools provided

**Happy Training! 🚀**
