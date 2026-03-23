# Training Landmark Converters for Cross-Dataset Compatibility

## Overview

This guide demonstrates how to train neural network models that convert between different facial landmark annotation formats. The primary use case is converting 68-point human face annotations to 48-point primate face annotations, enabling the use of large human face datasets for primate model training.

**Key applications:**
- Leverage COCO-WholeBody-Face dataset (68 points) for primate models (48 points)
- Enable cross-species transfer learning
- Unify different annotation standards
- Reduce annotation effort by 80%+

## Theory & Motivation

### The Problem

Different datasets use incompatible keypoint definitions:

- **Human datasets**: Often 68 points (dlib standard) or 106 points (MediaPipe)
- 
- **Primate datasets**: Custom 48-point or 49-point schemes
- 
- **Cross-species mapping**: Non-trivial due to anatomical differences

### The Solution

Train a neural network to learn the mapping between formats:

1. Collect paired annotations (same images with both formats)
   h
2. Train converter network on normalized coordinates
   
3. Apply to new data at inference time

## Installation

```bash
# Activate PrimateFace environment
conda activate primateface

# Install PyTorch (choose your CUDA version)
# CUDA 11.8:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Optional: For Graph Neural Network models
# uv pip install torch-geometric
```

## Data Preparation

### Required Format: COCO JSON

The training script expects COCO format with both source and target keypoints:

```json
{
    "images": [...],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "keypoints": [x1, y1, v1, ..., x68, y68, v68],  // 68-point source
            "num_keypoints": 68,
            "bbox": [x, y, width, height],
            "area": 12345,
            "keypoints_target": [x1, y1, v1, ..., x48, y48, v48]  // 48-point target
        }
    ],
    "categories": [...]
}
```

### Prepare Training Data

```python
import json
import numpy as np

def prepare_paired_annotations(source_json, target_json, output_json):
    """Combine source and target annotations into training format."""
    
    with open(source_json) as f:
        source_data = json.load(f)
    with open(target_json) as f:
        target_data = json.load(f)
    
    # Create mapping by image_id
    target_by_image = {ann['image_id']: ann for ann in target_data['annotations']}
    
    # Add target keypoints to source annotations
    for ann in source_data['annotations']:
        if ann['image_id'] in target_by_image:
            target_ann = target_by_image[ann['image_id']]
            ann['keypoints_target'] = target_ann['keypoints']
    
    # Save combined data
    with open(output_json, 'w') as f:
        json.dump(source_data, f)
    
    print(f"Created paired dataset with {len(source_data['annotations'])} samples")

# Example usage
prepare_paired_annotations(
    "human_68pt.json",
    "primate_48pt.json", 
    "paired_training.json"
)
```

## Model Architectures

The framework provides multiple architectures in `landmark-converter/src/models.py`:

| Model | Class | Parameters | Description |
|-------|-------|------------|-------------|
| `simple_linear` | `SimpleLinearConverter` | ~7K | Direct linear mapping, baseline |
| `mlp` | `KeypointConverterMLP` | ~85K | **Recommended**: 2-layer MLP with dropout |
| `mlp_attention` | `KeypointConverterMLPWithAttention` | ~51K | Self-attention enhanced MLP |
| `minimal_mlp` | `MinimalMLPConverter` | ~20K | Single hidden layer, fast |

## Training Pipeline

### Basic Training

Train the default MLP model for 68→48 conversion:

```bash
cd landmark-converter

# Basic training
python train.py \
    --model mlp \
    --coco_json paired_training.json \
    --output_dir results/ \
    --epochs 500 \
    --batch_size 64
```

### Advanced Training with Validation

```bash
python train.py \
    --model mlp \
    --coco_json paired_training.json \
    --output_dir results/ \
    --epochs 500 \
    --batch_size 64 \
    --val_split 0.15 \
    --test_split 0.05 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --augment \
    --num_source_kpt 68 \
    --num_target_kpt 48 \
    --gpu_id 0
```

### Training with Attention Model

For complex mappings, use the attention-enhanced model:

```bash
python train.py \
    --model mlp_attention \
    --coco_json paired_training.json \
    --output_dir results_attention/ \
    --epochs 1000 \
    --embed_dim 128 \
    --num_heads 4 \
    --mlp_hidden_dim 256
```

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 500 | Training epochs |
| `--batch_size` | 64 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--weight_decay` | 1e-4 | L2 regularization |
| `--val_split` | 0.0 | Validation split ratio |
| `--augment` | False | Enable data augmentation |
| `--early_stop_patience` | 50 | Early stopping patience |

## Model Training Code Example

```python
from landmark_converter.train import create_model, setup_device
from landmark_converter.src.training_pipeline import ModelTrainer
from landmark_converter.utils.data_utils import CocoPairedKeypointDataset

# Setup
device = setup_device(gpu_id=0)
num_source_kpts = 68
num_target_kpts = 48

# Create model
model = create_model(
    model_name='mlp',
    num_source_kpts=num_source_kpts,
    num_target_kpts=num_target_kpts,
    args=args  # Command-line arguments
)
model.to(device)

# Load data
dataset = CocoPairedKeypointDataset(
    coco_json_path='paired_training.json',
    img_dir='images/',
    num_source_kpt=num_source_kpts,
    num_target_kpt=num_target_kpts,
    transform=None
)

# Initialize trainer
trainer = ModelTrainer(
    model=model,
    device=device,
    output_dir='results/',
    learning_rate=1e-3,
    weight_decay=1e-4
)

# Train
trainer.train(
    train_dataset=dataset,
    val_dataset=None,
    epochs=500,
    batch_size=64
)
```

## Applying Trained Models

### Batch Inference

Apply the trained converter to new annotations:

```bash
python apply_model.py \
    --model_path results/best_model.pth \
    --coco_json test_68pt.json \
    --output_dir predictions/ \
    --batch_size 32 \
    --visualize
```

### Python API for Inference

```python
from landmark_converter.apply_model import load_model_from_checkpoint, predict_keypoints
import torch
import numpy as np

# Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, train_args = load_model_from_checkpoint("results/best_model.pth", device)

# Prepare input (68-point keypoints)
source_keypoints = np.array([[x1, y1], [x2, y2], ..., [x68, y68]])  # Shape: (68, 2)
bbox = np.array([100, 100, 200, 200])  # [x, y, width, height]

# Predict 48-point keypoints
predicted_keypoints = predict_keypoints(model, source_keypoints, bbox, device)
print(f"Predicted shape: {predicted_keypoints.shape}")  # (48, 2)
```

### Batch Conversion Script

Convert entire COCO dataset:

```python
import json
import torch
import numpy as np
from tqdm import tqdm

def convert_coco_dataset(model, source_json, output_json, device):
    """Convert all annotations in a COCO file."""
    
    with open(source_json) as f:
        data = json.load(f)
    
    converted_annotations = []
    
    for ann in tqdm(data['annotations'], desc="Converting"):
        # Extract source keypoints (68 points)
        kpts = ann['keypoints']
        source_kpts = np.array(kpts).reshape(-1, 3)[:, :2]  # Shape: (68, 2)
        
        # Get bbox for normalization
        bbox = np.array(ann['bbox'])
        
        # Predict target keypoints
        target_kpts = predict_keypoints(model, source_kpts, bbox, device)
        
        # Create new annotation
        new_ann = ann.copy()
        # Flatten with visibility flags (all visible)
        new_ann['keypoints'] = []
        for x, y in target_kpts:
            new_ann['keypoints'].extend([float(x), float(y), 2])
        new_ann['num_keypoints'] = len(target_kpts)
        
        converted_annotations.append(new_ann)
    
    # Update data
    data['annotations'] = converted_annotations
    
    # Update category info
    data['categories'][0]['keypoints'] = [f"point_{i}" for i in range(48)]
    
    # Save
    with open(output_json, 'w') as f:
        json.dump(data, f)
    
    print(f"Converted {len(converted_annotations)} annotations")

# Usage
model, _ = load_model_from_checkpoint("best_model.pth", device)
convert_coco_dataset(model, "human_68pt.json", "primate_48pt.json", device)
```

## Evaluation Metrics

The training pipeline tracks multiple metrics:

### Mean Per-Joint Position Error (MPJPE)

```python
from landmark_converter.utils.metrics import calculate_mpjpe

mpjpe = calculate_mpjpe(predicted_keypoints, ground_truth_keypoints)
print(f"MPJPE: {mpjpe:.2f} pixels")
```

### Percentage of Correct Keypoints (PCK)

```python
from landmark_converter.utils.metrics import calculate_pck

# PCK@0.2 (20% of bbox size)
pck = calculate_pck(
    predicted_keypoints,
    ground_truth_keypoints,
    bboxes,
    threshold=0.2
)
print(f"PCK@0.2: {pck:.1f}%")
```

## Visualization

### Training Progress

Monitor training with TensorBoard:

```bash
tensorboard --logdir results/logs/
```

### Prediction Visualization

The `apply_model.py` script generates visualizations:

```python
from landmark_converter.apply_model import generate_visualization

# Generate side-by-side comparison
fig = generate_visualization(
    image_path="primate.jpg",
    source_kpts_list=[source_keypoints],
    predicted_kpts_list=[predicted_keypoints]
)
fig.savefig("comparison.png")
```

## Advanced Techniques

### Data Augmentation

Enable augmentation for better generalization:

```python
from torchvision import transforms

augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    )
])

dataset = CocoPairedKeypointDataset(
    coco_json_path='paired_training.json',
    transform=augmentation
)
```

### Custom Loss Functions

Implement weighted loss for important keypoints:

```python
class WeightedKeypointLoss(nn.Module):
    def __init__(self, keypoint_weights):
        super().__init__()
        self.weights = torch.tensor(keypoint_weights)
    
    def forward(self, pred, target):
        # pred, target: [batch, num_kpts, 2]
        diff = (pred - target) ** 2
        weighted_diff = diff * self.weights.unsqueeze(0).unsqueeze(-1)
        return weighted_diff.mean()

# Eye and mouth points get higher weight
weights = [1.0] * 48
weights[36:48] = [2.0] * 12  # Eye region
loss_fn = WeightedKeypointLoss(weights)
```

### Multi-Species Support

Train converters for different species:

```bash
# Macaque-specific converter
python train.py \
    --model mlp_attention \
    --coco_json macaque_paired.json \
    --conversion_mode human_to_macaque \
    --num_source_kpt 68 \
    --num_target_kpt 49

# Chimpanzee-specific converter  
python train.py \
    --model mlp_attention \
    --coco_json chimp_paired.json \
    --conversion_mode human_to_chimp \
    --num_source_kpt 68 \
    --num_target_kpt 51
```

## Troubleshooting

### Poor Conversion Accuracy

1. **Try attention model**: Better for complex mappings
   ```bash
   python train.py --model mlp_attention --epochs 1000
   ```

2. **Increase training data**: Need at least 1000 paired samples

3. **Enable augmentation**: Improves generalization
   ```bash
   python train.py --augment --aug_prob 0.5
   ```

### CUDA Out of Memory

```bash
# Reduce batch size
python train.py --batch_size 16

# Use CPU
python train.py --gpu_id -1
```

### Validation Loss Not Decreasing

```bash
# Reduce learning rate
python train.py --lr 1e-4

# Add regularization
python train.py --weight_decay 1e-3

# Try different architecture
python train.py --model minimal_mlp
```

## Testing

Run unit tests to verify installation:

```bash
cd landmark-converter
python test_landmark_converter.py

# Or with pytest
pytest test_landmark_converter.py -v
```

## Next Steps

### Practical Tutorials
- [Lemur Video Timestamping](../../tutorials/notebooks/lemur_video_timestamping.ipynb) - Time series analysis with converted landmarks
- [Macaque Face Recognition](../../tutorials/notebooks/macaque_face_recognition.ipynb) - Face recognition with converted data
- [Macaque Gaze Following](../../tutorials/notebooks/macaque_gaze_following.ipynb) - Gaze analysis with converted points

### Core Workflows
- [MMPose Training](mmpose-training.md) - Train models with converted data
- [Pseudo-labeling Guide](pseudo-labeling.md) - Generate more training data
- [Evaluation Metrics](evaluation-metrics.md) - Measure conversion quality

## References

- Training script: `landmark-converter/train.py`
- Inference script: `landmark-converter/apply_model.py`
- Model architectures: `landmark-converter/src/models.py`
- Training pipeline: `landmark-converter/src/training_pipeline.py`
- Data utilities: `landmark-converter/utils/data_utils.py`
- Constants: `landmark-converter/constants.py`

## Detailed API Documentation

For comprehensive API reference, advanced usage patterns, and detailed parameter documentation, see the [Landmark Converter API Reference](../../api/landmark-converter.md).

This includes:
- Complete class and method documentation
- All CLI options and parameters
- Advanced configuration examples
- Performance optimization details
- Troubleshooting guides