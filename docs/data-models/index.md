# Data & Models

Datasets, pretrained models, and data preparation guides.

## Pretrained Models

### Available Models

| Model | Framework | Task | Download | Size |
|-------|-----------|------|----------|------|
| Cascade R-CNN R101-FPN | MMDetection | Face Detection | [HuggingFace](https://huggingface.co/fparodi/primateface-models/tree/main/detection) | 340 MB |
| HRNetV2-W18-DARK | MMPose | 68-pt Landmarks | [HuggingFace](https://huggingface.co/fparodi/primateface-models/tree/main/pose) | 38 MB |
| ViTPose-Base | MMPose | 68-pt Landmarks | [HuggingFace](https://huggingface.co/fparodi/primateface-models/tree/main/pose) | 1.2 GB |

### Quick Download

```bash
# Download all default models (Cascade R-CNN + HRNet)
python demos/download_models.py

# Or use the Python API (auto-downloads on first use)
import primateface
pf = primateface.PrimateFace()  # downloads HRNet (default)
pf = primateface.PrimateFace(pose_model="vitpose")  # downloads ViTPose
```

All models are hosted on [HuggingFace](https://huggingface.co/fparodi/primateface-models) and cached locally after first download.

## Datasets

### PrimateFace Dataset

Our comprehensive dataset includes:
- **60+ primate genera**
- **10,000+ annotated images**
- **68-point facial landmarks**
- **Species labels**

### Data Format

All annotations use COCO format:

```json
{
  "info": {
    "description": "PrimateFace Dataset",
    "version": "1.0"
  },
  "images": [
    {
      "id": 1,
      "file_name": "macaque_001.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 100, 200, 200],
      "keypoints": [x1,y1,v1, x2,y2,v2, ...],
      "num_keypoints": 68
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "primate_face",
      "keypoints": ["point_1", "point_2", ...],
      "skeleton": [[1,2], [2,3], ...]
    }
  ]
}
```

## Data Preparation

### Creating COCO Annotations

#### From Image Directory
```python
from gui.imgdir2coco_facedet import create_coco_dataset

create_coco_dataset(
    image_dir="path/to/images",
    output_json="annotations.json"
)
```

#### From Video Files
```python
from gui.viddir2coco_facedet import extract_and_annotate

extract_and_annotate(
    video_dir="path/to/videos",
    output_json="video_annotations.json",
    sample_rate=5  # fps
)
```

### Data Validation

```python
from evals.core.utils import validate_coco

# Check annotation integrity
issues = validate_coco("annotations.json")
if not issues:
    print("Dataset is valid!")
```

### Data Splitting

```python
from evals.core.utils import split_dataset

# Create train/val/test splits
split_dataset(
    input_json="full_dataset.json",
    output_dir="splits/",
    ratios=(0.7, 0.15, 0.15)
)
```

### Data Augmentation

For training, apply augmentations:

```python
augmentation_config = {
    "RandomFlip": {"p": 0.5},
    "RandomRotate": {"max_angle": 30},
    "ColorJitter": {"brightness": 0.2, "contrast": 0.2},
    "GaussianBlur": {"p": 0.1}
}
```

## Keypoint Definitions

### 68-Point System

Standard facial landmarks:

```python
FACIAL_LANDMARKS_68 = {
    "jaw": list(range(0, 17)),        # Jaw line
    "right_eyebrow": list(range(17, 22)),
    "left_eyebrow": list(range(22, 27)),
    "nose_bridge": list(range(27, 31)),
    "lower_nose": list(range(31, 36)),
    "right_eye": list(range(36, 42)),
    "left_eye": list(range(42, 48)),
    "outer_lip": list(range(48, 60)),
    "inner_lip": list(range(60, 68))
}
```

### 48-Point System

Primate-optimized landmarks:

```python
PRIMATE_LANDMARKS_48 = {
    "face_contour": list(range(0, 12)),   # Simplified jaw
    "right_eyebrow": list(range(12, 16)),
    "left_eyebrow": list(range(16, 20)),
    "nose": list(range(20, 28)),
    "right_eye": list(range(28, 34)),
    "left_eye": list(range(34, 40)),
    "mouth": list(range(40, 48))
}
```

## Dataset Statistics

### Species Distribution

Analysis of the PrimateFace dataset:

```python
from evals.plot_genus_distribution import analyze_dataset

stats = analyze_dataset("annotations.json")
print(f"Total images: {stats['n_images']}")
print(f"Total annotations: {stats['n_annotations']}")
print(f"Unique species: {stats['n_species']}")
```

### Quality Metrics

```python
from evals.core.metrics import dataset_quality

quality = dataset_quality("annotations.json")
print(f"Average keypoints per face: {quality['avg_keypoints']}")
print(f"Annotation completeness: {quality['completeness']:.1%}")
```

## External Datasets

### Compatible Datasets

PrimateFace can work with:
- **MacaquePose**: 13,000+ macaque images
- **ChimpFace**: Chimpanzee facial dataset
- **OpenMonkeyChallenge**: Multi-species competition data

### Converting External Data

```python
from landmark_converter.apply_model import convert_dataset

# Convert human face dataset
convert_dataset(
    input_json="human_faces.json",
    output_json="primate_compatible.json",
    converter_model="68to48_converter.pth"
)
```

## Model Zoo Integration

### Using HuggingFace Models

```python
from transformers import AutoModel

# Load from HuggingFace
model = AutoModel.from_pretrained("primateface/hrnet-w32")
```

### Model Cards

Each model includes:
- Training details
- Performance benchmarks
- Limitations
- Citation information

## See Also

- [Installation](../installation.md)
- [Getting Started](../getting-started/index.md)
- [User Guide](../user-guide/index.md)
- [API Reference](../api/index.md)