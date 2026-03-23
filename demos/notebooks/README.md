# PrimateFace Tutorial Notebooks

Local Jupyter notebooks for the applications described in the PrimateFace [paper](https://www.biorxiv.org/content/10.1101/2025.08.12.669927). Models are auto-downloaded from [HuggingFace](https://huggingface.co/fparodi/primateface-models) on first run.

| Notebook | Description |
|----------|-------------|
| [`lemur_video_timestamping.ipynb`](lemur_video_timestamping.ipynb) | Detect and timestamp primate faces in video for behavioral coding (BORIS export) |
| [`macaque_face_recognition.ipynb`](macaque_face_recognition.ipynb) | Closed-set face recognition pipeline: detection, alignment, ArcFace embeddings, SVM |
| [`howler_vocal_motor_coupling.ipynb`](howler_vocal_motor_coupling.ipynb) | Correlate facial kinematics (mouth aperture) with vocalizations via cross-correlation and SVR |
| [`infant_gaze_following.ipynb`](infant_gaze_following.ipynb) | Gaze-following heuristic for two-primate videos using Gazelle gaze estimation |
| [`landmark_demographics.ipynb`](landmark_demographics.ipynb) | Predict age and sex from 68-point facial landmarks (mandrill and chimpanzee datasets) |
| [`facial_action_discovery.ipynb`](facial_action_discovery.ipynb) | Unsupervised discovery of facial actions via wavelets + UMAP + watershed (MotionMapper-inspired) |
| [`quickstart.ipynb`](quickstart.ipynb) | Quick Start: 3-line API, Face object, visualization, export |

## Prerequisites

- Conda environment with PyTorch and CUDA (see main [README](../../README.md))
- MMDetection + MMPose (`mim install mmdet==3.2.0 mmpose==1.3.2`)
- `uv pip install huggingface-hub` (for model auto-download)
- For gaze notebook: [Gazelle](https://github.com/fkryan/gazelle) package

## Quick Start

```python
# Models download automatically when you run a notebook.
# To download manually:
from notebook_utils import download_models_hf
from pathlib import Path
download_models_hf(Path("../"))  # saves to demos/
```
