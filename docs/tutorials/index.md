# Tutorials

PrimateFace provides Jupyter notebook tutorials demonstrating real-world applications of primate facial analysis. Models are auto-downloaded from [HuggingFace](https://huggingface.co/fparodi/primateface-models) on first run.

## Quick Start

New to PrimateFace? Start here:

```python
import primateface
pf = primateface.PrimateFace()
faces = pf.analyze("monkey.jpg")
faces[0].head_pose   # (yaw, pitch, roll)
faces[0].kinematics  # 14 geometric features
```

See the [Quick Start notebook](https://github.com/KordingLab/PrimateFace/blob/main/demos/notebooks/quickstart.ipynb) for a full walkthrough.

## Application Notebooks

| Notebook | Species | Description |
|----------|---------|-------------|
| [Quick Start](https://github.com/KordingLab/PrimateFace/blob/main/demos/notebooks/quickstart.ipynb) | Spider monkey | 3-line API, Face object, visualization, export |
| [Lemur Video Timestamping](https://github.com/KordingLab/PrimateFace/blob/main/demos/notebooks/lemur_video_timestamping.ipynb) | Lemur | Detect and timestamp faces in video for behavioral coding |
| [Macaque Face Recognition](https://github.com/KordingLab/PrimateFace/blob/main/demos/notebooks/macaque_face_recognition.ipynb) | Macaque | Face recognition: ArcFace vs MegaDescriptor vs DINOv2 |
| [Howler Vocal-Motor Coupling](https://github.com/KordingLab/PrimateFace/blob/main/demos/notebooks/howler_vocal_motor_coupling.ipynb) | Howler monkey | Correlate facial kinematics with vocalizations |
| [Macaque Gaze Following](https://github.com/KordingLab/PrimateFace/blob/main/demos/notebooks/macaque_gaze_following.ipynb) | Macaque | Gaze-following heuristic with Gazelle |
| [Landmark Demographics](https://github.com/KordingLab/PrimateFace/blob/main/demos/notebooks/landmark_demographics.ipynb) | Mandrill, Chimp | Predict age and sex from 68-point facial landmarks |
| [Facial Action Discovery](https://github.com/KordingLab/PrimateFace/blob/main/demos/notebooks/facial_action_discovery.ipynb) | Howler monkey | Unsupervised action discovery (wavelets + UMAP + watershed) |

## Running Locally

```bash
# Clone and install
git clone https://github.com/KordingLab/PrimateFace.git
cd PrimateFace
pip install -e .

# Launch notebooks
jupyter notebook demos/notebooks/
```

Models download automatically on first use. See the [installation guide](../getting-started/installation.md) for full setup instructions.

## Need Help?

- Check [Troubleshooting](../troubleshooting.md)
- Report issues on [GitHub](https://github.com/KordingLab/PrimateFace/issues)
