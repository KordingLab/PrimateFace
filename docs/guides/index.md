# Workflow Guides

Step-by-step guides for common PrimateFace workflows.

## Quick Start

The fastest way to get started:

```python
import primateface
pf = primateface.PrimateFace()
faces = pf.analyze("monkey.jpg")
print(faces[0].head_pose)
```

Models download automatically from [HuggingFace](https://huggingface.co/fparodi/primateface-models). See the [Quick Start notebook](https://github.com/KordingLab/PrimateFace/blob/main/demos/notebooks/quickstart.ipynb) for a full walkthrough.

## Available Guides

| Workflow | Guide | Description |
|----------|-------|-------------|
| Inference | [Demos User Guide](../user-guide/core-workflows/demos.md) | Detection + pose on images and video |
| DINOv2 Features | [DINOv2 Guide](../user-guide/core-workflows/dinov2.md) | Extract and visualize DINOv2 embeddings |
| Pseudo-labeling | [GUI Guide](../user-guide/core-workflows/gui.md) | Interactive annotation refinement |
| Landmark Conversion | [Converter Guide](../user-guide/core-workflows/landmark-converter.md) | Convert between keypoint formats |
| Framework Integration | [Frameworks](../frameworks/index.md) | Export to DLC, SLEAP, NWB |

## CLI Usage

```bash
# Analyze an image
primateface analyze image.jpg --output result.jpg

# List available models
primateface models list
```

## Tutorials

For hands-on examples, see the [tutorial notebooks](../tutorials/index.md).
