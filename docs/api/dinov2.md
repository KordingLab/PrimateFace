# DINOv2 API - Feature Extraction & Analysis

The `dinov2` module provides self-supervised feature extraction using DINOv2 models, enabling data-efficient subset selection and visualization for primate face analysis.

## Quick Reference

```python
from dinov2.core import DINOv2Extractor
from dinov2.visualization import UMAPVisualizer, PatchVisualizer
from dinov2.selection import DiverseImageSelector
```

## Main Classes

### DINOv2Extractor

Core feature extraction interface for DINOv2 models.

```python
extractor = DINOv2Extractor(
    model_name="facebook/dinov2-base",
    device="cuda",
    batch_size=32,
    num_workers=4
)

# Extract from directory
embeddings, image_ids = extractor.extract_from_directory("./images/")

# Extract from CSV
embeddings, ids = extractor.extract_from_csv("data.csv")

# Save embeddings
extractor.save_embeddings(embeddings, ids, "features.pt")
```

**Model Options:**
- `facebook/dinov2-small` - 21M params, 384-dim (fast)
- `facebook/dinov2-base` - 86M params, 768-dim (recommended)
- `facebook/dinov2-large` - 300M params, 1024-dim (accurate)
- `facebook/dinov2-giant` - 1.1B params, 1536-dim (best)

### UMAPVisualizer

Dimensionality reduction and visualization of embeddings.

```python
visualizer = UMAPVisualizer(
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine"
)

# Compute UMAP projection
projection = visualizer.fit_transform(embeddings)

# Cluster embeddings
labels = visualizer.cluster(n_clusters=100)

# Generate visualizations
visualizer.plot_static("umap.png", dot_size=50)
visualizer.plot_interactive("umap.html", show_images=True)
```

### DiverseImageSelector

Active learning strategies for subset selection.

```python
selector = DiverseImageSelector(strategy="hybrid")

# Select diverse subset
indices = selector.select(
    embeddings,
    n_samples=1000,
    n_clusters=100,
    fps_ratio=0.5
)

# Get selected image IDs
selected_ids = [image_ids[i] for i in indices]
```

**Strategies:**
- `random` - Random sampling
- `cluster` - K-means based selection
- `fps` - Farthest point sampling
- `hybrid` - Combined approach (best)

## Common Usage Patterns

### Complete Pipeline

```python
from dinov2.core import DINOv2Extractor
from dinov2.visualization import UMAPVisualizer
from dinov2.selection import DiverseImageSelector

# 1. Extract features
extractor = DINOv2Extractor()
embeddings, ids = extractor.extract_from_directory("images/")

# 2. Visualize
visualizer = UMAPVisualizer()
visualizer.fit_transform(embeddings)
visualizer.cluster(n_clusters=50)
visualizer.plot_interactive("analysis.html")

# 3. Select subset
selector = DiverseImageSelector(strategy="hybrid")
indices = selector.select(embeddings, n_samples=500)
print(f"Selected {len(indices)} diverse images")
```

### Attention Visualization

```python
from dinov2.visualization import PatchVisualizer

visualizer = PatchVisualizer(model_name="facebook/dinov2-base")

# Visualize attention for single image
visualizer.visualize_attention(
    "primate.jpg",
    output_path="attention.png",
    layer=-1  # Last layer
)
```

### Batch Processing

```python
from pathlib import Path

# Process multiple directories
directories = ["dataset1/", "dataset2/", "dataset3/"]
all_embeddings = []

for dir_path in directories:
    embeddings, ids = extractor.extract_from_directory(dir_path)
    all_embeddings.append({
        "dir": dir_path,
        "embeddings": embeddings,
        "ids": ids
    })
```

## CLI Interface

The module provides a comprehensive CLI:

```bash
# Extract features
python -m dinov2.dinov2_cli extract --input images/ --output features.pt

# Visualize
python -m dinov2.dinov2_cli visualize --embeddings features.pt --output umap.html --interactive

# Select subset
python -m dinov2.dinov2_cli select --embeddings features.pt --n 1000 --strategy hybrid

# Generate patches
python -m dinov2.dinov2_cli patches --images images/ --output patches/
```

## Output Formats

### Embeddings File
```python
{
    'embeddings': torch.Tensor,  # Shape: (N, feature_dim)
    'image_ids': List[str],      # Length: N
    'model_name': str,           # e.g., "facebook/dinov2-base"
    'timestamp': str,            # ISO format
    'metadata': dict             # Additional info
}
```

### Selection Output
```text
# selected_images.txt
image_001.jpg
image_042.jpg
image_087.jpg
...
```

## Performance Considerations

### Memory Management

```python
# For large datasets
def process_in_chunks(image_dir, chunk_size=5000):
    extractor = DINOv2Extractor(batch_size=16)
    
    for chunk in chunk_directory(image_dir, chunk_size):
        embeddings = extractor.extract_from_list(chunk)
        torch.save(embeddings, f"chunk_{i}.pt")
        torch.cuda.empty_cache()
```

### Speed Optimization

```python
# Optimize for speed
extractor = DINOv2Extractor(
    model_name="facebook/dinov2-small",  # Smaller model
    batch_size=64,  # Larger batches
    num_workers=8   # More workers
)
```

## Integration Examples

### With Scikit-learn

```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Extract features
embeddings, _ = extractor.extract_from_directory("images/")

# Reduce dimensions with PCA
pca = PCA(n_components=50)
reduced = pca.fit_transform(embeddings.numpy())

# Cluster
kmeans = KMeans(n_clusters=10)
labels = kmeans.fit_predict(reduced)
```

### With PyTorch Dataset

```python
from torch.utils.data import DataLoader
from dinov2.core import ImageDataset

dataset = ImageDataset("images/", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

for batch in dataloader:
    features = extractor.extract_batch(batch)
```

## Detailed Documentation

For comprehensive technical documentation including:
- Full parameter descriptions
- Implementation details
- Advanced usage patterns
- Custom strategies
- Testing guidelines

See: Internals section below

## Related Modules

- [Demos](demos.md) - Detection and pose estimation
- [GUI Tools](gui.md) - Annotation with DINOv2
- [Evaluation](evaluation.md) - Model comparison

## Scientific Context

This module implements strategies from the PrimateFace paper:
- **Data efficiency**: DINOv2-guided vs random sampling
- **Clustering**: Natural grouping of primate faces
- **Attention**: Focus on facial features

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Slow extraction**: Increase batch size or use smaller model
3. **Import errors**: Install with `pip install -e ".[dinov2]"`
4. **UMAP errors**: Install `umap-learn` package

### Getting Help

- Check Internals section below
- See [GitHub Issues](https://github.com/KordingLab/PrimateFace/issues)
- Review unit tests in `test_dinov2.py`

## Internals

Implementation details for contributors and advanced users.

- Core extractor: `dinov2/core.py:DINOv2Extractor`
  - Loads DINOv2 backbones (small/base/large/giant) via Hugging Face.
  - Handles preprocessing, batching, device placement, and no-grad inference.
  - Supports directory/CSV inputs and persistence with metadata.

- CLI tooling: `dinov2/dinov2_cli.py`
  - `extract`: images/CSV → `embeddings.pt` (tensor + IDs + metadata).
  - `visualize`: UMAP projection and clustering to SVG/HTML; optional image hover.
  - `select`: `random|cluster|fps|hybrid` strategies; writes selected IDs/paths.

- Selection algorithms: `dinov2/selection.py:DiverseImageSelector`
  - `cluster`: k-means then proportional sampling per cluster.
  - `fps`: farthest-point sampling in embedding space.
  - `hybrid`: cluster centers + intra-cluster FPS for coverage and diversity.
  - Includes helpers to save selections and compute diversity/coverage metrics.

- Visualization utilities: `dinov2/visualization.py:UMAPVisualizer` and `PatchVisualizer`
  - UMAP fit/transform with configurable neighbors/dist metrics; static/interactive plots.
  - Attention/patch visualization for qualitative inspection.

- Constants: `dinov2/constants.py`
  - Default model names, feature dims, and recommended batch sizes.

- Performance tips
  - Prefer `facebook/dinov2-base` as default; reduce `--batch-size` on 8GB GPUs.
  - Increase `--num-workers` for I/O-bound workloads; pre-resize large images.
  - For massive datasets, stream in chunks and persist partial outputs.

- Troubleshooting
  - OOM → smaller model or batch size; consider CPU.
  - Slow extraction → smaller backbone, more workers, fewer augmentations.
  - Import errors → ensure extras installed: `uv pip install -e ".[dinov2]"`.

Related
- User guide workflow: `docs/user-guide/core-workflows/dinov2.md`
- Quick start: repository `README.md` and `dinov2/README.md`

## Next Steps

For practical workflows and step-by-step guides, see:
- [DINOv2-Guided Selection Guide](../user-guide/core-workflows/dinov2.md) - Complete workflows
- [Lemur Video Timestamping](https://github.com/KordingLab/PrimateFace/blob/main/demos/notebooks/lemur_video_timestamping.ipynb) - Time series analysis
- [Macaque Face Recognition](https://github.com/KordingLab/PrimateFace/blob/main/demos/notebooks/macaque_face_recognition.ipynb) - Face recognition
- [Pseudo-labeling Guide](../user-guide/core-workflows/pseudo-labeling.md) - Annotation workflows
