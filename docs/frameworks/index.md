# Framework Integration

PrimateFace's core pipeline uses **MMDetection** (Cascade R-CNN) for face detection and **MMPose** (HRNet/ViTPose) for 68-point landmark estimation. Results can be exported to other frameworks via `primateface.io`.

## Core Pipeline

The default PrimateFace pipeline:

```python
import primateface
pf = primateface.PrimateFace()  # auto-downloads MMDet + MMPose models
faces = pf.analyze("monkey.jpg")
```

See the [User Guide](../user-guide/core-workflows/demos.md) for detailed usage.

## Exporting to Other Frameworks

PrimateFace results can be exported to formats compatible with:

| Framework | Export Function | Format |
|-----------|----------------|--------|
| **DeepLabCut / Lightning Pose** | `primateface.io.to_dlc_csv()` | MultiIndex CSV (scorer/bodypart/coords) |
| **SLEAP** | `primateface.io.to_sleap()` | `.slp` file with 68-point skeleton |
| **NWB (Neurodata Without Borders)** | `primateface.io.to_nwb()` | `.nwb` with ndx-pose PoseEstimation |
| **COCO** | `primateface.io.to_coco_json()` | Standard COCO keypoints JSON |

```python
from primateface.io import to_dlc_csv, to_sleap, to_nwb

# Export to DeepLabCut format (also works with Lightning Pose)
to_dlc_csv(faces, "predictions.csv")

# Export to SLEAP .slp file
to_sleap(faces, "predictions.slp")  # requires: pip install sleap-io

# Export to NWB
to_nwb(faces, "session.nwb")  # requires: pip install pynwb ndx-pose
```

## Importing from Other Frameworks

```python
from primateface.io import from_dlc, from_sleap, from_nwb

df = from_dlc("dlc_predictions.h5")     # DLC/LP H5 or CSV
df = from_sleap("predictions.slp")       # SLEAP .slp
df = from_nwb("session.nwb")            # NWB file
```

## Framework-Specific Guides

- [MMDetection & MMPose](../user-guide/framework-integration/mmpose-mmdetection.md)
- [DeepLabCut](../user-guide/framework-integration/deeplabcut.md)
- [SLEAP](../user-guide/framework-integration/sleap.md)
- [Ultralytics (YOLO)](../user-guide/framework-integration/ultralytics.md)
