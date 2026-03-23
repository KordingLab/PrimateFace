"""Data export utilities for PrimateFace results.

Converts :class:`Face` objects to common interchange formats:
CSV, COCO JSON, DeepLabCut/Lightning Pose CSV/H5, SLEAP ``.slp``,
NWB, and pandas DataFrames.

The DLC-format functions (``to_dlc_csv``, ``to_dlc_h5``, ``from_dlc``)
are also compatible with **Lightning Pose**, which uses the same
MultiIndex CSV format (scorer / bodyparts / coords).

Example:
    >>> from primateface.io import to_csv, to_dataframe, to_sleap
    >>> faces = pf.analyze("monkey.jpg")
    >>> to_csv(faces, "results.csv", image_path="monkey.jpg")
    >>> to_sleap(faces, "results.slp")  # requires: pip install sleap-io
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .face import Face

# Dlib 68-point facial landmark connectivity (contour edges)
DLIB_68_EDGES: List[tuple] = (
    # Jaw contour: 0→1→...→16
    [(i, i + 1) for i in range(16)]
    # Right eyebrow: 17→...→21
    + [(i, i + 1) for i in range(17, 21)]
    # Left eyebrow: 22→...→26
    + [(i, i + 1) for i in range(22, 26)]
    # Nose bridge: 27→...→30
    + [(i, i + 1) for i in range(27, 30)]
    # Nose base: 31→...→35
    + [(i, i + 1) for i in range(31, 35)]
    # Right eye: 36→...→41→36 (closed)
    + [(i, i + 1) for i in range(36, 41)] + [(41, 36)]
    # Left eye: 42→...→47→42 (closed)
    + [(i, i + 1) for i in range(42, 47)] + [(47, 42)]
    # Outer mouth: 48→...→59→48 (closed)
    + [(i, i + 1) for i in range(48, 59)] + [(59, 48)]
    # Inner mouth: 60→...→67→60 (closed)
    + [(i, i + 1) for i in range(60, 67)] + [(67, 60)]
)


def to_dataframe(
    faces: List[Face],
    image_path: Optional[str] = None,
    include_kinematics: bool = True,
    include_head_pose: bool = True,
    include_quality: bool = True,
    include_symmetry: bool = True,
) -> pd.DataFrame:
    """Convert Face objects to a pandas DataFrame.

    One row per detected face, with columns for detection outputs and
    optionally all analysis features.

    Args:
        faces: List of Face objects from ``PrimateFace.analyze()``.
        image_path: Source image path (added as a column if provided).
        include_kinematics: Include kinematic feature columns.
        include_head_pose: Include yaw/pitch/roll columns.
        include_quality: Include quality metric columns.
        include_symmetry: Include symmetry columns.

    Returns:
        DataFrame with one row per face.
    """
    rows: List[Dict[str, Any]] = []
    for i, face in enumerate(faces):
        row: Dict[str, Any] = {
            "face_idx": i,
            "score": face.score,
            "bbox_x1": face.bbox[0],
            "bbox_y1": face.bbox[1],
            "bbox_x2": face.bbox[2],
            "bbox_y2": face.bbox[3],
        }

        if image_path:
            row["image"] = image_path

        # Keypoint coordinates as flat columns
        for j in range(68):
            row[f"kpt_{j}_x"] = float(face.keypoints[j, 0])
            row[f"kpt_{j}_y"] = float(face.keypoints[j, 1])
            row[f"kpt_{j}_score"] = float(face.keypoints[j, 2])

        if include_kinematics:
            for key, val in face.kinematics.items():
                row[key] = val

        if include_head_pose:
            yaw, pitch, roll = face.head_pose
            row["yaw"] = yaw
            row["pitch"] = pitch
            row["roll"] = roll

        if include_quality:
            for key, val in face.quality.items():
                row[f"quality_{key}"] = val

        if include_symmetry:
            row["symmetry_fa"] = face.symmetry
            for key, val in face.region_symmetry.items():
                row[f"symmetry_{key}"] = val

        rows.append(row)

    return pd.DataFrame(rows)


def to_csv(
    faces: List[Face],
    output_path: Union[str, Path],
    image_path: Optional[str] = None,
    **kwargs: Any,
) -> Path:
    """Export Face results to CSV.

    Args:
        faces: List of Face objects.
        output_path: Output CSV file path.
        image_path: Source image path (added as a column).
        **kwargs: Passed to :func:`to_dataframe`.

    Returns:
        Path to the written CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = to_dataframe(faces, image_path=image_path, **kwargs)
    df.to_csv(output_path, index=False)
    return output_path


def to_coco_json(
    faces: List[Face],
    output_path: Union[str, Path],
    image_path: Optional[str] = None,
    image_id: int = 1,
) -> Path:
    """Export Face results to COCO keypoints JSON format.

    Args:
        faces: List of Face objects.
        output_path: Output JSON file path.
        image_path: Source image filename (stored in the ``images`` list).
        image_id: Image ID for the COCO annotation.

    Returns:
        Path to the written JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = []
    if image_path:
        h, w = faces[0]._image.shape[:2] if faces else (0, 0)
        images.append({
            "id": image_id,
            "file_name": str(Path(image_path).name),
            "width": w,
            "height": h,
        })

    annotations = []
    for i, face in enumerate(faces):
        # COCO keypoints: [x1, y1, v1, x2, y2, v2, ...]
        kpts_flat = []
        num_visible = 0
        for j in range(68):
            x = float(face.keypoints[j, 0])
            y = float(face.keypoints[j, 1])
            s = float(face.keypoints[j, 2])
            # Map score to COCO visibility: 0=not labeled, 2=visible
            v = 2 if s > 0.3 else 0
            if v > 0:
                num_visible += 1
            kpts_flat.extend([x, y, v])

        x1, y1, x2, y2 = face.bbox.tolist()
        annotations.append({
            "id": i + 1,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO uses [x, y, w, h]
            "keypoints": kpts_flat,
            "num_keypoints": num_visible,
            "score": face.score,
            "area": (x2 - x1) * (y2 - y1),
            "iscrowd": 0,
        })

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": 1,
            "name": "primate_face",
            "keypoints": [f"point_{i}" for i in range(68)],
            "skeleton": [],
        }],
    }

    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    return output_path


def to_dlc_csv(
    faces: List[Face],
    output_path: Union[str, Path],
    scorer: str = "primateface",
) -> Path:
    """Export to DeepLabCut / Lightning Pose compatible CSV format.

    DLC/LP CSV has a MultiIndex header: scorer / bodypart / (x, y, likelihood).
    This exports single-frame data for each detected face. Output is compatible
    with both DeepLabCut and Lightning Pose prediction formats.

    Args:
        faces: List of Face objects.
        output_path: Output CSV file path.
        scorer: Name of the scorer/network (DLC convention).

    Returns:
        Path to the written CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bodyparts = [f"point_{i}" for i in range(68)]

    # Build MultiIndex columns: (scorer, bodypart, coord)
    columns = pd.MultiIndex.from_tuples(
        [(scorer, bp, coord) for bp in bodyparts for coord in ("x", "y", "likelihood")],
        names=["scorer", "bodyparts", "coords"],
    )

    rows = []
    for face in faces:
        row = []
        for j in range(68):
            row.extend([
                float(face.keypoints[j, 0]),
                float(face.keypoints[j, 1]),
                float(face.keypoints[j, 2]),
            ])
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_path)
    return output_path


def to_dlc_h5(
    faces: List[Face],
    output_path: Union[str, Path],
    scorer: str = "primateface",
) -> Path:
    """Export to DeepLabCut / Lightning Pose compatible HDF5 format.

    Args:
        faces: List of Face objects.
        output_path: Output H5 file path.
        scorer: Name of the scorer/network.

    Returns:
        Path to the written H5 file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bodyparts = [f"point_{i}" for i in range(68)]

    columns = pd.MultiIndex.from_tuples(
        [(scorer, bp, coord) for bp in bodyparts for coord in ("x", "y", "likelihood")],
        names=["scorer", "bodyparts", "coords"],
    )

    rows = []
    for face in faces:
        row = []
        for j in range(68):
            row.extend([
                float(face.keypoints[j, 0]),
                float(face.keypoints[j, 1]),
                float(face.keypoints[j, 2]),
            ])
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    df.to_hdf(output_path, key="df_with_missing", mode="w")
    return output_path


def from_coco_json(
    json_path: Union[str, Path],
) -> pd.DataFrame:
    """Load keypoints from a COCO JSON file into a DataFrame.

    Args:
        json_path: Path to COCO keypoints JSON file.

    Returns:
        DataFrame with one row per annotation, columns for bbox,
        score, and per-keypoint x/y/visibility.
    """
    with open(json_path) as f:
        coco = json.load(f)

    rows = []
    for ann in coco.get("annotations", []):
        row: Dict[str, Any] = {
            "image_id": ann.get("image_id"),
            "score": ann.get("score", 1.0),
        }
        bbox = ann.get("bbox", [0, 0, 0, 0])
        row["bbox_x1"] = bbox[0]
        row["bbox_y1"] = bbox[1]
        row["bbox_x2"] = bbox[0] + bbox[2]
        row["bbox_y2"] = bbox[1] + bbox[3]

        kpts = ann.get("keypoints", [])
        n_kpts = len(kpts) // 3
        for j in range(n_kpts):
            row[f"kpt_{j}_x"] = kpts[j * 3]
            row[f"kpt_{j}_y"] = kpts[j * 3 + 1]
            row[f"kpt_{j}_v"] = kpts[j * 3 + 2]

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SLEAP interop (requires sleap-io)
# ---------------------------------------------------------------------------

def _require_sleap_io():  # type: ignore[return]
    """Import sleap_io or raise a helpful error."""
    try:
        import sleap_io as sio
        return sio
    except ImportError:
        raise ImportError(
            "sleap-io is required for SLEAP import/export. "
            "Install with: uv pip install 'primateface[interop]'"
        )


def to_sleap(
    faces: List[Face],
    output_path: Union[str, Path],
    video_path: Optional[str] = None,
    frame_idx: int = 0,
) -> Path:
    """Export Face results to a SLEAP ``.slp`` file.

    Args:
        faces: List of Face objects.
        output_path: Output ``.slp`` file path.
        video_path: Optional video path for SLEAP video reference.
        frame_idx: Frame index for the labeled frame (default 0).

    Returns:
        Path to the written ``.slp`` file.
    """
    sio = _require_sleap_io()
    import numpy as np
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    node_names = [f"point_{i}" for i in range(68)]
    edge_names = [(node_names[s], node_names[d]) for s, d in DLIB_68_EDGES]
    skeleton = sio.Skeleton(nodes=node_names, edges=edge_names)

    video = None
    if video_path:
        video = sio.Video(filename=video_path)

    instances = []
    for face in faces:
        pts = face.keypoints[:, :2].astype(np.float64)
        scores = face.keypoints[:, 2].astype(np.float64)
        instance = sio.PredictedInstance.from_numpy(
            points=pts,
            point_confidences=scores,
            skeleton=skeleton,
        )
        instances.append(instance)

    lf = sio.LabeledFrame(
        video=video or sio.Video(filename="unknown"),
        frame_idx=frame_idx,
        instances=instances,
    )

    labels = sio.Labels(
        videos=[video] if video else [],
        skeletons=[skeleton],
        labeled_frames=[lf],
    )
    labels.save(str(output_path))
    return output_path


def from_sleap(
    slp_path: Union[str, Path],
) -> pd.DataFrame:
    """Load keypoints from a SLEAP ``.slp`` file into a DataFrame.

    Args:
        slp_path: Path to ``.slp`` file.

    Returns:
        DataFrame with columns: frame_idx, instance_idx, and
        per-node x/y/score columns.
    """
    sio = _require_sleap_io()
    labels = sio.load_file(str(slp_path))

    rows: List[Dict[str, Any]] = []
    for lf in labels.labeled_frames:
        for inst_idx, inst in enumerate(lf.instances):
            row: Dict[str, Any] = {
                "frame_idx": lf.frame_idx,
                "instance_idx": inst_idx,
            }
            pts = inst.numpy()  # (n_nodes, 2)
            for j in range(pts.shape[0]):
                row[f"kpt_{j}_x"] = float(pts[j, 0])
                row[f"kpt_{j}_y"] = float(pts[j, 1])
                score = 1.0
                if hasattr(inst, "point_confidences") and inst.point_confidences is not None:
                    score = float(inst.point_confidences[j])
                row[f"kpt_{j}_score"] = score
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# NWB interop (requires pynwb + ndx-pose)
# ---------------------------------------------------------------------------

def _require_nwb():
    """Import pynwb + ndx_pose or raise a helpful error."""
    try:
        import pynwb
        import ndx_pose
        return pynwb, ndx_pose
    except ImportError:
        raise ImportError(
            "pynwb and ndx-pose are required for NWB import/export. "
            "Install with: uv pip install 'primateface[interop]'"
        )


def to_nwb(
    faces: List[Face],
    output_path: Union[str, Path],
    session_description: str = "PrimateFace pose estimation",
    timestamps: Optional[List[float]] = None,
) -> Path:
    """Export Face results to an NWB file with ndx-pose PoseEstimation.

    Creates one ``PoseEstimationSeries`` per landmark (68 total) inside
    a ``PoseEstimation`` container.

    Args:
        faces: List of Face objects (treated as sequential frames).
        output_path: Output ``.nwb`` file path.
        session_description: NWB session description.
        timestamps: Per-face timestamps in seconds. Defaults to
            ``[0.0, 1.0, 2.0, ...]`` if not provided.

    Returns:
        Path to the written ``.nwb`` file.
    """
    pynwb, ndx_pose = _require_nwb()
    import numpy as np
    from datetime import datetime
    from uuid import uuid4

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = len(faces)
    if timestamps is None:
        ts = np.arange(n_frames, dtype=np.float64)
    else:
        ts = np.array(timestamps, dtype=np.float64)

    all_xy = np.zeros((n_frames, 68, 2), dtype=np.float64)
    all_conf = np.zeros((n_frames, 68), dtype=np.float64)
    for i, face in enumerate(faces):
        all_xy[i] = face.keypoints[:, :2]
        all_conf[i] = face.keypoints[:, 2]

    nodes = [ndx_pose.Node(name=f"point_{j}") for j in range(68)]
    edges = [ndx_pose.Edge(source=nodes[s], target=nodes[d]) for s, d in DLIB_68_EDGES]
    skeleton = ndx_pose.Skeleton(
        name="primate_face_68",
        nodes=nodes,
        edges=edges,
    )

    pose_series = []
    for j in range(68):
        series = ndx_pose.PoseEstimationSeries(
            name=f"point_{j}",
            description=f"Face landmark {j}",
            data=all_xy[:, j, :],
            timestamps=ts,
            confidence=all_conf[:, j],
            reference_frame="pixel coordinates",
        )
        pose_series.append(series)

    pose_est = ndx_pose.PoseEstimation(
        name="primateface_pose",
        description="68-point facial landmarks from PrimateFace",
        pose_estimation_series=pose_series,
        skeleton=skeleton,
    )

    nwbfile = pynwb.NWBFile(
        session_description=session_description,
        identifier=str(uuid4()),
        session_start_time=datetime.now(),
    )
    behavior = nwbfile.create_processing_module(
        name="behavior", description="Pose estimation data"
    )
    behavior.add(pose_est)

    with pynwb.NWBHDF5IO(str(output_path), "w") as io_nwb:
        io_nwb.write(nwbfile)

    return output_path


def from_nwb(
    nwb_path: Union[str, Path],
    pose_estimation_name: Optional[str] = None,
) -> pd.DataFrame:
    """Load keypoints from an NWB file into a DataFrame.

    Args:
        nwb_path: Path to ``.nwb`` file.
        pose_estimation_name: Name of the ``PoseEstimation`` container.
            If *None*, uses the first one found.

    Returns:
        DataFrame with columns: timestamp and per-keypoint x/y/confidence.
    """
    pynwb, _ = _require_nwb()

    with pynwb.NWBHDF5IO(str(nwb_path), "r") as io_nwb:
        nwbfile = io_nwb.read()

        pose_est = None
        for module_name in nwbfile.processing:
            module = nwbfile.processing[module_name]
            for container_name in module.data_interfaces:
                if pose_estimation_name and container_name != pose_estimation_name:
                    continue
                pose_est = module.data_interfaces[container_name]
                break
            if pose_est is not None:
                break

        if pose_est is None:
            raise ValueError("No PoseEstimation container found in NWB file")

        series_names = list(pose_est.pose_estimation_series.keys())
        if not series_names:
            return pd.DataFrame()

        first_series = pose_est.pose_estimation_series[series_names[0]]
        timestamps = first_series.timestamps[:]
        n_frames = len(timestamps)

        rows: List[Dict[str, Any]] = []
        for frame_idx in range(n_frames):
            row: Dict[str, Any] = {"timestamp": float(timestamps[frame_idx])}
            for j, name in enumerate(series_names):
                series = pose_est.pose_estimation_series[name]
                xy = series.data[frame_idx]
                conf = series.confidence[frame_idx]
                row[f"kpt_{j}_x"] = float(xy[0])
                row[f"kpt_{j}_y"] = float(xy[1])
                row[f"kpt_{j}_confidence"] = float(conf)
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DeepLabCut import
# ---------------------------------------------------------------------------

def from_dlc(
    h5_or_csv_path: Union[str, Path],
) -> pd.DataFrame:
    """Load keypoints from a DeepLabCut or Lightning Pose H5/CSV file.

    Args:
        h5_or_csv_path: Path to DLC/LP predictions (``.h5`` or ``.csv``).

    Returns:
        DLC-format DataFrame with MultiIndex columns
        (scorer / bodyparts / coords).
    """
    path = Path(h5_or_csv_path)
    if path.suffix.lower() in (".h5", ".hdf5"):
        return pd.read_hdf(str(path), "df_with_missing")
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(str(path), header=[0, 1, 2], index_col=0)
    else:
        raise ValueError(
            f"Unsupported file extension: {path.suffix}. "
            "Expected .h5, .hdf5, or .csv"
        )
