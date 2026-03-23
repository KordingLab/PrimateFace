"""Shared utilities for PrimateFace tutorial notebooks.

Provides environment checking, device detection, model path resolution,
and demo asset downloading for local notebook execution.
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional


def check_environment(require_packages: Optional[List[str]] = None) -> Dict[str, str]:
    """Check local environment and return device info.

    Verifies Python version, PyTorch, CUDA availability, and optional
    package imports. Warns on missing GPU but never raises.

    Args:
        require_packages: Optional list of package names to verify
            (e.g., ["mmdet", "mmpose", "gazelle"]).

    Returns:
        Dict with keys: python, torch, cuda, device, gpu_name, and
        per-package availability.
    """
    info: Dict[str, str] = {}

    # Python
    info["python"] = sys.version.split()[0]

    # PyTorch + CUDA
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["device"] = "cuda:0"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            info["gpu_memory_gb"] = f"{gpu_mem:.1f}"
        else:
            info["device"] = "cpu"
            info["gpu_name"] = "N/A"
            warnings.warn(
                "CUDA not available. Running on CPU -- inference will be "
                "significantly slower. A GPU is strongly recommended.",
                stacklevel=2,
            )
    except ImportError:
        info["torch"] = "NOT INSTALLED"
        info["cuda"] = "False"
        info["device"] = "cpu"
        info["gpu_name"] = "N/A"
        print("ERROR: PyTorch not found. Install it first.")

    # Optional packages
    if require_packages:
        for pkg in require_packages:
            try:
                mod = __import__(pkg)
                info[pkg] = getattr(mod, "__version__", "installed")
            except ImportError:
                info[pkg] = "NOT INSTALLED"
                print(f"WARNING: {pkg} not found. See demos/README.md for install instructions.")

    # Print summary
    print("=" * 55)
    print("PrimateFace Environment")
    print("=" * 55)
    print(f"  Python:   {info['python']}")
    print(f"  PyTorch:  {info.get('torch', '?')}")
    print(f"  CUDA:     {info.get('cuda', '?')}")
    print(f"  Device:   {info.get('device', '?')}")
    if info.get("gpu_name") and info["gpu_name"] != "N/A":
        print(f"  GPU:      {info['gpu_name']} ({info.get('gpu_memory_gb', '?')} GB)")
    if require_packages:
        for pkg in require_packages:
            status = info.get(pkg, "?")
            marker = "OK" if status != "NOT INSTALLED" else "MISSING"
            print(f"  {pkg}: {status} [{marker}]")
    print("=" * 55)

    return info


def get_device(prefer_gpu: bool = True) -> str:
    """Return the best available device string.

    Args:
        prefer_gpu: If True, returns 'cuda:0' when available.

    Returns:
        Device string ('cuda:0' or 'cpu').
    """
    import torch

    if prefer_gpu and torch.cuda.is_available():
        return "cuda:0"
    if prefer_gpu and not torch.cuda.is_available():
        warnings.warn(
            "GPU requested but CUDA not available. Falling back to CPU.",
            stacklevel=2,
        )
    return "cpu"


def resolve_model_paths(
    model_dir: Path, required: List[str]
) -> Dict[str, Path]:
    """Validate that required model files exist and return their paths.

    Args:
        model_dir: Directory where models were downloaded.
        required: List of expected filenames (e.g., ["mmdet_config.py",
            "mmdet_checkpoint.pth"]).

    Returns:
        Dict mapping filename to its absolute Path.

    Raises:
        FileNotFoundError: If any required file is missing, with
            instructions to run download_models.py.
    """
    model_dir = Path(model_dir)
    paths: Dict[str, Path] = {}
    missing: List[str] = []

    for filename in required:
        fp = model_dir / filename
        if fp.exists():
            paths[filename] = fp
            size_mb = fp.stat().st_size / (1024 * 1024)
            print(f"  Found: {filename} ({size_mb:.1f} MB)")
        else:
            missing.append(filename)

    if missing:
        raise FileNotFoundError(
            f"Missing model files in {model_dir}:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nDownload models using one of:\n"
            + f"  CLI:    python demos/download_models.py {model_dir}\n"
            + "  Python: from notebook_utils import download_models_hf\n"
            + f"          download_models_hf('{model_dir}')"
        )

    return paths


def download_demo_asset(
    gdrive_id: str, output_path: Path, description: str = ""
) -> Path:
    """Download a file from Google Drive if not already present.

    Args:
        gdrive_id: Google Drive file ID.
        output_path: Local path to save the file.
        description: Human-readable description for progress messages.

    Returns:
        Path to the downloaded (or already existing) file.
    """
    output_path = Path(output_path)
    if output_path.exists():
        print(f"  Already exists: {output_path.name}")
        return output_path

    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required to download demo assets. "
            "Install with: uv pip install gdown"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    desc_str = f" ({description})" if description else ""
    print(f"  Downloading{desc_str}: {output_path.name} ...")
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    gdown.download(url, str(output_path), quiet=False)
    return output_path


def download_models_hf(output_dir: Path) -> Dict[str, Path]:
    """Download PrimateFace models from Hugging Face Hub.

    Downloads detection and pose model checkpoints + configs to the
    specified directory. Skips files that already exist locally.

    Args:
        output_dir: Directory to save model files.

    Returns:
        Dict mapping local filenames to their Paths.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download models. "
            "Install with: uv pip install huggingface-hub"
        )

    import shutil

    # Import model constants from the centralized registry
    try:
        from primateface._model_registry import HF_REPO_ID, MODELS, LIBRARY_NAME, LIBRARY_VERSION
    except ImportError:
        _demos_dir = str(Path(__file__).resolve().parent.parent)
        if _demos_dir not in sys.path:
            sys.path.insert(0, _demos_dir)
        from model_registry import HF_REPO_ID, MODELS, LIBRARY_NAME, LIBRARY_VERSION  # type: ignore[no-redef]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    for local_name, (subfolder, hf_filename) in MODELS.items():
        local_path = output_dir / local_name
        if local_path.exists():
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"  Already exists: {local_name} ({size_mb:.1f} MB)")
            paths[local_name] = local_path
            continue

        print(f"  Downloading: {local_name} ...")
        cached = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=hf_filename,
            subfolder=subfolder,
            library_name=LIBRARY_NAME,
            library_version=LIBRARY_VERSION,
        )
        shutil.copy2(cached, str(local_path))
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"    Saved: {local_name} ({size_mb:.1f} MB)")
        paths[local_name] = local_path

    return paths


def init_models(
    det_config: str,
    det_checkpoint: str,
    pose_config: Optional[str] = None,
    pose_checkpoint: Optional[str] = None,
    device: str = "cuda:0",
):
    """Initialise MMDetection (and optionally MMPose) models."""
    from mmdet.apis import init_detector
    from mmpose.utils import adapt_mmdet_pipeline

    print("Loading MMDetection model...")
    detector = init_detector(det_config, det_checkpoint, device=device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    pose_model = None
    if pose_config and pose_checkpoint:
        from mmpose.apis import init_model as init_pose_estimator
        print("Loading MMPose model...")
        pose_model = init_pose_estimator(pose_config, pose_checkpoint, device=device)

    return detector, pose_model


def setup_publication_style() -> None:
    """Configure matplotlib for Nature-style publication figures.

    Sets large bold labels, minimal spines, colorblind-friendly defaults,
    and 300 dpi output.  Call once at the top of a notebook.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.prop_cycle": plt.cycler(color=[
            "#0072B2", "#D55E00", "#009E73", "#CC79A7",
            "#F0E442", "#56B4E9", "#E69F00", "#000000",
        ]),
    })
    print("Publication plot style configured (Nature-style).")


def save_fig(fig, name: str, out_dir: str = "figures", dpi: int = 300) -> None:
    """Save figure as PNG, SVG, and PDF.

    Args:
        fig: Matplotlib figure.
        name: Base filename (no extension).
        out_dir: Output directory.
        dpi: Resolution for PNG.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    print(f"Figure saved: {out / name} (.png/.svg/.pdf)")
