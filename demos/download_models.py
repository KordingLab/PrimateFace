#!/usr/bin/env python
"""Download PrimateFace models from Hugging Face Hub.

Models include detection (MMDetection) and pose estimation (MMPose) checkpoints
hosted at https://huggingface.co/fparodi/primateface-models.
"""

import shutil
import sys
import argparse
from pathlib import Path

try:
    from primateface._model_registry import (
        HF_REPO_ID, HF_REPO_URL, MODELS, LIBRARY_NAME, LIBRARY_VERSION,
    )
except ImportError:
    # Fallback for standalone script execution
    from model_registry import (  # type: ignore[no-redef]
        HF_REPO_ID, HF_REPO_URL, MODELS, LIBRARY_NAME, LIBRARY_VERSION,
    )


def download_models(output_dir: str = ".", force: bool = False) -> bool:
    """Download all PrimateFace models to the specified directory.

    Args:
        output_dir: Directory to save model files.
        force: If True, re-download even if files already exist.

    Returns:
        True if all downloads succeeded.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "ERROR: huggingface_hub is required to download models.\n"
            "Install with:  uv pip install huggingface-hub\n"
            "  or:          pip install huggingface-hub",
            file=sys.stderr,
        )
        return False

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading PrimateFace models to: {output_path.absolute()}")
    print(f"Source: {HF_REPO_URL}")
    print()

    for local_name, (subfolder, hf_filename) in MODELS.items():
        output_file = output_path / local_name
        model_type = "MMDetection" if "mmdet" in local_name else "MMPose"
        file_type = "config" if local_name.endswith(".py") else "checkpoint"

        if output_file.exists() and not force:
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"  Already exists: {local_name} ({size_mb:.1f} MB)")
            continue

        print(f"  Downloading {model_type} {file_type}: {local_name} ...")
        cached_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=hf_filename,
            subfolder=subfolder,
            library_name=LIBRARY_NAME,
            library_version=LIBRARY_VERSION,
        )

        # Copy from HF cache to the requested output directory
        shutil.copy2(cached_path, str(output_file))

        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"    Saved: {local_name} ({size_mb:.1f} MB)")

    print()
    print("All models downloaded successfully!")
    print("\nFiles:")
    for local_name in MODELS:
        file_path = output_path / local_name
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  - {local_name} ({size_mb:.1f} MB)")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download PrimateFace models from Hugging Face Hub"
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=".",
        help="Directory to save models (default: current directory)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )

    args = parser.parse_args()

    try:
        success = download_models(args.output_dir, force=args.force)
        return 0 if success else 1
    except Exception as e:
        print(f"Error downloading models: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
