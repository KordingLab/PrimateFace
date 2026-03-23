"""PrimateFace command-line interface.

Usage:
    primateface analyze image.jpg
    primateface analyze image.jpg --output result.jpg
    primateface analyze ./images/ --output results/
    primateface models list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_analyze(args: argparse.Namespace) -> int:
    """Run face analysis on image(s)."""
    from primateface import PrimateFace

    pf = PrimateFace(
        device=args.device,
        pose_model=args.pose_model,
        det_threshold=args.det_threshold,
    )

    input_path = Path(args.input)

    if input_path.is_dir():
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        images = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in extensions
        )
        if not images:
            print(f"No images found in {input_path}", file=sys.stderr)
            return 1

        output_dir = Path(args.output) if args.output else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        for img_path in images:
            faces = pf.analyze(str(img_path))
            print(f"{img_path.name}: {len(faces)} face(s)")
            for i, face in enumerate(faces):
                print(f"  [{i}] {face}")
            if output_dir and faces:
                out_path = output_dir / f"viz_{img_path.name}"
                PrimateFace.draw(faces, str(img_path), output=str(out_path))
                print(f"  -> {out_path}")
    else:
        faces = pf.analyze(str(input_path))
        print(f"{input_path.name}: {len(faces)} face(s)")
        for i, face in enumerate(faces):
            print(f"  [{i}] {face}")
            kin = face.kinematics
            print(f"       head_pose={face.head_pose}")
            print(f"       symmetry={face.symmetry:.4f}")
            print(f"       mouth_aperture={kin['mouth_aperture']:.4f}")

        if args.output and faces:
            PrimateFace.draw(faces, str(input_path), output=args.output)
            print(f"Visualization saved to {args.output}")

    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    """List available models."""
    from ._model_registry import MODEL_ENTRIES, HF_REPO_URL

    print(f"Models hosted at: {HF_REPO_URL}\n")
    print(f"{'Task':<12} {'Variant':<10} {'Type':<12} {'Description'}")
    print("-" * 70)
    for entry in MODEL_ENTRIES:
        print(
            f"{entry.task:<12} {entry.variant:<10} {entry.file_type:<12} "
            f"{entry.description}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="primateface",
        description="PrimateFace: Cross-species primate face analysis",
    )
    subparsers = parser.add_subparsers(dest="command")

    # -- analyze --
    p_analyze = subparsers.add_parser(
        "analyze", help="Detect and analyze primate faces in images"
    )
    p_analyze.add_argument("input", help="Image path or directory")
    p_analyze.add_argument(
        "--output", "-o", help="Output path for visualization (file or directory)"
    )
    p_analyze.add_argument(
        "--pose-model", default="hrnet",
        choices=["hrnet", "vitpose"],
        help="Pose model backend (default: hrnet)",
    )
    p_analyze.add_argument(
        "--device", default=None, help="Device (e.g. cuda:0, cpu)"
    )
    p_analyze.add_argument(
        "--det-threshold", type=float, default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )

    # -- models --
    p_models = subparsers.add_parser(
        "models", help="List available models"
    )
    p_models.add_argument(
        "action", nargs="?", default="list",
        choices=["list"], help="Action (default: list)",
    )

    args = parser.parse_args(argv)

    if args.command == "analyze":
        return _cmd_analyze(args)
    elif args.command == "models":
        return _cmd_models(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
