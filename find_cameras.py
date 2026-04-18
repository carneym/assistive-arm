#!/usr/bin/env python3
"""
Find all cameras connected to the system.

Scans OpenCV indices and /dev/video* devices, prints resolution/FPS info,
and saves a test snapshot from each detected camera.

Usage:
    python find_cameras.py
    python find_cameras.py --max-index 30
    python find_cameras.py --save-snapshots
    python find_cameras.py --preview       # show live preview windows (press q to close)
"""

import argparse
import platform
import sys
import time
from pathlib import Path

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV not installed.  Run:  pip install opencv-python")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not installed.  Run:  pip install numpy")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _open_camera(index_or_path) -> cv2.VideoCapture | None:
    """Try to open a camera; return None if it fails or has no stream."""
    cap = cv2.VideoCapture(index_or_path)
    if not cap.isOpened():
        cap.release()
        return None
    # Read one frame to confirm it's a real camera, not a ghost index
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def _camera_info(cap: cv2.VideoCapture, identifier) -> dict:
    """Extract metadata from an open VideoCapture."""
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    backend = cap.getBackendName() if hasattr(cap, "getBackendName") else "unknown"
    return {
        "identifier": identifier,
        "width": width,
        "height": height,
        "fps": round(fps, 1),
        "backend": backend,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Camera discovery
# ──────────────────────────────────────────────────────────────────────────────

def find_opencv_cameras(max_index: int = 20) -> list[dict]:
    """
    Scan integer indices 0..max_index-1.
    On Linux also try /dev/video* paths (avoids gaps in the index space).
    Returns list of camera info dicts.
    """
    found = []
    seen_indices = set()

    # On Linux, enumerate /dev/video* first to get stable device paths
    if platform.system() == "Linux":
        dev_paths = sorted(Path("/dev").glob("video*"), key=lambda p: p.name)
        for path in dev_paths:
            cap = _open_camera(str(path))
            if cap is not None:
                info = _camera_info(cap, str(path))
                # Also record the numeric index if we can derive it
                try:
                    idx = int(path.name.replace("video", ""))
                    info["index"] = idx
                    seen_indices.add(idx)
                except ValueError:
                    info["index"] = None
                found.append(info)
                cap.release()

    # Scan integer indices (covers macOS and fills gaps on Linux)
    print(f"Scanning indices 0..{max_index - 1} ...", end=" ", flush=True)
    for idx in range(max_index):
        if idx in seen_indices:
            continue
        cap = _open_camera(idx)
        if cap is not None:
            info = _camera_info(cap, idx)
            info["index"] = idx
            found.append(info)
            cap.release()
        # Brief progress indicator every 5 indices
        if idx % 5 == 4:
            print(".", end="", flush=True)
    print()

    return found


# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────

def print_camera_table(cameras: list[dict]):
    if not cameras:
        print("\n  No cameras detected.\n")
        return

    print(f"\n{'─' * 60}")
    print(f"  Found {len(cameras)} camera(s)")
    print(f"{'─' * 60}")
    header = f"  {'Identifier':<28} {'Res':>12} {'FPS':>6}  Backend"
    print(header)
    print(f"  {'-' * 56}")
    for cam in cameras:
        ident = str(cam["identifier"])
        res   = f"{cam['width']}x{cam['height']}"
        fps   = str(cam["fps"])
        backend = cam.get("backend", "")
        print(f"  {ident:<28} {res:>12} {fps:>6}  {backend}")
    print(f"{'─' * 60}\n")


def save_snapshots(cameras: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving snapshots to {output_dir}/")
    for cam in cameras:
        ident = cam["identifier"]
        cap = _open_camera(ident)
        if cap is None:
            print(f"  [SKIP] Could not re-open {ident}")
            continue
        # Grab a few frames to let auto-exposure settle
        for _ in range(5):
            cap.read()
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print(f"  [SKIP] No frame from {ident}")
            continue
        safe_name = str(ident).replace("/", "_").replace("\\", "_")
        path = output_dir / f"camera_{safe_name}.png"
        cv2.imwrite(str(path), frame)
        print(f"  Saved  {path}")


def show_previews(cameras: list[dict]):
    """Open a preview window for each camera. Press 'q' to quit all."""
    if not cameras:
        return
    caps = []
    for cam in cameras:
        cap = _open_camera(cam["identifier"])
        if cap:
            caps.append((str(cam["identifier"]), cap))

    if not caps:
        return

    print(f"Showing {len(caps)} preview window(s). Press 'q' to close.")
    try:
        while True:
            for win_name, cap in caps:
                ok, frame = cap.read()
                if ok:
                    cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        for _, cap in caps:
            cap.release()
        cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detect cameras connected to this system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--max-index", type=int, default=20,
        help="Highest OpenCV index to probe (default: 20)",
    )
    parser.add_argument(
        "--save-snapshots", action="store_true",
        help="Save one PNG snapshot per camera to outputs/camera_snapshots/",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/camera_snapshots"),
        help="Directory for snapshots (default: outputs/camera_snapshots)",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Show live preview windows (press q to close)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Camera Finder")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print("=" * 60)

    cameras = find_opencv_cameras(max_index=args.max_index)
    print_camera_table(cameras)

    if not cameras:
        sys.exit(0)

    # Print ready-to-use lerobot config snippets
    print("  Lerobot config snippets (copy into --robot.cameras):")
    print()
    for cam in cameras:
        ident = cam["identifier"]
        w, h, fps = cam["width"], cam["height"], cam["fps"]
        fps_int = int(fps) if fps > 0 else 30
        print(f"    front: {{type: opencv, index_or_path: {ident!r}, width: {w}, height: {h}, fps: {fps_int}}}")
    print()

    if args.save_snapshots:
        save_snapshots(cameras, args.output_dir)
        print()

    if args.preview:
        show_previews(cameras)


if __name__ == "__main__":
    main()
