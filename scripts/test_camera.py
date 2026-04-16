from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vision.camera_source import open_camera_source


def _run_probe(command: list[str], timeout_sec: float) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except FileNotFoundError:
        return False, f"{command[0]} not installed"
    except subprocess.TimeoutExpired:
        return False, "timed out"

    output = (result.stdout + result.stderr).strip()
    if result.returncode == 0:
        return True, output
    return False, output or f"exit code {result.returncode}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone camera diagnostic")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=640, help="Requested width")
    parser.add_argument("--height", type=int, default=480, help="Requested height")
    parser.add_argument("--fps", type=int, default=30, help="Requested fps")
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="First-frame timeout in seconds",
    )
    parser.add_argument(
        "--output",
        default="/tmp/walle-camera-test.jpg",
        help="Where to save the first captured frame",
    )
    parser.add_argument(
        "--skip-probes",
        action="store_true",
        help="Skip v4l2/gstreamer fallback probes",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    print(
        f"Trying shared camera source on index {args.camera} "
        f"({args.width}x{args.height} @ {args.fps}fps) ..."
    )
    opened = open_camera_source(
        args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        first_frame_timeout_sec=args.timeout,
    )

    if opened.source is not None and opened.first_frame is not None:
        import cv2  # noqa: PLC0415

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), opened.first_frame)
        print(f"Camera OK via {opened.backend_name}")
        print(f"Saved frame to {out_path}")
        print(f"Frame shape: {opened.first_frame.shape}")
        opened.source.release()
        return 0

    print(f"Camera open failed: {opened.error or 'unknown error'}")
    if args.skip_probes:
        return 1

    device = f"/dev/video{args.camera}"
    print()
    print("Extra probes:")

    ok, output = _run_probe(
        ["v4l2-ctl", "--device", device, "--list-formats-ext"],
        timeout_sec=max(args.timeout, 5.0),
    )
    print(f"- v4l2 list formats: {'OK' if ok else 'FAIL'}")
    if output:
        print(output)

    ok, output = _run_probe(
        [
            "gst-launch-1.0",
            "-q",
            "v4l2src",
            f"device={device}",
            "num-buffers=1",
            "!",
            "fakesink",
        ],
        timeout_sec=max(args.timeout, 5.0),
    )
    print(f"- gst-launch fakesink probe: {'OK' if ok else 'FAIL'}")
    if output:
        print(output)

    with tempfile.NamedTemporaryFile(suffix=".mjpg", delete=False) as handle:
        stream_path = handle.name

    ok, output = _run_probe(
        [
            "v4l2-ctl",
            f"--device={device}",
            (
                f"--set-fmt-video=width={args.width},height={args.height},"
                "pixelformat=MJPG"
            ),
            "--stream-mmap=3",
            "--stream-count=1",
            f"--stream-to={stream_path}",
        ],
        timeout_sec=max(args.timeout, 8.0),
    )
    size = Path(stream_path).stat().st_size if Path(stream_path).exists() else 0
    print(f"- v4l2 one-frame probe: {'OK' if ok and size > 0 else 'FAIL'}")
    print(f"  output file: {stream_path} ({size} bytes)")
    if output:
        print(output)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
