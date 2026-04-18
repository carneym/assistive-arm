#!/usr/bin/env python3
"""
SO-101 Teleoperation Practice Script

Guides you through setting up and running leader→follower teleoperation using
the lerobot library.  Also supports a lightweight "simulation" mode using only
the arm_controller module when lerobot is not available.

Usage — interactive wizard:
    python teleoperate.py

Usage — direct (skip prompts):
    lerobot_venv/bin/python teleoperate.py \\
        --leader-port /dev/tty.usbmodem58760431551 \\
        --follower-port /dev/tty.usbmodem58760431541 \\
        --camera-index 0 \\
        --display

Usage — simulation mode (no hardware needed):
    python teleoperate.py --sim
"""

import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Port discovery helpers (no lerobot dependency)
# ──────────────────────────────────────────────────────────────────────────────

def list_serial_ports() -> list[str]:
    """Return candidate serial/USB ports on this system."""
    try:
        from serial.tools import list_ports
        return sorted(p.device for p in list_ports.comports())
    except ImportError:
        pass
    # Fallback: glob /dev/
    if platform.system() in ("Linux", "Darwin"):
        candidates = []
        for pattern in ("ttyUSB*", "ttyACM*", "tty.usbmodem*", "tty.usbserial*",
                        "tty.wchusbserial*"):
            candidates.extend(str(p) for p in Path("/dev").glob(pattern))
        return sorted(set(candidates))
    return []


def auto_find_arm_port(baudrate: int = 1000000) -> str | None:
    """Ping servo ID-1 on each candidate port; return first that responds."""
    import time
    ports = list_serial_ports()
    for port in ports:
        try:
            import serial
            ser = serial.Serial(port=port, baudrate=baudrate, timeout=0.1)
            ping = bytes([0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB])
            ser.write(ping)
            ser.flush()
            time.sleep(0.02)
            resp = ser.read(10)
            ser.close()
            if len(resp) >= 4 and resp[0] == 0xFF and resp[1] == 0xFF:
                return port
        except Exception:
            continue
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Interactive wizard
# ──────────────────────────────────────────────────────────────────────────────

def _prompt(msg: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{msg}{suffix}: ").strip()
    return val if val else default


def _pick_port(label: str, ports: list[str]) -> str:
    """Let user pick from a numbered list or type their own."""
    if not ports:
        return _prompt(f"Enter {label} port")
    print(f"\n  Available ports:")
    for i, p in enumerate(ports):
        print(f"    {i}: {p}")
    choice = _prompt(f"Pick {label} port (number or path)", default=ports[0])
    if choice.isdigit() and int(choice) < len(ports):
        return ports[int(choice)]
    return choice


def wizard() -> dict:
    """Interactive setup wizard. Returns config dict."""
    print("\n" + "=" * 62)
    print("  SO-101 Teleoperation Setup Wizard")
    print("=" * 62)

    ports = list_serial_ports()
    if ports:
        print(f"\n  Detected serial ports: {', '.join(ports)}")
    else:
        print("\n  No serial ports detected — is the arm plugged in?")

    print("\n  Step 1: Leader arm (the arm you hold and move by hand)")
    leader_port = _pick_port("leader", ports)

    # Remove leader port from candidates for follower
    follower_candidates = [p for p in ports if p != leader_port]
    print("\n  Step 2: Follower arm (the arm that mirrors the leader)")
    follower_port = _pick_port("follower", follower_candidates)

    print("\n  Step 3: Camera (optional — leave blank to skip)")
    cam_raw = _prompt("Camera index or device path (e.g. 0, 1, /dev/video0)", default="")

    print("\n  Step 4: Options")
    display_raw = _prompt("Show motor values on screen? (y/n)", default="y")
    fps_raw = _prompt("Target FPS", default="30")
    duration_raw = _prompt("Run for N seconds (leave blank = run until Ctrl-C)", default="")

    return {
        "leader_port": leader_port,
        "follower_port": follower_port,
        "camera": cam_raw if cam_raw else None,
        "display": display_raw.lower().startswith("y"),
        "fps": int(fps_raw) if fps_raw.isdigit() else 30,
        "duration": float(duration_raw) if duration_raw else None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# lerobot command builder
# ──────────────────────────────────────────────────────────────────────────────

def build_lerobot_command(cfg: dict) -> list[str]:
    """
    Build the `python -m lerobot.scripts.lerobot_teleoperate` command.
    Expects lerobot source to be in ./lerobot/.
    """
    # Prefer lerobot_venv python if it exists
    venv_python = Path("lerobot_venv/bin/python")
    python = str(venv_python) if venv_python.exists() else sys.executable

    cmd = [
        python, "-m", "lerobot.scripts.lerobot_teleoperate",
        "--robot.type=so101_follower",
        f"--robot.port={cfg['follower_port']}",
        "--robot.id=follower",
        "--teleop.type=so101_leader",
        f"--teleop.port={cfg['leader_port']}",
        "--teleop.id=leader",
        f"--fps={cfg['fps']}",
        f"--display_data={'true' if cfg['display'] else 'false'}",
    ]

    if cfg.get("camera") is not None:
        cam = cfg["camera"]
        # Determine if it's an integer index or a path
        try:
            idx = int(cam)
            cam_repr = idx
        except ValueError:
            cam_repr = f'"{cam}"'
        cmd.append(
            f"--robot.cameras={{front: {{type: opencv, index_or_path: {cam_repr}, "
            f"width: 640, height: 480, fps: {cfg['fps']}}}}}"
        )

    if cfg.get("duration") is not None:
        cmd.append(f"--teleop_time_s={cfg['duration']}")

    return cmd


def run_lerobot(cfg: dict):
    """Launch lerobot teleoperation subprocess."""
    cmd = build_lerobot_command(cfg)

    print("\n" + "=" * 62)
    print("  Starting lerobot teleoperation")
    print("=" * 62)
    print(f"\n  Command:\n    " + " \\\n    ".join(cmd))
    print("\n  Press Ctrl-C to stop.\n")

    # Set PYTHONPATH so lerobot source is importable
    import os
    env = os.environ.copy()
    lerobot_src = str(Path("lerobot/src").resolve())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{lerobot_src}:{existing}" if existing else lerobot_src

    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n  Stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n  lerobot exited with error code {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"\n  ERROR: Could not find python at: {cmd[0]}")
        print("  Make sure lerobot_venv is set up:  python -m venv lerobot_venv")
        print("  Then:  lerobot_venv/bin/pip install -e lerobot/")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Simulation mode (no lerobot, no hardware)
# ──────────────────────────────────────────────────────────────────────────────

def run_simulation(duration: float | None = None):
    """
    Lightweight simulation using the local arm_controller module.
    Cycles through home → raised → wave → home to demonstrate the motion API.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from voice_pipeline.core.arm_controller import ArmController
    except ImportError as e:
        print(f"ERROR: Could not import arm_controller: {e}")
        sys.exit(1)

    print("\n" + "=" * 62)
    print("  Simulation Mode — no hardware required")
    print("  Cycling through: home → raised → wave → home")
    print("  Press Ctrl-C to stop.")
    print("=" * 62)

    ctrl = ArmController(enabled=True, run_diagnostic=True)

    moves = [
        ("go_home",   "Going home..."),
        ("raise_arm", "Raising arm..."),
        ("lower_arm", "Lowering arm..."),
    ]

    start = time.perf_counter()
    step = 0
    try:
        while True:
            action, label = moves[step % len(moves)]
            print(f"\n  [{step + 1}] {label}")

            class FakeIntent:
                class action:
                    value = None

            FakeIntent.action.value = action
            result = ctrl.execute(FakeIntent())
            print(f"      → {result.message}")

            step += 1
            time.sleep(1.5)

            if duration is not None and time.perf_counter() - start >= duration:
                break
    except KeyboardInterrupt:
        print("\n  Stopped by user.")
    finally:
        ctrl.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="SO-101 teleoperation practice script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--leader-port",   default=None, help="Serial port for the leader arm")
    ap.add_argument("--follower-port", default=None, help="Serial port for the follower arm")
    ap.add_argument("--camera-index",  default=None, help="Camera index or path (optional)")
    ap.add_argument("--display",       action="store_true", help="Show motor values on screen")
    ap.add_argument("--fps",           type=int, default=30, help="Control loop FPS (default: 30)")
    ap.add_argument("--duration",      type=float, default=None,
                    help="Run for this many seconds then exit")
    ap.add_argument("--sim",           action="store_true",
                    help="Simulation mode — no lerobot, no hardware required")
    args = ap.parse_args()

    if args.sim:
        run_simulation(duration=args.duration)
        return

    # Build config: use CLI args if provided, otherwise run wizard
    if args.leader_port and args.follower_port:
        cfg = {
            "leader_port": args.leader_port,
            "follower_port": args.follower_port,
            "camera": args.camera_index,
            "display": args.display,
            "fps": args.fps,
            "duration": args.duration,
        }
    else:
        cfg = wizard()

    # Auto-detect ports if the user left them blank
    if not cfg.get("leader_port") or not cfg.get("follower_port"):
        print("\n  Auto-detecting arm ports...")
        detected = auto_find_arm_port()
        if detected and not cfg.get("leader_port"):
            cfg["leader_port"] = detected
            print(f"  Leader port: {detected}")
        if not cfg.get("leader_port") or not cfg.get("follower_port"):
            print("\n  WARNING: Could not auto-detect ports.  Please specify them manually.")
            print("  Run:  python find_cameras.py   (for cameras)")
            print("  Run:  python -c \"from lerobot.scripts.lerobot_find_port import find_port; find_port()\"  (for arm ports)")

    print("\n  Configuration summary:")
    print(f"    Leader port:   {cfg.get('leader_port', 'not set')}")
    print(f"    Follower port: {cfg.get('follower_port', 'not set')}")
    print(f"    Camera:        {cfg.get('camera') or 'none'}")
    print(f"    Display data:  {cfg['display']}")
    print(f"    FPS:           {cfg['fps']}")
    print(f"    Duration:      {cfg.get('duration') or 'unlimited'}")

    run_lerobot(cfg)


if __name__ == "__main__":
    main()
