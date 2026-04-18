#!/usr/bin/env python3
"""
WALL-E hardware diagnostic CLI.

Pokes the motors / servo directly so you can figure out which layer
of the stack is breaking when the voice pipeline says "OK" but the
robot doesn't move. Runs two modes:

  * Via walle-hwc (default): connects to /tmp/walle_hw.sock as an
    ``IDENT cli`` client. Exercises the full walle stack except the
    voice frontend.

  * Direct serial (--direct): opens /dev/ttyCH341USB0 yourself.
    Bypasses walle-hwc entirely. Useful when you want to rule out
    the controller and talk to firmware raw.

Usage:
    # Interactive REPL via HWC (walle-hwc must be running):
    uv run scripts/hw_diag.py

    # Send one command and exit:
    uv run scripts/hw_diag.py drive 200 1500

    # Run a preset diagnostic sequence:
    uv run scripts/hw_diag.py --sequence motors

    # Bypass HWC, hit the Arduino straight:
    uv run scripts/hw_diag.py --direct drive 200 1500

The diagnostic sequence walks you through a checklist: query status,
try wheels individually, try head servo, then stop. Each step prints
the firmware reply so you can see whether the Arduino's state
actually matches what you told it to do.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from typing import Optional


DEFAULT_SOCKET = "/tmp/walle_hw.sock"
DEFAULT_SERIAL = "/dev/ttyCH341USB0"
DEFAULT_BAUD = 115200


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------


class HwcTransport:
    """Talks to walle-hwc over its Unix socket as IDENT cli."""

    def __init__(self, path: str = DEFAULT_SOCKET) -> None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(3.0)
        s.connect(path)
        s.sendall(b"IDENT cli\n")
        self._sock = s
        self._reader = s.makefile("rb", buffering=0)
        ready = self._reader.readline().decode().strip()
        if ready != "READY":
            raise RuntimeError(f"HWC didn't say READY: {ready!r}")

    def send(self, line: str) -> str:
        self._sock.sendall((line.strip() + "\n").encode())
        reply = self._reader.readline().decode("utf-8", errors="ignore").strip()
        return reply or "(empty)"

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass


class DirectTransport:
    """Opens pyserial on /dev/ttyCH341USB0 directly — bypasses HWC."""

    def __init__(
        self, port: str = DEFAULT_SERIAL, baud: int = DEFAULT_BAUD
    ) -> None:
        import serial

        self._ser = serial.Serial(port, baudrate=baud, timeout=0.3)
        # Arduino auto-resets on DTR — wait out the boot banner so the
        # first real command isn't lost.
        time.sleep(1.8)
        self._ser.reset_input_buffer()
        print(f"(direct) connected to {port} @ {baud}", file=sys.stderr)

    def send(self, line: str) -> str:
        line = line.strip()
        self._ser.reset_input_buffer()
        self._ser.write((line + "\n").encode())
        self._ser.flush()
        # Collect everything the firmware says for ~0.8 s.
        deadline = time.monotonic() + 0.8
        lines: list[str] = []
        while time.monotonic() < deadline:
            raw = self._ser.readline()
            if not raw:
                if lines:
                    break
                continue
            text = raw.decode("utf-8", errors="ignore").strip()
            if not text:
                if lines:
                    break
                continue
            lines.append(text)
        return " | ".join(lines) if lines else "(no reply)"

    def close(self) -> None:
        try:
            self._ser.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Diagnostic sequences
# ---------------------------------------------------------------------------


def _run_step(t, label: str, cmd: str, pause: float = 0.5) -> None:
    print(f"\n>>> {label}")
    print(f"    → {cmd}")
    reply = t.send(cmd)
    print(f"    ← {reply}")
    if pause > 0:
        time.sleep(pause)


def sequence_motors(t) -> None:
    """Exercise each wheel motor individually, then both together.

    If this shows `left=200 right=200 for 1500 ms` but the robot
    doesn't physically roll, it's hardware (battery / H-bridge /
    wiring). If `status` comes back with `left=0 right=0` despite
    you sending `drive 200 1500`, it's firmware / pin mapping.
    """
    _run_step(t, "query baseline", "status", pause=0)
    _run_step(t, "left wheel only (2 s)", "left 200 2000", pause=2.5)
    _run_step(t, "right wheel only (2 s)", "right 200 2000", pause=2.5)
    _run_step(t, "both wheels forward (2 s)", "drive 200 2000", pause=2.5)
    _run_step(t, "both wheels backward (2 s)", "drive -200 2000", pause=2.5)
    _run_step(t, "spin in place left (1 s)", "spin 200 1000", pause=1.5)
    _run_step(t, "spin in place right (1 s)", "spin -200 1000", pause=1.5)
    _run_step(t, "stop", "stop", pause=0.3)
    _run_step(t, "final status", "status", pause=0)


def sequence_head(t) -> None:
    """Walk the head servo through left / right / center with a small pause.

    Delta cap was removed in the last firmware update — if any of
    these return 'Rejected: ... jump too large' the firmware wasn't
    reflashed. Otherwise the servo should sweep visibly.
    """
    _run_step(t, "home", "home", pause=0.5)
    _run_step(t, "head left", "head pos 0", pause=1.0)
    _run_step(t, "head right", "head pos 100", pause=1.0)
    _run_step(t, "head center", "head pos 50", pause=0.3)


def sequence_all(t) -> None:
    sequence_head(t)
    sequence_motors(t)


SEQUENCES = {
    "motors": sequence_motors,
    "head": sequence_head,
    "all": sequence_all,
}


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------


def repl(t) -> None:
    print(
        "Type Arduino CLI commands (e.g. `drive 200 1500`, `status`, `stop`). "
        "Blank line or `quit` to exit.",
        file=sys.stderr,
    )
    while True:
        try:
            line = input("hw> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line or line.lower() in {"quit", "exit", "q"}:
            return
        try:
            print(f"    ← {t.send(line)}")
        except Exception as exc:
            print(f"    !! {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--direct", action="store_true",
        help="Bypass walle-hwc and open the serial port directly",
    )
    p.add_argument(
        "--socket-path", default=DEFAULT_SOCKET,
        help="walle-hwc UDS path (default %(default)s)",
    )
    p.add_argument(
        "--serial-port", default=DEFAULT_SERIAL,
        help="Serial port for --direct mode (default %(default)s)",
    )
    p.add_argument(
        "--sequence", choices=list(SEQUENCES.keys()),
        help="Run a preset diagnostic sequence and exit.",
    )
    p.add_argument(
        "command", nargs=argparse.REMAINDER,
        help="One Arduino CLI command; omit for REPL.",
    )
    args = p.parse_args(argv)

    transport_kind = "direct serial" if args.direct else f"HWC at {args.socket_path}"
    print(f"(using {transport_kind})", file=sys.stderr)

    if args.direct:
        t = DirectTransport(port=args.serial_port)
    else:
        t = HwcTransport(path=args.socket_path)

    try:
        if args.sequence:
            SEQUENCES[args.sequence](t)
        elif args.command:
            cmd = " ".join(args.command).strip()
            if cmd:
                print(f"    ← {t.send(cmd)}")
        else:
            repl(t)
    finally:
        t.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
