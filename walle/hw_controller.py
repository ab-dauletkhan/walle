"""
WALL-E Hardware Controller (walle-hwc)
--------------------------------------

A standalone process that owns the Arduino serial port and arbitrates
between three command sources:

  tracker  — the `track_and_turn_head.py` subprocess (30 Hz, low priority)
  stt      — the STT-direct path in walle (preemptive)
  llm      — the LLM DO-line dispatcher in walle (preemptive)
  cli      — ad-hoc debugging via `nc -U` (preemptive)

Clients connect to a Unix domain socket, declare their source with an
``IDENT <source>\\n`` handshake, then send Arduino CLI lines verbatim.
The controller forwards them to the firmware, subject to the arbiter:

  - ``tracker`` commands are dropped silently while any preemptive head
    command issued in the last 3 seconds is still "holding". This keeps
    LLM/STT head moves from being instantly overwritten by the face
    tracker's next tick.
  - All other sources' commands go through immediately, and any that
    target the head servo push the override window forward.

Wheel commands never conflict with head tracking (different actuators),
so they always go straight through.

Life cycle:

  - start.sh spawns walle-hwc before walle (main).
  - walle-hwc listens on the UDS path and spawns the tracker as a
    child subprocess, supervised: if the tracker dies the controller
    logs it and respawns it.
  - walle connects later, issues commands, and survives the controller
    going away (reconnect with backoff on the walle side).
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import List, Optional

from walle.serial_manager import SerialManager

_log = logging.getLogger("walle.hwc")

DEFAULT_SOCKET_PATH = "/tmp/walle_hw.sock"
DEFAULT_OVERRIDE_SEC = 3.0
DEFAULT_TRACKER_CMD = "uv run walle-face-track-head --edge-tpu --headless"


# ---------------------------------------------------------------------------
# Arbiter
# ---------------------------------------------------------------------------
# Head-class CLI verbs — any command starting with one of these pushes the
# override window. Wheel / status commands pass through freely.
_HEAD_VERBS = ("head", "scan", "home")


def _is_head_cli(cli: str) -> bool:
    first = cli.strip().split(" ", 1)[0].lower()
    return first in _HEAD_VERBS


class Arbiter:
    """Decides whether a given (source, cli) pair is admitted for forwarding.

    Single lock, trivial logic. Keep it simple — the 30 Hz tracker hits
    :meth:`admit` many times per second and we don't want any contention.
    """

    def __init__(self, override_sec: float = DEFAULT_OVERRIDE_SEC) -> None:
        self._override_sec = override_sec
        self._override_until = 0.0
        self._lock = threading.Lock()

    def admit(self, source: str, cli: str) -> bool:
        now = time.monotonic()
        with self._lock:
            if source == "tracker":
                return now >= self._override_until
            # stt / llm / cli
            if _is_head_cli(cli):
                self._override_until = now + self._override_sec
            return True

    def override_remaining(self) -> float:
        with self._lock:
            return max(0.0, self._override_until - time.monotonic())


# ---------------------------------------------------------------------------
# Tracker supervisor
# ---------------------------------------------------------------------------


class TrackerSupervisor:
    """Spawns the head-tracking subprocess and respawns it on exit.

    The subprocess talks back to us over the SAME Unix socket path with
    ``IDENT tracker``; supervisor only cares about liveness.
    """

    _RESPAWN_BACKOFF_SEC = 2.0

    def __init__(self, command: str, socket_path: str) -> None:
        self._command = command
        self._socket_path = socket_path
        self._running = False
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._supervise_loop, name="hwc-tracker-sup", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def is_alive(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    def _supervise_loop(self) -> None:
        while self._running:
            try:
                self._proc = self._spawn()
            except Exception:
                _log.exception("failed to spawn tracker")
                time.sleep(self._RESPAWN_BACKOFF_SEC)
                continue
            _log.info("tracker spawned (pid=%s)", self._proc.pid)
            self._proc.wait()
            if not self._running:
                return
            _log.warning(
                "tracker exited rc=%s; respawning in %.1fs",
                self._proc.returncode,
                self._RESPAWN_BACKOFF_SEC,
            )
            time.sleep(self._RESPAWN_BACKOFF_SEC)

    def _spawn(self) -> subprocess.Popen:
        argv = shlex.split(self._command) + [
            "--head-command-socket",
            self._socket_path,
        ]
        # Inherit stdout/stderr so the tracker's human-readable messages
        # show up in whatever log stream start.sh is capturing. The head
        # tick commands travel over the separate UDS, not stdout.
        return subprocess.Popen(argv, stdout=sys.stdout, stderr=sys.stderr)


# ---------------------------------------------------------------------------
# Socket server
# ---------------------------------------------------------------------------


class HWCServer:
    """Accepts UDS connections, handles one per thread."""

    def __init__(
        self,
        socket_path: str,
        serial_mgr: SerialManager,
        arbiter: Arbiter,
    ) -> None:
        self._socket_path = socket_path
        self._serial = serial_mgr
        self._arbiter = arbiter
        self._server_sock: Optional[socket.socket] = None
        self._running = False
        self._accept_thread: Optional[threading.Thread] = None
        self._client_threads: List[threading.Thread] = []

    def start(self) -> None:
        # Clean up any stale socket file from a previous crashed run.
        try:
            if os.path.exists(self._socket_path):
                os.unlink(self._socket_path)
        except Exception:
            _log.exception("could not remove stale socket file")

        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(self._socket_path)
        os.chmod(self._socket_path, 0o666)  # let any user of this host speak
        s.listen(16)
        self._server_sock = s
        self._running = True

        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="hwc-accept", daemon=True
        )
        self._accept_thread.start()
        _log.info("listening on %s", self._socket_path)

    def stop(self) -> None:
        self._running = False
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except Exception:
                pass
            self._server_sock = None
        try:
            if os.path.exists(self._socket_path):
                os.unlink(self._socket_path)
        except Exception:
            pass

    # -- internals --

    def _accept_loop(self) -> None:
        assert self._server_sock is not None
        while self._running:
            try:
                conn, _ = self._server_sock.accept()
            except OSError:
                if self._running:
                    _log.exception("accept failed")
                return
            t = threading.Thread(
                target=self._handle_client, args=(conn,), daemon=True,
                name="hwc-client",
            )
            t.start()
            self._client_threads.append(t)

    def _handle_client(self, conn: socket.socket) -> None:
        # makefile wraps the socket for line-buffered reading; any exception
        # just closes the connection. Clients reconnect with backoff.
        source = "cli"
        try:
            f = conn.makefile("rwb", buffering=0)
            ident_line = f.readline().decode("utf-8", errors="ignore").strip()
            source = self._parse_ident(ident_line)
            if source is None:
                # Client sent garbage / disconnected pre-IDENT. Best-
                # effort error reply; swallow BrokenPipe silently so
                # misbehaving probes don't leave tracebacks in the log.
                try:
                    conn.sendall(b"ERR bad IDENT\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                return
            _log.info("client connected: source=%s", source)
            if source != "tracker":
                try:
                    conn.sendall(b"READY\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

            while True:
                raw = f.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                self._dispatch(conn, source, line)
        except Exception:
            _log.exception("client handler error (source=%s)", source)
        finally:
            try:
                conn.close()
            except Exception:
                pass
            _log.info("client disconnected: source=%s", source)

    @staticmethod
    def _parse_ident(line: str) -> Optional[str]:
        parts = line.split()
        if len(parts) != 2:
            return None
        verb, source = parts[0].upper(), parts[1].lower()
        if verb != "IDENT":
            return None
        if source not in {"tracker", "stt", "llm", "cli"}:
            return None
        return source

    def _dispatch(self, conn: socket.socket, source: str, cli: str) -> None:
        # Arbiter gate first — drop tracker lines silently when an
        # override is active. No reply is sent to the tracker in any
        # case; it is a fire-hose, not a blocking caller.
        admit = self._arbiter.admit(source, cli)
        if source == "tracker":
            if admit:
                # Write-only path; firmware's ack is drained on the next
                # write_only call to keep the tty buffer from filling.
                self._serial.send_write_only(cli)
            return

        if not admit:
            # stt/llm/cli are never denied by the arbiter today. Future-
            # proofing: if that changes, respond sensibly.
            try:
                conn.sendall(b"ERR denied\n")
            except Exception:
                pass
            return

        # Blocking path — returns the firmware's reply line so the client
        # knows whether its command was accepted or rejected.
        reply = self._serial.send_command(cli)
        try:
            first = reply.split(";", 1)[0].strip() if reply else "OK"
            if first.startswith("OK"):
                conn.sendall(b"OK\n")
            elif first.startswith("Rejected"):
                conn.sendall(f"ERR {first}\n".encode("utf-8"))
            else:
                # Unknown prefix — echo the line back so diagnostic output
                # (e.g. status blocks) isn't lost.
                conn.sendall(f"{first}\n".encode("utf-8"))
        except Exception:
            _log.exception("failed to send reply")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WALL-E Hardware Controller")
    p.add_argument(
        "--socket-path",
        default=os.environ.get("WALLE_HW_SOCK", DEFAULT_SOCKET_PATH),
        help="Unix socket path (default %(default)s)",
    )
    p.add_argument(
        "--serial-port",
        default=os.environ.get("WALLE_SERIAL_PORT", "/dev/ttyCH341USB0"),
        help="Arduino serial port (default %(default)s)",
    )
    p.add_argument(
        "--baud-rate",
        type=int,
        default=int(os.environ.get("WALLE_SERIAL_BAUD", "115200")),
        help="Arduino baud (default %(default)s)",
    )
    p.add_argument(
        "--override-sec",
        type=float,
        default=DEFAULT_OVERRIDE_SEC,
        help=(
            "Seconds to suppress tracker commands after any stt/llm/cli "
            "head command (default %(default)s)"
        ),
    )
    p.add_argument(
        "--tracker-cmd",
        default=os.environ.get("WALLE_TRACKER_CMD", DEFAULT_TRACKER_CMD),
        help=(
            "Shell command used to spawn the face-tracking subprocess. "
            "The HWC appends --head-command-socket <path> to it. "
            "Set to empty string to skip spawning a tracker (useful for "
            "debugging the socket by hand). Default: %(default)r"
        ),
    )
    p.add_argument(
        "--log-level",
        default=os.environ.get("WALLE_HW_LOG_LEVEL", "INFO"),
        help="Python log level (default %(default)s)",
    )
    p.add_argument(
        "--allow-simulation",
        action="store_true",
        default=os.environ.get("WALLE_HW_ALLOW_SIM") == "1",
        help=(
            "Allow the controller to run with a simulated (unconnected) "
            "Arduino. Off by default — a missing serial port almost always "
            "means a cable / udev problem, and silently simulating it "
            "makes walle-hwc reply OK to drive commands that never reach "
            "the firmware. Set this (or WALLE_HW_ALLOW_SIM=1) when you "
            "genuinely want to test the protocol without hardware."
        ),
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    serial_mgr = SerialManager(port=args.serial_port, baud_rate=args.baud_rate)
    if not serial_mgr.is_connected():
        if args.allow_simulation:
            _log.warning(
                "serial not connected (port=%s); running in simulation "
                "mode (--allow-simulation set). Every drive/head command "
                "will return OK but the firmware will never see it.",
                args.serial_port,
            )
        else:
            _log.error(
                "serial port %s is not connected. Refusing to start in "
                "silent simulation mode — that's the trap that made "
                "`drive 200 1500 → OK` look fine while the robot sat "
                "still. Diagnose with:  ls /dev/tty{USB*,ACM*,CH341*}  "
                "and  lsusb | grep -i ch341.  Override with "
                "--allow-simulation if you really want the simulated path.",
                args.serial_port,
            )
            return 2

    arbiter = Arbiter(override_sec=args.override_sec)
    server = HWCServer(
        socket_path=args.socket_path,
        serial_mgr=serial_mgr,
        arbiter=arbiter,
    )
    server.start()

    tracker: Optional[TrackerSupervisor] = None
    if args.tracker_cmd.strip():
        tracker = TrackerSupervisor(
            command=args.tracker_cmd, socket_path=args.socket_path
        )
        tracker.start()
    else:
        _log.info("--tracker-cmd empty, not spawning a tracker subprocess")

    stop_event = threading.Event()

    def _halt(signum, _frame):
        _log.info("signal %s received, shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _halt)
    signal.signal(signal.SIGTERM, _halt)

    try:
        while not stop_event.is_set():
            stop_event.wait(1.0)
    finally:
        if tracker is not None:
            tracker.stop()
        server.stop()
        try:
            serial_mgr.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
