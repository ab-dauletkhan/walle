"""
WALL-E Serial Manager
Thread-safe shared serial connection to the Arduino CLI firmware
(`firmware/wall-e/motor_control_cli.ino`).

Protocol
--------
The firmware speaks a line-oriented text CLI at 115200 baud. Every command is
a single '\n'-terminated line, e.g. ``drive 80 500`` or ``head tick 300``.
Replies always start with either ``OK -> ...`` on success or ``Rejected: ...``
on validation failure, and may be followed by zero or more informational
lines (e.g. ``Timed move complete -> stopped.``). The ``status`` command
emits a multi-line block that ends with a blank line; we collect lines until
drain.

This module exposes one shared ``SerialManager`` that both the LLM tool
executor and the vision face tracker acquire through ``startup.py`` so they
don't race on the same ``/dev/ttyUSB...`` handle.
"""

import logging
import threading
import time
from typing import List, Optional

_log = logging.getLogger("walle.serial")

try:
    import serial

    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class SerialManager:
    """Thread-safe serial connection to the Arduino CLI firmware."""

    # Time to wait for a reply line after a write. The firmware responds
    # within a few ms; we only need a cushion for USB serial + Arduino
    # interrupt latency.
    _REPLY_TIMEOUT = 1.5
    _STATUS_TIMEOUT = 2.0

    def __init__(self, port: Optional[str] = None, baud_rate: int = 115200):
        self._lock = threading.Lock()
        self._conn: Optional["serial.Serial"] = None
        self._simulation = True
        self._port = port
        self._baud_rate = baud_rate

        if port and SERIAL_AVAILABLE:
            try:
                self._conn = serial.Serial(port=port, baudrate=baud_rate, timeout=0.2)
                self._simulation = False
                # Arduino auto-resets on DTR; give the sketch a moment to boot
                # and drain its startup banner so the first real command isn't
                # lost in the noise.
                time.sleep(1.8)
                try:
                    self._conn.reset_input_buffer()
                except Exception:
                    pass
                _log.info("Connected to %s @ %s baud", port, baud_rate)
            except Exception as e:
                _log.warning("Failed to connect to %s: %s", port, e)
                self._simulation = True
        else:
            reason = "no port specified" if not port else "pyserial not installed"
            _log.info("Simulation mode (%s)", reason)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def send_command(self, cmd: str) -> str:
        """Send one or more newline-separated CLI commands.

        Returns a human-readable status string combining each command's
        reply. Multi-line input (``"head tick 300\nspin 80 400"``) is
        split and each line is dispatched individually so a single bad
        line doesn't abort the rest.
        """
        if not cmd:
            return "Error: empty command"

        lines = [line.strip() for line in cmd.split("\n") if line.strip()]
        if not lines:
            return "Error: empty command"

        results: List[str] = []
        for line in lines:
            results.append(self._send_line(line))
        return "; ".join(results)

    def send_write_only(self, cmd: str) -> None:
        """Write one line without waiting for any firmware reply.

        Used by high-frequency callers like the vision head tracker
        (30 Hz): waiting for the ack round-trip (10-50 ms each) would
        cap the effective send rate at ~15 Hz and make the servo feel
        sluggish. The firmware still processes the command — we just
        don't block the caller for the response.

        Even though we don't want the reply, we still drain the RX
        buffer before/after writing. The firmware prints an "OK -> ..."
        line for every command, and at 30 Hz those replies quickly
        saturate the kernel tty buffer. Once the host-side buffer is
        full, the USB endpoint NAKs, the CH341 backs up, and the
        Arduino's Serial.print starts blocking — which stalls the
        firmware's main loop and makes the LLM's subsequent commands
        time out. Draining keeps the pipe flowing.

        Returns silently in simulation mode. Thread-safe.
        """
        line = cmd.strip()
        if not line:
            return
        with self._lock:
            if self._simulation:
                _log.debug("SIM (wo) >> %s", line)
                return
            if not (self._conn and self._conn.is_open):
                return
            try:
                pending = self._conn.in_waiting
                if pending:
                    self._conn.read(pending)
            except Exception:
                pass
            try:
                self._conn.write(f"{line}\n".encode())
                self._conn.flush()
            except Exception as exc:
                # Upgraded from DEBUG → WARNING. A silent write-only failure
                # is the classic "head tracker keeps logging sends but the
                # servo never moves" symptom. With this visible, a dead
                # integrated tracker surfaces as a run of warnings instead
                # of an invisible regression against the standalone.
                _log.warning("send_write_only '%s' failed: %s", line, exc)

    def send_raw(self, cmd: str, timeout: Optional[float] = None) -> List[str]:
        """Send one command and return the full reply as a list of lines.

        Use this when you need to parse a multi-line response like
        ``status``. Returns an empty list in simulation mode.
        """
        line = cmd.strip()
        if not line:
            return []
        with self._lock:
            if self._simulation:
                _log.debug("SIM >> %s", line)
                return []
            return self._write_and_collect(line, timeout or self._REPLY_TIMEOUT)

    def status(self) -> List[str]:
        """Query the firmware `status` command and return its reply block."""
        return self.send_raw("status", timeout=self._STATUS_TIMEOUT)

    def heartbeat(self) -> Optional[str]:
        """Best-effort connectivity probe. Returns ``None`` on failure."""
        if self._simulation:
            return "status (simulated)"
        lines = self.status()
        if not lines:
            return None
        return "\n".join(lines)

    def read_line(self, timeout: float = 0.5) -> Optional[str]:
        """Thread-safe single-line read (e.g. for draining unsolicited output)."""
        with self._lock:
            if self._simulation or not self._conn or not self._conn.is_open:
                return None
            try:
                old_timeout = self._conn.timeout
                self._conn.timeout = timeout
                line = self._conn.readline().decode("utf-8", errors="ignore").strip()
                self._conn.timeout = old_timeout
                return line or None
            except Exception:
                return None

    def is_connected(self) -> bool:
        return not self._simulation and self._conn is not None and self._conn.is_open

    @property
    def simulation(self) -> bool:
        return self._simulation

    def close(self) -> None:
        with self._lock:
            if self._conn and self._conn.is_open:
                try:
                    self._conn.write(b"stopall\n")
                    self._conn.flush()
                except Exception:
                    pass
                self._conn.close()
                _log.info("Connection closed")

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _send_line(self, line: str) -> str:
        with self._lock:
            if self._simulation:
                _log.debug("SIM >> %s", line)
                return "OK (simulated)"

            if not (self._conn and self._conn.is_open):
                _log.warning("Connection closed, simulating: %s", line)
                return "OK (connection closed, simulated)"

            try:
                reply_lines = self._write_and_collect(line, self._REPLY_TIMEOUT)
            except Exception as e:
                _log.error("Serial error on '%s': %s", line, e)
                return f"Serial Error: {e}"

            if not reply_lines:
                _log.warning("No reply for '%s' within %.1fs", line, self._REPLY_TIMEOUT)
                return "OK (no reply)"

            head = reply_lines[0]
            if head.startswith("OK"):
                return head
            if head.startswith("Rejected"):
                _log.warning("Firmware rejected '%s': %s", line, head)
                return head
            # Unknown prefix — pass the first line through so callers can log it.
            return head

    def _write_and_collect(self, line: str, timeout: float) -> List[str]:
        """Write one line and collect reply lines until the port goes quiet.

        Must be called with ``self._lock`` held. The firmware sends its reply
        in a burst, so we keep reading until ``timeout`` elapses OR until we
        see a blank line after at least one real reply line. We prefer the
        *first* non-empty line as the primary result.
        """
        assert self._conn is not None
        self._conn.reset_input_buffer()
        self._conn.write(f"{line}\n".encode())
        self._conn.flush()

        deadline = time.monotonic() + timeout
        collected: List[str] = []
        old_timeout = self._conn.timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._conn.timeout = min(remaining, 0.2)
                raw = self._conn.readline()
                if not raw:
                    # No more data within the short read window; if we already
                    # have a reply line we can return — otherwise keep waiting.
                    if collected:
                        break
                    continue
                text = raw.decode("utf-8", errors="ignore").strip()
                if not text:
                    if collected:
                        break
                    continue
                collected.append(text)
        finally:
            self._conn.timeout = old_timeout
        return collected
