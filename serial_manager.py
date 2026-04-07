"""
WALL-E Serial Manager
Thread-safe shared serial connection to Arduino.
Used by both the LLM tool pipeline (RobotControlExecutor) and the web API bridge.
"""
import logging
import time
import threading
from typing import Optional

_log = logging.getLogger("walle.serial")

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class SerialManager:
    """Thread-safe serial connection to Arduino."""

    def __init__(self, port: Optional[str] = None, baud_rate: int = 115200):
        self._lock = threading.Lock()
        self._conn: Optional[serial.Serial] = None
        self._simulation = True
        self._port = port
        self._baud_rate = baud_rate

        if port and SERIAL_AVAILABLE:
            try:
                self._conn = serial.Serial(port=port, baudrate=baud_rate, timeout=1)
                self._simulation = False
                _log.info("Connected to %s @ %s baud", port, baud_rate)
            except Exception as e:
                _log.warning("Failed to connect to %s: %s", port, e)
                self._simulation = True
        else:
            reason = "no port specified" if not port else "pyserial not installed"
            _log.info("Simulation mode (%s)", reason)

    # Max command length the Arduino buffer can handle (letter + up to 3 digits)
    _CMD_MAX_LEN = 4

    def send_command(self, cmd: str) -> str:
        """Thread-safe write to serial port. Returns status string.

        Validates that *cmd* fits the Arduino serial protocol before sending:
        single-char commands (e.g. ``q``) or a letter followed by up to 3 digits
        (e.g. ``G50``, ``Y100``, ``X-80``).
        """
        # --- input validation ---
        if not cmd:
            return "Error: empty command"

        # Allow multi-command strings separated by \n (e.g. "L80\nR80")
        parts = cmd.split("\n")
        results = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Validate: single letter, or letter + optional minus + 1-3 digits
            if len(part) > 5 or not part[0].isalpha():
                return f"Error: invalid command '{part}' (expected letter + optional number, max 4 chars)"
            if len(part) > 1:
                tail = part[1:]
                if tail.lstrip("-").isdigit() is False:
                    return f"Error: invalid command '{part}' (non-numeric payload)"
            results.append(self._send_single(part))

        return "; ".join(results) if results else "Error: no valid commands"

    # How long to wait for the Arduino ACK ("OK\n") after sending a command.
    _ACK_TIMEOUT = 2.0  # seconds

    def _send_single(self, cmd: str) -> str:
        """Send a single validated command and wait for Arduino ACK."""
        with self._lock:
            if self._simulation:
                _log.debug("SIM >> %s", cmd)
                return "OK (simulated)"

            try:
                if not (self._conn and self._conn.is_open):
                    _log.warning("Connection closed, simulating: %s", cmd)
                    return "OK (connection closed, simulated)"

                # Drain any stale data before writing
                self._conn.reset_input_buffer()
                self._conn.write(f"{cmd}\n".encode())
                self._conn.flush()

                # Wait for "OK" acknowledgment from Arduino
                deadline = time.monotonic() + self._ACK_TIMEOUT
                while time.monotonic() < deadline:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    old_timeout = self._conn.timeout
                    self._conn.timeout = min(remaining, 0.5)
                    try:
                        line = self._conn.readline().decode("utf-8", errors="ignore").strip()
                    finally:
                        self._conn.timeout = old_timeout
                    if line == "OK":
                        return "OK"
                    if line:
                        _log.debug("Arduino: %s", line)  # echo or status line

                _log.warning("No ACK for '%s' within %.1fs", cmd, self._ACK_TIMEOUT)
                return "OK (no ACK)"
            except Exception as e:
                _log.error("Error: %s", e)
                return f"Serial Error: {e}"

    def read_line(self, timeout: float = 0.5) -> Optional[str]:
        """Thread-safe read from serial port."""
        with self._lock:
            if self._simulation or not self._conn or not self._conn.is_open:
                return None
            try:
                old_timeout = self._conn.timeout
                self._conn.timeout = timeout
                line = self._conn.readline().decode("utf-8", errors="ignore").strip()
                self._conn.timeout = old_timeout
                return line if line else None
            except Exception:
                return None

    def heartbeat(self) -> Optional[str]:
        """Send status query to Arduino. Returns status string or None on failure."""
        with self._lock:
            if self._simulation:
                return "STATUS 248,560,140,475,270,250,290 M0,0 (simulated)"
            try:
                if not (self._conn and self._conn.is_open):
                    return None
                self._conn.reset_input_buffer()
                self._conn.write(b"?\n")
                self._conn.flush()
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    old_timeout = self._conn.timeout
                    self._conn.timeout = min(remaining, 0.5)
                    try:
                        line = self._conn.readline().decode("utf-8", errors="ignore").strip()
                    finally:
                        self._conn.timeout = old_timeout
                    if line.startswith("STATUS"):
                        return line
                _log.debug("No heartbeat response within 1s")
                return None
            except Exception as e:
                _log.debug("Heartbeat error: %s", e)
                return None

    def is_connected(self) -> bool:
        return not self._simulation and self._conn is not None and self._conn.is_open

    @property
    def simulation(self) -> bool:
        return self._simulation

    def close(self) -> None:
        with self._lock:
            if self._conn and self._conn.is_open:
                self._conn.close()
                _log.info("Connection closed")
