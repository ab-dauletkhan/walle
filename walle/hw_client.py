"""
Unix-domain-socket client for the Hardware Controller (walle-hwc).

Used by the STT-direct handlers and the LLM ``DO:`` dispatcher to ship
Arduino CLI lines to the controller, which arbitrates and forwards them
to the firmware. The client is thread-safe — one connection, one lock,
several reuse-call sites.

Failure policy: best-effort with bounded retries. If the controller is
down (not running, crashed, or socket file missing), send() returns an
``ERR`` string instead of raising. Callers log and move on; the robot
does not go catatonic just because hardware is temporarily unreachable.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from typing import Optional

_log = logging.getLogger("walle.hw_client")

DEFAULT_SOCKET_PATH = os.environ.get("WALLE_HW_SOCK", "/tmp/walle_hw.sock")
_DEFAULT_RETRIES = 3
_RETRY_BACKOFF_SEC = 0.3
_SOCK_TIMEOUT_SEC = 2.0


class HardwareClient:
    """Thread-safe UDS client for the walle-hwc server."""

    def __init__(
        self,
        identity: str,
        socket_path: str = DEFAULT_SOCKET_PATH,
        retries: int = _DEFAULT_RETRIES,
    ) -> None:
        if identity not in {"stt", "llm", "cli"}:
            # tracker is reserved for the subprocess started by HWC itself.
            raise ValueError(f"identity must be stt/llm/cli, got {identity!r}")
        self._identity = identity
        self._path = socket_path
        self._retries = retries
        self._lock = threading.Lock()
        self._sock: Optional[socket.socket] = None
        self._reader = None  # BufferedReader from makefile

    # -- Public API --------------------------------------------------------

    def send(self, cli_line: str) -> str:
        """Send one Arduino CLI line; return the controller's reply line.

        Returns ``"OK"`` on a bare OK, ``"ERR <reason>"`` on rejection, or
        ``"ERR hwc unreachable"`` if the controller couldn't be reached
        after the configured retries. Never raises for transient issues —
        callers log the reply and carry on.
        """
        line = cli_line.strip()
        if not line:
            return "OK"

        with self._lock:
            last_err: str = ""
            for attempt in range(self._retries):
                try:
                    if self._sock is None:
                        self._connect_locked()
                    assert self._sock is not None and self._reader is not None
                    self._sock.sendall(f"{line}\n".encode("utf-8"))
                    reply = self._reader.readline()
                    if not reply:
                        # EOF — server dropped us; reconnect and retry.
                        raise ConnectionError("empty reply (server closed)")
                    text = reply.decode("utf-8", errors="ignore").strip()
                    return text or "OK"
                except Exception as exc:
                    last_err = str(exc) or type(exc).__name__
                    _log.warning(
                        "HWC send %r failed (attempt %d/%d): %s",
                        line, attempt + 1, self._retries, last_err,
                    )
                    self._close_locked()
                    if attempt < self._retries - 1:
                        time.sleep(_RETRY_BACKOFF_SEC)
            return f"ERR hwc unreachable: {last_err}"

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def is_connected(self) -> bool:
        with self._lock:
            return self._sock is not None

    # -- Internals ---------------------------------------------------------

    def _connect_locked(self) -> None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(_SOCK_TIMEOUT_SEC)
        s.connect(self._path)
        s.sendall(f"IDENT {self._identity}\n".encode("utf-8"))
        # makefile gives us a line-oriented reader over the same fd.
        reader = s.makefile("rb", buffering=0)
        ready = reader.readline().decode("utf-8", errors="ignore").strip()
        if ready != "READY":
            s.close()
            raise ConnectionError(f"HWC did not say READY: {ready!r}")
        self._sock = s
        self._reader = reader

    def _close_locked(self) -> None:
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
            self._reader = None
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
