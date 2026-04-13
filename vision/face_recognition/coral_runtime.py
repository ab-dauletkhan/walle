from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

LOG = logging.getLogger("walle.coral_runtime")

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ENV_FLAG = "WALLE_CORAL_LOCAL"
PYTHON39_ENV = "WALLE_CORAL_PYTHON39"
DEFAULT_VENV = REPO_ROOT / ".venv-coral39" / "bin" / "python"


def running_on_python39_or_lower() -> bool:
    return sys.version_info < (3, 10)


def should_delegate_edge_tpu(edge_tpu: bool) -> bool:
    return (
        edge_tpu
        and not running_on_python39_or_lower()
        and os.environ.get(LOCAL_ENV_FLAG) != "1"
    )


def resolve_coral_python() -> Optional[str]:
    configured = os.environ.get(PYTHON39_ENV)
    candidates = [
        configured,
        str(DEFAULT_VENV) if DEFAULT_VENV.exists() else None,
        shutil.which("python3.9"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    return None


def build_reexec_env() -> dict[str, str]:
    env = os.environ.copy()
    env[LOCAL_ENV_FLAG] = "1"
    return env


def reexec_module_with_coral_python(module_name: str, argv: list[str]) -> int:
    coral_python = resolve_coral_python()
    if coral_python is None:
        raise RuntimeError(
            "Edge TPU requires a Python 3.9 Coral runtime, but none was found. "
            "Set WALLE_CORAL_PYTHON39 or create .venv-coral39 with scripts/setup_jetson_coral39.sh."
        )

    cmd = [coral_python, "-m", module_name, *argv]
    LOG.info("Re-executing %s via Coral runtime: %s", module_name, coral_python)
    return subprocess.call(cmd, cwd=str(REPO_ROOT), env=build_reexec_env())


class CoralWorkerClient:
    def __init__(self, timeout_ms: int = 5000):
        coral_python = resolve_coral_python()
        if coral_python is None:
            raise RuntimeError(
                "Coral worker requires a Python 3.9 runtime. "
                "Set WALLE_CORAL_PYTHON39 or create .venv-coral39 first."
            )

        self._timeout = timeout_ms / 1000.0
        self._proc = subprocess.Popen(
            [coral_python, "-m", "vision.face_recognition.coral_worker", "serve"],
            cwd=str(REPO_ROOT),
            env=build_reexec_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def request(self, payload: dict) -> dict:
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("Coral worker pipes are unavailable.")
        if self._proc.poll() is not None:
            stderr = ""
            if self._proc.stderr is not None:
                stderr = self._proc.stderr.read().strip()
            raise RuntimeError(
                f"Coral worker exited with code {self._proc.returncode}. {stderr}".strip()
            )

        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()

        line = self._proc.stdout.readline()
        if not line:
            stderr = ""
            if self._proc.stderr is not None:
                stderr = self._proc.stderr.read().strip()
            raise RuntimeError(
                f"Coral worker stopped without responding. {stderr}".strip()
            )

        response = json.loads(line)
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "Unknown Coral worker error"))
        return response

    def close(self) -> None:
        if self._proc.poll() is not None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=self._timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
