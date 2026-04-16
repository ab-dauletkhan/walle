"""
WALL-E robot control tool executor.

Thin adapter between the LLM tool-calling layer and the shared
``SerialManager`` that speaks the CLI firmware (``motor_control_cli.ino``).
All real serial I/O is delegated to ``SerialManager`` so the face tracker
and the LLM can share one port without racing.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from walle.tools.base_executor import BaseToolExecutor
from walle.tools.robot.catalog import RobotRuntime, build_default_robot_registry

_log = logging.getLogger("walle.robot")


_ROBOT_REGISTRY = build_default_robot_registry()


def get_robot_control_tools():
    return _ROBOT_REGISTRY.get_tool_schemas()


def get_robot_tool_names():
    return _ROBOT_REGISTRY.get_tool_names()


class RobotControlExecutor(BaseToolExecutor, RobotRuntime):
    """Executor for robot control tools.

    Always routes through a shared ``SerialManager``. If one isn't provided
    (e.g. tests), the executor runs in simulation mode and logs commands
    instead of writing to hardware.
    """

    def __init__(self, serial_manager=None, **_legacy_kwargs):
        # _legacy_kwargs absorbs old serial_port / baud_rate arguments so
        # existing callers keep working without creating a second pyserial
        # connection behind SerialManager's back.
        self._serial_manager = serial_manager
        self._registry = _ROBOT_REGISTRY
        self.simulation = serial_manager is None or serial_manager.simulation
        if serial_manager is None:
            _log.info("Robot running without a SerialManager (simulation mode)")
        elif self.simulation:
            _log.info("Robot using shared serial (simulation mode)")
        else:
            _log.info("Robot using shared serial connection")

    # ------------------------------------------------------------------ #
    # Tool dispatch
    # ------------------------------------------------------------------ #

    def execute(self, fn_name: str, args: dict) -> str:
        _log.info("tool=%s args=%s", fn_name, args)
        try:
            result = self._registry.execute(self, fn_name, args)
        except RuntimeError as exc:
            _log.warning("Tool %s failed: %s", fn_name, exc)
            return f"Error: {exc}"
        _log.info("result=%s", result)
        return result

    # ------------------------------------------------------------------ #
    # RobotRuntime protocol
    # ------------------------------------------------------------------ #

    def send_command(self, cmd: str) -> str:
        if self._serial_manager is None:
            _log.debug("[SERIAL SIM] >> %s", cmd)
            return "OK (simulated)"
        return self._serial_manager.send_command(cmd)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def heartbeat(self) -> Optional[str]:
        if self._serial_manager is None:
            return "status (simulated)"
        return self._serial_manager.heartbeat()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        # The SerialManager owns the port; nothing to tear down here.
        return
