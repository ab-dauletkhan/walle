"""
WALL-E robot control tools.
"""

from __future__ import annotations

import logging
import time

from walle.memory.base_executor import BaseToolExecutor
from walle.tools.robot.catalog import RobotRuntime, build_default_robot_registry

_log = logging.getLogger("walle.robot")

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


_ROBOT_REGISTRY = build_default_robot_registry()


def get_robot_control_tools():
    return _ROBOT_REGISTRY.get_tool_schemas()


def get_robot_tool_names():
    return _ROBOT_REGISTRY.get_tool_names()


class RobotControlExecutor(BaseToolExecutor, RobotRuntime):
    """Executor for robot control tools."""

    def __init__(self, serial_port=None, baud_rate=115200, serial_manager=None):
        self._serial_manager = serial_manager
        self._registry = _ROBOT_REGISTRY
        self.serial_connection = None
        self.simulation = True

        if serial_manager is not None:
            self.simulation = serial_manager.simulation
            if not self.simulation:
                _log.info("Robot using shared serial connection")
            else:
                _log.info("Robot using shared serial (simulation mode)")
        elif serial_port and SERIAL_AVAILABLE:
            try:
                self.serial_connection = serial.Serial(
                    port=serial_port,
                    baudrate=baud_rate,
                    timeout=1,
                )
                self.simulation = False
                _log.info("Connected to robot on %s", serial_port)
            except Exception as exc:
                _log.warning("Failed to connect to %s: %s", serial_port, exc)
                _log.warning("   Running in simulation mode")
                self.simulation = True
        else:
            _log.info("Running in simulation mode (no serial port specified)")
            self.simulation = True

    def execute(self, fn_name: str, args: dict) -> str:
        return self._registry.execute(self, fn_name, args)

    def send_command(self, cmd: str) -> str:
        if self._serial_manager is not None:
            return self._serial_manager.send_command(cmd)

        parts = [part.strip() for part in cmd.split("\n") if part.strip()]
        if not parts:
            return "Error: empty command"

        results = [self._send_single_command(part) for part in parts]
        return "; ".join(results)

    def _send_single_command(self, cmd: str) -> str:
        if self.simulation:
            _log.info("[SERIAL SIM] >> %s", cmd)
            return "OK (simulated)"

        try:
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.write(f"{cmd}\n".encode())
                self.serial_connection.flush()
                return "OK"
            _log.warning("Serial connection closed, simulating: %s", cmd)
            return "OK (connection closed, simulated)"
        except Exception as exc:
            _log.warning("Serial Error: %s", exc)
            return f"Serial Error: {exc}"

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def close(self):
        if self._serial_manager is not None:
            return
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            _log.info("Serial connection closed")
