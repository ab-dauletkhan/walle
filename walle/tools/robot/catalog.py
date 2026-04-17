"""
Robot tool catalog for the WALL-E CLI firmware
(`firmware/wall-e/motor_control_cli.ino`).

Every action emits one or more text-line commands from the firmware's help:

    help | status | home | off | stop | stopall | trim VALUE
    head center | head pos PCT | head tick TICK | head step DELTA | head off
    left SPEED [MS] | right SPEED [MS] | drive SPEED [MS]
    spin SPEED [MS] | tank LEFT RIGHT [MS] | mix MOVE TURN [MS]

Speed range is -200..200. Duration is optional and capped at 5000 ms by the
firmware. Head pan lives on PCA9685 channel 2 with ticks 150..550 and a
60-tick-per-command safety clamp, so we rely on `head step` for smooth
tracking updates instead of jumping absolute ticks.

Only actions that the hardware can actually perform are exposed. The old
arm / neck / eye servos were removed from the firmware, so the matching
tools (`express_emotion`, `wave_hello`, etc.) no longer exist here — the
LLM should not be given the ability to call commands that silently fail.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence


# Hardware limits mirrored from motor_control_cli.ino so the Python layer
# can clamp before hitting the firmware's "Rejected:" path.
MAX_MOTOR_SPEED = 200
MAX_TIMED_MOVE_MS = 5000
HEAD_TICK_MIN = 150
HEAD_TICK_MAX = 550
HEAD_TICK_CENTER = 350
HEAD_MAX_DELTA = 60


class RobotRuntime(Protocol):
    def send_command(self, cmd: str) -> str: ...

    def sleep(self, seconds: float) -> None: ...

    def heartbeat(self) -> Optional[str]: ...


@dataclass(frozen=True)
class RobotToolSpec:
    name: str
    description: str
    properties: dict
    required: tuple = ()

    def to_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": copy.deepcopy(self.properties),
                    "required": list(self.required),
                },
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(value, low: int, high: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = 0
    return max(low, min(high, v))


def _clamp_speed(value) -> int:
    return _clamp(value, -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED)


def _clamp_speed_percent(value) -> int:
    """LLM-facing speed is 0..100 for readability; clamp + scale to ticks."""
    percent = _clamp(value, 0, 100)
    return int(round(percent * MAX_MOTOR_SPEED / 100))


def _clamp_duration(value) -> int:
    if value is None:
        return 0
    return _clamp(value, 0, MAX_TIMED_MOVE_MS)


class RobotAction(ABC):
    def __init__(self, spec: RobotToolSpec):
        self.spec = spec

    @abstractmethod
    def execute(self, runtime: RobotRuntime, args: dict) -> str:
        raise NotImplementedError

    @staticmethod
    def _send_checked(runtime: RobotRuntime, cmd: str) -> str:
        """Send a line and raise if the firmware rejected it.

        Accepts both ``OK`` (old protocol, simulation) and ``OK -> ...`` (new
        CLI firmware). Multi-line replies are joined by ``;`` in SerialManager
        so we only need to check whether every segment starts with ``OK``.
        """
        status = str(runtime.send_command(cmd))
        for part in status.split(";"):
            part = part.strip()
            if not part or part.startswith("OK"):
                continue
            raise RuntimeError(f"{cmd}: {status}")
        return status


# ---------------------------------------------------------------------------
# Drive / wheels
# ---------------------------------------------------------------------------


class DriveAction(RobotAction):
    """High-level `drive` tool — dispatches to firmware CLI commands.

    direction=forward|backward → ``drive SPEED MS`` (both wheels same sign)
    direction=left|right       → ``spin SPEED MS`` (tank turn in place)
    """

    def execute(self, runtime: RobotRuntime, args: dict) -> str:
        direction = str(args.get("direction", "forward")).lower()
        speed_percent = _clamp(args.get("speed", 50), 0, 100)
        duration_ms = _clamp_duration(args.get("duration_ms", 0))
        speed_abs = _clamp_speed_percent(speed_percent)

        if direction == "forward":
            signed = speed_abs
            cli = f"drive {signed}"
        elif direction == "backward":
            signed = -speed_abs
            cli = f"drive {signed}"
        elif direction == "left":
            # spin SPEED uses left=-speed, right=+speed; positive = spin left
            cli = f"spin {speed_abs}"
        elif direction == "right":
            cli = f"spin {-speed_abs}"
        else:
            return f"Unknown direction: {direction}"

        if duration_ms > 0:
            cli = f"{cli} {duration_ms}"
            suffix = f" for {duration_ms}ms"
        else:
            suffix = " (continuous)"

        self._send_checked(runtime, cli)
        return f"Driving {direction} at {speed_percent}%{suffix}"


class TankDriveAction(RobotAction):
    """Direct left/right wheel speeds. Lets the vision/tracking loop steer."""

    def execute(self, runtime: RobotRuntime, args: dict) -> str:
        left = _clamp_speed(args.get("left", 0))
        right = _clamp_speed(args.get("right", 0))
        duration_ms = _clamp_duration(args.get("duration_ms", 0))
        cli = f"tank {left} {right}"
        if duration_ms > 0:
            cli = f"{cli} {duration_ms}"
        self._send_checked(runtime, cli)
        return f"tank left={left} right={right}" + (
            f" for {duration_ms}ms" if duration_ms else ""
        )


class StopMovementAction(RobotAction):
    def execute(self, runtime: RobotRuntime, args: dict) -> str:
        self._send_checked(runtime, "stop")
        return "Motors stopped"


class HomeAction(RobotAction):
    def execute(self, runtime: RobotRuntime, args: dict) -> str:
        global _head_tick_cache
        self._send_checked(runtime, "home")
        _head_tick_cache = HEAD_TICK_CENTER
        return "Head centered"


class StopAllAction(RobotAction):
    def execute(self, runtime: RobotRuntime, args: dict) -> str:
        global _head_tick_cache
        self._send_checked(runtime, "stopall")
        # `stopall` disables head PWM; next move will start from
        # wherever the servo physically rested, so drop the cache.
        _head_tick_cache = None
        return "Motors stopped, head servo off"


# ---------------------------------------------------------------------------
# Head pan (single servo on PCA9685 channel 2)
# ---------------------------------------------------------------------------


def _percent_to_tick(percent: int) -> int:
    percent = _clamp(percent, 0, 100)
    span = HEAD_TICK_MAX - HEAD_TICK_MIN
    return HEAD_TICK_MIN + int(round(span * percent / 100))


# Module-level cache of the last commanded head tick. The firmware's
# 60-tick-per-command delta is the real source of truth for what's
# valid, so tracking it here lets us split a big absolute move into
# multiple `head step` calls without a serial round-trip to query.
# `None` means "we haven't commanded it yet"; fall back to center.
_head_tick_cache: Optional[int] = None


def _walk_head_to_tick(action: "RobotAction", runtime: RobotRuntime, target: int) -> int:
    """Drive the head from the cached position to `target` via ±60 steps.

    Returns the final tick (== target on success). Mutates the module
    cache so subsequent calls know where the servo is.
    """
    global _head_tick_cache
    target = max(HEAD_TICK_MIN, min(HEAD_TICK_MAX, target))
    current = (
        _head_tick_cache
        if _head_tick_cache is not None
        else HEAD_TICK_CENTER
    )
    delta = target - current
    while abs(delta) > HEAD_MAX_DELTA:
        step = HEAD_MAX_DELTA if delta > 0 else -HEAD_MAX_DELTA
        action._send_checked(runtime, f"head step {step}")
        current += step
        delta = target - current
    if delta != 0:
        action._send_checked(runtime, f"head step {delta}")
        current += delta
    _head_tick_cache = current
    return current


class HeadPanAction(RobotAction):
    """Absolute head position as a percentage, 0=left, 50=center, 100=right.

    The firmware enforces max 60 ticks of jump per command. We emit a
    short sequence of `head step ±60` calls to walk the servo to the
    target, then a final step for the remainder. This matches the
    docstring promise that large swings don't require the LLM to
    micro-step.
    """

    def execute(self, runtime: RobotRuntime, args: dict) -> str:
        percent = _clamp(args.get("position", 50), 0, 100)
        _walk_head_to_tick(self, runtime, _percent_to_tick(percent))
        return f"Head pan -> {percent}%"


class HeadStepAction(RobotAction):
    """Relative head move by raw ticks, capped to firmware's ±60 safety."""

    def execute(self, runtime: RobotRuntime, args: dict) -> str:
        global _head_tick_cache
        delta = _clamp(args.get("delta", 0), -HEAD_MAX_DELTA, HEAD_MAX_DELTA)
        if delta == 0:
            return "Head step 0 (no-op)"
        self._send_checked(runtime, f"head step {delta}")
        base = (
            _head_tick_cache
            if _head_tick_cache is not None
            else HEAD_TICK_CENTER
        )
        _head_tick_cache = max(HEAD_TICK_MIN, min(HEAD_TICK_MAX, base + delta))
        return f"Head step {delta:+d}"


class ScanSurroundingsAction(RobotAction):
    _DELAYS = {"slow": 1.5, "normal": 0.8, "fast": 0.4}

    def execute(self, runtime: RobotRuntime, args: dict) -> str:
        speed = str(args.get("speed", "normal"))
        delay = self._DELAYS.get(speed, self._DELAYS["normal"])
        # Go 20% → 50% → 80% → 50%, walking each leg through the
        # firmware's 60-tick delta using the shared helper so no leg
        # gets rejected mid-scan.
        targets = (20, 50, 80, 50)
        for index, pct in enumerate(targets):
            _walk_head_to_tick(self, runtime, _percent_to_tick(pct))
            if index < len(targets) - 1:
                runtime.sleep(delay)
        return f"Scanned surroundings ({speed})"


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class GetStatusAction(RobotAction):
    """Query the firmware `status` block and compact it for the LLM."""

    def execute(self, runtime: RobotRuntime, args: dict) -> str:
        raw = runtime.heartbeat()
        if not raw:
            return "Status unavailable (no connection)"
        return self._summarize(raw)

    @staticmethod
    def _summarize(raw: str) -> str:
        head_tick: Optional[str] = None
        head_pct: Optional[str] = None
        left_motor: Optional[str] = None
        right_motor: Optional[str] = None
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith("tick ="):
                head_tick = stripped.split("=", 1)[1].strip()
            elif stripped.startswith("percent ="):
                head_pct = stripped.split("=", 1)[1].strip()
            elif stripped.startswith("left ="):
                left_motor = stripped.split("=", 1)[1].strip()
            elif stripped.startswith("right ="):
                right_motor = stripped.split("=", 1)[1].strip()
        if head_tick is None and left_motor is None:
            # Firmware didn't answer in the expected format — just echo it.
            return f"raw status: {raw[:200]}"
        moving = (left_motor not in (None, "0")) or (right_motor not in (None, "0"))
        return (
            f"motors={'moving' if moving else 'stopped'} "
            f"(left={left_motor or 0}, right={right_motor or 0}), "
            f"head tick={head_tick or '?'} ({head_pct or '?'}%)"
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class RobotToolRegistry:
    def __init__(self, actions: Sequence[RobotAction]):
        self._actions = list(actions)
        self._action_map = {action.spec.name: action for action in self._actions}

    def get_tool_schemas(self) -> List[dict]:
        return [action.spec.to_schema() for action in self._actions]

    def get_tool_names(self) -> List[str]:
        return [action.spec.name for action in self._actions]

    def execute(self, runtime: RobotRuntime, fn_name: str, args: dict) -> str:
        action = self._action_map.get(fn_name)
        if action is None:
            return f"Unknown tool: {fn_name}"
        return action.execute(runtime, args)


def build_default_robot_registry() -> RobotToolRegistry:
    return RobotToolRegistry(
        [
            DriveAction(
                RobotToolSpec(
                    "drive",
                    "Drive the robot's WHEELS to move the whole body. Use for: "
                    "'move forward/backward', 'go forward', 'turn left/right' "
                    "(left/right spin the body in place). DO NOT use for "
                    "'look left/right' — that is a head motion, use head_pan "
                    "instead. direction=forward/backward/left/right. speed is "
                    "0-100 (percent of max). duration_ms is optional, max 5000; "
                    "omit or set 0 for continuous motion that keeps running "
                    "until `stop_movement` is called or the firmware watchdog "
                    "fires after 3s.",
                    {
                        "direction": {
                            "type": "string",
                            "enum": ["forward", "backward", "left", "right"],
                        },
                        "speed": {"type": "integer", "minimum": 0, "maximum": 100},
                        "duration_ms": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": MAX_TIMED_MOVE_MS,
                        },
                    },
                    ("direction", "speed"),
                )
            ),
            StopMovementAction(
                RobotToolSpec(
                    "stop_movement",
                    "Immediately stop both wheel motors.",
                    {},
                )
            ),
            HeadPanAction(
                RobotToolSpec(
                    "head_pan",
                    "Turn just the HEAD horizontally (body stays still). Use "
                    "for: 'look left/right', 'turn your head', 'face the "
                    "door', 'look at me'. Do NOT use `drive` for these — "
                    "that spins the whole robot. position=0 is fully left, "
                    "50 is centered, 100 is fully right.",
                    {
                        "position": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                        }
                    },
                    ("position",),
                )
            ),
            HomeAction(
                RobotToolSpec(
                    "reset_to_neutral",
                    "Center the head back to its home position.",
                    {},
                )
            ),
            ScanSurroundingsAction(
                RobotToolSpec(
                    "scan_surroundings",
                    "Sweep the head left-center-right-center to look around.",
                    {
                        "speed": {
                            "type": "string",
                            "enum": ["slow", "normal", "fast"],
                        }
                    },
                    ("speed",),
                )
            ),
            GetStatusAction(
                RobotToolSpec(
                    "get_robot_status",
                    "Report the current head position and wheel motor speeds.",
                    {},
                )
            ),
        ]
    )
