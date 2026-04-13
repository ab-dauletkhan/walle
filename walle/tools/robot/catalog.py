"""
Robot tool catalog and command objects.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Sequence


class RobotRuntime(Protocol):
    def send_command(self, cmd: str) -> str: ...

    def sleep(self, seconds: float) -> None: ...

    def heartbeat(self) -> str | None: ...


@dataclass(frozen=True)
class RobotToolSpec:
    name: str
    description: str
    properties: dict
    required: tuple[str, ...] = ()

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


@dataclass(frozen=True)
class RobotPose:
    head_rotation: int | None = None
    neck_top: int | None = None
    neck_bottom: int | None = None
    left_arm: int | None = None
    right_arm: int | None = None

    def to_commands(self) -> list[str]:
        commands = []
        for prefix, value in (
            ("G", self.head_rotation),
            ("T", self.neck_top),
            ("B", self.neck_bottom),
            ("L", self.left_arm),
            ("R", self.right_arm),
        ):
            if value is not None:
                commands.append(f"{prefix}{max(0, min(100, int(value)))}")
        return commands


class RobotAction(ABC):
    def __init__(self, spec: RobotToolSpec):
        self.spec = spec

    @abstractmethod
    def execute(self, runtime: RobotRuntime, args: dict) -> str:
        raise NotImplementedError

    @staticmethod
    def _clamp_percent(value) -> int:
        return max(0, min(100, int(value)))

    @staticmethod
    def _clamp_position(value) -> int:
        return max(0, min(100, int(value)))

    @staticmethod
    def _send_checked(runtime: RobotRuntime, cmd: str) -> None:
        status = str(runtime.send_command(cmd))
        if not all(part.strip().startswith("OK") for part in status.split(";")):
            raise RuntimeError(f"{cmd}: {status}")

    def _apply_pose(self, runtime: RobotRuntime, pose: RobotPose) -> None:
        for cmd in pose.to_commands():
            self._send_checked(runtime, cmd)


class TimedAxisAction(RobotAction):
    def __init__(
        self, spec: RobotToolSpec, axis: str, direction: int, action_label: str
    ):
        super().__init__(spec)
        self._axis = axis
        self._direction = direction
        self._action_label = action_label

    def execute(self, runtime: RobotRuntime, args: dict) -> str:
        speed = self._clamp_percent(args.get("speed", 50))
        duration_ms = max(0, int(args.get("duration_ms", 0)))
        signed_speed = speed * self._direction
        self._send_checked(runtime, f"{self._axis}{signed_speed}")
        if duration_ms > 0:
            self._send_checked(runtime, f"D{duration_ms}")
            return f"{self._action_label} at {speed}% for {duration_ms}ms"
        return f"{self._action_label} at {speed}% (continuous)"


class SetPositionAction(RobotAction):
    def __init__(self, spec: RobotToolSpec, command_prefix: str, label: str):
        super().__init__(spec)
        self._command_prefix = command_prefix
        self._label = label

    def execute(self, runtime: RobotRuntime, args: dict) -> str:
        position = self._clamp_position(args.get("position", 50))
        self._send_checked(runtime, f"{self._command_prefix}{position}")
        return f"{self._label} to {position}"


class CoordinatedNeckAction(RobotAction):
    """Adapter from a single logical neck position to the two-neck-servo firmware protocol."""

    def execute(self, runtime: RobotRuntime, args: dict) -> str:
        position = self._clamp_position(args.get("position", 50))
        pose = RobotPose(
            neck_top=position,
            neck_bottom=100 - position,
        )
        self._apply_pose(runtime, pose)
        return f"Neck to {position}"


class SetBothArmsAction(RobotAction):
    def execute(self, runtime: RobotRuntime, args: dict) -> str:
        left = self._clamp_position(args.get("left", 50))
        right = self._clamp_position(args.get("right", 50))
        self._send_checked(runtime, f"L{left}\nR{right}")
        return f"Arms set to left={left}, right={right}"


class StopMovementAction(RobotAction):
    def execute(self, runtime: RobotRuntime, args: dict) -> str:
        self._send_checked(runtime, "q")
        return "Emergency stop"


class ScanSurroundingsAction(RobotAction):
    _DELAYS = {"slow": 1.5, "normal": 0.8, "fast": 0.4}

    def execute(self, runtime: RobotRuntime, args: dict) -> str:
        speed = str(args.get("speed", "normal"))
        delay = self._DELAYS.get(speed, self._DELAYS["normal"])
        commands = ("G20", "G50", "G80", "G50")
        for index, cmd in enumerate(commands):
            self._send_checked(runtime, cmd)
            if index < len(commands) - 1:
                runtime.sleep(delay)
        return f"Scanned surroundings ({speed})"


class ExpressEmotionAction(RobotAction):
    _EMOTION_COMMANDS = {
        "happy": RobotPose(neck_top=80, neck_bottom=20, left_arm=80, right_arm=80),
        "sad": RobotPose(neck_top=20, neck_bottom=80, left_arm=20, right_arm=20),
        "neutral": RobotPose(neck_top=50, neck_bottom=50, left_arm=40, right_arm=40),
    }

    def execute(self, runtime: RobotRuntime, args: dict) -> str:
        emotion = str(args.get("emotion", "neutral"))
        pose = self._EMOTION_COMMANDS.get(emotion, self._EMOTION_COMMANDS["neutral"])
        self._apply_pose(runtime, pose)
        return f"Expression set to {emotion}"


class WaveHelloAction(RobotAction):
    def execute(self, runtime: RobotRuntime, args: dict) -> str:
        self._send_checked(runtime, "L40")
        self._send_checked(runtime, "R80")
        runtime.sleep(0.3)
        for _ in range(2):
            self._send_checked(runtime, "R35")
            runtime.sleep(0.3)
            self._send_checked(runtime, "R80")
            runtime.sleep(0.3)
        self._send_checked(runtime, "R40")
        return "Waved hello"


class ResetToNeutralAction(RobotAction):
    def execute(self, runtime: RobotRuntime, args: dict) -> str:
        self._send_checked(runtime, "A0")
        return "Neutral position"


class GetStatusAction(RobotAction):
    """Query Arduino for current servo positions and motor state."""

    def execute(self, runtime: RobotRuntime, args: dict) -> str:
        raw = runtime.heartbeat()
        if not raw:
            return "Status unavailable (no connection)"
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> str:
        # FORMAT: "STATUS 248,560,140,475,270,250,290 M0,0"
        # Servos: head, neck_top, neck_bottom, eye_r, eye_l, arm_l, arm_r
        # Motors: left_speed, right_speed
        try:
            parts = raw.split()
            servos = parts[1].split(",")
            motors = parts[2].lstrip("M").split(",")
            motor_l, motor_r = int(motors[0]), int(motors[1])
            moving = motor_l != 0 or motor_r != 0
            return (
                f"motors={'moving' if moving else 'stopped'} "
                f"(left={motor_l}, right={motor_r}), "
                f"head={servos[0]}, neck_top={servos[1]}, neck_bottom={servos[2]}, "
                f"arm_left={servos[5]}, arm_right={servos[6]}"
            )
        except (IndexError, ValueError):
            return f"raw status: {raw}"


class RobotToolRegistry:
    def __init__(self, actions: Sequence[RobotAction]):
        self._actions = list(actions)
        self._action_map = {action.spec.name: action for action in self._actions}

    def get_tool_schemas(self) -> list[dict]:
        return [action.spec.to_schema() for action in self._actions]

    def get_tool_names(self) -> list[str]:
        return [action.spec.name for action in self._actions]

    def execute(self, runtime: RobotRuntime, fn_name: str, args: dict) -> str:
        action = self._action_map.get(fn_name)
        if action is None:
            return f"Unknown tool: {fn_name}"
        return action.execute(runtime, args)


def build_default_robot_registry() -> RobotToolRegistry:
    return RobotToolRegistry(
        [
            # --- Servo position tools ---
            SetPositionAction(
                RobotToolSpec(
                    "set_head_rotation",
                    "Call when the user asks to look left/right or face a direction. "
                    "Sets head rotation angle. Returns confirmation with final position.",
                    {
                        "position": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Head angle: 0=full left, 50=center, 100=full right. Example: 25=slightly left.",
                        }
                    },
                    ("position",),
                ),
                command_prefix="G",
                label="Head",
            ),
            CoordinatedNeckAction(
                RobotToolSpec(
                    "set_neck_position",
                    "Call when the user asks to look up/down or nod. "
                    "Sets neck tilt angle. Returns confirmation with final position.",
                    {
                        "position": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Neck tilt: 0=looking down, 50=level, 100=looking up. Example: 80=looking up.",
                        }
                    },
                    ("position",),
                )
            ),
            SetBothArmsAction(
                RobotToolSpec(
                    "set_both_arms",
                    "Call when the user asks to raise/lower arms or make a gesture. "
                    "Sets both arm positions independently. Returns confirmation.",
                    {
                        "left": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Left arm: 0=fully lowered, 50=horizontal, 100=fully raised.",
                        },
                        "right": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Right arm: 0=fully lowered, 50=horizontal, 100=fully raised.",
                        },
                    },
                    ("left", "right"),
                )
            ),
            # --- Timed movement tools ---
            TimedAxisAction(
                RobotToolSpec(
                    "drive_forward",
                    "Call when the user asks to move forward or approach something. "
                    "Drives forward for the specified duration then stops automatically. "
                    "Returns distance estimate. Reference: speed=50 for 1000ms covers ~30cm.",
                    {
                        "speed": {
                            "type": "integer",
                            "minimum": 10,
                            "maximum": 100,
                            "description": "Motor power in percent. 30=slow, 50=normal, 80=fast.",
                        },
                        "duration_ms": {
                            "type": "integer",
                            "minimum": 100,
                            "maximum": 5000,
                            "description": "Duration in ms, max 5000 per call. For longer moves, call multiple times. Example: 1000=1s.",
                        },
                    },
                    ("speed", "duration_ms"),
                ),
                axis="Y",
                direction=1,
                action_label="Driving forward",
            ),
            TimedAxisAction(
                RobotToolSpec(
                    "drive_backward",
                    "Call when the user asks to move backward or retreat. "
                    "Drives backward for the specified duration then stops automatically. "
                    "Returns distance estimate. Reference: speed=50 for 1000ms covers ~30cm.",
                    {
                        "speed": {
                            "type": "integer",
                            "minimum": 10,
                            "maximum": 100,
                            "description": "Motor power in percent. 30=slow, 50=normal, 80=fast.",
                        },
                        "duration_ms": {
                            "type": "integer",
                            "minimum": 100,
                            "maximum": 5000,
                            "description": "Duration in ms, max 5000 per call. For longer moves, call multiple times. Example: 1000=1s.",
                        },
                    },
                    ("speed", "duration_ms"),
                ),
                axis="Y",
                direction=-1,
                action_label="Driving backward",
            ),
            TimedAxisAction(
                RobotToolSpec(
                    "turn_left",
                    "Call when the user asks to turn/rotate left. "
                    "Rotates left in place for the specified duration then stops. "
                    "Returns angle estimate. Reference at speed=50: 500ms=~90°, 1000ms=~180°, 2000ms=~360°.",
                    {
                        "speed": {
                            "type": "integer",
                            "minimum": 10,
                            "maximum": 100,
                            "description": "Motor power in percent. 50=normal turning speed.",
                        },
                        "duration_ms": {
                            "type": "integer",
                            "minimum": 100,
                            "maximum": 5000,
                            "description": "Duration in ms, max 5000 per call. For longer turns, call multiple times. 500=~90°, 1000=~180°, 2000=~360°.",
                        },
                    },
                    ("speed", "duration_ms"),
                ),
                axis="X",
                direction=-1,
                action_label="Turning left",
            ),
            TimedAxisAction(
                RobotToolSpec(
                    "turn_right",
                    "Call when the user asks to turn/rotate right. "
                    "Rotates right in place for the specified duration then stops. "
                    "Returns angle estimate. Reference at speed=50: 500ms=~90°, 1000ms=~180°, 2000ms=~360°.",
                    {
                        "speed": {
                            "type": "integer",
                            "minimum": 10,
                            "maximum": 100,
                            "description": "Motor power in percent. 50=normal turning speed.",
                        },
                        "duration_ms": {
                            "type": "integer",
                            "minimum": 100,
                            "maximum": 5000,
                            "description": "Duration in ms, max 5000 per call. For longer turns, call multiple times. 500=~90°, 1000=~180°, 2000=~360°.",
                        },
                    },
                    ("speed", "duration_ms"),
                ),
                axis="X",
                direction=1,
                action_label="Turning right",
            ),
            # --- Instant actions ---
            StopMovementAction(
                RobotToolSpec(
                    "stop_movement",
                    "Call to immediately halt all drive motors. Use in emergencies or when the user says stop.",
                    {},
                )
            ),
            ExpressEmotionAction(
                RobotToolSpec(
                    "express_emotion",
                    "Call when the user asks to show a feeling or when contextually appropriate "
                    "(e.g. greet with happy, apologize with sad). "
                    "Sets head, neck, and arm positions to match the emotion.",
                    {
                        "emotion": {
                            "type": "string",
                            "enum": ["happy", "sad", "neutral"],
                            "description": "happy=head up, arms raised. sad=head down, arms lowered. neutral=relaxed centered pose.",
                        }
                    },
                    ("emotion",),
                )
            ),
            ScanSurroundingsAction(
                RobotToolSpec(
                    "scan_surroundings",
                    "Call when the user asks to look around or check the environment. "
                    "Sweeps head left→center→right→center. Returns completion status.",
                    {
                        "speed": {
                            "type": "string",
                            "enum": ["slow", "normal", "fast"],
                            "description": "Pause between positions: slow=1.5s, normal=0.8s, fast=0.4s.",
                        }
                    },
                    ("speed",),
                )
            ),
            WaveHelloAction(
                RobotToolSpec(
                    "wave_hello",
                    "Call when the user says hello/hi/greetings or asks the robot to wave. "
                    "Raises one arm and waves back and forth twice.",
                    {},
                )
            ),
            ResetToNeutralAction(
                RobotToolSpec(
                    "reset_to_neutral",
                    "Call to return all servos (head, neck, arms) to centered resting position. "
                    "Use after finishing a sequence of movements.",
                    {},
                )
            ),
            # --- Feedback ---
            GetStatusAction(
                RobotToolSpec(
                    "get_robot_status",
                    "Call after movement commands to verify execution completed. "
                    "Returns current servo positions and motor state (moving/stopped). "
                    "Use this to confirm the robot finished a maneuver before issuing the next one.",
                    {},
                )
            ),
        ]
    )
