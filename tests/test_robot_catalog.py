import unittest

from walle.tools.robot.catalog import build_default_robot_registry


class FakeRobotRuntime:
    def __init__(self, heartbeat_reply=None):
        self.commands = []
        self.sleeps = []
        self._heartbeat = heartbeat_reply

    def send_command(self, cmd: str) -> str:
        self.commands.append(cmd)
        return "OK -> test ack"

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)

    def heartbeat(self):
        return self._heartbeat


class TestRobotToolRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = build_default_robot_registry()
        self.runtime = FakeRobotRuntime()

    def test_registry_exposes_expected_tools(self):
        names = set(self.registry.get_tool_names())
        self.assertEqual(
            names,
            {
                "drive",
                "stop_movement",
                "head_pan",
                "reset_to_neutral",
                "scan_surroundings",
                "get_robot_status",
            },
        )

    def test_drive_forward_emits_cli_drive(self):
        result = self.registry.execute(
            self.runtime,
            "drive",
            {"direction": "forward", "speed": 50, "duration_ms": 500},
        )
        # 50% of MAX_MOTOR_SPEED=200 → 100 ticks
        self.assertEqual(self.runtime.commands, ["drive 100 500"])
        self.assertIn("forward", result)

    def test_drive_backward_uses_negative_speed(self):
        self.registry.execute(
            self.runtime,
            "drive",
            {"direction": "backward", "speed": 40, "duration_ms": 500},
        )
        self.assertEqual(self.runtime.commands, ["drive -80 500"])

    def test_drive_left_uses_positive_spin(self):
        self.registry.execute(
            self.runtime,
            "drive",
            {"direction": "left", "speed": 50},
        )
        self.assertEqual(self.runtime.commands, ["spin 100"])

    def test_drive_right_uses_negative_spin(self):
        self.registry.execute(
            self.runtime,
            "drive",
            {"direction": "right", "speed": 50, "duration_ms": 300},
        )
        self.assertEqual(self.runtime.commands, ["spin -100 300"])

    def test_head_pan_uses_head_pos(self):
        self.registry.execute(self.runtime, "head_pan", {"position": 70})
        self.assertEqual(self.runtime.commands, ["head pos 70"])

    def test_reset_to_neutral_uses_home(self):
        self.registry.execute(self.runtime, "reset_to_neutral", {})
        self.assertEqual(self.runtime.commands, ["home"])

    def test_stop_movement_uses_stop(self):
        self.registry.execute(self.runtime, "stop_movement", {})
        self.assertEqual(self.runtime.commands, ["stop"])

    def test_scan_surroundings_sweeps_head_pos(self):
        self.registry.execute(self.runtime, "scan_surroundings", {"speed": "fast"})
        self.assertEqual(
            self.runtime.commands,
            ["head pos 20", "head pos 50", "head pos 80", "head pos 50"],
        )
        self.assertEqual(self.runtime.sleeps, [0.4, 0.4, 0.4])

    def test_get_robot_status_summarizes_firmware_block(self):
        runtime = FakeRobotRuntime(
            heartbeat_reply=(
                "Head:\n"
                "  channel = 2\n"
                "  tick = 320\n"
                "  percent = 42\n"
                "  center tick = 350\n"
                "  range = 150..550\n"
                "Motors:\n"
                "  left = 80\n"
                "  right = -40\n"
            )
        )
        result = self.registry.execute(runtime, "get_robot_status", {})
        self.assertIn("moving", result)
        self.assertIn("left=80", result)
        self.assertIn("right=-40", result)
        self.assertIn("42", result)


if __name__ == "__main__":
    unittest.main()
