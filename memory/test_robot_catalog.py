import unittest

from memory.robot_catalog import build_default_robot_registry


class FakeRobotRuntime:
    def __init__(self):
        self.commands = []
        self.sleeps = []

    def send_command(self, cmd: str) -> str:
        self.commands.append(cmd)
        return "OK"

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


class TestRobotToolRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = build_default_robot_registry()
        self.runtime = FakeRobotRuntime()

    def test_registry_contains_runtime_actions(self):
        tool_names = self.registry.get_tool_names()
        self.assertIn("drive_backward", tool_names)
        self.assertIn("turn_right", tool_names)
        self.assertIn("set_neck_position", tool_names)

    def test_drive_backward_uses_negative_y_axis(self):
        result = self.registry.execute(
            self.runtime,
            "drive_backward",
            {"speed": 40, "duration_ms": 500},
        )

        self.assertEqual(self.runtime.commands, ["Y-40", "q"])
        self.assertEqual(self.runtime.sleeps, [0.5])
        self.assertIn("Driving backward", result)

    def test_set_both_arms_uses_combined_protocol(self):
        result = self.registry.execute(
            self.runtime,
            "set_both_arms",
            {"left": 10, "right": 90},
        )

        self.assertEqual(self.runtime.commands, ["L10\nR90"])
        self.assertEqual(result, "Arms set to left=10, right=90")

    def test_set_neck_position_uses_firmware_top_and_bottom_servos(self):
        result = self.registry.execute(
            self.runtime,
            "set_neck_position",
            {"position": 70},
        )

        self.assertEqual(self.runtime.commands, ["T70", "B30"])
        self.assertEqual(result, "Neck to 70")

    def test_wave_hello_uses_arm_commands(self):
        result = self.registry.execute(self.runtime, "wave_hello", {})

        self.assertEqual(self.runtime.commands[0], "L40")
        self.assertTrue(all(not cmd.startswith("B") for cmd in self.runtime.commands))
        self.assertTrue(all(cmd.startswith(("L", "R")) for cmd in self.runtime.commands))
        self.assertEqual(result, "Waved hello")

    def test_reset_to_neutral_uses_firmware_animation(self):
        result = self.registry.execute(self.runtime, "reset_to_neutral", {})

        self.assertEqual(self.runtime.commands, ["A0"])
        self.assertEqual(result, "Neutral position")


if __name__ == "__main__":
    unittest.main()
