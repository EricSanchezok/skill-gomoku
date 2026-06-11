"""SO101 pose sampler adapter for manual board calibration."""

from __future__ import annotations

import time
from typing import Any

from src.robot.controller import RobotPose


class SO101PoseSampler:
    """Read current SO101 LeRobot action poses while the arm is hand-guided."""

    coordinate_space = "lerobot_action"

    def __init__(
        self,
        port: str,
        robot_id: str,
        max_relative_target: float = 5.0,
        use_degrees: bool = True,
    ) -> None:
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        except ImportError as exc:
            raise ImportError(
                "SO101PoseSampler requires the lerobot package. "
                "Run this script inside the conda/uv environment that controls the arm."
            ) from exc

        config = SO101FollowerConfig(
            port=port,
            id=robot_id,
            max_relative_target=max_relative_target,
            disable_torque_on_disconnect=False,
            use_degrees=use_degrees,
        )
        self.robot: Any = SO101Follower(config)
        self._connected = False

    def connect(self) -> None:
        if not self._connected:
            self.robot.bus.connect()
            self._connected = True

    def disconnect(self) -> None:
        if self._connected:
            self.robot.bus.disconnect(disable_torque=False)
            self._connected = False

    def prepare_manual_guidance(self) -> None:
        self.connect()
        self.robot.bus.disable_torque(num_retry=5)
        print("SO101 torque disabled. You can now guide the arm by hand.")

    def read_current_pose(self) -> RobotPose:
        obs = self.robot.get_observation()
        pose = {key: float(value) for key, value in obs.items() if key.endswith(".pos")}
        if not pose:
            raise RuntimeError("No '.pos' keys found in SO101 observation")
        return pose

    def finish_manual_guidance(self, hold: bool = False) -> None:
        try:
            if hold:
                action = self.read_current_pose()
                self.robot.send_action(action)
                time.sleep(0.2)
                self.robot.bus.enable_torque(num_retry=5)
                print("SO101 is holding the last calibrated pose.")
        finally:
            self.disconnect()
