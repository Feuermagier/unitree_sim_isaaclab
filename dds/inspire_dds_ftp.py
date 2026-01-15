# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Gripper DDS communication class
Handle the state publishing and command receiving of the gripper
"""

import threading
from typing import Any, Dict, Optional
from dds.dds_base import DDSObject
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from .inspire_dds_lib._inspire_hand_ctrl import inspire_hand_ctrl
from .inspire_dds_lib._inspire_hand_state import inspire_hand_state
from .inspire_dds_lib import inspire_hand_defaut
import numpy as np


class InspireDDS(DDSObject):
    """Gripper DDS communication class - singleton pattern

    Features:
    - Publish the state of the gripper to DDS (rt/unitree_actuator/state)
    - Receive the control command of the gripper (rt/unitree_actuator/cmd)
    """

    def __init__(self, node_name: str = "inspire"):
        """Initialize the gripper DDS node"""
        # avoid duplicate initialization
        if hasattr(self, "_initialized"):
            return

        super().__init__()
        self.node_name = node_name

        self.msg_left = inspire_hand_defaut.get_inspire_hand_state()
        self.msg_right = inspire_hand_defaut.get_inspire_hand_state()

        self.last_cmd = {
            "positions": [0.0 for _ in range(12)],
            "velocities": [0.0 for _ in range(12)],
            "torques": [0.0 for _ in range(12)],
            "kp": [500.0 for _ in range(12)],
            "kd": [0.0 for _ in range(12)],
        }

        self._initialized = True

        # setup the shared memory
        self.setup_shared_memory(
            input_shm_name="isaac_inspire_state",  # read the state of the gripper from Isaac Lab
            input_size=1024,
            output_shm_name="isaac_inspire_cmd",  # output the command to Isaac Lab
            output_size=1024,  # output the command to Isaac Lab
        )

        print(f"[{self.node_name}] Inspire Hand DDS node initialized")

    def setup_publisher(self) -> bool:
        """Setup the publisher of the gripper"""
        try:
            self.left_publisher = ChannelPublisher(
                "rt/inspire_hand/state/l", inspire_hand_state
            )
            self.left_publisher.Init()

            self.right_publisher = ChannelPublisher(
                "rt/inspire_hand/state/r", inspire_hand_state
            )
            self.right_publisher.Init()

            print(f"[{self.node_name}] Inspire Hand state publishers initialized")
            return True
        except Exception as e:
            print(
                f"gripper_dds [{self.node_name}] Gripper state publishers initialization failed: {e}"
            )
            return False

    def setup_subscriber(self) -> bool:
        """Setup the subscriber of the gripper"""
        try:
            self.left_subscriber = ChannelSubscriber(
                "rt/inspire_hand/ctrl/l", inspire_hand_ctrl
            )
            self.left_subscriber.Init(
                lambda msg: self.dds_subscriber(msg, False, ""), 32
            )

            self.right_subscriber = ChannelSubscriber(
                "rt/inspire_hand/ctrl/r", inspire_hand_ctrl
            )
            self.right_subscriber.Init(
                lambda msg: self.dds_subscriber(msg, True, ""), 32
            )

            print(f"[{self.node_name}] Inspire Hand command subscribers initialized")
            return True
        except Exception as e:
            print(
                f"gripper_dds [{self.node_name}] Gripper command subscriber initialization failed: {e}"
            )
            return False

    def normalize(self, val, min_val, max_val):
        return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)

    def dds_publisher(self) -> Any:
        """Process the publish data: convert the Isaac Lab state to the DDS message

        Expected data format:
        {
            "positions": [2 gripper joint positions] (Isaac Lab joint angle range [-0.02, 0.03])
            "velocities": [2 gripper joint velocities],
            "torques": [2 gripper joint torques]
        }
        """
        try:
            data = self.input_shm.read_data()
            if data is None:
                return
            if all(key in data for key in ["positions", "velocities", "torques"]):
                positions = data["positions"]
                velocities = data["velocities"]
                torques = data["torques"]
                for i in range(6):
                    if i < 4:
                        q = self.normalize(float(positions[i]), 0.0, 1.7)
                    elif i == 4:
                        q = self.normalize(float(positions[i]), 0.0, 0.5)
                    else:
                        q = self.normalize(float(positions[i]), -0.1, 1.3)

                    self.msg_left.angle_act[i] = int(q * 1000)
                self.left_publisher.Write(self.msg_left)

                for i in range(6, 12):
                    if i < 10:
                        q = self.normalize(float(positions[i]), 0.0, 1.7)
                    elif i == 11:
                        q = self.normalize(float(positions[i]), 0.0, 0.5)
                    else:
                        q = self.normalize(float(positions[i]), -0.1, 1.3)

                    self.msg_left.angle_act[i - 6] = int(q * 1000)

                self.right_publisher.Write(self.msg_right)

        except Exception as e:
            print(f"inspire_dds [{self.node_name}] Error processing publish data: {e}")
            return None

    def denormalize(self, norm_val, min_val, max_val):
        return (1.0 - np.clip(norm_val, 0.0, 1.0)) * (max_val - min_val) + min_val

    def dds_subscriber(self, msg: inspire_hand_ctrl, right: bool, datatype: str = None):
        """Process the subscribe data: convert the DDS command to the Isaac Lab format"""
        try:
            offset = 6 if right else 0

            for i in range(6):
                if i < 4:
                    q = self.denormalize(float(msg.angle_set[i]) / 1000, 0.0, 1.7)
                elif i == 4:
                    q = self.denormalize(float(msg.angle_set[i]) / 1000, 0.0, 0.5)
                else:
                    q = self.denormalize(float(msg.angle_set[i]) / 1000, -0.1, 1.3)
                self.last_cmd["positions"][i + offset] = q
            self.output_shm.write_data(self.last_cmd)
        except Exception as e:
            print(
                f"inspire_dds [{self.node_name}] Error processing subscribe data: {e}"
            )
            return None

    def get_inspire_hand_command(self) -> Optional[Dict[str, Any]]:
        """Get the gripper control command

        Returns:
            Dict: the gripper command, return None if there is no new command
        """
        if self.output_shm:
            return self.output_shm.read_data()
        return None

    def write_inspire_state(self, positions, velocities, torques):
        """Write the gripper state to the shared memory

        Args:
            positions: the gripper joint position list or torch.Tensor (Isaac Lab joint angle)
            velocities: the gripper joint velocity list or torch.Tensor
            torques: the gripper joint torque list or torch.Tensor
        """
        try:
            # prepare the gripper data
            inspire_hand_data = {
                "positions": positions.tolist()
                if hasattr(positions, "tolist")
                else positions,
                "velocities": velocities.tolist()
                if hasattr(velocities, "tolist")
                else velocities,
                "torques": torques.tolist() if hasattr(torques, "tolist") else torques,
            }

            # write the input shared memory for publishing
            if self.input_shm:
                self.input_shm.write_data(inspire_hand_data)

        except Exception as e:
            print(
                f"gripper_dds [{self.node_name}] Error writing inspire hand state: {e}"
            )
