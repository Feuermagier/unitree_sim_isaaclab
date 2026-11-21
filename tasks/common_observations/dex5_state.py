# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
dex5 state
"""     
from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import sys
import os
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


import torch


_obs_cache = {
    "device": None,
    "batch": None,
    "hand_idx_t": None,
    "hand_idx_batch": None,
    "pos_buf": None,
    "vel_buf": None,
    "torque_buf": None,
    "dds_last_ms": 0,
    "dds_min_interval_ms": 20,
}

def get_robot_girl_joint_names() -> list[str]:
    return [
        "Roll_21L",
        "Pitch_22L",
        "Pitch_23L",
        "Pitch_24L",
        "Roll_31L",
        "Pitch_32L",
        "Pitch_33L",
        "Pitch_34L",
        "Roll_41L",
        "Pitch_42L",
        "Pitch_43L",
        "Pitch_44L",
        "Roll_51L",
        "Pitch_52L",
        "Pitch_53L",
        "Pitch_54L",
        "Yaw_11L",
        "Roll_12L",
        "Pitch_13L",
        "Pitch_14L",


        "Roll_21R",
        "Pitch_22R",
        "Pitch_23R",
        "Pitch_24R",
        "Roll_31R",
        "Pitch_32R",
        "Pitch_33R",
        "Pitch_34R",
        "Roll_41R",
        "Pitch_42R",
        "Pitch_43R",
        "Pitch_44R",
        "Roll_51R",
        "Pitch_52R",
        "Pitch_53R",
        "Pitch_54R",
        "Yaw_11R" ,
        "Roll_12R",
        "Pitch_13R",
        "Pitch_14R",
    ]

# global variable to cache the DDS instance
_dex5_dds = None
_dds_initialized = False

def _get_dex5_dds_instance():
    """get the DDS instance, delay initialization"""
    global _dex5_dds, _dds_initialized
    
    if not _dds_initialized or _dex5_dds is None:
        try:
            # dynamically import the DDS module
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dds'))
            from dds.dds_master import dds_manager
            
            _dex5_dds = dds_manager.get_object("dex5")
            print("[Observations dex5] DDS communication instance obtained")
            
            # register the cleanup function
            import atexit
            def cleanup_dds():
                try:
                    if _dex5_dds:
                        dds_manager.unregister_object("dex5")
                        print("[dex5_state] DDS communication closed correctly")
                except Exception as e:
                    print(f"[dex5_state] Error closing DDS: {e}")
            atexit.register(cleanup_dds)
            
        except Exception as e:
            print(f"[Observations dex5] Failed to get DDS instances: {e}")
            _dex5_dds = None
        
        _dds_initialized = True
    
    return _dex5_dds

def get_robot_dex5_joint_states(
    env: ManagerBasedRLEnv,
    enable_dds: bool = True,
) -> torch.Tensor:
    """get the robot gripper joint states and publish them to DDS
    
    Args:
        env: ManagerBasedRLEnv - reinforcement learning environment instance
        enable_dds: bool - whether to enable the DDS publish function
    
    Returns:
        torch.Tensor
    """
    # get the gripper joint positions, velocities, torques
    joint_pos = env.scene["robot"].data.joint_pos
    joint_vel = env.scene["robot"].data.joint_vel  
    joint_torque = env.scene["robot"].data.applied_torque
    device = joint_pos.device
    batch = joint_pos.shape[0]
    

    global _obs_cache
    if _obs_cache["device"] != device or _obs_cache["hand_idx_t"] is None:
        gripper_joint_indices=[i+27 for i in [0, 10, 20, 30, 1, 11, 21, 31, 2, 12, 22, 32, 3, 13, 23, 33, 4, 14, 24, 34, 5, 15, 25, 35, 6, 16, 26, 36, 7, 17, 27, 37, 8, 18, 28, 38, 9, 19, 29, 39]]
        _obs_cache["hand_idx_t"] = torch.tensor(gripper_joint_indices, dtype=torch.long, device=device)
        _obs_cache["device"] = device
        _obs_cache["batch"] = None
    idx_t = _obs_cache["hand_idx_t"]
    n = idx_t.numel()


    if _obs_cache["batch"] != batch or _obs_cache["hand_idx_batch"] is None:
        _obs_cache["hand_idx_batch"] = idx_t.unsqueeze(0).expand(batch, n)
        _obs_cache["pos_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _obs_cache["vel_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _obs_cache["torque_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _obs_cache["batch"] = batch

    idx_batch = _obs_cache["hand_idx_batch"]
    pos_buf = _obs_cache["pos_buf"]
    vel_buf = _obs_cache["vel_buf"]
    torque_buf = _obs_cache["torque_buf"]


    try:
        torch.gather(joint_pos, 1, idx_batch, out=pos_buf)
        torch.gather(joint_vel, 1, idx_batch, out=vel_buf)
        torch.gather(joint_torque, 1, idx_batch, out=torque_buf)
    except TypeError:
        pos_buf.copy_(torch.gather(joint_pos, 1, idx_batch))
        vel_buf.copy_(torch.gather(joint_vel, 1, idx_batch))
        torque_buf.copy_(torch.gather(joint_torque, 1, idx_batch))
    
    # publish to DDS (only publish the data of the first environment)
    if enable_dds and len(pos_buf) > 0:
        try:
            import time
            now_ms = int(time.time() * 1000)
            if now_ms - _obs_cache["dds_last_ms"] >= _obs_cache["dds_min_interval_ms"]:
                dex5_dds = _get_dex5_dds_instance()
                if dex5_dds:
                    pos = pos_buf[0].contiguous().cpu().numpy()
                    vel = vel_buf[0].contiguous().cpu().numpy()
                    torque = torque_buf[0].contiguous().cpu().numpy()
                    left_pos = pos[:20]
                    right_pos = pos[20:]
                    left_vel = vel[:20]
                    right_vel = vel[20:]
                    left_torque = torque[:20]
                    right_torque = torque[20:]
                    dex5_dds.write_hand_states(left_pos, left_vel, left_torque, right_pos, right_vel, right_torque)
                    _obs_cache["dds_last_ms"] = now_ms
        except Exception as e:
            print(f"dex5_state [dex5_state] Failed to write to DDS: {e}")
    
    return pos_buf

