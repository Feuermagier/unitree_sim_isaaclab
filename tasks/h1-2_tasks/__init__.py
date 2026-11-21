
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""Unitree G1 robot task module
contains various task implementations for the G1 robot, such as pick and place, motion control, etc.
"""

# use relative import

from . import pick_place_cylinder_h12_27dof_inspire
from . import stack_rgyblock_h12_27dof_inspire
from . import pick_place_redblock_h12_27dof_inspire
from . import pick_place_cylinder_h12_27dof_inspire_wholebody
from . import pick_place_cylinder_h12_27dof_dex5_wholebody
from . import pick_place_cylinder_h12_27dof_dex5_fix_base


# export all modules
__all__ = [
        "pick_place_cylinder_h12_27dof_inspire",
        "pick_place_cylinder_h12_27dof_inspire_wholebody",
        "stack_rgyblock_h12_27dof_inspire",
        "pick_place_redblock_h12_27dof_inspire",
        "pick_place_cylinder_h12_27dof_dex5_wholebody",
        "pick_place_cylinder_h12_27dof_dex5_fix_base",
]