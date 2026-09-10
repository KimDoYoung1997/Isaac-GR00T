# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""FFW-SH5 left arm only — 8-dim action, chunk 50.

The dataset (``learner1119/260820``) stores 16-dim state/action vectors::

    arm_l_joint1..7   0:7       arm_r_joint1..7    8:15
    gripper_l_joint1  7:8       gripper_r_joint1  15:16

Only the left side moves. Verified over all 44,136 frames: the left dims carry
std 0.17-0.38 (range 0.88-2.08 rad), the right dims std 5e-6 to 2.2e-3 with a
full range of at most 0.0088 rad -- encoder noise, not command.
``gripper_r_joint1`` is constant to the bit.

So this config declares the left 8 only. ``modality.json`` is a slicing spec, so
the right 8 dims are never read and the parquet files need no rewriting. GR00T
N1.5 was trained on this data without the trim and spent half its action head on
dead signal (``ffw_sg2_n15_260820_left_*``); this is that fix.

The hand is an SH5 hand, not a parallel gripper. The teleop stack interpolates
open/close into a single 0-1 scalar, which is what ``gripper_l_joint1`` holds,
so it behaves as 1-dof and needs no per-finger groups. Default min/max
normalisation maps 0-1 onto [-1, 1] exactly.

chunk 50 = 2.5 s at 20 fps. Use with ``--action_horizon 50``.

ABSOLUTE, not RELATIVE: ACT and pi05 were trained on this same data with
absolute joint targets (``local/posco_260820_left``, 8-dim, chunk 20/50), so
keeping ABSOLUTE makes all three directly comparable. The earlier N1.7
``psc_left`` run used RELATIVE for the arm -- a different choice on different
data, not a precedent to follow here.

``state`` keeps the same 8 dims as ``action``. There is nothing else to keep:
unlike the 26-dim SH5 config, the uncommanded dims here belong to the right arm,
which never moves, so they carry no contact information either.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


def _abs_non_eef() -> ActionConfig:
    """Absolute joint targets -- what the FFW-SH5 teleop stack records."""
    return ActionConfig(
        rep=ActionRepresentation.ABSOLUTE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
    )


ffw_sh5_left8_h50 = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_head"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_arm", "left_gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(50)),
        modality_keys=[
            "left_arm",       # 0:7
            "left_gripper",   # 7:8
        ],
        action_configs=[_abs_non_eef() for _ in range(2)],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}


register_modality_config(ffw_sh5_left8_h50, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
