# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""FFW-SH5 0826 — 26-dim action, chunk 40, RELATIVE arms + ABSOLUTE fingers.

Variant of ffw_sh5_0826_used26_h40_config.py that reparameterizes the arm
actions as RELATIVE (action minus current state at train time, added back at
decode). Rationale: N1.7's pretraining corpus is dominated by delta/relative
action distributions (delta-EEF OXE/DROID, relative-arm SO100), so the shared
DiT trunk has adapted to zero-centered targets; full-range absolute joint
targets are out-of-family for it. The finger groups stay ABSOLUTE — they are
0~1.29 flexion command levels, gripper-like signals, the same reason
unitree_g1 keeps its hands ABSOLUTE under RELATIVE arms.

Everything else — 26 commanded dims, chunk 40 (native horizon), full 54-dim
state, single cam_head frame — matches the absolute baseline run
(gr00t_n17_ffw_sh5_0826_b200) for a clean pairwise comparison.

Gotcha: meta/relative_stats.json is coupled to (rep, horizon). The file the
absolute run generated carries no per-step stats for these arm keys — delete
it from the dataset before training/eval so rank 0 regenerates it.
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


def _cfg(rep: ActionRepresentation) -> ActionConfig:
    return ActionConfig(rep=rep, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)


ffw_sh5_0826_used26_h40_relarm_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_head"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(40)),
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_hand_j14_16",
            "left_hand_j18_20",
            "right_hand_j14_16",
            "right_hand_j18_20",
        ],
        action_configs=[
            _cfg(ActionRepresentation.RELATIVE),  # left_arm
            _cfg(ActionRepresentation.RELATIVE),  # right_arm
            _cfg(ActionRepresentation.ABSOLUTE),  # left_hand_j14_16
            _cfg(ActionRepresentation.ABSOLUTE),  # left_hand_j18_20
            _cfg(ActionRepresentation.ABSOLUTE),  # right_hand_j14_16
            _cfg(ActionRepresentation.ABSOLUTE),  # right_hand_j18_20
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}


register_modality_config(
    ffw_sh5_0826_used26_h40_relarm_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)
