# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""FFW-SH5 rev1, 0826 merge — 26-dim action, chunk 40, dual arm + hands.

Mirrors the RLDX-1 config `ffw_sh5_rev1_config_used26_h40.py` used for the
run this one is compared against, on the identical train/holdout split
(rlwrld_demo_0826_train 72 ep / rlwrld_demo_0826_holdout 9 ep).

chunk 40 = 4.0 s at 10 fps and is exactly the N1.7 action head's native
horizon (gr00t/configs/model/gr00t_n1d7.py: action_horizon = 40), so the
action mask is all-ones and nothing is padded. At deployment execute the
first 20 (2.0 s) and re-query.

The raw vectors are 57-dim, but measured over all 33,018 frames of the 0826
merge, `action` is identically 0.000 on 28 of the 54 arm+hand dims: every
finger joint except 14/15/16 and 18/19/20 on each hand, and head/lift never
receive a command. The real control space is 26-dim:

    left_arm 7 + right_arm 7 + 4 x 3 commanded finger-flexion joints.

The `*_hand_j14_16` / `*_hand_j18_20` slices already exist in the dataset's
meta/modality.json (indices 27:30 / 31:34 and 47:50 / 51:54).

`state` deliberately keeps the full 20-dim hands: proprioception is free —
the uncommanded finger joints still deflect and that reading may carry
contact information the commands do not. Only the output side is trimmed.

All actions are ABSOLUTE joint targets (verified against the raw parquet),
so relative_stats.json carries no per-step content for these keys.
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
    return ActionConfig(
        rep=ActionRepresentation.ABSOLUTE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
    )


ffw_sh5_0826_used26_h40_config = {
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
        action_configs=[_abs_non_eef() for _ in range(6)],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}


register_modality_config(
    ffw_sh5_0826_used26_h40_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)
