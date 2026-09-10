# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""FFW-SH5 rev1 — 26-dim action, chunk 40. GR00T port of the RLDX-1 config.

Mirrors ``rldx/configs/data/ffw_sh5_rev1_config_used26_h40.py`` so the two
models are trained on the same control space and the same horizon, and their
held-out numbers are directly comparable. Use with ``--action_horizon 40``.

chunk 40 = 4.0 s at 10 fps; execute only the first 20 at deployment for the
intended 2.0 s of committed motion. That 20/40 ratio is a client-side choice,
not baked into the checkpoint.

Why 26 and not 57
-----------------
The raw vectors are 57-dim but the teleop stack drives far less. Verified over
all 33,018 frames of ``rlwrld_demo_0826_merge``: 28 of the 54 arm+hand dims are
identically constant — every finger joint except 14/15/16 and 18/19/20 on each
hand (joint 17, the spread axis, is constant too, which is why each hand splits
into two groups rather than one).

``head_joint1/2`` and ``lift_joint`` (54:57) are not bit-identical across the
dataset, but their full range is 0.003 / 0.00007 rad against 0.12-0.71 rad on
the finger joints — encoder quantisation, not command. They stay out.

Training the action head on all 54 is not wrong (q99 normalisation collapses the
constant dims), it just spends decoder width and gradient on dims that carry no
signal.

``state`` deliberately keeps the full 20-dim hands. Proprioception is free: the
uncommanded finger joints still deflect and may carry contact information the
commands do not. Only the *output* side is trimmed. state and action modality
keys are consumed independently, so they do not have to match.

Requires the four extra action groups in the dataset's ``meta/modality.json``::

    left_hand_j14_16   27:30      right_hand_j14_16   47:50
    left_hand_j18_20   31:34      right_hand_j18_20   51:54

The copy on the Hub does **not** have them; add them before training or the
loader fails to resolve ``left_hand_j14_16``.
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
    """Absolute joint targets — what the FFW-SH5 teleop stack records."""
    return ActionConfig(
        rep=ActionRepresentation.ABSOLUTE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
    )


ffw_sh5_used26_h40 = {
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
            "left_arm",             # 0:7
            "right_arm",            # 7:14
            "left_hand_j14_16",     # 27:30
            "left_hand_j18_20",     # 31:34
            "right_hand_j14_16",    # 47:50
            "right_hand_j18_20",    # 51:54
        ],
        action_configs=[_abs_non_eef() for _ in range(6)],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}


register_modality_config(ffw_sh5_used26_h40, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
