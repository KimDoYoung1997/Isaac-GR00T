# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""FFW-SH5 left arm only -- 8-dim action, chunk 50, **relative** arm.

RELATIVE 짝. 절대 좌표판은 ``ffw_sh5_left8_h50_config.py`` 이고, 데이터셋 슬라이싱
(``left_arm`` 0:7, ``left_gripper`` 7:8), horizon 50, 카메라, 언어 키까지 전부
같다. 다른 것은 ``left_arm`` 의 표현 하나뿐이다.

왜 두 벌을 만드는가
-------------------
행동 청킹 논문이 absolute qpos 가 아니라 delta action 으로 학습한다. 같은 데이터
같은 horizon 에서 표현만 바꿔 4개 런(ABS/REL × tune_visual on/off)을 돌려 비교한다.

``left_arm`` 만 RELATIVE, ``left_gripper`` 는 ABSOLUTE
------------------------------------------------------
gripper 는 SH5 hand 의 쥠/폄을 0-1 로 interpolation 한 스칼라다. 여기에 delta 를
씌우면 "0.03 만큼 더 쥐어라" 가 되는데, 원래 신호가 이산적인 열림/닫힘이라
누적 오차가 그대로 남고 정규화 범위도 무너진다. 이전 N1.7 ``psc_left`` 런도
``left_arm`` RELATIVE / ``left_hand`` ABSOLUTE 로 같은 선택을 했다.

RELATIVE 가 실제로 적용되려면
-----------------------------
1. 모델 쪽 ``use_relative_action=True`` 가 함께 켜져야 한다. 처리 코드가
   ``action_config.rep == RELATIVE and self.use_relative_action`` 로 **둘 다**
   확인한다 (``gr00t/data/state_action/state_action_processor.py``). 한쪽만 켜면
   조용히 absolute 로 학습된다.
2. ``meta/relative_stats.json`` 이 있어야 한다. 이 config 를 넘겨 생성한다::

       python gr00t/data/stats.py \
           --dataset-path ~/ku_doyoung/datasets/posco_260820 \
           --embodiment-tag new_embodiment \
           --modality-config-path examples/FFW_SH5/ffw_sh5_left8_h50_rel_config.py

   ``generate_rel_stats`` 는 modality config 에서 rep 이 RELATIVE 인 키만 골라
   통계를 만든다. 그래서 ABS config 로 돌리면 ``relative_stats.json`` 은 비어 있다.

변환 기준점은 **state 의 마지막 타임스텝**이다 (``reference_state =
state[state_key][-1]``). 즉 관측 시점 관절각 기준의 delta 이고, chunk 안의 50 스텝이
모두 같은 기준점을 쓴다 -- 스텝 간 차분이 아니다.

.. warning::
   이 파일과 ABS 판은 둘 다 ``EmbodimentTag.NEW_EMBODIMENT`` 로 등록한다. 한
   프로세스에서 동시에 import 하면 나중 것이 이긴다. 런마다 ``--modality-config-path``
   로 하나만 넘길 것.
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


def _rel_non_eef() -> ActionConfig:
    """관측 시점 관절각 기준의 delta."""
    return ActionConfig(
        rep=ActionRepresentation.RELATIVE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
    )


def _abs_non_eef() -> ActionConfig:
    """절대 관절 목표값."""
    return ActionConfig(
        rep=ActionRepresentation.ABSOLUTE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
    )


ffw_sh5_left8_h50_rel = {
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
            "left_arm",       # 0:7   RELATIVE
            "left_gripper",   # 7:8   ABSOLUTE
        ],
        action_configs=[_rel_non_eef(), _abs_non_eef()],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}


register_modality_config(ffw_sh5_left8_h50_rel, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
