# FFW SH5 데이터로 GR00T N1.7 파인튜닝

이 문서는 다음 데이터셋을 이용해 GR00T N1.7을 파인튜닝하는 절차를 정리한다.

- 저장소: `/workspace/GR00TN17/Isaac-GR00T`
- 베이스 모델: `nvidia/GR00T-N1.7-3B`
- 데이터셋: `/workspace/dataset/merge_ffw_sh5_rev1_260627_psc_left`
- Embodiment tag: `NEW_EMBODIMENT`
- Modality config: `examples/FFW_SH5/ffw_sh5_config.py` (기본), `ffw_sh5_config_abs.py` (absolute ablation)
- 실행 스크립트: `finetune_ffw_sh5.sh` 외 dropout/action ablation 3종 (아래 §2.1)
- 기본 출력 경로: `ffw_sh5_n17-checkpoint1`

## 1. 적용한 modality

데이터셋의 `meta/modality.json`과 GR00T 공식 가이드
[`getting_started/data_config.md`](getting_started/data_config.md)를 기준으로
`examples/FFW_SH5/ffw_sh5_config.py`를 작성했다.

공식 문서의 핵심 규칙은 다음과 같다.

- modality config는 `video`, `state`, `action`, `language` 4개 key를 가진다.
- 각 `modality_keys`는 `meta/modality.json`의 key와 **정확히 일치**해야 한다.
- `action`에는 `modality_keys`와 **같은 길이·순서**의 `action_configs`가 필요하다.
- `delta_indices`는 시간축 샘플링을 정의한다. action horizon을 바꾸면
  `meta/relative_stats.json`도 같이 맞춰야 한다.
- custom robot은 `.py` config를 만든 뒤 `register_modality_config()`로 등록하고,
  파인튜닝 시 `--modality-config-path`로 넘긴다.

### 입력 영상

- `cam_head` 한 대만 사용
- 현재 프레임만 사용: `delta_indices=[0]`
- 원본 LeRobot key: `observation.images.cam_head`
- 데이터셋에는 wrist camera 정보도 있지만 현재 `meta/modality.json`에 등록되어 있지 않아 사용하지 않는다.

### State

- `left_arm`: index `[0:7]`, 7 DoF
- `left_hand`: index `[7:27]`, 20 DoF
- 총 27 DoF
- 현재 state만 사용: `delta_indices=[0]`

### Action

- 예측 horizon: 30 step (action chunk size)
- `left_arm`: 7 DoF, `RELATIVE`
- `left_hand`: 20 DoF, `ABSOLUTE`
- joint-space 데이터이므로 두 항목 모두 `NON_EEF`, `DEFAULT` 형식

공식 SO100 예제와 같은 패턴으로 팔은 relative action, 손은 absolute action으로
설정했다. 데이터셋 parquet에는 absolute joint target이 저장되어 있고,
`RELATIVE` arm action은 processor가 현재 state 기준 delta로 변환한다.
손가락 target은 drift를 피하기 위해 absolute action으로 유지한다.

### Language

- key: `annotation.human.task_description`
- `meta/modality.json`에서 `task_index`로 연결되고 실제 문장은
  `meta/tasks.jsonl`에서 읽는다.

## 1.1 `data_config.md` ↔ `ffw_sh5_config.py` 대응표

공식 문서의 `ModalityConfig` / `ActionConfig` 규칙이 아래처럼 반영되어 있다.

| 공식 문서 개념 | `meta/modality.json` | `ffw_sh5_config.py` | 반영 내용 |
|---|---|---|---|
| video key | `cam_head` | `"modality_keys": ["cam_head"]` | head camera 1대만 사용 |
| video time | - | `delta_indices=[0]` | 현재 프레임만 사용 |
| state key | `left_arm`, `left_hand` | `"modality_keys": ["left_arm", "left_hand"]` | 27 DoF state slice |
| state time | - | `delta_indices=[0]` | 현재 state만 사용 |
| action key | `left_arm`, `left_hand` | `"modality_keys": ["left_arm", "left_hand"]` | 27 DoF action slice |
| action horizon | - | `delta_indices=list(range(30))` | 30-step chunk 예측 |
| action rep | - | arm=`RELATIVE`, hand=`ABSOLUTE` | arm delta / hand absolute |
| action type/format | - | `NON_EEF`, `DEFAULT` | joint-space 제어 |
| language key | `human.task_description` | `"annotation.human.task_description"` | task instruction 사용 |
| registration | - | `register_modality_config(..., NEW_EMBODIMENT)` | custom embodiment 등록 |

### `meta/modality.json`과 config key 연결

```json
{
  "state": {
    "left_arm": {"start": 0, "end": 7},
    "left_hand": {"start": 7, "end": 27}
  },
  "action": {
    "left_arm": {"start": 0, "end": 7},
    "left_hand": {"start": 7, "end": 27}
  },
  "video": {
    "cam_head": {"original_key": "observation.images.cam_head"}
  },
  "annotation": {
    "human.task_description": {"original_key": "task_index"}
  }
}
```

위 json의 key 이름을 그대로 `ffw_sh5_config.py`의 `modality_keys`에 넣었다.
GR00T loader는 이 key를 lookup한 뒤 parquet의 concat state/action 배열에서
해당 slice를 꺼내고, config에 지정한 normalization / relative 변환을 적용한다.

### `ffw_sh5_config.py`에서 공식 패턴을 따른 부분

```python
ffw_sh5_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_head"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_arm", "left_hand"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(30)),
        modality_keys=["left_arm", "left_hand"],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(ffw_sh5_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
```

체크 포인트:

- `action.modality_keys` 2개 ↔ `action_configs` 2개 순서 일치
- 첫 번째 action key `left_arm` → `RELATIVE`
- 두 번째 action key `left_hand` → `ABSOLUTE`
- language key는 `annotation.` prefix 포함 (`data_config.md` 규칙)
- custom robot이므로 `EmbodimentTag.NEW_EMBODIMENT`로 등록

### action chunk 30과 stats의 관계

공식 문서 [`getting_started/data_config.md`](getting_started/data_config.md)에
따르면 action horizon(`delta_indices`)을 바꾸면 `meta/relative_stats.json`도
같은 길이로 다시 만들어야 한다. 현재 config는 30-step horizon을 쓰므로
`meta/relative_stats.json`도 30-step 기준으로 맞춰 두었다.

- `left_arm`만 relative stats를 가진다. hand는 absolute action이라 별도
  relative stats가 필요 없다.
- horizon을 16이나 8로 바꾸면 config 수정 후 stats 재생성이 필요하다.
- 다만 학습 시작 시 `DatasetFactory`가 stats를 자동 생성하기도 하므로,
  horizon을 바꾸지 않았다면 아래 stats 명령은 선택 사항이다.

## 2. 생성한 파일

### `examples/FFW_SH5/ffw_sh5_config.py`

N1.7의 `register_modality_config()` API를 사용해 위 modality를
`EmbodimentTag.NEW_EMBODIMENT`로 등록한다. 작성 기준은 N1.7의
[`getting_started/data_config.md`](getting_started/data_config.md)이며,
N1.6 FFW SH5에서 검증된 27 DoF left arm/hand 구조를 그대로 옮겼다.

N1.5의 `--data-config` 방식과 달리 N1.7에서는 파인튜닝할 때 이 파일을
`--modality-config-path`로 넘겨야 한다.

### `examples/FFW_SH5/ffw_sh5_config_abs.py`

`ffw_sh5_config.py`와 동일한 video/state/language 설정이지만, `left_arm`과
`left_hand` **모두** `ABSOLUTE` action으로 학습한다. N1.6 Hub에 올린 absolute
체크포인트와 action representation을 맞추기 위한 ablation용 config다.

### 2.1 학습 스크립트 비교

동일 dataset·학습 스케줄(batch 64, max 100k, save 50k, horizon 30) 위에서
**action representation**과 **state dropout**만 바꿔 4개 GPU에 병렬 학습한다.
나머지 하이퍼파라미터(lr, warmup, weight decay, color jitter 등)는
`examples/finetune.sh` / N1.7 기본값을 공통으로 쓴다.

| 스크립트 | GPU | modality config | state dropout | 출력 디렉터리 | W&B experiment |
|---|---|---|---|---|---|
| `finetune_ffw_sh5.sh` | 1 | `ffw_sh5_config.py` (arm REL, hand ABS) | **0.2** (N1.7 기본) | `ffw_sh5_n17-checkpoint1` | `ffw_sh5_n17_psc_left` |
| `finetune_ffw_sh5_abs.sh` | 0 | `ffw_sh5_config_abs.py` (arm/hand ABS) | 0.2 | `ffw_sh5_n17-checkpoint2_abs` | `ffw_sh5_n17_psc_left_abs` |
| `finetune_ffw_sh5_drop0.sh` | 2 | `ffw_sh5_config.py` | **0.0** | `ffw_sh5_n17-checkpoint1_drop0` | `ffw_sh5_n17_psc_left_drop0` |
| `finetune_ffw_sh5_drop05.sh` | 3 | `ffw_sh5_config.py` | **0.5** | `ffw_sh5_n17-checkpoint1_drop05` | `ffw_sh5_n17_psc_left_drop05` |

#### 무엇이 다른가

**Action representation (`finetune_ffw_sh5.sh` vs `finetune_ffw_sh5_abs.sh`)**

- 기본(`ffw_sh5_config.py`): 팔 7 DoF는 `RELATIVE`(현재 state 대비 delta), 손 20 DoF는
  `ABSOLUTE`. N1.6 FFW SH5에서 쓰던 SO100-style 패턴.
- absolute(`ffw_sh5_config_abs.py`): 팔·손 모두 joint absolute target. parquet에
  저장된 값을 그대로 예측.

**State dropout (`drop0` / 기본 / `drop05`)**

학습 중 일정 확률로 proprioceptive state를 0으로 대체한다. N1.7은 processor와
action head 양쪽에서 dropout을 적용해, state 없이 vision(+language)만으로도
action을 예측하도록 regularize한다.

- `0.0`: state를 항상 사용. N1.6 학습과 동일한 조건에 가깝다.
- `0.2`: N1.7 `FinetuneConfig` 기본값.
- `0.5`: dropout을 강하게 걸어 vision 의존도를 더 높인다. 공식 SimplerEnv
  예제에서도 sim2real/generalization 실험에 0.5를 쓴다.

#### 왜 이렇게 나눠 학습하는가

한 번에 하나만 바꿔 **open-loop / closed-loop / sim2real**에서 어떤 조합이
FFW SH5에 맞는지 비교하려는 목적이다.

| 실험 | 기대 |
|---|---|
| `finetune_ffw_sh5.sh` | N1.7 공식 기본 + N1.6 검증 modality. **메인 baseline**. |
| `finetune_ffw_sh5_abs.sh` | arm absolute가 relative delta보다 tracking·배포에 유리한지 확인. N1.6 absolute Hub 모델과 직접 비교 가능. |
| `finetune_ffw_sh5_drop0.sh` | state에 항상 의존할 때의 upper bound. dropout regularization 효과를 분리해 본다. |
| `finetune_ffw_sh5_drop05.sh` | state 누락·노이즈·sim2real gap에 robust한 policy 기대. vision grounding이 강해지지만, state가 꼭 필요한 fine manipulation에서는 drop0/기본보다 불리할 수 있다. |

실행 예:

```bash
cd /workspace/GR00TN17/Isaac-GR00T

# GPU별 병렬 (각 터미널)
bash finetune_ffw_sh5_abs.sh      # GPU 0
bash finetune_ffw_sh5.sh          # GPU 1
bash finetune_ffw_sh5_drop0.sh    # GPU 2
bash finetune_ffw_sh5_drop05.sh   # GPU 3
```

공통 학습 스케줄:

- global batch size: 64
- max steps: 100000
- save steps: 50000
- action chunk / horizon: 30
- dataloader workers: 4
- learning rate: `1e-4` (`examples/finetune.sh`)
- warmup ratio: `0.05` (`examples/finetune.sh`)
- weight decay: `1e-5` (`examples/finetune.sh`)
- color jitter: brightness 0.3, contrast 0.4, saturation 0.5, hue 0.08
- W&B: 기본 활성화

## 3. N1.7 전용 uv 환경 준비

N1.7 환경은 다음 경로에 따로 생성된다.

```bash
/workspace/GR00TN17/Isaac-GR00T/.venv
```

N1.6 환경과 공유하지 않는다.

현재 checkout에는 일부 wheel이 Git LFS pointer 상태일 수 있다. 이 상태에서
`uv sync`를 실행하면 `Invalid zip file structure` 오류가 발생한다. 먼저 Git
LFS 파일을 받아야 한다.

```bash
cd /workspace/GR00TN17/Isaac-GR00T

apt-get update
apt-get install -y git-lfs ffmpeg libpython3.10
git lfs install
git lfs pull

uv sync
source .venv/bin/activate
```

`ffmpeg`만 설치하면 부족할 수 있다. Ubuntu 22.04에서는 `torchcodec` native
library가 `libpython3.10.so.1.0`과 시스템 FFmpeg 4.4 shared library
(`libavutil.so.56`)를 함께 필요로 한다. 아래로 import가 되는지 확인한다.

```bash
uv run python -c "import torchcodec; from torchcodec.decoders import VideoDecoder; print('torchcodec OK')"
```

`Could not load libtorchcodec` 또는 `libpython3.10.so.1.0: cannot open shared
object file`가 나오면 `apt install -y libpython3.10`을 추가로 설치한다.

환경 확인:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import gr00t; print('GR00T import OK')"
```

## 4. 전체 데이터셋 받기

현재 로컬 데이터셋에는 `meta/` 파일만 있고 실제 parquet 및 mp4 파일이 없을
수 있다. 아래 명령으로 Hugging Face dataset repo 전체를 받는다.

```bash
hf download \
  --repo-type dataset \
  learner1119/merge_ffw_sh5_rev1_260627_psc_left \
  --local-dir /workspace/dataset/merge_ffw_sh5_rev1_260627_psc_left
```

다운로드 확인:

```bash
test -n "$(compgen -G '/workspace/dataset/merge_ffw_sh5_rev1_260627_psc_left/data/chunk-*/episode_*.parquet')"
test -n "$(compgen -G '/workspace/dataset/merge_ffw_sh5_rev1_260627_psc_left/videos/chunk-*/*/episode_*.mp4')"
```

`finetune_ffw_sh5.sh`도 데이터 파일이 없으면 학습을 시작하지 않고 원인을
출력하도록 구성했다.

## 5. 통계 확인 및 재생성

공식 [`getting_started/data_config.md`](getting_started/data_config.md)는
action horizon 변경 시 stats 재생성을 요구한다. 현재 config는 action chunk 30,
`left_arm=RELATIVE`, `left_hand=ABSOLUTE`이므로 `meta/relative_stats.json`도
30-step 기준으로 맞춰 두었다.

통계를 수동으로 다시 만들고 싶을 때만 아래 명령을 실행하면 된다. 보통은
학습 시작 시 자동 생성되므로 필수 단계는 아니다.

```bash
cd /workspace/GR00TN17/Isaac-GR00T

uv run python gr00t/data/stats.py \
  --dataset-path /workspace/dataset/merge_ffw_sh5_rev1_260627_psc_left \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/FFW_SH5/ffw_sh5_config.py
```

Action horizon을 변경하면 기존 `meta/relative_stats.json`을 제거한 후 반드시
이 명령으로 통계를 재생성해야 한다.

## 6. 파인튜닝 실행

환경과 dataset 준비가 끝나면 저장소 루트에서 실행한다. §2.1의 4개 스크립트는
GPU만 다르게 잡혀 있어 동시에 돌릴 수 있다.

먼저 학습을 시작하지 않고 파일 준비 상태만 검사할 수 있다.

```bash
cd /workspace/GR00TN17/Isaac-GR00T
PREFLIGHT_ONLY=1 bash finetune_ffw_sh5.sh
PREFLIGHT_ONLY=1 bash finetune_ffw_sh5_abs.sh
PREFLIGHT_ONLY=1 bash finetune_ffw_sh5_drop0.sh
PREFLIGHT_ONLY=1 bash finetune_ffw_sh5_drop05.sh
```

검사를 통과하면 본 학습을 실행한다. baseline만 돌릴 때:

```bash
cd /workspace/GR00TN17/Isaac-GR00T
bash finetune_ffw_sh5.sh
```

스크립트 내부에서 `uv run`을 사용하므로 별도로 `.venv`를 활성화하지 않아도
N1.7 전용 환경이 사용된다.

기본 실행과 동일한 실제 파인튜닝 명령은 다음과 같다.

```bash
CUDA_VISIBLE_DEVICES=1 \
NUM_GPUS=1 \
GLOBAL_BATCH_SIZE=64 \
MAX_STEPS=100000 \
SAVE_STEPS=50000 \
USE_WANDB=1 \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /workspace/dataset/merge_ffw_sh5_rev1_260627_psc_left \
  --modality-config-path examples/FFW_SH5/ffw_sh5_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir ./ffw_sh5_n17-checkpoint1 \
  --experiment-name ffw_sh5_n17_psc_left \
  --wandb-project finetune-gr00t-n1d7-ffw-sh5
```

## 7. 실행 옵션 변경

`finetune_ffw_sh5.sh`의 기본값은 환경변수로 덮어쓸 수 있다.

W&B 없이 짧게 smoke test:

```bash
CUDA_VISIBLE_DEVICES=1 \
USE_WANDB=0 \
GLOBAL_BATCH_SIZE=4 \
MAX_STEPS=10 \
SAVE_STEPS=10 \
OUTPUT_DIR=/tmp/ffw_sh5_n17_smoke \
bash finetune_ffw_sh5.sh
```

GPU 0번 사용:

```bash
CUDA_VISIBLE_DEVICES=0 bash finetune_ffw_sh5.sh
```

다른 출력 경로 사용:

```bash
OUTPUT_DIR=/workspace/GR00TN17/Isaac-GR00T/ffw_sh5_n17-checkpoint2 \
bash finetune_ffw_sh5.sh
```

메모리 부족 시:

```bash
GLOBAL_BATCH_SIZE=16 bash finetune_ffw_sh5.sh
```

## 8. 학습 결과

각 스크립트마다 동일한 step interval로 체크포인트가 생성된다.

```text
ffw_sh5_n17-checkpoint1/           # finetune_ffw_sh5.sh
ffw_sh5_n17-checkpoint2_abs/       # finetune_ffw_sh5_abs.sh
ffw_sh5_n17-checkpoint1_drop0/    # finetune_ffw_sh5_drop0.sh
ffw_sh5_n17-checkpoint1_drop05/   # finetune_ffw_sh5_drop05.sh
├── checkpoint-50000/
└── checkpoint-100000/
```

50000 step마다 저장하고 `save_total_limit=5`가 적용된다. 기본적으로 optimizer state까지 저장한다. 모델 파일만
저장하려면 `examples/finetune.sh` 호출에 `--save-only-model`을 추가해야 한다.

## 9. Open-loop 평가

예를 들어 50000 step 체크포인트를 평가하려면:

```bash
cd /workspace/GR00TN17/Isaac-GR00T

uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path /workspace/dataset/merge_ffw_sh5_rev1_260627_psc_left \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path ./ffw_sh5_n17-checkpoint1/checkpoint-50000 \
  --traj-ids 0 \
  --action-horizon 30 \
  --steps 400 \
  --modality-keys left_arm left_hand
```

체크포인트에 modality config와 dataset statistics가 저장되므로 평가 시
`--modality-config-path`는 필요하지 않다.

## 10. 학습 전 최종 체크리스트

- [x] N1.7 공식 `n1.7-release` 소스 확인
- [x] FFW SH5 27 DoF modality config 생성
- [x] `data_config.md` 규칙에 맞게 `ffw_sh5_config.py` 작성
- [x] `meta/modality.json` key와 config `modality_keys` 1:1 매칭
- [x] arm relative / hand absolute / action chunk 30 적용
- [x] N1.7 학습 wrapper 생성
- [x] batch 64 / max 100k / save 50k 유지
- [x] baseline + absolute + dropout ablation 스크립트 4종 (GPU 0–3)
- [x] Git LFS 파일 다운로드
- [x] N1.7 전용 `.venv` 및 `uv sync` 완료
- [x] PyTorch 2.7.1+cu128, CUDA 4개 GPU, modality config 로드 확인
- [x] dataset parquet 65개 및 mp4 130개 전체 다운로드
- [x] N1.7 stats 도구로 dataset/modality 호환성 확인
- [ ] 10-step smoke test
- [ ] 본 학습 실행

Dataset 본문을 받은 뒤 10-step smoke test를 통과하면 본 학습을 시작할 수 있다.
