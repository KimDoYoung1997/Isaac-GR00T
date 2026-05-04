#!/usr/bin/env python3
"""
LeRobot v2.1 로컬 데이터셋에서 observation.state / action 을 54차원 → 27차원으로 줄입니다.
codebase_version 은 v2.1 그대로 두고, 메타·파케이만 갱신합니다 (v3.0 변환 없음).

기본 슬라이스(이름 순서는 meta/info.json 과 동일해야 함):
  - arm_r_joint1–7  + finger_r_joint1–20
  - 즉 인덱스 7:14 와 34:53 (ffw_sh5_rev1 기본 레이아웃)

갱신 대상:
  - data/chunk-*/episode_*.parquet 의 observation.state, action
  - meta/episodes_stats.jsonl 의 에피소드별 observation.state / action 벡터 통계
  - meta/info.json 의 해당 feature names / shape
  - meta/stats.json 의 observation.state, action 블록 (데이터 전체를 다시 스캔해 재계산, 나머지 키는 유지)

권장 파이프라인:
  1) v2.1 데이터셋을 로컬에 받기 (예: huggingface_hub.snapshot_download)
  2) 본 스크립트로 54→27
  3) visualize.ipynb 로 확인

사용 예:
  # merge 원본은 그대로 두고 복사본에서 54→27 후 Hub 업로드
  python 03_down_dimension_v2.py \\
    --dataset-dir /workspace/dataset/merge_ffw_sh5_rev1_20260504_pick_up_a_red_cylinder_and_place_it_on_the_basket \\
    --output-dir /workspace/dataset/merge_ffw_sh5_rev1_20260504_pick_up_a_red_cylinder_and_place_it_on_the_basket_27dof \\
    --push-to-hub --push-repo-id learner1119/원하는_repo_이름_27dof

  # 원본 폴더에서 직접 갱신 (백업 .bak_down_dim 생성)
  python 03_down_dimension_v2.py --dataset-dir /path/to/v2.1_dataset

  python 03_down_dimension_v2.py --help
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

from lerobot.datasets.utils import STATS_PATH, create_lerobot_dataset_card, write_json

# ffw_sh5_rev1_hand_test_edit 의 observation.state / action 이름 순서 기준
DEFAULT_SLICE_INDICES = list(range(7, 14)) + list(range(34, 54))


def _backup_file(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak_down_dim")
    shutil.copy2(path, bak)


def _slice_list_column(col: pa.Array, indices: list[int]) -> pa.Array:
    out: list[list[float]] = []
    for i in range(len(col)):
        v = col[i].as_py()
        if v is None:
            out.append([])
            continue
        arr = np.asarray(v, dtype=np.float64)
        out.append(arr[indices].astype(np.float32).tolist())
    return pa.array(out, type=pa.list_(pa.float32()))


def _process_data_parquet(path: Path, indices: list[int], full_dim: int) -> None:
    table = pq.read_table(path)
    if "observation.state" not in table.column_names or "action" not in table.column_names:
        raise ValueError(f"{path}: observation.state / action 컬럼이 없습니다.")
    arrays = []
    names = []
    for name in table.column_names:
        if name in ("observation.state", "action"):
            col = table[name]
            for i in range(len(col)):
                arr = np.asarray(col[i].as_py(), dtype=np.float64)
                if arr.size != full_dim:
                    raise ValueError(f"{path} row {i} {name}: len {arr.size} != {full_dim}")
            arrays.append(_slice_list_column(col, indices))
        else:
            arrays.append(table[name])
        names.append(name)
    pq.write_table(pa.Table.from_arrays(arrays, names=names), path)


def _process_episodes_stats_jsonl(path: Path, indices: list[int], full_dim: int) -> None:
    out_lines: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            st = row.get("stats", {})
            for key in ("observation.state", "action"):
                if key not in st:
                    continue
                blk = st[key]
                for field in ("min", "max", "mean", "std"):
                    if field not in blk:
                        continue
                    vec = blk[field]
                    arr = np.asarray(vec, dtype=np.float64).ravel()
                    if arr.size != full_dim:
                        raise ValueError(
                            f"{path} ep={row.get('episode_index')} {key}.{field}: len {arr.size} != {full_dim}"
                        )
                    blk[field] = arr[indices].astype(np.float64).tolist()
            out_lines.append(json.dumps(row, ensure_ascii=False))
    with open(path, "w", encoding="utf-8") as wf:
        wf.write("\n".join(out_lines) + ("\n" if out_lines else ""))


def _list_v21_data_parquets(dataset_root: Path) -> list[Path]:
    paths = sorted(dataset_root.glob("data/chunk-*/episode_*.parquet"))
    if paths:
        return paths
    alt = sorted(dataset_root.glob("data/chunk-*/*.parquet"))
    return [p for p in alt if p.name.startswith("episode_")]


def _recompute_global_vector_stats(
    dataset_root: Path, keys: list[str], dim: int
) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {k: {"min": None, "max": None, "mean": None, "std": None, "count": [0]} for k in keys}
    data_files = _list_v21_data_parquets(dataset_root)
    if not data_files:
        raise FileNotFoundError(f"data parquet 없음: {dataset_root / 'data'}")

    chunks: dict[str, list[np.ndarray]] = {k: [] for k in keys}

    for fp in data_files:
        t = pq.read_table(fp, columns=keys)
        n = len(t)
        for k in keys:
            for i in range(n):
                arr = np.asarray(t[k][i].as_py(), dtype=np.float64)
                if arr.size != dim:
                    raise ValueError(f"{fp} {k} row {i}: len {arr.size} != {dim}")
                chunks[k].append(arr)

    for k in keys:
        X = np.stack(chunks[k], axis=0)
        out[k]["min"] = X.min(axis=0).tolist()
        out[k]["max"] = X.max(axis=0).tolist()
        out[k]["mean"] = X.mean(axis=0).tolist()
        out[k]["std"] = X.std(axis=0).tolist()
        out[k]["count"] = [int(X.shape[0])]
    return out


def _update_info_json(info_path: Path, indices: list[int], old_names: list[str]) -> dict:
    with open(info_path) as f:
        info = json.load(f)
    new_names = [old_names[i] for i in indices]
    new_shape = [len(indices)]
    for key in ("observation.state", "action"):
        if key not in info["features"]:
            raise KeyError(f"info.json 에 {key} 가 없습니다.")
        feat = info["features"][key]
        names = feat.get("names")
        if names is None or len(names) != len(old_names):
            raise ValueError(f"{key}: names 길이가 기대와 다릅니다.")
        feat["names"] = new_names
        feat["shape"] = new_shape
    return info


def _push_to_hub(local_root: Path, repo_id: str, *, private: bool) -> None:
    """로컬 LeRobot 데이터셋 폴더를 HF dataset repo 로 업로드 (LeRobotDataset 미사용)."""
    if not repo_id or "/" not in repo_id:
        raise ValueError("push_repo_id 는 org-or-user/dataset-name 형식이어야 합니다.")
    info_path = local_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"meta/info.json 없음: {info_path}")
    with open(info_path, encoding="utf-8") as f:
        hub_info = json.load(f)

    hub_api = HfApi()
    hub_api.create_repo(repo_id=repo_id, private=private, repo_type="dataset", exist_ok=True)
    print(f"Hub 업로드 중… {repo_id} <- {local_root}")
    hub_api.upload_folder(
        repo_id=repo_id,
        folder_path=str(local_root),
        repo_type="dataset",
        allow_patterns=None,
        ignore_patterns=["images/"],
    )
    card = create_lerobot_dataset_card(
        tags=None,
        dataset_info=hub_info,
        license="apache-2.0",
    )
    card.push_to_hub(repo_id=repo_id, repo_type="dataset")
    print(f"Hub 업로드 완료: {repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LeRobot v2.1: state/action 차원 축소 (기본 54→27, 포맷 유지)")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(
            "/workspace/dataset/merge_ffw_sh5_rev1_20260504_pick_up_a_red_cylinder_and_place_it_on_the_basket"
        ),
        help="입력 v2.1 데이터셋 루트 (codebase_version v2.1, 기본값: merge_ffw_sh5_rev1_20260504…)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="지정 시 dataset-dir 전체를 이 경로로 복사(.cache 제외)한 뒤 여기서만 차원 축소·푸시 (원본 유지)",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="--output-dir 이 이미 있으면 삭제 후 다시 복사",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default="",
        help="쉼표로 구분한 0-based 인덱스 (비우면 기본: 7-13,34-53). 예: 7,8,9,...,53",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="info.json / stats.json 백업(.bak_down_dim) 생략",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="차원 축소 후 Hugging Face Hub 에 dataset 으로 업로드",
    )
    parser.add_argument(
        "--push-repo-id",
        type=str,
        default="",
        help="--push-to-hub 일 때 필수. 예: learner1119/my_robot_dataset_27dof",
    )
    parser.add_argument(
        "--push-private",
        action="store_true",
        help="--push-to-hub 일 때 비공개 repo 로 생성/업로드 (기본: 공개)",
    )
    args = parser.parse_args()

    if args.push_to_hub:
        rid = (args.push_repo_id or "").strip()
        if not rid or "/" not in rid:
            raise SystemExit("--push-to-hub 이면 --push-repo-id 에 user-or-org/name 형식으로 지정하세요.")

    src_root = args.dataset_dir.resolve()
    if args.output_dir is not None:
        root = args.output_dir.resolve()
        if root.exists():
            if not args.overwrite_output:
                raise SystemExit(
                    f"--output-dir 이 이미 있습니다: {root}\n"
                    "비우거나 --overwrite-output 으로 덮어쓰세요."
                )
            shutil.rmtree(root)
        print(f"복사 중 (원본 유지): {src_root} -> {root}")
        shutil.copytree(
            src_root,
            root,
            symlinks=False,
            ignore=shutil.ignore_patterns(".cache"),
        )
    else:
        root = src_root

    info_path = root / "meta" / "info.json"
    stats_path = root / STATS_PATH

    if not info_path.is_file():
        raise SystemExit(f"없음: {info_path}")

    with open(info_path) as f:
        info = json.load(f)

    if info.get("codebase_version") != "v2.1":
        raise SystemExit(f"codebase_version 이 v2.1 이 아닙니다: {info.get('codebase_version')}")

    old_names = info["features"]["observation.state"]["names"]
    if old_names != info["features"]["action"]["names"]:
        raise SystemExit("observation.state 와 action 의 joint 이름 순서가 다릅니다. 수동으로 맞춘 뒤 실행하세요.")

    full_dim = int(info["features"]["observation.state"]["shape"][0])
    if args.indices.strip():
        indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    else:
        indices = DEFAULT_SLICE_INDICES

    for i in indices:
        if i < 0 or i >= full_dim:
            raise SystemExit(f"인덱스 범위 오류: {i} not in [0, {full_dim})")

    new_dim = len(indices)
    print("입력:", src_root)
    print("작업 루트:", root)
    print(f"slice {full_dim} -> {new_dim}, indices={indices[:8]}{'...' if len(indices) > 8 else ''}")

    if full_dim == new_dim:
        print("이미 observation.state 가 목표 차원과 같습니다. 종료합니다.")
        return

    data_files = _list_v21_data_parquets(root)
    ep_stats_path = root / "meta" / "episodes_stats.jsonl"

    if not data_files:
        raise SystemExit("data/chunk-*/episode_*.parquet 이 없습니다.")

    if not args.no_backup:
        _backup_file(info_path)
        if stats_path.is_file():
            _backup_file(stats_path)
        if ep_stats_path.is_file():
            _backup_file(ep_stats_path)

    for fp in data_files:
        print("  data:", fp.relative_to(root))
        _process_data_parquet(fp, indices, full_dim)

    if ep_stats_path.is_file():
        print("  meta:", ep_stats_path.relative_to(root))
        _process_episodes_stats_jsonl(ep_stats_path, indices, full_dim)
    else:
        print("  [warn] meta/episodes_stats.jsonl 없음 — 스킵")

    new_info = _update_info_json(info_path, indices, old_names)
    write_json(new_info, info_path)
    print("  wrote:", info_path.relative_to(root))

    # stats.json: 벡터 통계 재계산 후 기존 나머지 유지
    old_stats = {}
    if stats_path.is_file():
        with open(stats_path) as f:
            old_stats = json.load(f)

    vec_stats = _recompute_global_vector_stats(root, ["observation.state", "action"], new_dim)
    merged = dict(old_stats)
    merged["observation.state"] = vec_stats["observation.state"]
    merged["action"] = vec_stats["action"]
    write_json(merged, stats_path)
    print("  wrote:", stats_path.relative_to(root))

    if args.push_to_hub:
        _push_to_hub(
            root,
            args.push_repo_id.strip(),
            private=bool(args.push_private),
        )

    print("완료 (v2.1 유지). visualize.ipynb 에서 DATASET_ROOT =", root)


if __name__ == "__main__":
    main()
