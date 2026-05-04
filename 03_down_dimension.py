#!/usr/bin/env python3
"""
LeRobot v3.0 로컬 데이터셋에서 observation.state / action 을 54차원 → 27차원으로 줄입니다.

기본 슬라이스(이름 순서는 meta/info.json 과 동일해야 함):
  - arm_r_joint1–7  + finger_r_joint1–20
  - 즉 인덱스 7:14 와 34:54 (ffw_sh5_rev1_hand_test_edit 기본 레이아웃)

갱신 대상:
  - data/chunk-*/file-*.parquet 의 observation.state, action
  - meta/episodes/chunk-*/file-*.parquet 의 stats/observation.state/*, stats/action/* (count 제외)
  - meta/info.json 의 해당 feature names / shape
  - meta/stats.json 의 observation.state, action 블록 (데이터 전체를 다시 스캔해 재계산, 나머지 키는 유지)

권장 파이프라인:
  1) download.py 로 v2.1 받기
  2) convert_lerobot_dataset_v21_to_v30.py 로 v3.0 변환
  3) 본 스크립트로 54→27
  4) visualize.ipynb 로 확인

사용 예:
  python dataset/down_dimension.py \\
    --dataset-dir /workspace/Lerobot-MujoCo-VLA-Tutorial/dataset/learner1119/ffw_sh5_rev1_hand_test_edit

  python dataset/down_dimension.py --help
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.utils import STATS_PATH, write_json

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


def _process_episodes_parquet(path: Path, indices: list[int], full_dim: int) -> None:
    table = pq.read_table(path)
    arrays = []
    names = []
    for name in table.column_names:
        col = table[name]
        if name.startswith("stats/observation.state/") or name.startswith("stats/action/"):
            suffix = name.split("/")[-1]
            if suffix == "count":
                arrays.append(col)
            else:
                for i in range(len(col)):
                    v = col[i].as_py()
                    if v is None:
                        continue
                    arr = np.asarray(v, dtype=np.float64)
                    if arr.size != full_dim:
                        raise ValueError(f"{path} {name} row {i}: len {arr.size} != {full_dim}")
                arrays.append(_slice_list_column(col, indices))
        else:
            arrays.append(col)
        names.append(name)
    pq.write_table(pa.Table.from_arrays(arrays, names=names), path)


def _recompute_global_vector_stats(
    dataset_root: Path, keys: list[str], dim: int
) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {k: {"min": None, "max": None, "mean": None, "std": None, "count": [0]} for k in keys}
    data_files = sorted(dataset_root.glob("data/chunk-*/file-*.parquet"))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="LeRobot v3.0: state/action 차원 축소 (기본 54→27)")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/workspace/Lerobot-MujoCo-VLA-Tutorial/dataset/learner1119/ffw_sh5_rev1_hand_test_edit"),
        help="meta/, data/, videos/ 가 있는 v3.0 데이터셋 루트",
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
    args = parser.parse_args()

    dataset_root = args.dataset_dir.resolve()
    info_path = dataset_root / "meta" / "info.json"
    stats_path = dataset_root / STATS_PATH

    if not info_path.is_file():
        raise SystemExit(f"없음: {info_path}")

    with open(info_path) as f:
        info = json.load(f)

    if info.get("codebase_version") != "v3.0":
        raise SystemExit(f"codebase_version 이 v3.0 이 아닙니다: {info.get('codebase_version')}")

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
    print(f"dataset_root={dataset_root}")
    print(f"slice {full_dim} -> {new_dim}, indices={indices[:8]}{'...' if len(indices) > 8 else ''}")

    data_files = sorted(dataset_root.glob("data/chunk-*/file-*.parquet"))
    ep_files = sorted(dataset_root.glob("meta/episodes/chunk-*/file-*.parquet"))

    if not data_files:
        raise SystemExit("data/chunk-*/*.parquet 이 없습니다.")

    if not args.no_backup:
        _backup_file(info_path)
        if stats_path.is_file():
            _backup_file(stats_path)

    for fp in data_files:
        print("  data:", fp.relative_to(dataset_root))
        _process_data_parquet(fp, indices, full_dim)

    for fp in ep_files:
        print("  episodes:", fp.relative_to(dataset_root))
        _process_episodes_parquet(fp, indices, full_dim)

    new_info = _update_info_json(info_path, indices, old_names)
    write_json(new_info, info_path)
    print("  wrote:", info_path.relative_to(dataset_root))

    # stats.json: 벡터 통계 재계산 후 기존 나머지 유지
    old_stats = {}
    if stats_path.is_file():
        with open(stats_path) as f:
            old_stats = json.load(f)

    vec_stats = _recompute_global_vector_stats(
        dataset_root, ["observation.state", "action"], new_dim
    )
    merged = dict(old_stats)
    merged["observation.state"] = vec_stats["observation.state"]
    merged["action"] = vec_stats["action"]
    write_json(merged, stats_path)
    print("  wrote:", stats_path.relative_to(dataset_root))

    print("완료. visualize.ipynb 에서 DATASET_ROOT 를 이 폴더로 두고 확인하세요.")


if __name__ == "__main__":
    main()
