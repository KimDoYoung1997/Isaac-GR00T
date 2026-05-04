#!/usr/bin/env python3
"""
로컬 LeRobot 데이터셋(meta/info.json의 codebase_version v2.1)을 v3.0으로 변환합니다.
Hub에 푸시하지 않으며, 변환 결과는 디스크상 동일 repo_id 경로에 남습니다.

공식 구현: lerobot.datasets.v30.convert_dataset_v21_to_v30.convert_dataset

사용 예:
  python /root/convert_lerobot_dataset_v21_to_v30.py \\
    --dataset-dir /workspace/Lerobot-MujoCo-VLA-Tutorial/dataset/ffw_sh5_rev1_hand_test_edit

  python /root/convert_lerobot_dataset_v21_to_v30.py --help
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset
from lerobot.utils.utils import init_logging


def resolve_dataset_root_for_converter(
    dataset_dir: Path,
    repo_id: str,
) -> tuple[Path, Path]:
    """
    convert_dataset(repo_id, root=parent) 는 실제 데이터 경로가 (parent / repo_id) 일 때만 동작합니다.

    Returns:
        parent: --root 로 넘길 부모 디렉터리
        dataset_root: 변환 대상 데이터셋 루트 (parent / repo_id)
    """
    dataset_dir = dataset_dir.resolve()
    last = repo_id.split("/")[-1]

    if dataset_dir.name != last:
        raise SystemExit(
            f"데이터셋 폴더 이름이 repo_id의 마지막 구간과 다릅니다.\n"
            f"  dataset_dir.name={dataset_dir.name!r}, expected {last!r}\n"
            f"  또는 --dataset-dir 을 meta/ 가 있는 루트로 지정하세요."
        )

    parent = dataset_dir.parent
    expected = (parent / repo_id).resolve()

    if dataset_dir == expected:
        return parent, expected

    expected.parent.mkdir(parents=True, exist_ok=True)
    if expected.exists():
        raise SystemExit(
            f"이미 변환 레이아웃 경로가 있습니다. 수동으로 정리 후 다시 실행하세요:\n  {expected}"
        )

    shutil.move(str(dataset_dir), str(expected))
    return parent, expected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LeRobot v2.1 로컬 데이터셋 → v3.0 변환 (push_to_hub=False)"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/workspace/Lerobot-MujoCo-VLA-Tutorial/dataset/ffw_sh5_rev1_hand_test_edit"),
        help="meta/, data/, videos/ 가 있는 디렉터리 (v2.1)",
    )
    parser.add_argument(
        "--repo-id",
        default="learner1119/ffw_sh5_rev1_hand_test_edit",
        help="허브 repo_id; 로컬 경로는 (dataset-dir의 부모)/(repo_id) 가 됩니다.",
    )
    args = parser.parse_args()

    init_logging()

    parent, dataset_root = resolve_dataset_root_for_converter(
        args.dataset_dir,
        args.repo_id,
    )

    convert_dataset(
        repo_id=args.repo_id,
        root=str(parent),
        push_to_hub=False,
    )

    print("변환 완료.")
    print(f"  v3.0 데이터셋 루트: {dataset_root}")
    print("  백업 폴더가 있다면 같은 부모 아래 `*_old` 이름일 수 있습니다 (스크립트 동작 확인).")


if __name__ == "__main__":
    main()

