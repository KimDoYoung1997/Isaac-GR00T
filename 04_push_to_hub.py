#!/usr/bin/env python3
"""
로컬 LeRobot v2.1 데이터셋 폴더를 Hugging Face Hub (dataset repo)에 업로드합니다.
03_down_dimension_v2.py 의 --push-to-hub 와 동일한 방식입니다.

사전 준비: huggingface-cli login 또는 환경 변수 HF_TOKEN

예:
  python3 04_push_to_hub.py \\
    --repo-id learner1119/merge_ffw_sh5_rev1_20260504_27dof_pick_up_a_red_cylinder_and_place_it_on_the_basket

  python3 04_push_to_hub.py --repo-id org/name --private
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

from lerobot.datasets.utils import create_lerobot_dataset_card


def push_dataset_folder(
    local_root: Path,
    repo_id: str,
    *,
    private: bool,
    ignore_images_dir: bool,
) -> None:
    if not repo_id or "/" not in repo_id:
        raise ValueError("repo_id 는 user-or-org/dataset-name 형식이어야 합니다.")
    local_root = local_root.resolve()
    info_path = local_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"meta/info.json 없음: {info_path}")
    with open(info_path, encoding="utf-8") as f:
        hub_info = json.load(f)

    hub_api = HfApi()
    hub_api.create_repo(repo_id=repo_id, private=private, repo_type="dataset", exist_ok=True)
    print(f"Hub 업로드 중… {repo_id} <- {local_root}")
    ignore_patterns = ["images/"] if ignore_images_dir else None
    hub_api.upload_folder(
        repo_id=repo_id,
        folder_path=str(local_root),
        repo_type="dataset",
        allow_patterns=None,
        ignore_patterns=ignore_patterns,
    )
    card = create_lerobot_dataset_card(
        tags=None,
        dataset_info=hub_info,
        license="apache-2.0",
    )
    card.push_to_hub(repo_id=repo_id, repo_type="dataset")
    print(f"Hub 업로드 완료: https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LeRobot 로컬 데이터셋 폴더 → Hugging Face Hub 업로드")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(
            "/workspace/dataset/merge_ffw_sh5_rev1_20260504_pick_up_a_red_cylinder_and_place_it_on_the_basket_27dof"
        ),
        help="업로드할 데이터셋 루트 (meta/info.json 필수)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HF dataset repo. 예: learner1119/my_dataset_name",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="비공개 repo 로 생성/업로드 (기본: 공개)",
    )
    parser.add_argument(
        "--upload-images-dir",
        action="store_true",
        help="기본은 images/ 디렉터리를 제외합니다. 레거시 이미지까지 올릴 때만 지정하세요.",
    )
    args = parser.parse_args()

    root = args.dataset_dir
    if not root.is_dir():
        raise SystemExit(f"데이터셋 디렉터리가 없습니다: {root}")

    push_dataset_folder(
        root,
        args.repo_id.strip(),
        private=bool(args.private),
        ignore_images_dir=not args.upload_images_dir,
    )


if __name__ == "__main__":
    main()
