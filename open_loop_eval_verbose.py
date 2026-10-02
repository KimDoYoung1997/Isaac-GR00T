#!/usr/bin/env python3
"""Run gr00t.eval.open_loop_eval with its result lines actually visible.

The upstream script reports MSE/MAE through logging.info, but its
logging.basicConfig call runs after transformers has configured the root
logger, so it can be a no-op and every result line is swallowed — the same
failure mode RLDX-1's verbose wrapper exists for. force=True overrides the
existing handler configuration.

Usage mirrors the upstream script:

    uv run python open_loop_eval_verbose.py --model_path <ckpt> \
        --dataset_path dataset/rlwrld_demo_0826_holdout \
        --embodiment_tag new_embodiment --action_horizon 40 \
        --traj_ids 0 1 2 3 4 5 6 7 8 --steps 1000
"""

import logging

import tyro

from gr00t.eval import open_loop_eval

logging.basicConfig(level=logging.INFO, force=True)

open_loop_eval.main(tyro.cli(open_loop_eval.ArgsConfig))
