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
from pathlib import Path

import tyro

from gr00t.eval import open_loop_eval

logging.basicConfig(level=logging.INFO, force=True)

# Upstream saves every trajectory's plot to the same --save_plot_path, so a
# multi-trajectory run leaves only the last one. Suffix the trajectory id:
# plot.jpeg -> plot_traj0.jpeg, plot_traj1.jpeg, ...
_plot = open_loop_eval.plot_trajectory_results


def _plot_per_traj(*args, traj_id, save_plot_path, **kwargs):
    path = Path(save_plot_path)
    path = path.with_name(f"{path.stem}_traj{traj_id}{path.suffix}")
    return _plot(*args, traj_id=traj_id, save_plot_path=str(path), **kwargs)


open_loop_eval.plot_trajectory_results = _plot_per_traj

open_loop_eval.main(tyro.cli(open_loop_eval.ArgsConfig))
