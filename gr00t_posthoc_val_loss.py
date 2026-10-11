"""Held-out / train-probe flow-matching loss of saved GR00T N1.7 checkpoints (post hoc).

Mirror of RLDX-1's run_scripts/eval/posthoc_val_loss.py: every `stride`-th
usable frame of the held-out dataset plus the same kind of sample from
`probe_episodes` evenly spaced training episodes, `draws` fixed noise/time
draws per frame, model and processor in eval mode (checkpoint statistics, eval
image transforms, no state dropout). Loss is the model's own masked
flow-matching MSE (Gr00tN1d7ActionHead.forward), i.e. what the trainer logs.

    python gr00t_posthoc_val_loss.py --ckpts DIR [DIR ...] --heldout DS --train DS --out out.jsonl
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

import gr00t.model  # noqa: F401  (registers the model/processor classes)
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag


def usable_steps(ds, ep, stride):
    n = ds.get_effective_episode_length(ep)  # frames whose 40-step action chunk fits
    return list(range(0, n, stride))


@torch.no_grad()
def score(model, ds, items, draws, bs, dev):
    tot, n = 0.0, 0
    cache = {}
    for d in range(draws):
        for b in range(0, len(items), bs):
            chunk = items[b:b + bs]
            feats = []
            for ep, s in chunk:
                if ep not in cache:
                    cache = {ep: ds.episode_loader[ep]}  # one episode at a time
                feats.append(ds.get_datapoint(cache[ep], s))
            batch = ds.processor.collator(feats)["inputs"]
            with torch.random.fork_rng(devices=[dev]), torch.autocast("cuda", dtype=torch.bfloat16):
                torch.manual_seed(1_000_003 * (d + 1) + 7919 * chunk[0][0] + chunk[0][1])
                out = model(inputs=batch)
            loss = out["loss"] if "loss" in out else out[0]
            tot += loss.double().item() * len(chunk); n += len(chunk)
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="new_embodiment")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--probe-episodes", type=int, default=9)
    a = ap.parse_args()
    tag = EmbodimentTag.resolve(a.tag)
    dev = torch.device("cuda:0")
    for ck in a.ckpts:
        t0 = time.time()
        mdir = Path(ck)
        model = AutoModel.from_pretrained(mdir, torch_dtype=torch.float32)
        model.eval().to(dev)  # fp32 weights + bf16 autocast ~ the trainer's numerics
        ah = model.action_head
        _orig_sample_time = ah.sample_time
        ah.sample_time = lambda bsz, device, dtype: _orig_sample_time(bsz, device, torch.float32).to(dtype)
        proc = AutoProcessor.from_pretrained(mdir)  # processor_config.json + statistics.json at the ckpt root
        proc.eval()
        mcfg = proc.get_modality_configs()[tag.value]
        res = {"ckpt": ck}
        for name, path, probe in (("heldout", a.heldout, False), ("train_probe", a.train, True)):
            ds = ShardedSingleStepDataset(dataset_path=path, embodiment_tag=tag, modality_configs=mcfg,
                                          video_backend="torchcodec", episode_sampling_rate=1.0)
            ds.processor = proc
            eps = sorted(range(len(ds.episode_loader)))
            if probe:
                eps = np.asarray(eps)[np.linspace(0, len(eps) - 1, a.probe_episodes).round().astype(int)].tolist()
            items = [(e, s) for e in eps for s in usable_steps(ds, e, a.stride)]
            res[name + "_loss"] = score(model, ds, items, a.draws, a.bs, dev)
            res[name + "_frames"] = len(items)
        res["seconds"] = round(time.time() - t0, 1)
        print(json.dumps(res), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")
        del model, proc; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
