"""Personalized-weight transplantation diagnostic (Table 4 and the composed figures in the Discussion).

For one test participant, copy one module's per-participant fine-tuned weights into the
released generic TDS checkpoint of the benchmark and save the hybrid checkpoint. Each hybrid is
then evaluated on that participant with the benchmark's own `emg2qwerty.train ... train=False`
(see `eval/as_run/tp_regreedy.sh` for greedy and `eval/as_run/tp_reks.sh` for beam + keystroke LM).

The generic and per-participant fine-tuned TDS checkpoints are the ones released with the
benchmark (`generic.ckpt`, `personalized-finetuned/user{0..7}.ckpt`).

Usage:
  python eval/transplant.py --generic models/generic.ckpt \
      --personalized-dir models/personalized-finetuned --user user0 --out-dir out/transplant
"""
import argparse
from pathlib import Path

import torch


def _tds_block(key: str):
    """Index of the TDS sub-block a parameter belongs to (even = temporal conv, odd = channel-mixing FC)."""
    if not key.startswith("model.3."):
        return None
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p.isdigit() and i > 0 and not parts[i - 1].isdigit() and parts[i - 1] not in ("model",):
            return int(p)
    return None


CONFIGS = {
    "head": lambda k: k.startswith("model.4."),                                   # classification head
    "mlp": lambda k: k.startswith("model.1."),                                    # rotation-invariant MLP
    "tds_conv": lambda k: _tds_block(k) is not None and _tds_block(k) % 2 == 0,   # TDS temporal convolutions
    "tds_fc": lambda k: _tds_block(k) is not None and _tds_block(k) % 2 == 1,     # TDS channel-mixing (FC) blocks
    "head_plus_fc": lambda k: k.startswith("model.4.")
        or (_tds_block(k) is not None and _tds_block(k) % 2 == 1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generic", required=True, type=Path)
    ap.add_argument("--personalized-dir", required=True, type=Path)
    ap.add_argument("--user", default="user0")
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    gen_ckpt = torch.load(a.generic, map_location="cpu")
    per_sd = torch.load(a.personalized_dir / f"{a.user}.ckpt", map_location="cpu")["state_dict"]
    gen_sd = gen_ckpt["state_dict"]
    for name, pred in CONFIGS.items():
        patched = {k: (per_sd[k] if (pred(k) and k in per_sd) else v) for k, v in gen_sd.items()}
        n = sum(1 for k in gen_sd if pred(k) and k in per_sd)
        out = dict(gen_ckpt)
        out["state_dict"] = patched
        torch.save(out, a.out_dir / f"{a.user}_{name}.ckpt")
        print(f"{a.user}/{name}: {n} tensors")


if __name__ == "__main__":
    main()
