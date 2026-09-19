# Checkpoints (GitHub Release `v1.0`)

Weights-only Lightning checkpoints (`state_dict`, `hyper_parameters`, `epoch`,
`global_step`; optimizer state removed). Every one is the validation-best
checkpoint of a single run with seed 1501, selected by `val/CER` on the four
cross-user validation participants. Load with `checkpoint=<path>` in `emg2qwerty.train`
(set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` on torch ≥ 2.6).

| File | Table 2 row | Tree / model config | Epoch | Greedy / Beam+Wiki / Beam+KS (test CER %) | MD5 |
|---|---|---|---|---|---|
| `splashnet_raw_ep54.ckpt` | SplashNet retrained, no aug, no EMA | `splashnet` / `tds_conv_ctc` | 54 | 46.82 / 38.46 / 36.38 | 940e8b8d264645b111fa71b7bfbb6e8c |
| `splashnet_aug_ep123.ckpt` | SplashNet + aug | `splashnet` / `tds_conv_ctc` | 123 | 44.70 / 36.36 / 33.94 | 523c3e652494c633efa2847a00958a1c |
| `splashnet_ema_ep63.ckpt` | SplashNet + EMA | `splashnet` / `tds_conv_ctc` | 63 | 44.69 / 36.07 / 33.81 | 413eec1730bc4e51d9fdfc508e4a0e26 |
| `splashnet_aug_ema_ep76.ckpt` | SplashNet + aug + EMA (full recipe) | `splashnet` / `tds_conv_ctc` | 76 | 44.13 / 35.83 / 32.90 | fc817ebf3beb0e78a0a0ba3499552053 |
| `conformer_raw_ep36.ckpt` | Conformer retrained, no aug, no EMA | `conformer` / `conformer_ctc` | 36 | 41.38 / 32.60 / 30.70 | 00e4b2b86f236c564102604ac90a47ee |
| `conformer_aug_ep65.ckpt` | Conformer + aug | `conformer` / `conformer_ctc` | 65 | 41.32 / 32.81 / 31.17 | 4334eefab4b9f7d02f63cf309bb9d619 |
| `conformer_ema_ep42.ckpt` | Conformer + EMA | `conformer` / `conformer_ctc` | 42 | 39.60 / 31.09 / 28.98 | 84eb94df074ae1d5fd0b32430b2eff7f |
| `conformer_aug_ema_ep48.ckpt` | Conformer + aug + EMA (full recipe; **28.52**) | `conformer` / `conformer_ctc` (un-windowed) or `conformer_ctc_tail` (windowed, `TAIL_DELAY=62` → 28.48) | 48 | 39.70 / 30.25 / 28.52 | f5c4e863b921a634e5414354c05f9eb3 |

Sizes: Conformer 76.8 MB each, SplashNet 10.4 MB each (349 MB total).

Not redistributed here (download from the original authors): the released
SplashNet checkpoint (`models/splashnet.ckpt` in the SplashNet authors' repository),
the benchmark's TDS generic and per-participant fine-tuned checkpoints (emg2qwerty
`models/`), and Lee et al.'s Conformer (no checkpoint was published).
