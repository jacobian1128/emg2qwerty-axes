# Decoder and Encoder Axes of the Cross-User Gap in Zero-Shot sEMG Typing

Code, recipes, language models, checkpoints and per-participant results for the
paper of the same title (S. Park, Korea National Open University). Everything runs on the public
[emg2qwerty](https://github.com/facebookresearch/emg2qwerty) benchmark.

Headline: a Conformer retrained from public code with targeted augmentation and
EMA weight averaging, decoded by beam search under a keystroke character 6-gram
built from the benchmark's own training keylogs, reaches **28.52 % zero-shot CER**
on the eight test participants (published state of the art 35.67 %). The same
checkpoint decoded with a windowed encoder whose future input is bounded to
0.62 s gives 28.48 %.

## Layout

| Path | Contents |
|---|---|
| `conformer/` | emg2qwerty fork with the Conformer encoder (Lee et al.'s public code) plus our EMA callback, augmentations, and the windowed (tail-stream) encoder |
| `splashnet/` | emg2qwerty fork with the SplashNet architecture and our training-recipe changes |
| `recipes/` | launcher scripts of every training run in the design-space table, Table 2 (`RECIPES.md` explains them) |
| `lm/` | keystroke-corpus extraction, LM builders, and the four 6-gram binaries used in the paper |
| `eval/` | 8-participant evaluator, validation delay sweep, windowed-encoder checks, FLOP count, paired statistics |
| `results/` | per-participant CER for every Table 2 cell and every control |
| `checkpoints/` | table and checksums of the eight released checkpoints (files on the Release page) |
| `paper_figures/` | scripts of the pipeline, decoder-ladder and per-participant figures |

## Setup

1. Create the benchmark environment once (`conda env create -f splashnet/environment.yml`,
   or the CUDA variant you use) and download the emg2qwerty dataset into
   `splashnet/data/` (both trees read the same folder via `dataset.root`).
2. `pip install -e splashnet` and `pip install -e conformer` are not needed:
   scripts set `PYTHONPATH` to the tree they use.
3. Download the checkpoints from the `v1.0` Release into `checkpoints/`
   (see `checkpoints/README.md` for the mapping and MD5 sums).
4. For Beam+Wiki rows build the stock WikiText prior with `splashnet/scripts/lm/build_char_lm.sh 6`
   (KenLM required). The keystroke priors are already in `lm/bins/`.

## Reproduce one number

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1     # torch >= 2.6
# Table 2 corner cell: Conformer full recipe, Beam+KS  -> mean 28.52
eval/eval_8users.sh conformer conformer_ctc checkpoints/conformer_aug_ema_ep48.ckpt ks out/e3_ks lm/bins/keystroke-6gram-charlm.bin
# same checkpoint, windowed encoder (0.5-s commit delay) -> 28.48
TAIL_DELAY=62 ACCEL=gpu eval/eval_8users.sh conformer conformer_ctc_tail checkpoints/conformer_aug_ema_ep48.ckpt ks out/e3_win_ks lm/bins/keystroke-6gram-charlm.bin
# SplashNet full recipe, Beam+KS -> 32.90
eval/eval_8users.sh splashnet tds_conv_ctc checkpoints/splashnet_aug_ema_ep76.ckpt ks out/sn_ks lm/bins/keystroke-6gram-charlm.bin
```

The un-windowed rows run on CPU; the windowed row recomputes a 5-s window every
96 ms (61× the single-pass compute) and should be run with a GPU (`ACCEL=gpu`).
`PAR=3` evaluates three participants at once.

Paired statistics: `python eval/paired_stats.py` on two per-participant vectors
(bootstrap CI, exact Wilcoxon signed-rank).

## Training

See `RECIPES.md` for the exact overrides of all eight runs and the decoding
settings; `recipes/*/as_run/` holds the launchers as they were executed.

## Notes on the public Conformer code

`conformer/emg2qwerty/conformer.py` keeps Lee et al.'s classes unmodified. Two
facts documented in the paper matter for anyone reusing it: the positional
encoding is added after the transpose to batch-first, so it acts along the batch
axis and the model carries no explicit temporal position (order comes from the
convolution modules); and the training code exposes a causal flag but runs with
it disabled and a symmetric convolution, so the published causal figure cannot
be reproduced from the release. The windowed encoder (`TailStreamConformerModule`)
gives every window of a session the positional constant that a single pass
gives that session.

## License and attribution

CC BY-NC-SA 4.0, inherited from emg2qwerty (`LICENSE`). Third-party code and
what was changed are listed in `NOTICE.md`. The dataset is not redistributed.

## Citation

S. Park, "Decoder and encoder axes of the cross-user gap in zero-shot sEMG typing" (2026). A full reference will be added here when the paper is published.
