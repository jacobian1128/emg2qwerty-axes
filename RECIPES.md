# Recipes: every row of the design-space table (Table 2)

All runs: one seed (1501), 150 epochs of the benchmark's linear-warmup cosine
schedule, Adam, benchmark defaults for everything not listed. Validation uses
four held-out participants that are in neither the training nor the test set
(`splashnet/config/user/generic.yaml`, `conformer/config/user/generic_xval.yaml`).
The deployed checkpoint is the epoch with the lowest `val/CER` (Lightning
`ModelCheckpoint`, the benchmark's own rule). The `as_run/` folders keep the exact
launcher scripts that produced the paper's checkpoints (with machine paths and
Telegram notifications left in); the commands below are the same overrides.

## SplashNet (tree `splashnet/`, model `tds_conv_ctc` = SplashNet architecture)

Global batch 256 = `batch_size=32` × `accumulate_grad_batches=8`, as published.
`transforms=log_spectrogram` (default) is the published recipe with its aggressive
channel masking; `transforms=log_spectrogram_shift2` adds our augmentation
(fractional band rotation ±1.5 electrode pitches, per-channel gain jitter σ=0.2,
resample warp 0.1, magnitude warp 0.2); `--config-name base_ema` adds the EMA
callback (decay 0.999, updated every optimizer step; validation, checkpointing
and test use the averaged weights).

```bash
cd splashnet; export PYTHONPATH=$PWD
COMMON="user=generic dataset.root=$PWD/data batch_size=32 +trainer.accumulate_grad_batches=8 \
        trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=150 num_workers=8"
# raw reproduction  (best ep54  → 46.82 / 38.46 / 36.38)
python -m emg2qwerty.train $COMMON hydra.run.dir=runs/splashnet_raw
# + aug            (best ep123 → 44.70 / 36.36 / 33.94)
python -m emg2qwerty.train $COMMON transforms=log_spectrogram_shift2 hydra.run.dir=runs/splashnet_aug
# + EMA            (best ep63  → 44.69 / 36.07 / 33.81)
python -m emg2qwerty.train --config-name base_ema $COMMON hydra.run.dir=runs/splashnet_ema
# + aug + EMA      (best ep76  → 44.13 / 35.83 / 32.90)  ← full recipe
python -m emg2qwerty.train --config-name base_ema $COMMON transforms=log_spectrogram_shift2 hydra.run.dir=runs/splashnet_aug_ema
```

## Conformer (tree `conformer/`, model `conformer_ctc`)

Lee et al.'s public Conformer code (3 blocks, 6 heads, kernel 31, 15.35M
parameters) in its acausal form; batch 32 without accumulation; the benchmark's
standard augmentation by default, `transforms=aug2` for ours.

```bash
cd conformer; export PYTHONPATH=$PWD
COMMON="model=conformer_ctc user=generic_xval seed=1501 dataset.root=$PWD/../splashnet/data \
        batch_size=32 num_workers=8 trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=150"
# raw reproduction  (best ep36 → 41.38 / 32.60 / 30.70)
python -m emg2qwerty.train $COMMON hydra.run.dir=runs/conformer_raw
# + aug            (best ep65 → 41.32 / 32.81 / 31.17)
python -m emg2qwerty.train $COMMON transforms=aug2 hydra.run.dir=runs/conformer_aug
# + EMA            (best ep42 → 39.60 / 31.09 / 28.98)
python -m emg2qwerty.train --config-name base_ema $COMMON hydra.run.dir=runs/conformer_ema
# + aug + EMA      (best ep48 → 39.70 / 30.25 / 28.52)  ← full recipe, the paper's record
python -m emg2qwerty.train --config-name base_ema $COMMON transforms=aug2 hydra.run.dir=runs/conformer_aug_ema
```

### Run-to-run variation (two further runs of the full recipe)

Same recipe as the full-recipe run above, with the log-spectrogram and SpecAugment
computed on the GPU instead of in the data-loader workers (`emg2qwerty/gpu_frontend.py`;
it agrees with the CPU transform to about 5e-7 and adds no parameters, so checkpoints are
interchangeable). Seed 1501 repeats the paper run on this input path; seed 1502 is a new seed.
Launchers as executed: `recipes/conformer/as_run/launch_e3g_1501.sh`, `launch_e3g_1502.sh`.

```bash
# run 2 (best ep14 -> 40.42 / - / 28.60), run 3 with seed=1502 (best ep35 -> 39.21 / - / 27.99)
python -m emg2qwerty.train --config-name base_ema model=conformer_ctc_gpufe transforms=aug2_raw \
    user=generic_xval seed=1501 dataset.root=$PWD/../splashnet/data batch_size=32 num_workers=4 \
    trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=150 hydra.run.dir=runs/conformer_aug_ema_gpufe_s1501
```

Their checkpoints are evaluated like the others (`eval/as_run/e3g_eval_chain*.sh`, greedy and
Beam+KS); per-participant results are in `results/runs_full_recipe_per_participant.csv`.
These two checkpoints are not part of the Release.

## Decoding (all rows)

Every checkpoint is decoded three ways with `eval/eval_8users.sh`: greedy, beam
search under the stock WikiText-103 character 6-gram (build it with
`lm/build_char_lm.sh 6` inside the tree, as in the benchmark), and the same beam
search under the keystroke 6-gram `lm/bins/keystroke-6gram-charlm.bin`. Beam
width 50, LM weight 2.0, insertion bonus 2.0 (benchmark defaults, not re-tuned).

## Windowed encoder (last row of Table 2; windowed-encoder figure)

Same checkpoint `conformer_aug_ema_ep48.ckpt`, model `conformer_ctc_tail`,
`TAIL_DELAY=62` frames (0.5 s at 125 Hz), window 624 frames:

```bash
TAIL_DELAY=62 ACCEL=gpu eval/eval_8users.sh conformer conformer_ctc_tail \
    checkpoints/conformer_aug_ema_ep48.ckpt ks out/win_ks lm/bins/keystroke-6gram-charlm.bin   # 28.48
```

`eval/val_delay_sweep.sh` reproduces the delay curve on the validation
participants (d = 0, 0.2, 0.5, 1, 2 s); `eval/windowed_checks.py` runs the
exactness, batch-invariance, FLOP and positional-constant checks reported in the paper (Results, composition).
`TAIL_PE=cycle` restores the public code's batch-position constants for the
comparison reported there.

## Causal attempts (not in the tables; discussed in Results, composition)

`recipes/conformer/as_run/launch_e3c_150.sh` (per-frame causal mask, model
`conformer_ctc_causal`), `launch_e3p_150.sh` (0.1-s lookahead,
`conformer_ctc_100ms`), `launch_e1c_150.sh`, and `launch_e3ft.sh` (fine-tuning
the acausal checkpoint under a 96-ms band mask). All stalled near validation CER
70 and were stopped; they are kept for the record.

## Transplantation diagnostic (supporting analyses)

Uses the benchmark's released TDS checkpoints (`generic.ckpt`, `personalized-finetuned/user{0..7}.ckpt`)
and the benchmark's default `tds_conv_ctc` model, run from the `conformer/` tree.

```bash
for n in 0 1 2 3 4 5 6 7; do
  python eval/transplant.py --generic models/generic.ckpt \
      --personalized-dir models/personalized-finetuned --user user$n --out-dir out/transplant
done
# evaluate each hybrid on its own participant, one output directory per cell:
#   eval/as_run/tp_regreedy.sh (greedy), eval/as_run/tp_reks.sh (beam + keystroke 6-gram)
python eval/transplant_stats.py      # recovery, bootstrap CI, exact Wilcoxon
```

## Language models (`lm/`)

```bash
lm/build_keystroke_lm.sh              # keystroke-6gram-charlm.bin  (528,039 word tokens)
lm/build_keystroke_lm.sh dedup        # every prompt shared with a test session removed
lm/build_keystroke_lm.sh vocabuniform # each of the 25,980 word types once
lm/build_keystroke_lm.sh wikivocab    # WikiText-103 restricted to those word types
```

The four `.bin` files used in the paper are included in `lm/bins/`. Requires
KenLM (`lmplz`, `build_binary`) and the benchmark dataset at `splashnet/data`
(or `DATA_ROOT`).
