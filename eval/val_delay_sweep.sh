#!/bin/bash
# Commit-delay sweep of the windowed encoder on the 4 validation participants (greedy CER).
# Reproduces the validation delay curve of the windowed-encoder figure: d in {0, 25, 62, 125, 250} frames = {0, 0.2, 0.5, 1, 2} s at 125 Hz.
# valu0..valu3 configs put each validation participant's sessions in the test split.
#   eval/val_delay_sweep.sh <ckpt> [out_file]     env: DATA_ROOT, ACCEL (gpu recommended)
set -u
CKPT=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); OUTF=${2:-val_delay_sweep.txt}
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-$REPO/splashnet/data}; ACCEL=${ACCEL:-gpu}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
: > "$OUTF"
for D in 0 25 62 125 250; do
  for U in valu0 valu1 valu2 valu3; do
    TMP=$(mktemp -d)
    TAIL_DELAY=$D PYTHONPATH=$REPO/conformer python -m emg2qwerty.train \
      model=conformer_ctc_tail user=$U train=False checkpoint="$CKPT" dataset.root="$DATA_ROOT" \
      trainer.accelerator=$ACCEL trainer.devices=1 num_workers=2 decoder=ctc_greedy \
      ++trainer.limit_val_batches=1 ++trainer.num_sanity_val_steps=0 hydra.run.dir="$TMP" > "$TMP/log" 2>&1
    echo "D=$D $U CER=$(grep -ao "test/CER.: [0-9.]*" "$TMP/log" | tail -1 | awk -F': ' '{print $2}')" | tee -a "$OUTF"
    rm -rf "$TMP"
  done
done
