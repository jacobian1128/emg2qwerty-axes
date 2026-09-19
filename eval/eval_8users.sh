#!/bin/bash
# Evaluate one checkpoint on the 8 emg2qwerty test participants.
#
#   eval_8users.sh <tree> <model> <ckpt> <decoder> <out_dir> [lm_bin]
#     tree    : conformer | splashnet            (which code tree under this repo)
#     model   : conformer_ctc | conformer_ctc_tail | tds_conv_ctc
#     decoder : greedy | wiki | ks               (wiki/ks need lm_bin)
#   env: DATA_ROOT (default <repo>/splashnet/data), ACCEL (cpu|gpu, default cpu),
#        TAIL_DELAY (frames, windowed encoder only; 62 = 0.5 s), PAR (parallel users, default 1)
#
# Examples (Table 2 cells):
#   eval_8users.sh conformer conformer_ctc      ckpt/conformer_aug_ema_ep48.ckpt ks   out/e3_ks   lm/bins/keystroke-6gram-charlm.bin   # 28.52
#   TAIL_DELAY=62 ACCEL=gpu eval_8users.sh conformer conformer_ctc_tail ckpt/conformer_aug_ema_ep48.ckpt ks out/e3_win_ks lm/bins/keystroke-6gram-charlm.bin   # 28.48
#   eval_8users.sh splashnet tds_conv_ctc       ckpt/splashnet_aug_ema_ep76.ckpt  ks   out/sn_ks   lm/bins/keystroke-6gram-charlm.bin   # 32.90
set -u
TREE=$1; MODEL=$2; CKPT=$(cd "$(dirname "$3")" && pwd)/$(basename "$3"); DEC=$4; OUT=$5; LM=${6:-}
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-$REPO/splashnet/data}
ACCEL=${ACCEL:-cpu}; PAR=${PAR:-1}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
case $DEC in
  greedy) DARGS="decoder=ctc_greedy" ;;
  wiki|ks) [ -n "$LM" ] || { echo "lm_bin required for $DEC"; exit 1; }
           DARGS="decoder=ctc_beam decoder.lm_path=$(cd "$(dirname "$LM")" && pwd)/$(basename "$LM")" ;;
  *) echo "decoder must be greedy|wiki|ks"; exit 1 ;;
esac
mkdir -p "$OUT"
run_one() {
  local u=$1
  PYTHONPATH=$REPO/$TREE python -m emg2qwerty.train \
    model=$MODEL user=$u train=False checkpoint="$CKPT" dataset.root="$DATA_ROOT" \
    trainer.accelerator=$ACCEL trainer.devices=1 num_workers=2 $DARGS \
    ++trainer.limit_val_batches=1 ++trainer.num_sanity_val_steps=0 \
    hydra.run.dir="$OUT/$u" > "$OUT/$u.out" 2>&1
  echo "$u CER=$(grep -ao "test/CER.: [0-9.]*" "$OUT/$u.out" | tail -1 | awk -F': ' '{print $2}')"
}
i=0
for u in user0 user1 user2 user3 user4 user5 user6 user7; do
  run_one $u & i=$((i+1)); [ $((i % PAR)) -eq 0 ] && wait
done
wait
python3 - "$OUT" << 'PY'
import re, sys, glob, statistics as st
xs = []
for f in sorted(glob.glob(f"{sys.argv[1]}/user*.out")):
    m = re.findall(r"'test/CER': ([0-9.]+)", open(f, "rb").read().decode("utf8", "ignore"))
    if m: xs.append(float(m[-1]))
print(f"mean {st.mean(xs):.2f} sd {st.stdev(xs):.2f} n={len(xs)}" if len(xs) > 1 else "FAIL")
PY
