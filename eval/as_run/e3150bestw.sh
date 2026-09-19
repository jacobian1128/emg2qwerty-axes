#!/bin/bash
# E1 프로브 v2 — upstream 레포에서 greedy 8유저, 3레인 (val 스킵)
set -u
export MAMBA_ROOT_PREFIX=$HOME/micromamba OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
MM=/opt/homebrew/bin/micromamba
cd $HOME/upstream-conf
OUT=$HOME/emg-eval/results/e3150bestw
mkdir -p "$OUT"
push() {
  local TOK CHAT
  TOK=$(cat "$HOME/.telegram_bot_token"); CHAT=$(cat "$HOME/.telegram_chat_id")
  curl -s --max-time 8 -X POST "https://api.telegram.org/bot$TOK/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode text="[e3150bestw] $1" > /dev/null || true
}
run_one() {
  local u=$1
  PYTHONPATH=$HOME/upstream-conf nice -n 5 $MM run -n emg2qwerty python -m emg2qwerty.train \
    model=conformer_ctc user=$u train=False trainer.accelerator=cpu \
    checkpoint=$HOME/emg-eval/ckpts/e3150_best.ckpt \
    dataset.root=$HOME/splashnet-ws/data \
    decoder=ctc_beam decoder.lm_path=/Users/jacob/splashnet-ws/models/lm/wikitext-103-6gram-charlm.bin num_workers=1 +trainer.limit_val_batches=0 \
    hydra.run.dir="$OUT/$u" > "$OUT/$u.out" 2>&1
  local V
  V=$(python3 -c "
import re
t = open(\"$OUT/$u.out\", \"rb\").read().decode(\"utf8\", \"ignore\")
m = re.findall(r\"\x27test/CER\x27: ([0-9.]+)\", t)
print(m[-1] if m else \"FAIL\")")
  echo "beamks $u CER=$V"; [ "$V" = FAIL ] && push "$u → FAIL" || true
}
push "E3-150 best(ep48) Wiki 시작 (upstream rig, 3레인)"
i=0
for u in user0 user1 user2 user3 user4 user5 user6 user7; do
  run_one $u & i=$((i+1)); [ $((i % 3)) -eq 0 ] && wait
done
wait
AVG=$(python3 -c "
import re, glob
xs = []
for f in glob.glob(\"$OUT/user*.out\"):
    t = open(f, \"rb\").read().decode(\"utf8\", \"ignore\")
    m = re.findall(r\"\x27test/CER\x27: ([0-9.]+)\", t)
    if m: xs.append(float(m[-1]))
print(f\"{sum(xs)/len(xs):.2f} (n={len(xs)})\" if xs else \"FAIL\")")
echo "AVG=$AVG"; push "★ E1 e1bestw 평균 = $AVG"
echo e1bestw_DONE
