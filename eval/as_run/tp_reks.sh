#!/bin/bash
# Transplant diagnostic re-evaluation (2026-10-29): beam+KS test CER, 8 users x {generic, personalized} + user0 tds_fc (users 1-7 tds_fc KS already clean in g1fill).
# Every cell gets its own hydra.run.dir (the 8/17 runs collided on second-resolution dirs). 3 lanes. Telegram: star averages and FAIL only.
set -u
export MAMBA_ROOT_PREFIX=$HOME/micromamba
MM=/opt/homebrew/bin/micromamba
REPO=/Users/jacob/Library/CloudStorage/Dropbox/Program/VSCode/emg2qwerty
TP=$HOME/emg2qwerty-logs/transplant
OUTROOT=$HOME/emg-eval/results/tp_reks
SUM=$OUTROOT/summary.tsv
mkdir -p "$OUTROOT"; : > "$SUM"
push() {
  local TOK CHAT
  TOK=$(cat "$HOME/.telegram_bot_token"); CHAT=$(cat "$HOME/.telegram_chat_id")
  curl -s --max-time 8 -X POST "https://api.telegram.org/bot$TOK/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode text="[studio·transplant] $1" > /dev/null || true
}
cer_of() { python3 - "$1" <<'PY'
import re, sys
try:
    t = open(sys.argv[1], "rb").read().decode("utf8", "ignore")
    m = re.findall(r"'test/CER': ([0-9.]+)", t); print(m[-1] if m else "")
except Exception:
    print("")
PY
}
ckpt_for() {  # <cond> <user>
  case $1 in
    generic) echo "$REPO/models/generic.ckpt" ;;
    personalized) echo "$REPO/models/personalized-finetuned/$2.ckpt" ;;
    *) echo "$TP/${2}_$1.ckpt" ;;
  esac
}
JOBS=""
for c in generic personalized; do for n in 0 1 2 3 4 5 6 7; do JOBS="$JOBS $c:user$n"; done; done
JOBS="$JOBS tds_fc:user0"
run_one() {  # <cond> <user>
  local c=$1 u=$2 OUT=$OUTROOT/$1/$2 V
  mkdir -p "$OUT"
  cd "$REPO" && nice -n 5 $MM run -n emg2qwerty python -m emg2qwerty.train \
    user=$u train=False trainer.accelerator=cpu checkpoint="$(ckpt_for $c $u)" \
    dataset.root=/Users/jacob/splashnet-ws/data decoder=ctc_beam decoder.lm_path=$REPO/models/lm/keystroke-6gram-charlm.bin num_workers=2 \
    hydra.run.dir="$OUT" > "$OUT/eval.out" 2>&1
  V=$(cer_of "$OUT/eval.out"); printf "%s\t%s\t%s\n" "$c" "$u" "${V:-FAIL}" >> "$SUM"
  [ -z "$V" ] && push "$c $u FAIL"
}
# 3 lanes, continuous queue
for j in $JOBS; do
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 20; done
  run_one "${j%%:*}" "${j##*:}" &
  sleep 5
done
wait
for c in generic personalized; do
  AVG=$(awk -F'\t' -v c=$c '$1==c && $3!="FAIL"{s+=$3;n++} END{if(n) printf "%.2f (n=%d)", s/n, n}' "$SUM")
  push "★ $c beam+KS mean $AVG"
done
echo TP_REKS_DONE >> "$SUM.done"
