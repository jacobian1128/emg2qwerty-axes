#!/bin/bash
# Transplant diagnostic re-evaluation (2026-10-29): greedy test CER, 8 users x {generic, personalized, 5 grafted modules}.
# Every cell gets its own hydra.run.dir (the 8/17 runs collided on second-resolution dirs). 3 lanes. Telegram: star averages and FAIL only.
set -u
export MAMBA_ROOT_PREFIX=$HOME/micromamba
MM=/opt/homebrew/bin/micromamba
REPO=/Users/jacob/Library/CloudStorage/Dropbox/Program/VSCode/emg2qwerty
TP=$HOME/emg2qwerty-logs/transplant
OUTROOT=$HOME/emg-eval/results/tp_regreedy
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
for c in generic personalized head mlp tds_conv tds_fc head_plus_fc; do
  i=0
  for u in user0 user1 user2 user3 user4 user5 user6 user7; do
    (
      OUT=$OUTROOT/$c/$u; mkdir -p "$OUT"
      cd "$REPO" && nice -n 5 $MM run -n emg2qwerty python -m emg2qwerty.train \
        user=$u train=False trainer.accelerator=cpu checkpoint="$(ckpt_for $c $u)" \
        dataset.root=/Users/jacob/splashnet-ws/data decoder=ctc_greedy num_workers=2 \
        hydra.run.dir="$OUT" > "$OUT/eval.out" 2>&1
      V=$(cer_of "$OUT/eval.out"); printf "%s\t%s\t%s\n" "$c" "$u" "${V:-FAIL}" >> "$SUM"
      [ -z "$V" ] && push "$c $u FAIL"
    ) &
    sleep 3
    i=$((i+1)); [ $((i % 3)) -eq 0 ] && wait
  done
  wait
  AVG=$(awk -F'\t' -v c=$c '$1==c && $3!="FAIL"{s+=$3;n++} END{if(n) printf "%.2f (n=%d)", s/n, n}' "$SUM")
  push "★ $c greedy mean $AVG"
done
echo TP_REGREEDY_DONE >> "$SUM.done"
