#!/bin/bash
# E2: Conformer + EMA — E1 레시피에 base_ema만 추가 (150ep, seed 1501)
set -u
BASE=/srv/emg/runConfE3_150
LOG=$BASE/train.log
mkdir -p $BASE/run
push() {
  local TOK CHAT
  TOK=$(cat "$HOME/.telegram_bot_token"); CHAT=$(cat "$HOME/.telegram_chat_id")
  curl -s --max-time 8 -X POST "https://api.telegram.org/bot$TOK/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode text="[ws·E3] $1" > /dev/null || true
}
echo "RUNCONFE3150_STARTED $(date -u +%FT%TZ)" >> "$LOG"
push "E3 발사 — Conformer+EMA+aug2, 150ep, seed 1501"
nohup /home/jacob/tg_line_watch.sh ws-ConfE3150 "$LOG" "val/CER" "RUNCONFE3150_EXIT" 600 skip-history > $BASE/watch.log 2>&1 &
PYTHONPATH=/srv/emg/upstream-conf micromamba run -n emg2qwerty-cu128 python -m emg2qwerty.train \
  --config-name base_ema \
  model=conformer_ctc transforms=aug2 user=generic_xval seed=1501 \
  dataset.root=/srv/emg/splashnet-ws/data \
  batch_size=32 num_workers=8 \
  trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=150 \
  hydra.run.dir=$BASE/run >> "$LOG" 2>&1
RC=$?
echo "RUNCONFE3150_EXIT rc=$RC $(date -u +%FT%TZ)" >> "$LOG"
push "RUNCONFE3150_EXIT rc=$RC"
