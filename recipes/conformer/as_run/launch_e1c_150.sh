#!/bin/bash
# E1c: E1 재현 레시피의 causal판 — 공개코드 기본 + model=conformer_ctc_causal (150ep, seed 1501)
# 목적: Lee et al. causal 43.21의 재현 게이트 (謹 val-best greedy가 43.21±0.5 근방인지)
set -u
BASE=/srv/emg/runConfE1c_150
LOG=$BASE/train.log
mkdir -p $BASE/run
push() {
  local TOK CHAT
  TOK=$(cat "$HOME/.telegram_bot_token"); CHAT=$(cat "$HOME/.telegram_chat_id")
  curl -s --max-time 8 -X POST "https://api.telegram.org/bot$TOK/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode text="[ws·E1c] $1" > /dev/null || true
}
echo "RUNCONFE1C150_STARTED $(date -u +%FT%TZ)" >> "$LOG"
push "E1c 발사 — causal Conformer 재현(레시피=E1), 150ep, seed 1501"
nohup /home/jacob/tg_line_watch.sh ws-ConfE1c150 "$LOG" "val/CER" "RUNCONFE1C150_EXIT" 600 skip-history > $BASE/watch.log 2>&1 &
PYTHONPATH=/srv/emg/upstream-conf micromamba run -n emg2qwerty-cu128 python -m emg2qwerty.train \
  model=conformer_ctc_causal user=generic_xval seed=1501 \
  dataset.root=/srv/emg/splashnet-ws/data \
  batch_size=32 num_workers=8 \
  trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=150 \
  hydra.run.dir=$BASE/run >> "$LOG" 2>&1
RC=$?
echo "RUNCONFE1C150_EXIT rc=$RC $(date -u +%FT%TZ)" >> "$LOG"
push "RUNCONFE1C150_EXIT rc=$RC"
