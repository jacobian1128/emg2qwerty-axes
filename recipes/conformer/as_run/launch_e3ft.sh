#!/bin/bash
# E3ft: E3-best(acausal 28.52 가중치)를 96ms 밴드 마스크로 파인튜닝 — 방출 타이밍 교정 (20ep, lr 1e-4)
set -u
BASE=/srv/emg/runConfE3ft
LOG=$BASE/train.log
mkdir -p $BASE/run
push() {
  local TOK CHAT
  TOK=$(cat "$HOME/.telegram_bot_token"); CHAT=$(cat "$HOME/.telegram_chat_id")
  curl -s --max-time 8 -X POST "https://api.telegram.org/bot$TOK/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode text="[ws·E3ft] $1" > /dev/null || true
}
echo "RUNCONFE3FT_STARTED $(date -u +%FT%TZ)" >> "$LOG"
push "E3ft 발사 — E3-best → 96ms 마스크 파인튜닝, 20ep, lr 1e-4"
PYTHONPATH=/srv/emg/upstream-conf micromamba run -n emg2qwerty-cu128 python -m emg2qwerty.train \
  --config-name base_ema \
  model=conformer_ctc_100ms transforms=aug2 user=generic_xval seed=1501 \
  checkpoint=/srv/emg/e3ft_init.ckpt \
  dataset.root=/srv/emg/splashnet-ws/data \
  batch_size=32 num_workers=8 \
  optimizer.lr=1e-4 lr_scheduler.scheduler.warmup_epochs=1 \
  trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=20 \
  hydra.run.dir=$BASE/run >> "$LOG" 2>&1
RC=$?
echo "RUNCONFE3FT_EXIT rc=$RC $(date -u +%FT%TZ)" >> "$LOG"
push "RUNCONFE3FT_EXIT rc=$RC"
