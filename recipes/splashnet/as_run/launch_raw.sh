#!/bin/bash
# A‴ 풀레시피 대조군: from-scratch FC SplashNet (포크 기본값 = user generic·150ep·ACM specaug·seed 1501)
# + 글로벌 256 (batch 32 × accum 8), CTC only (현재 포크에 GRL 코드 없음 확인됨)
# 사용: launch_a3.sh [env=emg2qwerty-cu128] [epochs=150]
set -u
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV=${1:-emg2qwerty-cu128}
EPOCHS=${2:-150}
BASE=/srv/emg/runA3
RUN=$BASE/run
REPO=/srv/emg/splashnet-ws
LOG=$BASE/train.log
mkdir -p "$RUN"
cd "$RUN"
push() {
  local TOK CHAT
  TOK=$(cat "$HOME/.telegram_bot_token"); CHAT=$(cat "$HOME/.telegram_chat_id")
  curl -s --max-time 8 -X POST "https://api.telegram.org/bot$TOK/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode text="[ws·A3대조군] $1" > /dev/null || true
}
echo "RUNA3_STARTED env=$ENV epochs=$EPOCHS $(date -u +%FT%TZ)" >> "$LOG"
push "A3 풀레시피 대조군 학습 시작 — batch 32×accum8(글로벌 256), ${EPOCHS}ep, ACM, CTC only"
PYTHONPATH=$REPO /usr/local/bin/micromamba run -n $ENV python -m emg2qwerty.train \
  user=generic dataset.root=/srv/emg/splashnet-ws/data \
  batch_size=32 +trainer.accumulate_grad_batches=8 \
  trainer.accelerator=gpu trainer.devices=1 \
  trainer.max_epochs=$EPOCHS num_workers=8 \
  hydra.run.dir=$RUN >> "$LOG" 2>&1
RC=$?
echo "RUNA3_EXIT rc=$RC $(date -u +%FT%TZ)" >> "$LOG"
push "RUNA3_EXIT rc=$RC"
