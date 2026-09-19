#!/bin/bash
# 증강 라이더: A3 풀레시피 + 소수점 전극 시프트(±1.5피치) + 채널 게인 지터(σ0.2)
# transforms=log_spectrogram_shift2 하나만 A3와 다름. 에폭별 텔레그램 자체 장전.
set -u
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV=${1:-emg2qwerty-cu128}
EPOCHS=${2:-150}
BASE=/srv/emg/runA3aug2
RUN=$BASE/run
REPO=/srv/emg/splashnet-ws
LOG=$BASE/train.log
mkdir -p "$RUN"
cd "$RUN"
push() {
  local TOK CHAT
  TOK=$(cat "$HOME/.telegram_bot_token"); CHAT=$(cat "$HOME/.telegram_chat_id")
  curl -s --max-time 8 -X POST "https://api.telegram.org/bot$TOK/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode text="[ws·A3aug2] $1" > /dev/null || true
}
echo "RUNA3AUG2_STARTED env=$ENV epochs=$EPOCHS $(date -u +%FT%TZ)" >> "$LOG"
push "증강 라이더 학습 시작 — aug2: 시프트+지터+리샘플워프0.1+크기워프0.2 (${EPOCHS}ep)"
nohup /home/jacob/tg_line_watch.sh "ws·A3aug2" "$LOG" "val/CER" "RUNA3AUG2_EXIT" 600 \
  > "$BASE/watch.log" 2>&1 < /dev/null &
PYTHONPATH=$REPO /usr/local/bin/micromamba run -n $ENV python -m emg2qwerty.train --config-name base_ema \
  user=generic transforms=log_spectrogram_shift2 \
  dataset.root=/srv/emg/splashnet-ws/data \
  batch_size=32 +trainer.accumulate_grad_batches=8 \
  trainer.accelerator=gpu trainer.devices=1 \
  trainer.max_epochs=$EPOCHS num_workers=16 \
  hydra.run.dir=$RUN >> "$LOG" 2>&1
RC=$?
echo "RUNA3AUG2_EXIT rc=$RC $(date -u +%FT%TZ)" >> "$LOG"
push "RUNA3AUG2_EXIT rc=$RC"
