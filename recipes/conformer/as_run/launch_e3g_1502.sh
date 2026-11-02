#!/bin/bash
# E3g: E3 recipe (Conformer + EMA + aug2, 150 ep) on the GPU front-end path, seed 1502, GPU1, 4 workers.
# seed 1501 = reproduction of the paper run on the new path; seed 1502 = seed variance.
set -u
BASE=/srv/emg/runConfE3g_s1502
LOG=$BASE/train.log
mkdir -p $BASE/run
push() {
  local TOK CHAT
  TOK=$(cat "$HOME/.telegram_bot_token"); CHAT=$(cat "$HOME/.telegram_chat_id")
  curl -s --max-time 8 -X POST "https://api.telegram.org/bot$TOK/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode text="[ws·E3g-s1502] $1" > /dev/null || true
}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
echo "RUNCONFE3G_S1502_STARTED $(date -u +%FT%TZ)" >> "$LOG"
push "E3g 발사 — Conformer+EMA+aug2 (GPU 프런트엔드), 150ep, seed 1502, GPU1, 4 workers"
nohup /home/jacob/tg_val_watch.sh ws-E3g-s1502 $BASE RUNCONFE3G_S1502_EXIT 600 150 > $BASE/watch.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/srv/emg/upstream-conf micromamba run -n emg2qwerty-cu128 python -m emg2qwerty.train \
  --config-name base_ema \
  model=conformer_ctc_gpufe transforms=aug2_raw user=generic_xval seed=1502 \
  dataset.root=/srv/emg/splashnet-ws/data \
  batch_size=32 num_workers=4 \
  trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=150 \
  hydra.run.dir=$BASE/run >> "$LOG" 2>&1
RC=$?
echo "RUNCONFE3G_S1502_EXIT rc=$RC $(date -u +%FT%TZ)" >> "$LOG"
push "RUNCONFE3G_S1502_EXIT rc=$RC"
