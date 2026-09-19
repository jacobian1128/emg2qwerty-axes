#!/bin/bash
# 창 길이 스윕 (d=62 고정, val 4유저 greedy): valD0 완료 대기 → E3ft 재개분 일시정지 → 스윕 → E3ft 재재개
set -u
until [ -f /srv/emg/val_d0.txt ] && [ "$(wc -l < /srv/emg/val_d0.txt)" -ge 4 ]; do sleep 60; done
sleep 30
XDG_RUNTIME_DIR=/run/user/1000 systemctl --user stop e3ft-r.service tgval-e3ft-r.service 2>/dev/null
sleep 5
push() {
  local TOK CHAT
  TOK=$(cat "$HOME/.telegram_bot_token"); CHAT=$(cat "$HOME/.telegram_chat_id")
  curl -s --max-time 8 -X POST "https://api.telegram.org/bot$TOK/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode text="$1" > /dev/null || true
}
rm -f /srv/emg/winsweep.txt
for cfg in "624 32" "1250 16" "3750 4" "7500 1"; do
  set -- $cfg; SEG=$1; SB=$2
  for U in valu0 valu1 valu2 valu3; do
    rm -rf /tmp/winsweep
    TAIL_DELAY=62 TAIL_SEG=$SEG TAIL_SEG_BATCH=$SB PYTHONPATH=/srv/emg/upstream-conf micromamba run -n emg2qwerty-cu128 python -m emg2qwerty.train \
      model=conformer_ctc_tail user=$U train=False checkpoint=/srv/emg/e3150_ep48_stage.ckpt \
      dataset.root=/srv/emg/splashnet-ws/data trainer.accelerator=gpu trainer.devices=1 num_workers=2 \
      decoder=ctc_greedy ++trainer.limit_val_batches=1 ++trainer.num_sanity_val_steps=0 \
      hydra.run.dir=/tmp/winsweep > /tmp/winsweep_${SEG}_$U.log 2>&1
    echo "SEG=$SEG $U CER=$(grep -ao "test/CER.: [0-9.]*" /tmp/winsweep_${SEG}_$U.log | tail -1 | awk -F": " "{print \$2}")" >> /srv/emg/winsweep.txt
  done
  push "[ws·winsweep] SEG=$SEG(${SEG}프레임): $(grep "SEG=$SEG " /srv/emg/winsweep.txt | awk "{print \$3}" | tr "\n" " ")"
done
XDG_RUNTIME_DIR=/run/user/1000 systemd-run --user --collect --unit=e3ft-r2 bash /srv/emg/launch_e3ft.sh
sleep 5
XDG_RUNTIME_DIR=/run/user/1000 systemd-run --user --collect --unit=tgval-e3ft-r2 /home/jacob/tg_val_watch.sh ws-ConfE3ft /srv/emg/runConfE3ft RUNCONFE3FT_EXIT 300 20
echo WINSWEEP_DONE
