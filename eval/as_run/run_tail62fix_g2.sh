#!/bin/bash
set -u
until grep -q TAIL62FIX_KS_DONE /srv/emg/tail62fixks.log 2>/dev/null; do sleep 120; done
OUT=/srv/emg/results_tail62fix; mkdir -p "$OUT/g2"
push() { local TOK CHAT; TOK=$(cat "$HOME/.telegram_bot_token"); CHAT=$(cat "$HOME/.telegram_chat_id")
  curl -s --max-time 8 -X POST "https://api.telegram.org/bot$TOK/sendMessage" -d chat_id="$CHAT" --data-urlencode text="$1" > /dev/null || true; }
for i in 0 1 2 3 4 5 6 7; do u=user$i
  TAIL_DELAY=62 PYTHONPATH=/srv/emg/upstream-conf micromamba run -n emg2qwerty-cu128 python -m emg2qwerty.train \
    model=conformer_ctc_tail user=$u train=False checkpoint=/srv/emg/e3150_ep48_stage.ckpt \
    dataset.root=/srv/emg/splashnet-ws/data trainer.accelerator=gpu trainer.devices=1 num_workers=2 \
    decoder=ctc_greedy ++trainer.limit_val_batches=1 ++trainer.num_sanity_val_steps=0 hydra.run.dir="$OUT/g2/$u" > "$OUT/g2_$u.out" 2>&1
done
AVG=$(python3 -c "
import re, glob, statistics as st
xs=[]
for f in sorted(glob.glob(\"$OUT/g2_user*.out\")):
    t=open(f,\"rb\").read().decode(\"utf8\",\"ignore\"); m=re.findall(r\"\x27test/CER\x27: ([0-9.]+)\",t)
    if m: xs.append(float(m[-1]))
print(\" \".join(f\"{v:.2f}\" for v in xs) + f\" | avg {sum(xs)/len(xs):.2f} ± {st.stdev(xs):.2f}\" if len(xs)>1 else \"FAIL\")")
push "[ws·tail62fix·g2] 최종코드 greedy 8유저: $AVG"
echo TAIL62FIX_G2_DONE
