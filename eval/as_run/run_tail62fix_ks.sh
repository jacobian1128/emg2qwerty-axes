#!/bin/bash
set -u
OUT=/srv/emg/results_tail62fix; mkdir -p "$OUT"
push() { local TOK CHAT; TOK=$(cat "$HOME/.telegram_bot_token"); CHAT=$(cat "$HOME/.telegram_chat_id")
  curl -s --max-time 8 -X POST "https://api.telegram.org/bot$TOK/sendMessage" -d chat_id="$CHAT" --data-urlencode text="$1" > /dev/null || true; }
BASE_KS=(38.33 42.26 19.73 24.78 32.78 13.23 31.45 25.61)
push "[ws·tail62fix·ks] 고정 PE beamKS 8유저 시작 (GPU 인코더 + CPU 빔)"
for i in 0 1 2 3 4 5 6 7; do u=user$i
  TAIL_DELAY=62 PYTHONPATH=/srv/emg/upstream-conf micromamba run -n emg2qwerty-cu128 python -m emg2qwerty.train \
    model=conformer_ctc_tail user=$u train=False checkpoint=/srv/emg/e3150_ep48_stage.ckpt \
    dataset.root=/srv/emg/splashnet-ws/data trainer.accelerator=gpu trainer.devices=1 num_workers=2 \
    decoder=ctc_beam decoder.lm_path=/srv/emg/splashnet-ws/models/lm/keystroke-6gram-charlm.bin \
    ++trainer.limit_val_batches=1 ++trainer.num_sanity_val_steps=0 hydra.run.dir="$OUT/ks/$u" > "$OUT/ks_$u.out" 2>&1
  V=$(grep -ao "test/CER.: [0-9.]*" "$OUT/ks_$u.out" | tail -1 | awk -F": " "{print \$2}")
  if [ -n "$V" ]; then D=$(python3 -c "print(f\"{$V-${BASE_KS[$i]}:+.2f}\")"); push "[ws·tail62fix·ks] $u → $(printf %.2f $V) | 통짜 ${BASE_KS[$i]} | Δ$D"; else push "[ws·tail62fix·ks] $u → FAIL"; fi
done
AVG=$(python3 -c "
import re, glob, statistics as st
xs=[]
for f in sorted(glob.glob(\"$OUT/ks_user*.out\")):
    t=open(f,\"rb\").read().decode(\"utf8\",\"ignore\"); m=re.findall(r\"\x27test/CER\x27: ([0-9.]+)\",t)
    if m: xs.append(float(m[-1]))
print(f\"{sum(xs)/len(xs):.2f} ± {st.stdev(xs):.2f} (n={len(xs)})\" if len(xs)>1 else \"FAIL\")")
push "[ws·tail62fix·ks] ★ 평균 $AVG (통짜 KS 28.52)"
echo TAIL62FIX_KS_DONE
