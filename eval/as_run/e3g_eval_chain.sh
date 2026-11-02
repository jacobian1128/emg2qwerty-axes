#!/bin/bash
# E3g seed runs — studio 8-user evaluation chain (upstream-conf rig, CPU, 3 lanes).
# Order: s1501 KS -> s1502 KS -> s1501 greedy -> s1502 greedy. Telegram: star averages and FAIL only.
set -u
export MAMBA_ROOT_PREFIX=$HOME/micromamba OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
MM=/opt/homebrew/bin/micromamba
KSLM=$HOME/emg-eval/lm/keystroke-6gram-charlm.bin
cd $HOME/upstream-conf
push() {
  local TOK CHAT
  TOK=$(cat "$HOME/.telegram_bot_token"); CHAT=$(cat "$HOME/.telegram_chat_id")
  curl -s --max-time 8 -X POST "https://api.telegram.org/bot$TOK/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode text="[studio·E3g] $1" > /dev/null || true
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
run_set() {  # <tag> <ckpt> <ks|g>
  local TAG=$1 CKPT=$2 DEC=$3 OUT=$HOME/emg-eval/results/$1 DARGS
  mkdir -p "$OUT"
  if [ "$DEC" = ks ]; then DARGS="decoder=ctc_beam decoder.lm_path=$KSLM"; else DARGS="decoder=ctc_greedy"; fi
  local i=0
  for u in user0 user1 user2 user3 user4 user5 user6 user7; do
    (
      PYTHONPATH=$HOME/upstream-conf nice -n 5 $MM run -n emg2qwerty python -m emg2qwerty.train \
        model=conformer_ctc user=$u train=False trainer.accelerator=cpu checkpoint="$CKPT" \
        dataset.root=$HOME/splashnet-ws/data $DARGS num_workers=1 +trainer.limit_val_batches=0 \
        hydra.run.dir="$OUT/$u" > "$OUT/$u.out" 2>&1
      V=$(cer_of "$OUT/$u.out"); echo "$TAG $u CER=${V:-FAIL}"
      [ -z "$V" ] && push "$TAG $u FAIL"
    ) &
    i=$((i+1)); [ $((i % 3)) -eq 0 ] && wait
  done
  wait
  local AVG
  AVG=$(python3 - "$OUT" <<'PY'
import re, sys, statistics as st
xs = []
for u in range(8):
    try:
        t = open(f"{sys.argv[1]}/user{u}.out", "rb").read().decode("utf8", "ignore")
        m = re.findall(r"'test/CER': ([0-9.]+)", t)
        if m: xs.append(float(m[-1]))
    except Exception:
        pass
print(f"{st.mean(xs):.2f} ± {st.stdev(xs):.2f} (n={len(xs)})" if len(xs) > 1 else "FAIL")
PY
)
  echo "$TAG AVG=$AVG"; push "★ $TAG 8유저 평균 = $AVG"
}
push "E3g 평가 체인 시작 — s1501(ep14)·s1502(ep35) × KS·greedy, 3레인"
run_set e3g_s1501_ks $HOME/emg-eval/ckpts/e3g_s1501_best_ep14.ckpt ks
run_set e3g_s1502_ks $HOME/emg-eval/ckpts/e3g_s1502_best_ep35.ckpt ks
run_set e3g_s1501_g  $HOME/emg-eval/ckpts/e3g_s1501_best_ep14.ckpt g
run_set e3g_s1502_g  $HOME/emg-eval/ckpts/e3g_s1502_best_ep35.ckpt g
echo E3G_EVAL_CHAIN_DONE
push "E3G_EVAL_CHAIN_DONE"
