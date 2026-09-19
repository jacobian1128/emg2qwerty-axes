#!/bin/bash
# 8유저 평가 러너 v2 — 워처 없이 자체 추출·자체 텔레그램 푸시
# 사용: run_eval_v2.sh <ckpt> <태그> [동시성=3]
#   env RUN_EVAL_MODEL=tds_attn_branch_ctc  (attention ckpt일 때; 기본은 stock)
set -u
CKPT=$1; TAG=$2; PAR=${3:-3}
MODEL=${RUN_EVAL_MODEL:-}
export MAMBA_ROOT_PREFIX=$HOME/micromamba
MM=/opt/homebrew/bin/micromamba
WS=$HOME/splashnet-ws
KSLM=$HOME/emg-eval/lm/keystroke-6gram-charlm.bin
OUT=$HOME/emg-eval/results/$TAG
mkdir -p "$OUT"
cd "$WS"
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6

push() {
  local TOK CHAT
  TOK=$(cat "$HOME/.telegram_bot_token"); CHAT=$(cat "$HOME/.telegram_chat_id")
  curl -s --max-time 8 -X POST "https://api.telegram.org/bot$TOK/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode text="[$TAG] $1" > /dev/null || true
}

extract_cer() {  # 견고 추출: 바이너리 오판·경합 무관
  python3 - "$1" <<'PY'
import re, sys
try:
    t = open(sys.argv[1], "rb").read().decode("utf8", "ignore")
    m = re.findall(r"'test/CER': ([0-9.]+)", t)
    print(m[-1] if m else "")
except Exception:
    print("")
PY
}

# GRL disc 키 자동 스트립
STRIPPED=$HOME/emg-eval/ckpts/.stripped.$TAG.ckpt
rm -f "$STRIPPED"
$MM run -n emg2qwerty python - "$CKPT" "$STRIPPED" <<'PY'
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu")
sd = ck["state_dict"]
dropped = [k for k in list(sd) if k.startswith("disc.")]
for k in dropped:
    sd.pop(k)
if dropped:
    torch.save(ck, dst); print(f"disc stripped: {len(dropped)}")
else:
    print("no disc keys")
PY
[ -f "$STRIPPED" ] && CKPT="$STRIPPED"

run_one() {
  local u=$1 dec=$2 args=""
  case $dec in
    beamks) args="decoder=ctc_beam decoder.lm_path=$KSLM" ;;
    *)      args="decoder=ctc_greedy" ;;
  esac
  local mopt=""
  [ -n "$MODEL" ] && mopt="model=$MODEL"
  nice -n 5 $MM run -n emg2qwerty python -m emg2qwerty.train \
    $mopt user=$u train=False trainer.accelerator=cpu +trainer.limit_val_batches=0 \
    checkpoint="$CKPT" $args num_workers=0 \
    hydra.run.dir="$OUT/$dec/$u" \
    > "$OUT/$dec.$u.out" 2>&1
  local rc=$?
  sleep 1
  local cer
  cer=$(extract_cer "$OUT/$dec.$u.out")
  echo "[$(date +%H:%M:%S)] $TAG $dec $u rc=$rc CER=${cer:-FAIL}"
  [ -z "${cer:-}" ] && push "$dec $u FAIL (rc=$rc)" || true
}

push "평가 시작 (동시 $PAR${MODEL:+, model=$MODEL})"
for dec in ${DECS:-greedy beamks}; do
  i=0
  for u in user0 user1 user2 user3 user4 user5 user6 user7; do
    run_one $u $dec &
    i=$((i+1))
    if [ $((i % PAR)) -eq 0 ]; then wait; fi
  done
  wait
  # 디코더별 평균 계산 + 푸시
  AVG=$(python3 - "$OUT" "$dec" <<'PY'
import re, sys, glob
out, dec = sys.argv[1], sys.argv[2]
xs = []
for u in range(8):
    try:
        t = open(f"{out}/{dec}.user{u}.out", "rb").read().decode("utf8", "ignore")
        m = re.findall(r"'test/CER': ([0-9.]+)", t)
        xs.append(float(m[-1]))
    except Exception:
        pass
print(f"{sum(xs)/len(xs):.2f} (n={len(xs)})" if xs else "FAIL")
PY
)
  echo "$TAG $dec AVG=$AVG"
  push "★ $dec 8유저 평균 = $AVG"
done
echo "$TAG EVAL_ALL_DONE"
push "EVAL_ALL_DONE"
