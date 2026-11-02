#!/bin/bash
# 게임 1 스코어보드 8유저 확장 — 평가 전용 빈칸 채우기 (스튜디오 CPU)
# user1~7: transplant(tds_fc) × {greedy, beam+wiki, beam+KS} + randominit × {beam+KS}
set -u
REPO=/Users/jacob/Library/CloudStorage/Dropbox/Program/VSCode/emg2qwerty
LOG=$HOME/emg-eval/game1_fill.log
export MAMBA_ROOT_PREFIX=$HOME/micromamba
MM=/opt/homebrew/bin/micromamba

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

run_eval(){  # $1=user $2=ckpt $3=greedy|wiki|ks $4=tag
  local u=$1 ckpt=$2 dec=$3 tag=$4 args=""
  case $dec in
    greedy) args="decoder=ctc_greedy" ;;
    wiki)   args="decoder=ctc_beam" ;;
    ks)     args="decoder=ctc_beam decoder.lm_path=$REPO/models/lm/keystroke-6gram-charlm.bin" ;;
  esac
  local out=$HOME/emg-eval/results/g1fill/$tag
  mkdir -p "$out"
  (cd "$REPO" && nice -n 5 $MM run -n emg2qwerty python -m emg2qwerty.train \
     user=$u train=False trainer.accelerator=cpu \
     dataset.root=/Users/jacob/splashnet-ws/data checkpoint="$ckpt" $args \
     num_workers=2 hydra.run.dir="$out" > "$out/eval.out" 2>&1)
  local cer=$(grep -oE "'test/CER': [0-9.]+" "$out/eval.out" | tail -1 | grep -oE "[0-9.]+$")
  log "RESULT $tag test/CER=${cer:-FAIL}"
}

log "G1FILL_START"
for n in 1 2 3 4 5 6 7; do
  u=user$n
  $MM run -n emg2qwerty python "$REPO/research/transplant.py" --user $u >> "$LOG" 2>&1
  T=$HOME/emg2qwerty-logs/transplant/${u}_tds_fc.ckpt
  for d in greedy wiki ks; do run_eval $u "$T" $d "transplant_${u}_${d}" & done
  run_eval $u "$REPO/models/personalized-randominit/$u.ckpt" ks "randominit_${u}_ks" &
  wait
  log "USER_DONE $u"
done
log "G1FILL_ALL_DONE"
