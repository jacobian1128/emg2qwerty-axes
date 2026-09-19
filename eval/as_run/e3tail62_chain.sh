#!/bin/bash
# E3-best 꼬리정렬 스트리밍(지연 62프레임=0.5s) 본평가: beamKS → greedy 순차, 3레인
set -u
bash $HOME/emg-eval/run_e3tail62ks.sh
bash $HOME/emg-eval/run_e3tail62g.sh
echo E3TAIL62_ALL_DONE
