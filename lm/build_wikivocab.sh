#!/bin/bash
# 리뷰어 #3 어휘 통제군: WikiText-103을 학습 keylog 어휘(25,980단어)로 제한한 6-gram char LM
export MAMBA_ROOT_PREFIX=$HOME/micromamba
cd $HOME/emg-eval/lm
/opt/homebrew/bin/micromamba run -n emg2qwerty python build_wikivocab_lm.py 2> build.err || { echo PREPROCESS_FAIL >> build.log; exit 1; }
$HOME/kenlm/build/bin/lmplz -o 6 --discount_fallback < wikivocab_train.txt > wikivocab-6gram-charlm.arpa 2>> build.err || { echo LMPLZ_FAIL >> build.log; exit 1; }
$HOME/kenlm/build/bin/build_binary wikivocab-6gram-charlm.arpa wikivocab-6gram-charlm.bin 2>> build.err || { echo BINARY_FAIL >> build.log; exit 1; }
echo WIKIVOCAB_LM_DONE >> build.log
