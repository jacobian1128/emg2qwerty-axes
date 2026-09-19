#!/bin/bash
# Build the keystroke 6-gram character LM (and the three control LMs) with KenLM.
# Prerequisites: KenLM built at $KENLM (default ~/kenlm/build/bin); the benchmark
# dataset extracted at <repo>/splashnet/data (or set DATA_ROOT); python env of
# the benchmark (h5py, yaml). WikiText controls also need `datasets` and `nltk`.
#
#   ./build_keystroke_lm.sh            # keystroke-6gram-charlm.bin
#   ./build_keystroke_lm.sh dedup      # dedup-6gram-charlm.bin (overlapping prompts removed)
#   ./build_keystroke_lm.sh vocabuniform
#   ./build_keystroke_lm.sh wikivocab  # WikiText-103 restricted to the keystroke vocabulary
set -eu
KENLM=${KENLM:-$HOME/kenlm/build/bin}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HERE/bins}; mkdir -p "$OUT"
WHICH=${1:-keystroke}
case $WHICH in
  keystroke)
    python "$HERE/extract_keystroke_corpus.py" > "$OUT/keystroke_corpus.txt"
    TRAIN="$OUT/keystroke_corpus.txt" ;;
  dedup)
    python "$HERE/extract_dedup_corpus.py" > "$OUT/keystroke_corpus_dedup.txt"
    TRAIN="$OUT/keystroke_corpus_dedup.txt" ;;
  vocabuniform)
    [ -f "$OUT/keystroke_corpus.txt" ] || python "$HERE/extract_keystroke_corpus.py" > "$OUT/keystroke_corpus.txt"
    python "$HERE/build_vocabuniform.py" "$OUT/keystroke_corpus.txt" > "$OUT/vocabuniform_train.txt"
    TRAIN="$OUT/vocabuniform_train.txt" ;;
  wikivocab)
    [ -f "$OUT/keystroke_corpus.txt" ] || python "$HERE/extract_keystroke_corpus.py" > "$OUT/keystroke_corpus.txt"
    KS_CORPUS="$OUT/keystroke_corpus.txt" WIKIVOCAB_OUT="$OUT/wikivocab_train.txt" python "$HERE/build_wikivocab_lm.py"
    TRAIN="$OUT/wikivocab_train.txt" ;;
  *) echo "unknown target $WHICH"; exit 1 ;;
esac
"$KENLM/lmplz" -o 6 --discount_fallback < "$TRAIN" > "$OUT/$WHICH-6gram-charlm.arpa"
"$KENLM/build_binary" "$OUT/$WHICH-6gram-charlm.arpa" "$OUT/$WHICH-6gram-charlm.bin"
echo "built $OUT/$WHICH-6gram-charlm.bin"
