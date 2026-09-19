"""Vocabulary-uniform control corpus: every word type of the keystroke corpus
exactly once (25,980 types), in the character-spaced one-word-per-line format
of preprocess_char_lm.py. A 6-gram fit on this file shares the keystroke LM's
n-gram inventory but not its counts (the paper's second vocabulary control, Results, decoder axis).

Usage: python build_vocabuniform.py keystroke_corpus.txt > vocabuniform_train.txt
"""
import sys
seen = set()
for line in open(sys.argv[1], encoding="utf8", errors="ignore"):
    w = line.strip().replace(" ", "")
    if w and w not in seen:
        seen.add(w)
for w in sorted(seen):
    print(" ".join(w))
sys.stderr.write(f"VOCAB={len(seen)}\n")
