"""WikiText-103 restricted to the keystroke-corpus vocabulary (25,980 types; keeps 90.0% of
WikiText alpha tokens). Output: character-spaced one word per line. Env: KS_CORPUS, WIKIVOCAB_OUT."""
import os, sys, nltk, datasets
from pathlib import Path
KS = Path(os.environ.get("KS_CORPUS", "bins/keystroke_corpus.txt"))
OUT_PATH = Path(os.environ.get("WIKIVOCAB_OUT", "bins/wikivocab_train.txt"))
vocab=set()
for line in open(KS, encoding="utf8", errors="ignore"):
    w=line.strip().replace(" ","")
    if w: vocab.add(w)
sys.stderr.write(f"VOCAB={len(vocab)}\n"); sys.stderr.flush()
ALPHA=set("abcdefghijklmnopqrstuvwxyz")
nltk.download("punkt", quiet=True)
try: nltk.download("punkt_tab", quiet=True)
except Exception: pass
wt=datasets.load_dataset("Salesforce/wikitext","wikitext-103-raw-v1",split="all")
sys.stderr.write(f"WIKITEXT_LINES={len(wt)}\n"); sys.stderr.flush()
tot=kept=0
out=open(OUT_PATH,"w")
for line in wt["text"]:
    for word in nltk.word_tokenize(line):
        w=word.lower()
        if not w or not all(c in ALPHA for c in w): continue
        tot+=1
        if w in vocab:
            kept+=1; out.write(" ".join(w)+"\n")
out.close()
sys.stderr.write(f"TOTAL_ALPHA_WORDS={tot} KEPT={kept} KEPT_FRAC={kept/max(tot,1):.4f}\n")
print("PREPROCESS_DONE")
