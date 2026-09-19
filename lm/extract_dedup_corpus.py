"""B: keystroke corpus with overlap-deduplication.

Rebuild the LM corpus while EXCLUDING keystrokes typed during prompts whose
normalized text also appears in any test user's test sessions (the 99
overlapping sentences found by the audit). Output format identical to
extract_keystroke_corpus.py.

Usage: python research/extract_dedup_corpus.py > models/lm/keystroke_corpus_dedup.txt
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import h5py
import yaml

import os
REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATA_ROOT", REPO / "splashnet" / "data"))
CONFIG = Path(os.environ.get("CONFIG_DIR", REPO / "splashnet" / "config" / "user"))
LM_VOCAB = set("abcdefghijklmnopqrstuvwxyz")


def norm_sentence(text: str) -> str:
    return " ".join(re.findall(r"[a-z]+", text.lower()))


def session_prompts(name: str):
    with h5py.File(DATA / f"{name}.hdf5", "r") as f:
        prompts = json.loads(f["emg2qwerty"].attrs["prompts"])
    for p in prompts:
        payload = p.get("payload")
        if payload and payload.get("text"):
            s = norm_sentence(payload["text"])
            if s:
                yield s, float(p["start"]), float(p["end"])


def effective_text(keystrokes, exclude_spans):
    buf = []
    for k in keystrokes:
        t = float(k["start"])
        if any(a <= t <= b for a, b in exclude_spans):
            continue
        key = k["key"]
        if key == "Key.backspace":
            if buf:
                buf.pop()
        elif key == "Key.space":
            buf.append(" ")
        elif key == "Key.enter":
            buf.append("\n")
        elif len(key) == 1:
            buf.append(key)
    return "".join(buf)


def words_from_text(text: str):
    word = []
    for ch in text.lower():
        if ch in LM_VOCAB:
            word.append(ch)
        else:
            if word:
                yield "".join(word)
                word = []
    if word:
        yield "".join(word)


def main() -> None:
    gen = yaml.safe_load((CONFIG / "generic.yaml").read_text())
    train_sessions = [e["session"] for e in gen["dataset"]["train"]]

    test_sentences: set[str] = set()
    for u in range(8):
        cfg = yaml.safe_load((CONFIG / f"user{u}.yaml").read_text())
        for e in cfg["dataset"]["test"]:
            for s, _, _ in session_prompts(e["session"]):
                test_sentences.add(s)
    print(f"test prompt pool: {len(test_sentences)}", file=sys.stderr)

    n_words = n_dropped = 0
    for i, name in enumerate(train_sessions):
        with h5py.File(DATA / f"{name}.hdf5", "r") as f:
            keystrokes = json.loads(f["emg2qwerty"].attrs["keystrokes"])
        spans = [(a, b) for s, a, b in session_prompts(name)
                 if s in test_sentences]
        n_dropped += len(spans)
        for w in words_from_text(effective_text(keystrokes, spans)):
            print(" ".join(w))
            n_words += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(train_sessions)}", file=sys.stderr)
    print(f"done: {n_words} words, {n_dropped} overlapping prompt spans dropped",
          file=sys.stderr)


if __name__ == "__main__":
    main()
