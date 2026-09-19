# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Extract a character-LM training corpus from the keylogger streams of the
generic model's TRAIN split sessions (training users only; no test-user data).

Output format matches preprocess_char_lm.py: one word per line, characters
space-delimited, lowercase ascii alpha only. Backspaces are APPLIED
(screen-buffer reconstruction) so the corpus reflects the effective text the
decoder's backspace-aware LM rollback would score. Uncorrected typos remain.

Usage: python extract_keystroke_corpus.py > keystroke_corpus.txt
"""
import json
import sys
from pathlib import Path

import h5py
import yaml

import os
REPO = Path(__file__).resolve().parents[1]
# benchmark dataset (hdf5 files) and the user split configs; default: the splashnet tree
DATA = Path(os.environ.get("DATA_ROOT", REPO / "splashnet" / "data"))
CONFIG = Path(os.environ.get("CONFIG_DIR", REPO / "splashnet" / "config" / "user"))

LM_VOCAB = set("abcdefghijklmnopqrstuvwxyz")


def effective_text(keystrokes: list[dict]) -> str:
    """Replay the keylog applying backspaces (screen-buffer reconstruction)."""
    buf: list[str] = []
    for k in keystrokes:
        key = k["key"]
        if key == "Key.backspace":
            if buf:
                buf.pop()
        elif key == "Key.space":
            buf.append(" ")
        elif key == "Key.enter":
            buf.append("\n")
        elif len(key) == 1:  # regular character key
            buf.append(key)
        # other modifiers (Key.shift etc.) produce no character
    return "".join(buf)


def words_from_text(text: str):
    word: list[str] = []
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
    cfg = yaml.safe_load((CONFIG / "generic.yaml").read_text())
    train_sessions = [e["session"] for e in cfg["dataset"]["train"]]
    print(f"train sessions: {len(train_sessions)}", file=sys.stderr)

    n_words = 0
    for i, name in enumerate(train_sessions):
        with h5py.File(DATA / f"{name}.hdf5", "r") as f:
            keystrokes = json.loads(f["emg2qwerty"].attrs["keystrokes"])
        for w in words_from_text(effective_text(keystrokes)):
            print(" ".join(w))
            n_words += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(train_sessions)} sessions, {n_words} words",
                  file=sys.stderr)
    print(f"done: {n_words} words", file=sys.stderr)


if __name__ == "__main__":
    main()
