"""System overview block diagram (matplotlib, vector PDF):
sEMG -> SplashNet encoder -> CTC beam search (+ keystroke 6-gram) -> keystream.
Our two levers (training recipe, keystroke LM) are orange; the 8B LLM rescorer
is a dashed gray side branch that never reaches the output (diagnostic only).
Taps above the chain give measured zero-shot CER: released encoder -> retrained (ours).
  micromamba run -n emg2qwerty python fig_pipeline.py   ->  fig_pipeline.pdf
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).parent

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
})

BLUE_D = "#1f4e79"
BLUE_M = "#4a7fb5"
BLUE_L = "#dbe7f2"
ORANGE = "#e2711d"
ORANGE_L = "#fbe3d0"
INK = "#15202e"
GRAY = "#8a8f98"
GRAY_L = "#eceff3"
GRAY_E = "#b9c0c9"
EDGE = "#9db8cc"


def block(ax, x0, x1, y0, y1, face, edge, label, sub=None, tcol=None,
          lw=1.4, fs=12, ls="-", sub_fs=6.8, dy=0.17):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=0.12",
        facecolor=face, edgecolor=edge, linewidth=lw, linestyle=ls, zorder=3))
    yc = (y0 + y1) / 2
    tcol = tcol or "white"
    if sub:
        ax.text((x0 + x1) / 2, yc + dy, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tcol, zorder=4)
        ax.text((x0 + x1) / 2, yc - dy - 0.09, sub, ha="center", va="center",
                fontsize=sub_fs, color=tcol, zorder=4, alpha=0.92,
                linespacing=1.25)
    else:
        ax.text((x0 + x1) / 2, yc, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tcol, zorder=4)


def arrow(ax, A, B, color=INK, lw=1.8, ls="-", rad=0.0, z=2):
    ax.add_patch(FancyArrowPatch(
        A, B, connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>,head_length=0.55,head_width=0.28",
        mutation_scale=9, shrinkA=1, shrinkB=1,
        linewidth=lw, linestyle=ls, color=color, zorder=z))


fig, ax = plt.subplots(figsize=(7.4, 2.75))
Y0, Y1 = -0.65, 0.65   # main-chain block height

# ── input: sEMG mini waveforms ─────────────────────────
block(ax, 0.0, 1.5, Y0, Y1, "white", EDGE, "", None)
t = np.linspace(0, 1, 160)
rng = np.random.default_rng(3)
for i, yb in enumerate((0.30, 0.02, -0.26)):
    w = (np.sin(2 * np.pi * (6 + 2 * i) * t + i)
         + 0.55 * rng.standard_normal(t.size))
    w = np.convolve(w, np.ones(5) / 5, mode="same")
    ax.plot(0.14 + 1.22 * t, yb + 0.085 * w, lw=0.8, color=BLUE_M, zorder=4)
ax.text(0.75, Y0 - 0.28, "sEMG · 2×16 ch · 2 kHz", ha="center",
        fontsize=7.8, color=INK)

# ── main chain: encoder -> beam search -> keystream ─────
EX0, EX1 = 2.1, 5.3
BX0, BX1 = 5.9, 8.9
KX0, KX1 = 9.5, 11.3
block(ax, EX0, EX1, Y0, Y1, BLUE_D, BLUE_D, "Encoder (free slot)",
      "SplashNet · Conformer (retrained)", fs=9.5)
block(ax, BX0, BX1, Y0, Y1, BLUE_M, BLUE_M, "CTC beam search",
      "beam 50 · backspace-aware", fs=9.5)
block(ax, KX0, KX1, Y0, Y1, "white", EDGE, "keystream",
      "incl. backspace", tcol=INK, fs=9.5)

arrow(ax, (1.5, 0), (EX0, 0))
arrow(ax, (EX1, 0), (BX0, 0))
ax.text((EX1 + BX0) / 2, 0.78, "posteriors\n$T\\times 99$", ha="center",
        va="bottom", fontsize=7.0, color=INK)
arrow(ax, (BX1, 0), (KX0, 0))

# ── our two levers (orange), below the chain ───────────
LY0, LY1 = -2.65, -1.45
block(ax, EX0, EX1, LY0, LY1, ORANGE_L, ORANGE, "training recipe",
      "T-aug · W-aug · EMA\n(encoder axis, ours)",
      tcol=INK, fs=9, lw=1.6, sub_fs=6.6, dy=0.22)
arrow(ax, ((EX0 + EX1) / 2, LY1), ((EX0 + EX1) / 2, Y0), color=ORANGE, lw=1.8)

block(ax, BX0 - 0.4, BX1 + 0.4, LY0, LY1, ORANGE, ORANGE, "keystroke 6-gram LM",
      "shallow fusion · LM-state rewind on delete\n(decoder axis, ours)",
      fs=9, sub_fs=6.6, dy=0.22)
arrow(ax, ((BX0 + BX1) / 2, LY1), ((BX0 + BX1) / 2, Y0), color=ORANGE, lw=1.8)

# ── diagnostic side branch (gray, dashed, never reaches output) ──
DX0, DX1 = 9.7, 13.9
block(ax, DX0, DX1, LY0, LY1, GRAY_L, GRAY_E, "8B LLM rescoring",
      "rescores the window 50-best — diagnostic only,\n"
      "never part of the reported system",
      tcol="#5b6470", fs=9, lw=1.2, ls=(0, (4, 2)), sub_fs=6.6, dy=0.22)
arrow(ax, (BX1 - 0.35, Y0), (DX0 + 0.7, LY1), color=GRAY, lw=1.4,
      ls=(0, (3, 2)), rad=0.25)
ax.text(9.45, -1.0, "$n$-best (50)", ha="left", va="top",
        fontsize=6.8, color=GRAY)

# ── taps: the three decoding exits (numbers live in Tables I/III) ──
def tap(x, y_text, label, color, ls="-"):
    arrow(ax, (x, Y1), (x, y_text + 0.18), color=color, lw=1.4, ls=ls)
    ax.text(x, y_text + 0.34, label, ha="center", va="bottom",
            fontsize=8.0, color=INK)

TY = 1.15
tap((EX0 + EX1) / 2, TY, "greedy (argmax)", BLUE_D)
tap((BX0 + BX1) / 2, TY, "beam + WikiText (published)", BLUE_D)
tap((KX0 + KX1) / 2, TY, "beam + keystroke LM (ours)", ORANGE)

ax.set_xlim(-0.15, 14.05)
ax.set_ylim(-2.85, 2.0)
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout(pad=0.15)
fig.savefig(HERE / "fig_pipeline.pdf")
print("wrote", HERE / "fig_pipeline.pdf")
