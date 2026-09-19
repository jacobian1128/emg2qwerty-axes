"""Decoder-ladder figure (matplotlib, vector PDF).

Vertical CER axis (lower = higher = better). Solid rungs = measured;
dashed = pending (vertical position schematic). Two ceilings above the
ladder; hatched region = encoder territory. Writes fig_ladder.pdf.
  micromamba run -n emg2qwerty python gen_fig_ladder.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
})

BLUE = "#1f4e79"
LIGHT = "#9db8cc"
ORANGE = "#e2711d"
GRAY = "#8a8f98"
INK = "#15202e"

RAIL_L, RAIL_R = 0.15, 0.35
LBL_X = 0.38

# (CER, 이름, 부가설명, 상태)  상태: measured | pending
rungs = [
    (44.80, "Greedy = beam-only", "44.80/44.79", "measured"),
    (35.77, "Beam + WikiText 6-gram", "35.77", "measured"),
    (33.33, "Beam + keystroke 6-gram", "33.33", "best"),
]
oracle_y, win_y = 32.9, 29.85  # 세션(개략, 여력<0.1) / 윈도우(실측 8유저 평균)

fig, ax = plt.subplots(figsize=(3.5, 3.9))

# 인코더 전용 영역: 윈도우 oracle 위쪽 전부
ax.axhspan(26.4, win_y, facecolor="#eceff3", hatch="///",
           edgecolor="#c6cdd6", linewidth=0)
ax.text(0.5, 26.9,
        "Beyond the evaluated decoders",
        ha="center", va="center", fontsize=8, color="#5b6470",
        style="italic")

# 세션 oracle (실측: 여력 <0.1)
ax.hlines(oracle_y, 0.06, 0.94, color=BLUE, lw=1.6, linestyle=(0, (6, 3)))
ax.text(0.06, oracle_y - 1.05,
        "Oracle@50, session — bounds any $n$-best rescorer",
        fontsize=7.5, color=BLUE, ha="left", va="bottom", fontweight="bold")
ax.text(0.06, oracle_y - 1.55,
        "measured headroom 0.03 CER over beam+KS (8 participants)",
        fontsize=7, color=GRAY, ha="left", va="bottom", style="italic")

# 윈도우 oracle (실측: 3–4.5pt 보이나 98% 매장 — 8B 회수 0)
ax.hlines(win_y, 0.06, 0.94, color=BLUE, lw=1.6, linestyle=(0, (2, 2)))
ax.text(0.06, win_y - 1.02,
        "Oracle@50, window — 29.85: 3.4 CER visible (8 participants)",
        fontsize=7.5, color=BLUE, ha="left", va="bottom", fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.2))
ax.text(0.06, win_y - 1.62,
        "but in 99.5% the reference scores below the hypothesis (deficit 16–84)",
        fontsize=7, color="#5b6470", ha="left", va="bottom", style="italic",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.2))
ax.text(0.06, win_y - 2.24,
        "8B LLM rescorer, diagnostic only: 0.03 CER recovered ($\\to$ 33.19)",
        fontsize=7, color="#5b6470", ha="left", va="bottom", style="italic",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.2))

# 오라클 라벨 → 해당 점선 포인터 (아래 방향)
for ly in (oracle_y, win_y):
    ax.annotate("", xy=(0.028, ly - 0.08), xytext=(0.028, ly - 1.55),
                arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22",
                                mutation_scale=8, color=BLUE, lw=1.2))

# 사다리 세로 레일 (greedy ~ LLM 단)
top_rung, bot_rung = rungs[-1][0], rungs[0][0]
for x in (RAIL_L, RAIL_R):
    ax.vlines(x, top_rung, bot_rung, color=LIGHT, lw=1.8, zorder=1)

# 가로대 + 배지형 라벨 (파이프라인 그림 블록과 색 톤 일치)
from matplotlib.patches import FancyBboxPatch
for cer, name, val, status in rungs:
    if status == "best":
        color, lw, fill = ORANGE, 2.8, "#fbe3d0"
    else:
        color, lw, fill = BLUE, 2.4, "#e2ecf5"
    ax.hlines(cer, RAIL_L, RAIL_R, color=color, lw=lw, zorder=3)
    ax.add_patch(FancyBboxPatch(
        (LBL_X, cer - 0.52), 0.995 - LBL_X, 1.04,
        boxstyle="round,pad=0,rounding_size=0.02",
        facecolor=fill, edgecolor=color, linewidth=0.9,
        mutation_aspect=0.05, zorder=2))
    weight = "bold" if status == "best" else "normal"
    ax.text(LBL_X + 0.02, cer, name, fontsize=8.0, color=INK, ha="left",
            va="center", fontweight=weight, zorder=4)
    ax.text(0.975, cer, val, fontsize=8.4, color=color, ha="right",
            va="center", fontweight="bold", zorder=4)

# 측정 완료 디코더 이득 브래킷 (greedy → KS)
bx = 0.095
ax.annotate("", xy=(bx, 33.33), xytext=(bx, 44.80),
            arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.25",
                            mutation_scale=10, color=ORANGE, lw=1.8))
ax.text(bx - 0.022, (44.80 + 33.33) / 2, "measured decoder gain $-$11.5",
        fontsize=7, color=ORANGE, ha="center", va="center", rotation=90)

# 단 사이 델타 (오른쪽 여백을 정보로)
def delta(hi, lo, label):
    ax.annotate("", xy=(0.52, lo + 0.62), xytext=(0.52, hi - 0.62),
                arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.24",
                                mutation_scale=9, color=ORANGE, lw=1.5))
    ax.text(0.555, (hi + lo) / 2, label, fontsize=7.5, color=ORANGE,
            ha="left", va="center")

delta(44.80, 35.77, "generic language prior:  $-$9.03")
delta(35.77, 33.33, "domain matching:  $-$2.44")

# 축
ax.set_xlim(0, 1)
ax.set_ylim(46.3, 26.4)   # 아래 = 나쁨(44.8), 위 = 좋음
ax.set_yticks([45, 40, 35, 30])
ax.set_ylabel("Zero-shot test CER (%)  —  lower is better, upward",
              fontsize=8, color=INK)
ax.tick_params(axis="y", labelsize=7.5, colors=INK)
ax.set_xticks([])
for s in ("top", "right", "bottom"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#9db8cc")

ax.invert_yaxis()

fig.tight_layout(pad=0.3)
fig.savefig(HERE / "fig_ladder.pdf")
print("wrote", HERE / "fig_ladder.pdf")