# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Released SplashNet, published decoder (Beam+Wiki); our re-measurement. Mean 35.77.
rel  = np.array([38.02,41.08,26.02,36.13,40.43,23.85,44.77,35.84])
# Conformer (ours), Beam+KS, validation-best checkpoint (E3, ep. 48). Mean 28.52.
ours = np.array([38.33,42.26,19.73,24.78,32.78,13.23,31.45,25.61])
order = np.argsort(rel)  # released 낮은 유저부터 왼→오
C_REL, C_WIN, C_LOSE = "#4C78A8", "#54A24B", "#E45756"

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(5.0, 4.5))
xs = np.arange(len(order))
for x, i in zip(xs, order):
    win = ours[i] < rel[i]
    c = C_WIN  # Conformer(ours) always green, even when higher
    ax.plot([x, x], [rel[i], ours[i]], color=c, lw=2.4, zorder=1, alpha=0.85, solid_capstyle="round")
    ax.scatter(x, rel[i], s=32, color=C_REL, edgecolor="black", lw=0.7, zorder=3)
    ax.scatter(x, ours[i], s=32, color=c, edgecolor="black", lw=0.7, zorder=3)
    d = ours[i] - rel[i]
    ytxt = min(rel[i], ours[i]) - 1.1 if win else max(rel[i], ours[i]) + 1.1
    ax.annotate(f"{d:+.2f}", (x, ytxt), ha="center",
                va=("top" if win else "bottom"), fontsize=9, color=c, fontweight="bold")
# 평균 열
xm = len(order) + 0.7
ax.axvline(len(order) - 0.35 + 0.5, color="0.8", lw=1)
ax.plot([xm, xm], [rel.mean(), ours.mean()], color=C_WIN, lw=2.4, zorder=2)
ax.scatter(xm, rel.mean(), marker="D", s=45, color=C_REL, edgecolor="black", zorder=3)
ax.scatter(xm, ours.mean(), marker="D", s=45, color=C_WIN, edgecolor="black", zorder=3)
ax.annotate(f"{rel.mean():.2f}", (xm + 0.25, rel.mean()), va="center", fontsize=10, color=C_REL, fontweight="bold")
ax.annotate(f"{ours.mean():.2f}", (xm + 0.25, ours.mean()), va="center", fontsize=10, color=C_WIN, fontweight="bold")
ax.set_xticks(list(xs) + [xm])
ax.set_xticklabels([f"P{i}" for i in order] + ["Mean"], fontsize=11)
ax.set_ylabel("Character-Error Rate (%)", fontsize=12)
h = [plt.Line2D([], [], marker="o", ls="", mec="black", mfc=C_REL, label="SplashNet, released (Beam$+$Wiki)"),
     plt.Line2D([], [], marker="o", ls="", mec="black", mfc=C_WIN, label="Conformer, ours (Beam$+$KS)")]
ax.legend(handles=h, loc="upper left", fontsize=9, frameon=True)
ax.yaxis.grid(True, ls="--", alpha=0.4)
ax.set_axisbelow(True)
ax.set_ylim(11.0, 46)
ax.set_xlim(-0.7, xm + 1.4)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
plt.tight_layout()
for ext in ["pdf", "png"]:
    plt.savefig(f"fig_dumbbell_wiki_v.{ext}", dpi=200, bbox_inches="tight")
print("SAVED")
