#!/usr/bin/env python3
"""Statistics of the transplantation diagnostic (Table 4 and Discussion) from
results/transplant_per_participant.csv: per-module paired test against the generic model,
per-participant recovery of the personalization gap with a bootstrap 95% CI (1e4 resamples,
seed 0), and the residual gap to full fine-tuning. Tests are exact two-sided Wilcoxon
signed-rank tests over the eight test participants (eval/paired_stats.py).

Usage: python eval/transplant_stats.py [results/transplant_per_participant.csv]
"""
import csv, random, statistics as st, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_stats import wilcoxon_exact_two_sided as wx

path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "results/transplant_per_participant.csv")
R = {}
for row in csv.DictReader(open(path)):
    R[(row["transplanted_module"], row["decoder"])] = [float(row[f"P{i}"]) for i in range(8)]

def boot(x):
    random.seed(0)
    b = sorted(sum(random.choice(x) for _ in x) / len(x) for _ in range(10000))
    return b[250], b[9749]

for dec, modules in (("greedy", ["head", "mlp", "tds_conv", "tds_fc", "head_plus_fc"]), ("beam_ks", ["tds_fc"])):
    g, p = R[("generic", dec)], R[("personalized", dec)]
    gap = [a - b for a, b in zip(g, p)]
    print(f"== {dec}: generic {st.mean(g):.2f}, personalized {st.mean(p):.2f}")
    for m in modules:
        x = R[(m, dec)]
        d = [a - b for a, b in zip(x, g)]
        rec = [(a - b) / c for a, b, c in zip(g, x, gap)]
        lo, hi = boot(rec)
        print(f"{m:13s} CER {st.mean(x):6.2f}  recovery(of means) {100*(st.mean(g)-st.mean(x))/st.mean(gap):5.1f}%  "
              f"per-participant {100*st.mean(rec):5.1f}% CI [{100*lo:.1f}, {100*hi:.1f}] range {100*min(rec):.1f}-{100*max(rec):.1f}  "
              f"improved {sum(v < 0 for v in d)}/8  p_two_sided {wx(d):.4f}")
    x = R[("tds_fc", dec)]
    e = [a - b for a, b in zip(x, p)]
    lo, hi = boot(e)
    print(f"tds_fc - personalized {st.mean(e):+.2f} CI [{lo:+.2f}, {hi:+.2f}]  transplant better {sum(v < 0 for v in e)}/8  p_two_sided {wx(e):.4f}")
