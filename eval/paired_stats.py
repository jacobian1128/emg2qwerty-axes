#!/usr/bin/env python3
"""쌍체 통계: 유저별 차이, 부트스트랩 95% CI(1e4), 정확 양측 Wilcoxon 부호순위(n<=12 전수 열거). 부분집합 지원.
사용: paired_stats.py "a1 a2 ... a8" "b1 ... b8" [--subset 1,3,4,6,7]  (검정: a - b)"""
import sys, random, itertools, statistics as st
def wilcoxon_exact_two_sided(d):
    d = [x for x in d if x != 0]
    n = len(d)
    if n == 0: return 1.0
    ranks = {}
    order = sorted(range(n), key=lambda i: abs(d[i]))
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j+1]]) == abs(d[order[i]]): j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1): ranks[order[k]] = r
        i = j + 1
    w_pos = sum(ranks[i] for i in range(n) if d[i] > 0)
    w_neg = sum(ranks[i] for i in range(n) if d[i] < 0)
    w = min(w_pos, w_neg)
    cnt = 0; total = 0
    rk = [ranks[i] for i in range(n)]
    for signs in itertools.product([0, 1], repeat=n):
        wp = sum(r for r, sg in zip(rk, signs) if sg)
        wn = sum(rk) - wp
        total += 1
        if min(wp, wn) <= w: cnt += 1
    return cnt / total
def main():
    a = [float(x) for x in sys.argv[1].split()]
    b = [float(x) for x in sys.argv[2].split()]
    idx = list(range(len(a)))
    if "--subset" in sys.argv:
        idx = [int(x) for x in sys.argv[sys.argv.index("--subset") + 1].split(",")]
    d = [a[i] - b[i] for i in idx]
    random.seed(0)
    boots = []
    for _ in range(10000):
        smp = [random.choice(d) for _ in d]
        boots.append(sum(smp) / len(smp))
    boots.sort()
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
    wins = sum(1 for x in d if x < 0)
    print(f"n={len(d)} mean_diff={sum(d)/len(d):+.3f} sd={st.stdev(d) if len(d)>1 else 0:.3f} "
          f"CI95=[{lo:+.2f},{hi:+.2f}] wins(a<b)={wins}/{len(d)} p_two_sided={wilcoxon_exact_two_sided(d):.4f}")
    print("diffs:", " ".join(f"{x:+.2f}" for x in d))
if __name__ == "__main__": main()
