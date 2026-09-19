# Per-participant results

`table2_per_participant.csv` — every cell of the paper's design-space table (Table 2): test CER (%)
of the eight test participants P0–P7 (emg2qwerty `user0`–`user7`), then the
participant mean and sample s.d. (n−1). Rows: encoder × source × aug × EMA ×
lookahead × checkpoint × decoder (`greedy`, `beam_wiki` = stock WikiText-103
6-gram, `beam_ks` = keystroke 6-gram). The windowed-encoder rows (lookahead
0.62 s) are the fixed-positional-constant run reported in the paper.

`controls_per_participant.csv` — decoder-axis controls on the released SplashNet
checkpoint (beam without LM; deduplicated keystroke prior; WikiText restricted to
the keystroke vocabulary; vocabulary-uniform prior), the deduplicated prior on the
full-recipe Conformer, and the per-frame-causal Conformer probe.

Per-participant values are rounded to 0.01, so a mean recomputed from the CSV can
differ from the paper by ±0.01 (e.g. 33.95 vs 33.94). Paired statistics in
the paper (`eval/paired_stats.py`) were computed from the unrounded values.

Every value comes from `emg2qwerty.train ... train=False` on the un-modified
benchmark test sessions; beam settings are the benchmark defaults (width 50,
LM weight 2.0, insertion bonus 2.0).
