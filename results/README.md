# Per-participant results

`table2_per_participant.csv` — every cell of the paper's design-space table (Table 2): test CER (%)
of the eight test participants P0–P7 (emg2qwerty `user0`–`user7`), then the
participant mean and sample s.d. (n−1). Rows: encoder × source × aug × EMA ×
lookahead × checkpoint × decoder (`greedy`, `beam_wiki` = stock WikiText-103
6-gram, `beam_ks` = keystroke 6-gram). The windowed-encoder rows (lookahead
0.62 s) are the fixed-positional-constant run reported in the paper.

`runs_full_recipe_per_participant.csv` — the full-recipe Conformer trained three times
(run-to-run variation table): run 1 is the Table 2 run (seed 1501, input pipeline in the data
loader); runs 2 and 3 compute the log-spectrogram and SpecAugment on the GPU
(`conformer_ctc_gpufe`, `transforms=aug2_raw`) with seeds 1501 and 1502. Greedy and Beam+KS,
validation-best checkpoints.

`transplant_per_participant.csv` — personalized-weight transplantation diagnostic on the
benchmark's TDS encoder: test CER of the generic checkpoint, of the generic checkpoint with one
module's per-participant fine-tuned weights grafted in (`eval/transplant.py`), and of the full
per-participant fine-tuned checkpoint; greedy for every module, Beam+KS for generic, channel-mixing
(`tds_fc`) and personalized. Each cell was evaluated in its own output directory
(`eval/as_run/tp_regreedy.sh`, `eval/as_run/tp_reks.sh`; Beam+KS `tds_fc` for participants 1–7 from
`eval/as_run/game1_fill.sh`). Statistics: `python eval/transplant_stats.py`.

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
