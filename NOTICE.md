# Notices and attribution

This repository redistributes and modifies code from third parties. All of it is
released under the same license as the upstream benchmark, **CC BY-NC-SA 4.0**
(see `LICENSE`), because that license's ShareAlike clause applies to the forks.

| Component | Origin | What we changed |
|---|---|---|
| `conformer/`, `splashnet/` (everything not listed below) | [facebookresearch/emg2qwerty](https://github.com/facebookresearch/emg2qwerty) (Meta Platforms, CC BY-NC-SA 4.0), commit `3200d91` | forks; see the per-tree file lists below |
| `conformer/emg2qwerty/conformer.py` (classes `PositionalEncoding` … `ConformerModule`) | public training code of Lee et al., "Typing on the wrist: …", arXiv 2511.18213 (CMU course project, unreviewed) | appended, without modifying the original classes: `CausalConformer*`, `LookaheadConformer*`, `TailStreamConformerModule` (windowed encoder), `_FixedIndexPE` |
| `splashnet/emg2qwerty/modules.py` (SplashNet blocks) | Jacobs et al., SplashNet ([public code](https://github.com/facebookresearch/emg2qwerty) fork by the authors) | used as released; training recipe changes are in configs and `transforms.py` |
| `lm/build_char_lm.sh`, `lm/preprocess_char_lm.py` | emg2qwerty `scripts/lm/` | unchanged (used for the stock WikiText-103 prior) |
| `lm/extract_keystroke_corpus.py`, `lm/extract_dedup_corpus.py` | this work | keystroke corpus from the benchmark's training keylogs |

Files that differ from upstream in `conformer/`: `emg2qwerty/lightning.py`
(`EMAWeightAveraging` callback), `emg2qwerty/train.py`, `emg2qwerty/transforms.py`
(`FractionalBandRotation`, `PerChannelGainJitter`, `RandomResampleWarp`,
`MagnitudeWarp`), `emg2qwerty/conformer.py` (new), `config/base_ema.yaml`,
`config/model/conformer_ctc*.yaml`, `config/transforms/aug2.yaml`,
`config/user/generic_xval.yaml`, `config/user/valu0-3.yaml`.

Files that differ from upstream in `splashnet/`: `emg2qwerty/lightning.py`,
`emg2qwerty/modules.py`, `emg2qwerty/transforms.py`, `config/base_ema.yaml`,
`config/model/tds_conv_ctc.yaml` (SplashNet architecture), `config/model/tds_attn_branch_ctc.yaml`,
`tds_hat.yaml`, `tds_rnnt.yaml` (probes, not used in the paper's tables),
`config/transforms/log_spectrogram*.yaml`, `config/user/generic.yaml` (cross-user validation split),
`config/user/otheruser*.yaml`.

The emg2qwerty dataset itself is not redistributed; download it from the benchmark.
