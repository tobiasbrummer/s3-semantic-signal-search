# Saved Result Artifacts

Reproducible JSON artifacts for the headline numbers cited in the
project's main README.

## `onset_combined/`

Output of `experiments/research/onset_combined_test.py` -- the script
that produces the Recall@10 / latency comparison between Pooled,
Combined (dense+SPLADE RRF), Combined+Onset, and token-level
brute-force on BEIR SciFact.

| Run | Dataset | Backend | Headline |
|---|---|---|---|
| `onset_combined_scifact_20260525_190730Z.json` | BEIR SciFact, 1000 docs, 70 queries (post-filter) | local: `jinaai/jina-embeddings-v3` + `naver/splade-cocondenser-ensembledistil` on a single A100 | Pooled 85.7% / Combined 94.3% / **Combined+Onset 92.9%** / Token-BF 94.3%; Combined+Onset runs at **174 ms vs 1631 ms** for Token-BF, i.e. **~1/9 the compute**, at a **-1.4 pp** recall cost vs Token-BF and a **-1.4 pp** recall cost vs plain Combined. |

### How to read these numbers

Two findings sit inside the table:

1. **Combined (dense+SPLADE via RRF) matches token-level brute force**
   on Recall@10 (94.3% vs 94.3%) at roughly **1/6** the compute
   (256 ms vs 1631 ms). The headline "matches token-BF" claim sits
   entirely on this line; onset segmentation does not need to fire.

2. **Onset segmentation buys additional speed** -- Combined+Onset is
   ~32% faster than Combined (174 ms vs 256 ms), pushing the speedup
   vs Token-BF to ~1/9. The cost is **-1.4 pp recall** vs both
   Combined and Token-BF. So onset is a *tunable knob* (more speed for
   a small recall hit), not a free win.

Original write-ups that say "Combined+Onset matches token-level
brute-force recall at ~1/9 compute" are conflating the two findings.
The honest framing is: **Combined matches; Onset adds speed at a small
recall cost.**

### Caveats baked into these numbers

- **Single run**, no multi-seed averaging.
- **Corpus truncation interacts with query filtering**: the eval script
  truncates the corpus to 1000 docs, then keeps only the queries whose
  relevant docs survive that truncation. Easy queries (relevant doc in
  the first 1000) are over-represented vs. a uniform-random query
  sample. The 70 evaluated queries are the post-filter set.
- The `config` block in the JSON records the exact backend, models, and
  dataset sizes used.

### Reproducing

On a CUDA-capable pod (A100 used here; ~60 s model load + ~45 s index
build for the 1000-doc SciFact slice; full eval <2 min):

```bash
pip install -r requirements.txt
S3_BACKEND=local python experiments/research/onset_combined_test.py
```

The script writes a fresh `onset_combined/onset_combined_<dataset>_<UTC>.json`
on every run, so this archived file is one specific run from 2026-05-25
UTC.
