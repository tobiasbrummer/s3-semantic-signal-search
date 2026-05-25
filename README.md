# S3 - Semantic Signal Search

Research project exploring signal-processing approaches to semantic search. Treats documents as continuous signals rather than discrete chunks, applying audio-engineering principles (onset detection, spectral analysis, interference patterns) to text retrieval.

## Core Idea

```
Traditional:  Doc -> [Chunk1, Chunk2, Chunk3] -> Search on chunks
Signal:       Doc -> Signal[t=0...T] -> Search at any granularity
```

Instead of arbitrary 512-token chunking, S3 uses:
- **Token-level embeddings** for fine-grained semantic resolution
- **Onset detection** to find natural segment boundaries (no arbitrary cuts)
- **Combined retrieval** (dense + SPLADE sparse) with onset-based refinement

## Key Results (BEIR/SciFact, 1000 docs, 70 queries after filtering)

Saved artifact: [`results/onset_combined/onset_combined_scifact_20260525_190730Z.json`](results/onset_combined/onset_combined_scifact_20260525_190730Z.json)
(jina-embeddings-v3 + SPLADE-cocondenser, A100, 2026-05-25).

| Method | Recall@10 | Time | vs Pooled | vs Token-BF |
|--------|-----------|------|-----------|-------------|
| Pooled (baseline) | 85.7% | 4 ms | -- | -8.6 pp |
| Combined (dense + SPLADE RRF) | **94.3%** | 256 ms | +8.6 pp | **0.0 pp** |
| Combined + Onset | 92.9% | **174 ms** | +7.1 pp | -1.4 pp |
| Token Brute-Force | 94.3% | 1631 ms | +8.6 pp | -- |

Two distinct findings sit in this table; the previous wording
("Combined+Onset matches token-BF at ~1/9 compute") conflated them.
The honest reading is:

1. **Combined (dense + SPLADE via RRF) matches token-level brute-force
   recall** -- 94.3% vs 94.3% -- at roughly **1/6** the compute
   (256 ms vs 1631 ms). The "matches" claim lives entirely on this line.

2. **Onset segmentation adds speed** -- Combined+Onset is ~32% faster
   than Combined (174 ms vs 256 ms), pushing the speedup vs Token-BF
   to **~1/9**. The cost is -1.4 pp recall vs both Combined and
   Token-BF. So Onset is a tunable knob (more speed for a small recall
   hit), not a free win.

### Running the benchmark

There are two backends, selected via the `S3_BACKEND` env var:

```bash
# Default: HTTP backend. Needs two text-embeddings-inference services up:
#   :8200  pooled jina-embeddings-v3
#   :8202  SPLADE token-level jina-embeddings-v3
# docker-compose.yaml here is the dev container for the engine, NOT
# the eval services. Bring those up separately.
python experiments/research/onset_combined_test.py

# Self-contained: jina-v3 loaded directly via transformers, no HTTP needed.
# What produced the saved artifact above.
S3_BACKEND=local python experiments/research/onset_combined_test.py
```

### Caveats worth knowing about

- **Single run** on the BEIR SciFact slice, no multi-seed averaging.
- **Corpus truncation interacts with query filtering.** The eval script
  truncates the corpus to 1000 docs and then keeps only the queries
  whose relevant docs survive that truncation. Easy queries (relevant
  doc in the first 1000) are over-represented vs. a uniform-random
  query sample. The 70 evaluated queries are the post-filter set.
- The two backends will not produce byte-identical scores -- different
  inference paths into jina-v3, different attention implementations
  (PyTorch native vs FlashAttention vs TEI's batched kernels) -- but
  they should agree to within ~1 pp on the head-line numbers.

## Project Structure

```
s3-semantic-signal-search/
├── src/
│   ├── engine/                # Core semantic engine
│   │   ├── engine.py          # Main engine (onset detection, token search)
│   │   ├── engine_v2.py       # V2 with word-level pooling
│   │   ├── analyzer.py        # Document analysis pipeline
│   │   ├── critic.py          # LLM-based evaluation
│   │   └── cli.py             # Command line interface
│   └── api/                   # Web interface (FastAPI)
├── experiments/
│   ├── research/              # ~65 research scripts (retrieval, compression,
│   │                          #   spectral analysis, audio analogies)
│   └── dsp-lab/               # DSP experiments (frankenstein engines,
│                              #   slerp stitching, normalization tests)
├── docs/
│   ├── S3_ARCHITECTURE.md     # Main architecture document
│   ├── signal_pipeline_concept.org
│   ├── wave.org               # Wave-based memory concept
│   └── specs.org              # Engine specifications
├── tests/                     # Engine tests + benchmarks
├── docker-compose.yaml
├── Dockerfile
└── requirements.txt
```

## Research Areas

This project explores several novel concepts:

1. **Semantic onset detection**: Finding natural topic boundaries in text using embedding derivative analysis (analogous to audio onset detection)
2. **Wave-based memory**: Complex-valued embeddings where phase encodes negation/context, enabling interference-based retrieval
3. **DSP-inspired stitching**: Using SLERP interpolation for smooth embedding transitions across document segments
4. **Hybrid token-level retrieval**: SPLADE sparse + dense mean-pooled candidate retrieval, followed by token-level refinement in onset segments

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # sentence segmentation
```

For docker-based dev: `MODELS_DIR=$HOME/.cache/huggingface docker compose up`
(set `MODELS_DIR` so HF cache persists across rebuilds).

## Usage

```bash
# CLI
python -m src.engine compare old.pdf new.pdf

# API
python src/api/app_api.py
```

## Related Research

- Jina Embeddings v3 (task-specific vector spaces)
- SPLADE sparse retrieval
- Audio onset detection (librosa)
- Complex-valued neural networks for NLP
