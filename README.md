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

## Key Results (BEIR/SciFact, 1000 docs, ~70 queries)

| Method | Recall@10 | Time | vs Baseline |
|--------|-----------|------|-------------|
| Pooled (baseline) | 84.3% | 10ms | -- |
| Combined+Onset | **94.3%** | **373ms** | **+10pp** |
| Token Brute-Force | 94.3% | 3475ms | +10pp |

**Combined+Onset matches token-level brute-force recall at ~1/9 the
brute-force compute.** Run: `experiments/research/onset_combined_test.py`.
Requires two HTTP embedding services running locally:

- `localhost:8200` -- pooled (mean) sentence embeddings (e.g. text-embeddings-inference with `jinaai/jina-embeddings-v3`)
- `localhost:8202` -- SPLADE token-level sparse embeddings (e.g. text-embeddings-inference with `naver/splade-cocondenser-ensembledistil`)

The `docker-compose.yaml` here is the dev container for the engine, NOT
the eval services -- those need to be brought up separately.

### Caveats worth knowing about

- **Single run** on the BEIR SciFact slice, no multi-seed averaging.
- **Corpus truncation interacts with query filtering.** The eval script
  truncates the corpus to 1000 docs and then filters queries to those
  whose relevant docs survive the truncation. Easy queries (relevant
  doc is in the first 1000) are over-represented vs. a uniform-random
  query sample. The numbers above are honest for the *filtered* query
  set; they should not be read as "94.3% recall on SciFact in general".
- **No saved JSON artifact** for this run yet -- the result is in shell
  output from the original execution. Re-run from a cold pod with the
  two TEI services up to reproduce.

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
