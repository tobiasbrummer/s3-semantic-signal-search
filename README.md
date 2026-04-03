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

## Key Results (SciFact, 1000 docs, 70 queries)

| Method | Recall@10 | Time | vs Baseline |
|--------|-----------|------|-------------|
| Pooled (baseline) | 84.3% | 10ms | -- |
| Combined+Onset | **94.3%** | **373ms** | **+10%** |
| Token Brute-Force | 94.3% | 3475ms | +10% |

Combined+Onset achieves token-level recall at 10x the speed of brute force.

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
```

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
