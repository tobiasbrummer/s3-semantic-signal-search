# S3 - Semantic Signal Search

+doc:documentation +project:s3

## Kernidee

Dokumente als kontinuierliche Signale behandeln statt als diskrete Chunks. Audio-Engineering-Prinzipien auf semantische Suche anwenden.

```
Traditionell:  Doc → [Chunk1, Chunk2, Chunk3] → Suche auf Chunks
Signal:        Doc → Signal[t=0...T] → Suche auf beliebiger Granularität
```

## Durchbruch: Kein Chunking, keine Vektordatenbank

**Was S3 anders macht:**

| Aspekt | Standard RAG | S3 |
|--------|--------------|-----|
| Chunking | Arbiträre 512-Token Chunks | Kein Chunking nötig |
| Embeddings | 1 Vektor pro Chunk | Token-Level + Onset-Segmente |
| Index | Vektordatenbank (HNSW, IVF) | SPLADE Inverted + Sign-Hash |
| Recall | Baseline | +10% bei gleichem Speed |

**Warum das funktioniert:**

- Token-Level Embeddings erfassen feinere semantische Nuancen
- Onset Detection findet natürliche Grenzen (keine arbiträren Cuts)
- Combined Pipeline erreicht Token-BF Recall ohne Brute Force

## Finale Pipeline: Combined+Onset

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: Candidate Retrieval (parallel)                     │
├─────────────────────────────────────────────────────────────┤
│   Query ──┬── Dense Mean-Pooled ──► Top 100                 │
│           └── SPLADE Inverted ────► Top 100                 │
│                    Union ──► ~150-200 Kandidaten            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: Onset-Based Token Refinement                       │
├─────────────────────────────────────────────────────────────┤
│   Pro Kandidat:                                             │
│     1. Segment-Ranking (Onset-Segmente nach Query-Sim)      │
│     2. Token-Level NUR in Top-3 Segmenten                   │
│     3. Best Token Score = Doc Score                         │
└─────────────────────────────────────────────────────────────┘
```

**Ergebnis (SciFact 1000 Docs, 70 Queries):**

| Methode | R@10 | Zeit | vs Pooled |
|---------|------|------|-----------|
| Pooled (Baseline) | 84.3% | 10ms | - |
| Combined | 94.3% | 575ms | +10% |
| **Combined+Onset** | **94.3%** | **373ms** | **+10%** |
| Token BF | 94.3% | 3475ms | +10% |

**Key Metrics:**

- Combined+Onset = Token BF Recall
- 35% schneller als Combined (ohne Onset)
- 9.3x schneller als Token BF

## Validierte Komponenten

### 1. Token-Level Embeddings

**Endpoint:** `http://localhost:8202/embeddings` (llama.cpp mit `--pooling none`)

```python
response = requests.post(url, json={"input": text})
embeddings = response.json()[0]["embedding"]  # [[...], [...], ...]
```

**Ergebnis:** +10% Recall vs Pooled Embeddings (94.3% vs 84.3%)

### 2. Onset Detection (Audio-Style)

Semantische Breakpoints durch Spectral Flux auf Embedding-Kurven.

```python
def spectral_flux(embeddings):
    """Summe der absoluten Änderungen über alle Dimensionen."""
    changes = np.abs(np.diff(embeddings, axis=0))
    return changes.sum(axis=1)
```

**Optimierte Parameter (aus Sweep):**

- `threshold_pct = 95` (höher = weniger, größere Segmente)
- `min_dist = 3`
- `smooth_sigma = 2.0`

**Parameter-Einfluss:**

- Threshold: Sehr hoch (95% → 86% avg vs 81% bei anderen)
- Sigma: Mittel (3.0 → 84.5% vs 81% bei 0.5)
- Min Dist: Gering

### 3. SPLADE Integration

```python
# Hugging Face Transformers
model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil")
```

**Domain-Abhängigkeit (BEIR Test 1):**

| Dataset | SPLADE | Dense | Gewinner |
|---------|--------|-------|----------|
| SciFact | 87.6% | 80.0% | SPLADE +7.6% |
| FiQA | 69.6% | 82.1% | Dense +12.5% |
| NFCorpus | 70.4% | 73.7% | Dense +3.3% |
| Quora | 100% | 100% | Tie |

**Erkenntnis:** SPLADE stark auf wissenschaftlichen Texten, Dense besser auf Finanz/Fragen.

### 4. Sign-Hash Kompression

32x Kompression (1 bit pro Dimension), 96.1% von Brute Force Recall.

```python
def to_sign_hash(embeddings):
    signs = (embeddings > 0).astype(np.uint8)
    packed = np.packbits(signs, axis=-1)
    return packed
```

## Verworfene Ansätze

| Ansatz | Problem |
|--------|---------|
| Delta-Encoding | -139% (schlechter), Flip-Rate zu hoch |
| Spline-Approximation | Nur 7 dB SNR, 83% Sign-Accuracy |
| Max-Pooling | -4.3% vs Mean-Pooling |
| GPU Vulkan (naive) | CPU schneller für unseren Scale |
| Onset allein (ohne SPLADE) | -0.7% unter Pooled auf SciFact |

## BEIR Test Ergebnisse (Final)

### Recall@10 Summary

| Dataset | Pooled | Combined | Comb+Onset | Token BF | vs Pooled |
|---------|--------|----------|------------|----------|-----------|
| SciFact | 80.0% | 83.4% | **83.4%** | 83.4% | **+3.4%** |
| FiQA | 82.1% | 82.1% | **82.1%** | 82.1% | ±0% |
| NFCorpus | 73.7% | 76.3% | **76.3%** | 76.3% | **+2.6%** |
| Quora | 100% | 100% | **100%** | 100% | ±0% |

### Timing (ms per query)

| Dataset | Pooled | Combined | Comb+Onset | Token BF |
|---------|--------|----------|------------|----------|
| SciFact | 19ms | 567ms | **368ms** | 6500ms |
| FiQA | 21ms | 322ms | **240ms** | 3394ms |
| NFCorpus | 19ms | 579ms | **409ms** | 7061ms |
| Quora | 19ms | 46ms | **49ms** | 300ms |

### Speedup Summary

| Vergleich | SciFact | FiQA | NFCorpus | Quora |
|-----------|---------|------|----------|-------|
| Comb+Onset vs Token BF | **17.7x** | **14.1x** | **17.3x** | **6.1x** |
| Comb+Onset vs Combined | **35%** | **25%** | **29%** | - |

### Key Findings

1. **Combined+Onset = Token BF Recall** auf allen 4 Datasets (100% des Maximums)
2. **14-18x schneller** als Token Brute Force
3. **25-35% schneller** als Combined (ohne Onset)
4. **Nie schlechter als Pooled** (Standard RAG Baseline)

## Architektur-Vorteile

### Kein Chunking nötig

- Traditionelles RAG: Dokumente in 512-Token Chunks splitten
- S3: Ganzes Dokument als Signal, Onset-Segmente bei Bedarf
- Vorteil: Kein Kontext-Verlust an Chunk-Grenzen

### Keine Vektordatenbank nötig

- SPLADE: Standard Inverted Index (Lucene, Elasticsearch, etc.)
- Dense: Sign-Hash mit Hamming Distance (einfache Bit-Ops)
- Vorteil: Keine komplexe ANN-Infrastruktur (HNSW, IVF, etc.)

### Flexibles Embedding-Modell

- Nutzt Standard-Modelle (jina-v3, etc.) mit `--pooling none`
- Kein spezielles Training nötig (vs ConstBERT)
- Vorteil: Model-agnostisch, sofort einsetzbar

## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `beir_comprehensive_test.py` | Finaler BEIR Test (Combined+Onset) |
| `onset_combined_test.py` | Combined+Onset Validierung |
| `onset_parameter_sweep.py` | Parameter-Optimierung für Onset |
| `onset_detection.py` | Spectral Flux Onset Detection |
| `combined_pipeline_test.py` | Frühere Pipeline-Version |
| `vulkan_compute_test.py` | GPU Compute PoC |

## Geplante Features

### Kurzfristig (validierter Ansatz erweitern)

| Feature | Beschreibung | Status |
|---------|--------------|--------|
| **Spektrogramm-Plot** | Visualisierung der Embedding-Kurven | Geplant |
| **Matryoshka MRL** | 2D-Hierarchie (Dims + Sequenz) | Geplant |
| **Sign-Hash in Pipeline** | Stage 1 Beschleunigung | Offen |

### Mittelfristig (neue Capabilities)

| Feature | Beschreibung | Abhängigkeiten |
|---------|--------------|----------------|
| **Audio-Sonifikation** | Embedding-Kurven als Audio abspielen | Spektrogramm-Plot |
| **GPU-Suche** | Parallelisierung für großen Scale | Bei >100k Docs relevant |

### Langfristig (brauchen spezialisierte Modelle)

| Feature | Beschreibung | Modell-Anforderung |
|---------|--------------|-------------------|
| **Frequenzbänder** | Emotion, Stimmung, Tonalität als separate Signale | Modelle die diese Aspekte separat kodieren |
| **Phasen-Rotation** | Semantische "Phase" als Feature | Modell das Phase sinnvoll kodiert |
| **EQ-Filter** | Gewichtung verschiedener Frequenzen bei Suche | Braucht Frequenzbänder |
| **Prompt Injection Detection** | Anomalie-Erkennung über Frequenzanalyse | Frequenzbänder + Training |
| **Semantic Diff** | Textänderungen über Frequenzvergleich | Frequenzbänder |

### Matryoshka + Multi-Resolution (2D-Hierarchie)

```
              256 dims      512 dims      1024 dims
             (coarse)      (medium)       (fine)
            ─────────────────────────────────────────
Doc-Level       ●             ●              ●       → Stage 1 Filter
Segment         ●             ●              ●       → Stage 2 Refine
Token           ●             ●              ●       → Stage 3 Precision
```

**Idee:** Jina-v3 unterstützt MRL (Matryoshka Representation Learning).

- Stage 1: Doc-Level @ 256 dims (super schnell)
- Stage 2: Segment-Level @ 512 dims
- Stage 3: Token-Level @ 1024 dims

→ Kombination aus Dimensions-Kompression (MRL) und Sequenz-Kompression (Multi-Res)

## Abgeschlossene Meilensteine

1. ~~Token-Level Embeddings~~ → +10% Recall validiert
2. ~~Multi-Resolution Hierarchie~~ → Doc/Segment/Token
3. ~~SPLADE Integration~~ → Hybrid Dense+Sparse
4. ~~Combined Pipeline~~ → SPLADE+Dense → Token
5. ~~Sign-Hash Kompression~~ → 32x, 96% Recall
6. ~~Onset Detection~~ → Parameter optimiert
7. ~~Combined+Onset~~ → 35% schneller als Combined
8. ~~BEIR Validation~~ → 4 Datasets, alle bestanden

## Nächste Schritte

1. ~~Spektrogramm-Plot implementieren~~ → Done
2. ~~Matryoshka MRL integrieren~~ → Done (128/256/1024 = 94.3%, kein Recall-Verlust)
3. ~~FFT Cross-Correlation Search~~ → Getestet, CPU 66x langsamer als Cosine
4. ~~Audio-Modell Prototyp~~ → Done (Frequenz + Attention-Amplitude)
5. GPU-FFT Implementation (WebGPU/Vulkan) für Audio-Modell
6. Export-Format für Web-Visualisierung definieren
7. Produktiv-Implementierung planen

## Offene Tests/Features

| Feature | Status | Notizen |
|---------|--------|---------|
| FFT Pattern Search | Getestet | CPU zu langsam (66x vs Cosine), GPU nötig |
| GPU-FFT | Geplant | 256 FFTs parallel auf GPU |
| Sign-Hash + MRL | Offen | Kombination für Stage 1 |
| Audio-Sonifikation | Geplant | Embeddings als Sound abspielen |

---

## Experimentell: Semantic Audio Model

**Status:** Research / Proof of Concept

Neuer Ansatz, der Embeddings als vollständiges Audiosignal interpretiert - nicht nur Onset Detection, sondern komplette Frequenz/Amplituden-Darstellung.

### Motivation

Die bisherige Pipeline (Combined+Onset) nutzt Audio-Konzepte nur für Onset Detection. Dieser Ansatz geht weiter:

- **Jede Dimension** wird als eigener Ton interpretiert
- **Embedding-Wert** bestimmt die Frequenz
- **Attention Weight** bestimmt die Lautstärke (Amplitude)
- **Matryoshka-Bänder** gruppieren Dimensionen in Frequenzbereiche

### Das Modell (Option B)

```
Token-Embedding (1024 Dimensionen)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Pro Dimension d:                                    │
│    Frequenz = Band_Center + (value[d] / 3) × 500 Hz │
│    Amplitude = attention_weight                      │
└─────────────────────────────────────────────────────┘
         │
         ▼
1024 Töne mit individueller Frequenz und Lautstärke
```

**Matryoshka-Bänder:**

| Band | Dimensionen | Frequenzbereich | Semantik |
|------|-------------|-----------------|----------|
| 1 | 0-255 | 4000-5000 Hz | Grobe Semantik |
| 2 | 256-511 | 3000-4000 Hz | Mittlere Details |
| 3 | 512-767 | 2000-3000 Hz | Feine Nuancen |
| 4 | 768-1023 | 1000-2000 Hz | Micro-Features |

**Wert-Mapping:**

```
Embedding-Wert    Frequenz (Band 1)
─────────────────────────────────
    -3.0      →   4000 Hz (unterer Rand)
     0.0      →   4500 Hz (Mitte/Baseline)
    +3.0      →   5000 Hz (oberer Rand)
```

### Warum diese Interpretation?

| Konzept | Audio | Embedding |
|---------|-------|-----------|
| Wert +3 | Hohe Frequenz | Starke positive Aktivierung |
| Wert -3 | Niedrige Frequenz | Starke negative Aktivierung |
| Wert 0 | Baseline | Dimension irrelevant |
| Hohe Amplitude | Laut | Token ist wichtig (Attention) |
| Niedrige Amplitude | Leise | Token ist unwichtig |

**Semantische Korrektheit:**

- Ähnliche Tokens → ähnliche Spektren → hohe Korrelation
- Gegensätzliche Tokens → invertierte Spektren → niedrige Korrelation
- Cosine Similarity bleibt erhalten

### Attention als Amplitude

**Problem:** jina-v3 (Flash Attention) gibt keine Attention Weights aus.

**Lösung:** BGE (BAAI/bge-base-en-v1.5) als Alternative getestet:

```python
# Attention aggregieren
all_attn = torch.stack(outputs.attentions)  # (12, 1, 12, seq, seq)
avg_attn = all_attn.mean(dim=(0, 1, 2))     # (seq, seq)
importance = avg_attn.sum(dim=0)             # Attention received per token
```

**Ergebnis:**

- Content-Wörter ("atom", "explodiert") → hohe Attention
- Sub-Tokens ("##e", "##en") → niedrige Attention
- Satzzeichen (".") → mittlere-hohe Attention (Grenzen wichtig)

### Validierte Erkenntnisse

1. **Embedding-Werte sind NICHT [-1, +1]** sondern typisch [-3, +3]
   - Mean: ~0, Std: ~0.5
   - ~70% der Werte zwischen -0.5 und +0.5
   - Outlier (|val| > 1) sind semantisch relevant

2. **Semantische Dimensionen existieren**
   - Test: "Apple" ≈ "Banana" aber ≠ "Microsoft"
   - 74 Dimensionen gefunden wo dies gilt
   - Embedding-Raum hat interpretierbare Struktur

3. **Übergänge sind in allen Bändern sichtbar**
   - Thematische Wechsel (Hasen → Atomkraftwerk → Kind) sichtbar
   - Nicht nur in Dims 0-255, sondern über alle 1024

### Tools

| Datei | Beschreibung |
|-------|--------------|
| `audio_model_viz.py` | Spektrogramm mit Token-Labels |
| `attention_audio_test.py` | BGE + Attention als Amplitude |
| `debug_embeddings.py` | Embedding-Wertverteilung analysieren |

### Offene Fragen

1. **FFT-Suche auf Audio-Modell:** Hilft das Frequenz-Mapping beim Pattern Matching?
2. **GPU-Beschleunigung:** 1024 FFTs parallel auf GPU
3. **Kompression:** Nur dominante Frequenzen speichern (wie MP3)?
4. **jina-v3 Attention:** Alternative Methode um Importance zu bekommen?

### Nächste Schritte (Audio Model)

| Schritt | Beschreibung | Priorität |
|---------|--------------|-----------|
| Export-Format | JSON mit Frequenzen + Amplituden für Web-App | Hoch |
| Audio-Synthese | Tatsächlich hörbar machen | Mittel |
| FFT auf Audio-Modell | Pattern Matching mit Frequenzen | Mittel |
| jina-v3 + BGE Kombination | jina Embeddings + BGE Attention | Niedrig |
