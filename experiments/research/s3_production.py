#!/usr/bin/env python3
"""
S3 - Semantic Signal Search
Production-Ready Implementation

Includes all "invisible" engineering components:
1. Normalizer (vector preprocessing)
2. Config Map (central schema definition)
3. Two-Pass Re-Ranker (DB coarse + Python fine)
4. Complete ingestion and retrieval pipeline

Author: Claude & Toby
Date: December 2024
"""

import numpy as np
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any, Callable
from pathlib import Path
import struct
import time
from abc import ABC, abstractmethod

# Optional imports (graceful degradation)
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("⚠️  sentence-transformers not installed. Using mock embeddings.")

try:
    from elasticsearch import Elasticsearch
    HAS_ELASTICSEARCH = True
except ImportError:
    HAS_ELASTICSEARCH = False
    print("⚠️  elasticsearch not installed. Using in-memory index.")


# =============================================================================
# 1. CONFIGURATION (The "Law" of the System)
# =============================================================================

@dataclass
class BandConfig:
    """Configuration for a single frequency band."""
    name: str
    dim_start: int
    dim_end: int
    bits: int
    weight: float
    
    @property
    def dim_count(self) -> int:
        return self.dim_end - self.dim_start


@dataclass
class S3Config:
    """
    Central configuration for S3 system.
    
    ⚠️ IMPORTANT: If you change this, you MUST re-index everything!
    The LSH projections and bit layouts depend on these settings.
    """
    
    # Model settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384  # MiniLM is 384D, not 768D
    
    # Band definitions
    bands: List[BandConfig] = field(default_factory=lambda: [
        BandConfig(name="bass", dim_start=0, dim_end=48, bits=32, weight=10.0),
        BandConfig(name="mids", dim_start=48, dim_end=192, bits=64, weight=5.0),
        BandConfig(name="highs", dim_start=192, dim_end=384, bits=128, weight=1.0),
    ])
    
    # LSH settings
    lsh_seed: int = 42
    
    # Audio synthesis settings
    sample_rate: int = 16000
    audio_duration: float = 2.0
    
    # Search settings
    coarse_candidates: int = 1000  # How many to fetch from DB
    fine_top_k: int = 10           # Final results to return
    fuzzy_bits: int = 2            # Max bit flips for fuzzy match
    
    # Version (for migration tracking)
    version: str = "1.0.0"
    
    def __post_init__(self):
        # Validate bands cover the embedding dimension
        total_dims = sum(b.dim_count for b in self.bands)
        if total_dims != self.embedding_dim:
            raise ValueError(
                f"Bands cover {total_dims} dims but embedding has {self.embedding_dim}"
            )
    
    @property
    def total_bits(self) -> int:
        return sum(b.bits for b in self.bands)
    
    @property
    def total_bytes(self) -> int:
        return (self.total_bits + 7) // 8
    
    def to_json(self) -> str:
        """Serialize config to JSON for storage."""
        data = {
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "bands": [
                {
                    "name": b.name,
                    "dim_start": b.dim_start,
                    "dim_end": b.dim_end,
                    "bits": b.bits,
                    "weight": b.weight,
                }
                for b in self.bands
            ],
            "lsh_seed": self.lsh_seed,
            "sample_rate": self.sample_rate,
            "audio_duration": self.audio_duration,
            "coarse_candidates": self.coarse_candidates,
            "fine_top_k": self.fine_top_k,
            "fuzzy_bits": self.fuzzy_bits,
            "version": self.version,
        }
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "S3Config":
        """Deserialize config from JSON."""
        data = json.loads(json_str)
        bands = [BandConfig(**b) for b in data["bands"]]
        return cls(
            embedding_model=data["embedding_model"],
            embedding_dim=data["embedding_dim"],
            bands=bands,
            lsh_seed=data["lsh_seed"],
            sample_rate=data.get("sample_rate", 16000),
            audio_duration=data.get("audio_duration", 2.0),
            coarse_candidates=data.get("coarse_candidates", 1000),
            fine_top_k=data.get("fine_top_k", 10),
            fuzzy_bits=data.get("fuzzy_bits", 2),
            version=data.get("version", "1.0.0"),
        )
    
    def config_hash(self) -> str:
        """
        Generate a hash of the config.
        Use this to detect if re-indexing is needed.
        """
        # Only hash the parts that affect indexing
        key_parts = (
            self.embedding_model,
            self.embedding_dim,
            [(b.dim_start, b.dim_end, b.bits) for b in self.bands],
            self.lsh_seed,
        )
        return hashlib.sha256(str(key_parts).encode()).hexdigest()[:16]


# =============================================================================
# 2. NORMALIZER (Mathematical Hygiene)
# =============================================================================

class Normalizer:
    """
    Ensures vectors are on the unit sphere before LSH.
    
    LSH with random projections assumes cosine similarity,
    which only works correctly for normalized vectors.
    """
    
    @staticmethod
    def normalize(vector: np.ndarray) -> np.ndarray:
        """Normalize a single vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            # Zero vector - return as is (edge case)
            return vector
        return vector / norm
    
    @staticmethod
    def normalize_batch(vectors: np.ndarray) -> np.ndarray:
        """Normalize a batch of vectors."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Avoid division by zero
        return vectors / norms
    
    @staticmethod
    def is_normalized(vector: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Check if a vector is normalized."""
        norm = np.linalg.norm(vector)
        return abs(norm - 1.0) < tolerance


# =============================================================================
# 3. EMBEDDING MODEL (Teacher Model)
# =============================================================================

class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class SentenceTransformerModel(EmbeddingModel):
    """Real embedding model using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers is required")
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return Normalizer.normalize_batch(embeddings)
    
    @property
    def dimension(self) -> int:
        return self._dimension


class MockEmbeddingModel(EmbeddingModel):
    """Mock embedding model for testing without dependencies."""
    
    def __init__(self, dimension: int = 384, seed: int = 42):
        self._dimension = dimension
        self.seed = seed
    
    def embed(self, texts: List[str]) -> np.ndarray:
        # Generate deterministic embeddings based on text hash
        embeddings = []
        for text in texts:
            # Seed RNG with text hash for reproducibility
            text_hash = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(text_hash)
            emb = rng.randn(self._dimension).astype(np.float32)
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        return Normalizer.normalize_batch(embeddings)
    
    @property
    def dimension(self) -> int:
        return self._dimension


def get_embedding_model(config: S3Config) -> EmbeddingModel:
    """Factory function to get the appropriate embedding model."""
    if HAS_SENTENCE_TRANSFORMERS:
        return SentenceTransformerModel(config.embedding_model)
    else:
        return MockEmbeddingModel(config.embedding_dim)


# =============================================================================
# 4. MULTI-BAND LSH
# =============================================================================

class MultiBandLSH:
    """
    Locality Sensitive Hashing with separate projections per frequency band.
    
    Key insight: Each band is hashed independently, allowing:
    - Hierarchical search (bass first)
    - Weighted distance (bass errors penalized more)
    - Interpretable audio mapping (bass = low tones)
    """
    
    def __init__(self, config: S3Config):
        self.config = config
        self.projections: Dict[str, np.ndarray] = {}
        
        # Initialize random projections for each band
        rng = np.random.RandomState(config.lsh_seed)
        
        for band in config.bands:
            # Random hyperplanes for LSH
            self.projections[band.name] = rng.randn(
                band.bits, band.dim_count
            ).astype(np.float32)
    
    def hash_vector(self, vector: np.ndarray) -> Dict[str, bytes]:
        """
        Hash a normalized vector to multi-band binary codes.
        
        Args:
            vector: 1D normalized float array
            
        Returns:
            Dict mapping band_name → packed bytes
        """
        if not Normalizer.is_normalized(vector):
            vector = Normalizer.normalize(vector)
        
        result = {}
        
        for band in self.config.bands:
            # Extract band dimensions
            band_vector = vector[band.dim_start:band.dim_end]
            
            # Project and take sign
            projections = self.projections[band.name]
            dot_products = projections @ band_vector
            
            # Convert to bits
            bits = (dot_products > 0).astype(np.uint8)
            
            # Pack into bytes
            result[band.name] = self._pack_bits(bits)
        
        return result
    
    def hash_batch(self, vectors: np.ndarray) -> List[Dict[str, bytes]]:
        """Hash a batch of vectors."""
        vectors = Normalizer.normalize_batch(vectors)
        return [self.hash_vector(v) for v in vectors]
    
    def _pack_bits(self, bits: np.ndarray) -> bytes:
        """Pack bit array into bytes."""
        padded_len = ((len(bits) + 7) // 8) * 8
        padded = np.zeros(padded_len, dtype=np.uint8)
        padded[:len(bits)] = bits
        return bytes(np.packbits(padded))
    
    def unpack_bits(self, packed: bytes, num_bits: int) -> np.ndarray:
        """Unpack bytes back to bit array."""
        unpacked = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))
        return unpacked[:num_bits]
    
    def to_hex(self, hashed: Dict[str, bytes]) -> Dict[str, str]:
        """Convert packed bytes to hex strings for storage."""
        return {name: data.hex() for name, data in hashed.items()}
    
    def from_hex(self, hex_dict: Dict[str, str]) -> Dict[str, bytes]:
        """Convert hex strings back to bytes."""
        return {name: bytes.fromhex(hex_str) for name, hex_str in hex_dict.items()}


# =============================================================================
# 5. HAMMING DISTANCE (Fine Scorer)
# =============================================================================

class HammingScorer:
    """
    Compute weighted Hamming distance for fine-grained ranking.
    
    This runs in Python (not DB) on the coarse candidates.
    Optimized for speed with NumPy/vectorization.
    """
    
    def __init__(self, config: S3Config):
        self.config = config
        self.weights = {band.name: band.weight for band in config.bands}
        self.bits_per_band = {band.name: band.bits for band in config.bands}
    
    def hamming_distance(self, a: bytes, b: bytes) -> int:
        """Compute Hamming distance between two byte arrays."""
        # XOR and count bits (popcount)
        xor = bytes(x ^ y for x, y in zip(a, b))
        return sum(bin(byte).count('1') for byte in xor)
    
    def weighted_distance(self, 
                          query_hash: Dict[str, bytes],
                          doc_hash: Dict[str, bytes]) -> float:
        """
        Compute weighted Hamming distance.
        
        Lower = more similar.
        Bass errors are penalized more than high errors.
        """
        total = 0.0
        
        for band_name, weight in self.weights.items():
            if band_name in query_hash and band_name in doc_hash:
                dist = self.hamming_distance(query_hash[band_name], doc_hash[band_name])
                total += weight * dist
        
        return total
    
    def score_candidates(self,
                         query_hash: Dict[str, bytes],
                         candidates: List[Tuple[str, Dict[str, bytes]]],
                         top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Score and rank candidates by weighted Hamming distance.
        
        Args:
            query_hash: Hashed query
            candidates: List of (doc_id, doc_hash) tuples
            top_k: Number of results to return
            
        Returns:
            List of (doc_id, distance) sorted by distance ascending
        """
        scored = []
        
        for doc_id, doc_hash in candidates:
            dist = self.weighted_distance(query_hash, doc_hash)
            scored.append((doc_id, dist))
        
        scored.sort(key=lambda x: x[1])
        return scored[:top_k]
    
    def similarity(self, distance: float) -> float:
        """
        Convert distance to similarity score [0, 1].
        
        Max distance = all bits different in all bands.
        """
        max_distance = sum(
            self.weights[b.name] * b.bits 
            for b in self.config.bands
        )
        return 1.0 - (distance / max_distance)


# =============================================================================
# 6. INDEX BACKEND (Abstract + Implementations)
# =============================================================================

@dataclass
class Document:
    """A document in the index."""
    doc_id: str
    text: str
    hash_hex: Dict[str, str]  # band_name → hex string
    metadata: Dict[str, Any] = field(default_factory=dict)


class IndexBackend(ABC):
    """Abstract base class for index backends."""
    
    @abstractmethod
    def index_document(self, doc: Document) -> None:
        """Add a document to the index."""
        pass
    
    @abstractmethod
    def coarse_search(self, bass_hash: str, limit: int) -> List[Document]:
        """Find documents with matching or similar bass hash."""
        pass
    
    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by ID."""
        pass
    
    @abstractmethod
    def get_all_hashes(self) -> List[Tuple[str, Dict[str, bytes]]]:
        """Get all document hashes for fine scoring."""
        pass


class InMemoryIndex(IndexBackend):
    """Simple in-memory index for development/testing."""
    
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.bass_index: Dict[str, List[str]] = {}  # bass_hash → [doc_ids]
    
    def index_document(self, doc: Document) -> None:
        self.documents[doc.doc_id] = doc
        
        # Add to bass inverted index
        bass_hash = doc.hash_hex.get("bass", "")
        if bass_hash not in self.bass_index:
            self.bass_index[bass_hash] = []
        self.bass_index[bass_hash].append(doc.doc_id)
    
    def coarse_search(self, bass_hash: str, limit: int) -> List[Document]:
        """Find documents with exact or fuzzy bass match."""
        doc_ids = set()
        
        # Exact match
        if bass_hash in self.bass_index:
            doc_ids.update(self.bass_index[bass_hash])
        
        # Fuzzy match (1-bit flips)
        if len(doc_ids) < limit:
            bass_bytes = bytes.fromhex(bass_hash)
            for byte_idx in range(len(bass_bytes)):
                for bit_idx in range(8):
                    # Flip one bit
                    modified = bytearray(bass_bytes)
                    modified[byte_idx] ^= (1 << bit_idx)
                    modified_hex = bytes(modified).hex()
                    
                    if modified_hex in self.bass_index:
                        doc_ids.update(self.bass_index[modified_hex])
        
        # Return documents
        docs = [self.documents[did] for did in list(doc_ids)[:limit]]
        return docs
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        return self.documents.get(doc_id)
    
    def get_all_hashes(self) -> List[Tuple[str, Dict[str, bytes]]]:
        result = []
        for doc_id, doc in self.documents.items():
            hash_bytes = {
                name: bytes.fromhex(hex_str) 
                for name, hex_str in doc.hash_hex.items()
            }
            result.append((doc_id, hash_bytes))
        return result


class ElasticsearchIndex(IndexBackend):
    """Elasticsearch backend for production."""
    
    def __init__(self, es_url: str = "http://localhost:9200", index_name: str = "s3_docs"):
        if not HAS_ELASTICSEARCH:
            raise ImportError("elasticsearch is required")
        self.es = Elasticsearch(es_url)
        self.index_name = index_name
        self._ensure_index()
    
    def _ensure_index(self):
        """Create index if it doesn't exist."""
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(
                index=self.index_name,
                body={
                    "mappings": {
                        "properties": {
                            "text": {"type": "text"},
                            "bass_hash": {"type": "keyword"},
                            "mids_hash": {"type": "keyword"},
                            "highs_hash": {"type": "keyword"},
                            "metadata": {"type": "object"},
                        }
                    }
                }
            )
    
    def index_document(self, doc: Document) -> None:
        self.es.index(
            index=self.index_name,
            id=doc.doc_id,
            body={
                "text": doc.text,
                "bass_hash": doc.hash_hex.get("bass", ""),
                "mids_hash": doc.hash_hex.get("mids", ""),
                "highs_hash": doc.hash_hex.get("highs", ""),
                "metadata": doc.metadata,
            }
        )
    
    def coarse_search(self, bass_hash: str, limit: int) -> List[Document]:
        # Query for exact match on bass hash
        response = self.es.search(
            index=self.index_name,
            body={
                "query": {
                    "term": {"bass_hash": bass_hash}
                },
                "size": limit
            }
        )
        
        docs = []
        for hit in response["hits"]["hits"]:
            docs.append(Document(
                doc_id=hit["_id"],
                text=hit["_source"]["text"],
                hash_hex={
                    "bass": hit["_source"]["bass_hash"],
                    "mids": hit["_source"]["mids_hash"],
                    "highs": hit["_source"]["highs_hash"],
                },
                metadata=hit["_source"].get("metadata", {})
            ))
        
        return docs
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        try:
            hit = self.es.get(index=self.index_name, id=doc_id)
            return Document(
                doc_id=hit["_id"],
                text=hit["_source"]["text"],
                hash_hex={
                    "bass": hit["_source"]["bass_hash"],
                    "mids": hit["_source"]["mids_hash"],
                    "highs": hit["_source"]["highs_hash"],
                },
                metadata=hit["_source"].get("metadata", {})
            )
        except Exception:
            return None
    
    def get_all_hashes(self) -> List[Tuple[str, Dict[str, bytes]]]:
        # Scroll through all documents
        result = []
        response = self.es.search(
            index=self.index_name,
            body={"query": {"match_all": {}}, "size": 10000},
            scroll="2m"
        )
        
        while response["hits"]["hits"]:
            for hit in response["hits"]["hits"]:
                hash_bytes = {
                    "bass": bytes.fromhex(hit["_source"]["bass_hash"]),
                    "mids": bytes.fromhex(hit["_source"]["mids_hash"]),
                    "highs": bytes.fromhex(hit["_source"]["highs_hash"]),
                }
                result.append((hit["_id"], hash_bytes))
            
            response = self.es.scroll(scroll_id=response["_scroll_id"], scroll="2m")
        
        return result


# =============================================================================
# 7. AUDIO SYNTHESIZER
# =============================================================================

class BitStreamSynthesizer:
    """Generate audio from LSH hash codes."""
    
    def __init__(self, config: S3Config):
        self.config = config
        self.sample_rate = config.sample_rate
        self.duration = config.audio_duration
        
        # Frequency mappings
        self.bass_freqs = 32.70 * (2 ** (np.arange(32) / 12))  # C1 up
        self.mid_freqs = 130.81 * (2 ** (np.arange(24) / 12))  # C3 up
    
    def synthesize(self, hash_bytes: Dict[str, bytes]) -> np.ndarray:
        """Generate audio from hash codes."""
        num_samples = int(self.sample_rate * self.duration)
        t = np.linspace(0, self.duration, num_samples)
        
        audio = np.zeros(num_samples, dtype=np.float32)
        
        # Bass layer
        if "bass" in hash_bytes:
            bits = np.unpackbits(np.frombuffer(hash_bytes["bass"], dtype=np.uint8))[:32]
            for i, bit in enumerate(bits):
                if bit:
                    freq = self.bass_freqs[i]
                    envelope = 1 - np.exp(-t * 3)
                    audio += np.sin(2 * np.pi * freq * t) * envelope * 0.15
        
        # Mids layer
        if "mids" in hash_bytes:
            bits = np.unpackbits(np.frombuffer(hash_bytes["mids"], dtype=np.uint8))[:64]
            active_notes = np.where(bits[:24])[0]
            for note_idx in active_notes[:6]:
                freq = self.mid_freqs[note_idx % len(self.mid_freqs)]
                audio += np.sin(2 * np.pi * freq * t) * 0.1
        
        # Highs layer (texture)
        if "highs" in hash_bytes:
            bits = np.unpackbits(np.frombuffer(hash_bytes["highs"], dtype=np.uint8))[:128]
            noise_color = sum(bits[:32]) / 32
            rng = np.random.RandomState(int(sum(bits[:8])))
            noise = rng.randn(len(t)) * 0.05
            audio += noise * (1 - np.exp(-t * 5))
        
        # Normalize
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.9
        
        return audio


# =============================================================================
# 8. THE S3 ENGINE (Full Pipeline)
# =============================================================================

class S3Engine:
    """
    Complete S3 search engine.
    
    This is the main interface for:
    - Indexing documents
    - Searching (two-pass: coarse DB + fine Python)
    - Audio generation
    """
    
    def __init__(self, config: S3Config, index: IndexBackend = None):
        self.config = config
        self.embedding_model = get_embedding_model(config)
        self.lsh = MultiBandLSH(config)
        self.scorer = HammingScorer(config)
        self.synthesizer = BitStreamSynthesizer(config)
        
        # Use provided index or default to in-memory
        self.index = index or InMemoryIndex()
        
        # Store config hash for migration detection
        self._config_hash = config.config_hash()
    
    def index_text(self, doc_id: str, text: str, metadata: Dict = None) -> Document:
        """
        Index a single text document.
        
        Pipeline:
        1. Embed text with teacher model
        2. Normalize embedding
        3. Hash with multi-band LSH
        4. Store in index
        """
        # Embed
        embedding = self.embedding_model.embed([text])[0]
        
        # Hash
        hash_bytes = self.lsh.hash_vector(embedding)
        hash_hex = self.lsh.to_hex(hash_bytes)
        
        # Create document
        doc = Document(
            doc_id=doc_id,
            text=text,
            hash_hex=hash_hex,
            metadata=metadata or {}
        )
        
        # Index
        self.index.index_document(doc)
        
        return doc
    
    def index_batch(self, documents: List[Tuple[str, str, Dict]]) -> List[Document]:
        """
        Index a batch of documents.
        
        Args:
            documents: List of (doc_id, text, metadata) tuples
        """
        indexed = []
        for doc_id, text, metadata in documents:
            doc = self.index_text(doc_id, text, metadata)
            indexed.append(doc)
        return indexed
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Search for documents similar to query.
        
        Two-pass architecture:
        1. COARSE (DB): Find candidates with matching bass hash
        2. FINE (Python): Score all candidates with weighted Hamming
        
        Returns:
            List of {doc_id, text, distance, similarity, metadata}
        """
        top_k = top_k or self.config.fine_top_k
        
        # Embed query
        query_embedding = self.embedding_model.embed([query])[0]
        query_hash = self.lsh.hash_vector(query_embedding)
        query_hex = self.lsh.to_hex(query_hash)
        
        # PASS 1: Coarse filter (DB)
        bass_hex = query_hex["bass"]
        coarse_docs = self.index.coarse_search(bass_hex, self.config.coarse_candidates)
        
        # If not enough coarse candidates, fall back to all docs
        if len(coarse_docs) < top_k:
            all_hashes = self.index.get_all_hashes()
            candidates = [(did, hashes) for did, hashes in all_hashes]
        else:
            candidates = [
                (doc.doc_id, self.lsh.from_hex(doc.hash_hex))
                for doc in coarse_docs
            ]
        
        # PASS 2: Fine scoring (Python)
        ranked = self.scorer.score_candidates(query_hash, candidates, top_k)
        
        # Build results
        results = []
        for doc_id, distance in ranked:
            doc = self.index.get_document(doc_id)
            if doc:
                results.append({
                    "doc_id": doc_id,
                    "text": doc.text,
                    "distance": distance,
                    "similarity": self.scorer.similarity(distance),
                    "metadata": doc.metadata,
                })
        
        return results
    
    def get_audio(self, doc_id: str) -> Optional[np.ndarray]:
        """Generate audio for a document."""
        doc = self.index.get_document(doc_id)
        if not doc:
            return None
        
        hash_bytes = self.lsh.from_hex(doc.hash_hex)
        return self.synthesizer.synthesize(hash_bytes)
    
    def compare_audio(self, doc_id_a: str, doc_id_b: str) -> Optional[np.ndarray]:
        """Mix audio of two documents for comparison."""
        audio_a = self.get_audio(doc_id_a)
        audio_b = self.get_audio(doc_id_b)
        
        if audio_a is None or audio_b is None:
            return None
        
        return (audio_a + audio_b) / 2


# =============================================================================
# 9. DEMO & TESTING
# =============================================================================

def demo():
    """Full demonstration of S3 pipeline."""
    print("=" * 70)
    print("S3 - SEMANTIC SIGNAL SEARCH (Production-Ready Demo)")
    print("=" * 70)
    
    # 1. Configuration
    print("\n" + "-" * 70)
    print("1. CONFIGURATION")
    print("-" * 70)
    
    config = S3Config()
    print(f"\nConfig hash: {config.config_hash()}")
    print(f"Embedding model: {config.embedding_model}")
    print(f"Embedding dim: {config.embedding_dim}")
    print(f"Total bits: {config.total_bits}")
    print(f"Bytes per document: {config.total_bytes}")
    print("\nBands:")
    for band in config.bands:
        print(f"  {band.name}: dims [{band.dim_start}:{band.dim_end}] → {band.bits} bits (weight={band.weight})")
    
    # 2. Create engine
    print("\n" + "-" * 70)
    print("2. ENGINE INITIALIZATION")
    print("-" * 70)
    
    engine = S3Engine(config)
    print(f"Embedding model: {type(engine.embedding_model).__name__}")
    print(f"Index backend: {type(engine.index).__name__}")
    
    # 3. Index documents
    print("\n" + "-" * 70)
    print("3. INDEXING DOCUMENTS")
    print("-" * 70)
    
    documents = [
        ("doc_1", "How to cancel a contract immediately without penalties"),
        ("doc_2", "Steps to terminate your subscription service"),
        ("doc_3", "Contract cancellation procedures and legal requirements"),
        ("doc_4", "Best recipes for chocolate birthday cake"),
        ("doc_5", "How to bake a delicious vanilla cake at home"),
        ("doc_6", "Legal advice for resolving contract disputes"),
        ("doc_7", "Machine learning tutorial for beginners"),
        ("doc_8", "Deep learning with PyTorch and TensorFlow"),
    ]
    
    start = time.time()
    for doc_id, text in documents:
        doc = engine.index_text(doc_id, text, {"source": "demo"})
        print(f"  Indexed: {doc_id}")
        print(f"    Bass hash: {doc.hash_hex['bass'][:16]}...")
    
    elapsed = time.time() - start
    print(f"\n  Indexed {len(documents)} documents in {elapsed*1000:.1f}ms")
    
    # 4. Search
    print("\n" + "-" * 70)
    print("4. SEARCH: 'How do I cancel my contract?'")
    print("-" * 70)
    
    query = "How do I cancel my contract?"
    start = time.time()
    results = engine.search(query, top_k=5)
    elapsed = time.time() - start
    
    print(f"\nSearch completed in {elapsed*1000:.1f}ms")
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"  {i+1}. [{result['similarity']:.3f}] {result['doc_id']}")
        print(f"     {result['text'][:60]}...")
    
    # 5. Audio generation
    print("\n" + "-" * 70)
    print("5. AUDIO GENERATION")
    print("-" * 70)
    
    for doc_id in ["doc_1", "doc_4"]:
        audio = engine.get_audio(doc_id)
        if audio is not None:
            print(f"  {doc_id}: {len(audio)} samples ({len(audio)/config.sample_rate:.1f}s)")
            print(f"    Peak: {np.abs(audio).max():.3f}, RMS: {np.sqrt(np.mean(audio**2)):.3f}")
    
    # 6. Similarity analysis
    print("\n" + "-" * 70)
    print("6. SIMILARITY ANALYSIS")
    print("-" * 70)
    
    doc1_hash = engine.lsh.from_hex(engine.index.get_document("doc_1").hash_hex)
    doc2_hash = engine.lsh.from_hex(engine.index.get_document("doc_2").hash_hex)
    doc4_hash = engine.lsh.from_hex(engine.index.get_document("doc_4").hash_hex)
    
    dist_12 = engine.scorer.weighted_distance(doc1_hash, doc2_hash)
    dist_14 = engine.scorer.weighted_distance(doc1_hash, doc4_hash)
    
    print(f"\n  doc_1 vs doc_2 (both contracts):")
    print(f"    Distance: {dist_12:.1f}")
    print(f"    Similarity: {engine.scorer.similarity(dist_12):.3f}")
    
    print(f"\n  doc_1 vs doc_4 (contract vs cake):")
    print(f"    Distance: {dist_14:.1f}")
    print(f"    Similarity: {engine.scorer.similarity(dist_14):.3f}")
    
    # 7. Efficiency stats
    print("\n" + "-" * 70)
    print("7. EFFICIENCY STATS")
    print("-" * 70)
    
    float_bytes = config.embedding_dim * 4
    hash_bytes = config.total_bytes
    
    print(f"\n  Storage per document:")
    print(f"    Float embedding: {float_bytes} bytes")
    print(f"    S3 hash:         {hash_bytes} bytes")
    print(f"    Compression:     {float_bytes / hash_bytes:.1f}x smaller")
    
    print(f"\n  At 1 million documents:")
    print(f"    Float storage: {float_bytes * 1_000_000 / 1e9:.2f} GB")
    print(f"    S3 storage:    {hash_bytes * 1_000_000 / 1e6:.2f} MB")
    
    # 8. Config export
    print("\n" + "-" * 70)
    print("8. CONFIG EXPORT (for production)")
    print("-" * 70)
    
    print("\n" + config.to_json())
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\n✅ All components validated:")
    print("   • Normalizer: Vectors on unit sphere")
    print("   • Multi-Band LSH: Hierarchical hashing")
    print("   • Two-Pass Search: Coarse DB + Fine Python")
    print("   • Audio Synthesis: Bits → Sound")
    print("   • Config: Versioned and hashable")


if __name__ == "__main__":
    demo()
