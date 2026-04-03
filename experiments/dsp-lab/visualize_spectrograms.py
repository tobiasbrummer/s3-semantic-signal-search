import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from frankenstein_engine import FrankensteinEncoder

# Setup
if torch.backends.mps.is_available(): DEVICE = "mps"
elif torch.cuda.is_available(): DEVICE = "cuda"
else: DEVICE = "cpu"

def plot_spectrogram_comparison(engine, doc_text, query_text, filename):
    """
    Erstellt ein Doppel-Spektrogramm: Oben Dokument, Unten Query.
    X-Achse: Tokens (Zeit)
    Y-Achse: Embedding-Dimensionen (Frequenzen)
    """
    print(f"📊 Generiere Spektrogramm für: '{doc_text[:30]}...' vs '{query_text}'")
    
    # Embeddings holen (wir nutzen encode_query für beide, um die volle Sequenz ohne Kompression zu sehen)
    doc_emb, doc_tokens = engine.encode_query(doc_text)
    query_emb, query_tokens = engine.encode_query(query_text)
    
    # Tensor -> Numpy [Seq, Dim]
    # Wir nehmen den Absolutwert für die "Lautstärke"
    doc_data = torch.abs(doc_emb[0]).cpu().detach().numpy().T
    query_data = torch.abs(query_emb[0]).cpu().detach().numpy().T
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 1]})
    
    # 1. Dokument Plot
    im1 = ax1.imshow(doc_data, aspect='auto', cmap='viridis', origin='lower')
    ax1.set_title(f"DOKUMENT: {doc_text}", fontsize=12)
    ax1.set_ylabel("Embedding Dim (0-640)")
    # Token Labels auf X-Achse
    ax1.set_xticks(range(len(doc_tokens)))
    ax1.set_xticklabels([t.replace('▁', '_') for t in doc_tokens], rotation=45, ha='right', fontsize=8)
    
    # 2. Query Plot
    im2 = ax2.imshow(query_data, aspect='auto', cmap='viridis', origin='lower')
    ax2.set_title(f"QUERY: {query_text}", fontsize=12)
    ax2.set_ylabel("Embedding Dim (0-640)")
    ax2.set_xticks(range(len(query_tokens)))
    ax2.set_xticklabels([t.replace('▁', '_') for t in query_tokens], rotation=45, ha='right', fontsize=8)
    
    plt.colorbar(im1, ax=[ax1, ax2], label="Absolute Amplitude")
    plt.tight_layout()
    
    plt.savefig(filename, dpi=150)
    print(f"💾 Gespeichert als: {filename}")
    plt.close()

if __name__ == "__main__":
    print(f"🏗️ Lade Engine auf {DEVICE}...")
    engine = FrankensteinEncoder(
        "google/t5gemma-2-270m-270m", 
        "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
        device=DEVICE
    )

    # Test-Paare basierend auf deinem Corpus und den Suchanfragen
    # Paar 1: Ein semantisch korrekter Treffer (theoretisch)
    doc1 = "Radioaktiver Abfall ist ein Problem der Atomkraft."
    query1 = "Gefahr durch Atommüll"
    
    # Paar 2: Die Halluzination (Mietvertrag vs Atommüll)
    doc2 = "Der Mietvertrag wurde fristgerecht gekündigt."
    query2 = "Gefahr durch Atommüll"

    # Zusätzliches Paar: Hasenkinder
    doc3 = "Kleine Hasenkinder tollen über die Wiese."
    query3 = "Baby Bettruhe"

    plot_spectrogram_comparison(engine, doc1, query1, "spectrogram_match_atomic.png")
    plot_spectrogram_comparison(engine, doc2, query2, "spectrogram_hallucination_contract.png")
    plot_spectrogram_comparison(engine, doc3, query3, "spectrogram_rabbits.png")

    print("\n✅ Visualisierung abgeschlossen.")
