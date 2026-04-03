import torch
import torch.nn.functional as F
from frankenstein_engine_v2 import FrankensteinEngineV2
import fitz
import numpy as np

def center_vectors(vectors):
    """Zentriert Vektoren durch Abzug des Mittelwerts und Re-Normalisierung."""
    mean_vec = torch.stack(vectors).mean(dim=0)
    centered = [F.normalize(v - mean_vec, p=2, dim=0) for v in vectors]
    return centered, mean_vec

if __name__ == "__main__":
    if torch.backends.mps.is_available(): DEVICE = "mps"
    elif torch.cuda.is_available(): DEVICE = "cuda"
    else: DEVICE = "cpu"

    engine = FrankensteinEngineV2(device=DEVICE)
    doc = fitz.open("contract.pdf")
    full_text = " ".join([page.get_text() for page in doc])
    
    # Wort-Daten extrahieren
    word_data = engine.encode(full_text)
    raw_vecs = [w["vec"].float() for w in word_data]
    
    print(f"\n📊 BERECHNE KONTRAST-BOOST (Zentrierung)...")
    centered_vecs, bias_vec = center_vectors(raw_vecs)
    
    # Test-Query: Bankgeheimnis
    query_raw = engine.encode("Bankgeheimnis", is_query=True)[0]["vec"].float()
    # WICHTIG: Die Query muss mit dem GLEICHEN Bias zentriert werden wie das Dokument
    query_centered = F.normalize(query_raw - bias_vec, p=2, dim=0)

    # Vergleichs-Funktion
    def get_top_matches(q_vec, d_vecs, words, label):
        sims = []
        for i, dv in enumerate(d_vecs):
            sim = torch.dot(q_vec, dv).item()
            sims.append((sim, words[i]["text"]))
        sims.sort(key=lambda x: x[0], reverse=True)
        
        print(f"\nTop 5 Matches ({label}):")
        for i, (sim, text) in enumerate(sims[:10]):
            print(f"  {i+1}. '{text:<15}' | Sim: {sim:.4f}")

    get_top_matches(query_raw, raw_vecs, word_data, "VORHER")
    get_top_matches(query_centered, centered_vecs, word_data, "NACH KONTRAST-BOOST")
