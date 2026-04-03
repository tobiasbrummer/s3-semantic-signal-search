import torch
import torch.nn.functional as F
from frankenstein_engine import FrankensteinEncoder

# Setup
if torch.backends.mps.is_available(): DEVICE = "mps"
elif torch.cuda.is_available(): DEVICE = "cuda"
else: DEVICE = "cpu"

def analyze_token_relation(engine, query_text, doc_text, target_q_token_part):
    print(f"\n🔍 ANALYSE: '{query_text}' vs '{doc_text}'")
    print("=" * 70)
    
    # 1. Daten holen
    # Wir brauchen die rohen T5 Embeddings UND die Gewichte getrennt
    
    # A. Query Seite
    sparse_q = engine._get_sparse_weights(query_text)
    t5_in_q = engine.t5_tokenizer(query_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        t5_emb_q = engine.t5_model.get_encoder()(input_ids=t5_in_q.input_ids).last_hidden_state
    q_tokens = engine.t5_tokenizer.convert_ids_to_tokens(t5_in_q.input_ids[0])
    
    # B. Doc Seite
    sparse_d = engine._get_sparse_weights(doc_text)
    t5_in_d = engine.t5_tokenizer(doc_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        t5_emb_d = engine.t5_model.get_encoder()(input_ids=t5_in_d.input_ids).last_hidden_state
    d_tokens = engine.t5_tokenizer.convert_ids_to_tokens(t5_in_d.input_ids[0])

    # 2. Target Token in Query finden
    q_idx = -1
    for i, t in enumerate(q_tokens):
        if target_q_token_part.lower() in t.lower():
            q_idx = i
            break
    
    if q_idx == -1:
        print(f"❌ Target '{target_q_token_part}' nicht in Query gefunden: {q_tokens}")
        return

    q_vec_raw = t5_emb_q[0, q_idx]
    
    # 3. Best Match im Dokument suchen (basierend auf reiner T5-Richtung)
    q_norm = F.normalize(q_vec_raw.unsqueeze(0), p=2, dim=1)
    d_norms = F.normalize(t5_emb_d[0], p=2, dim=1)
    
    similarities = torch.matmul(q_norm, d_norms.T)[0]
    best_d_idx = torch.argmax(similarities).item()
    max_sim = similarities[best_d_idx].item()
    
    d_vec_raw = t5_emb_d[0, best_d_idx]
    
    # 4. Gewichte extrahieren (Fusion-Logik simulieren)
    # (Wir nehmen hier vereinfacht das Gewicht aus der Engine-Logik)
    _, q_tokens_fused = engine.encode_query(query_text)
    _, d_tokens_fused = engine.encode_query(doc_text)
    
    # Die Gewichte sind in der Magnitude der fused Vektoren versteckt
    fused_q, _ = engine.encode_query(query_text)
    fused_d, _ = engine.encode_query(doc_text)
    
    q_weight = torch.norm(fused_q[0, q_idx]).item() / torch.norm(q_vec_raw).item()
    d_weight = torch.norm(fused_d[0, best_d_idx]).item() / torch.norm(d_vec_raw).item()

    # 5. Resultat präsentieren
    print(f"Token Match: Query['{q_tokens[q_idx]}'] <-> Doc['{d_tokens[best_d_idx]}']")
    print(f"--- DENSE (Richtung) ---")
    print(f"Raw T5 Cosine Similarity:  {max_sim:.4f}")
    if max_sim > 0.8: print("⚠️  T5 hält diese Token für fast identisch!")
    
    print(f"\n--- SPARSE (Gewichtung) ---")
    print(f"Query Token Gewicht:       {q_weight:.2f}")
    print(f"Doc Token Gewicht:         {d_weight:.2f}")
    
    print(f"\n--- FUSION (Effekt) ---")
    fused_sim = F.cosine_similarity(fused_q[0, q_idx].unsqueeze(0), fused_d[0, best_d_idx].unsqueeze(0)).item()
    print(f"Fused Cosine Similarity:   {fused_sim:.4f}")
    
    # Dot Product (Das ist das, was die Suche nutzt!)
    dot_product = torch.dot(fused_q[0, q_idx], fused_d[0, best_d_idx]).item()
    print(f"FINAL DOT PRODUCT (Score): {dot_product:.2f}")

if __name__ == "__main__":
    engine = FrankensteinEncoder(
        "google/t5gemma-2-4b-4b", 
        "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
        device=DEVICE
    )

    # Fall A: Die Halluzination
    analyze_token_relation(
        engine, 
        "Gefahr durch Atommüll", 
        "Der Mietvertrag wurde fristgerecht gekündigt.", 
        "At"
    )

    # Fall B: Ein guter Treffer zum Vergleich
    analyze_token_relation(
        engine, 
        "Gefahr durch Atommüll", 
        "Radioaktiver Abfall ist ein Problem der Atomkraft.", 
        "At"
    )
    
    # Fall C: Baby vs Kind
    analyze_token_relation(
        engine, 
        "Baby Bettruhe", 
        "Hier schläft ein Kind.", 
        "Baby"
    )
