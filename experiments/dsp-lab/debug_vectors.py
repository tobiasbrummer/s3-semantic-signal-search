import torch
import torch.nn.functional as F
from frankenstein_engine import FrankensteinEncoder

# Setup
if torch.backends.mps.is_available(): DEVICE = "mps"
elif torch.cuda.is_available(): DEVICE = "cuda"
else: DEVICE = "cpu"

print(f"🏗️ Lade Engine auf {DEVICE}...")
engine = FrankensteinEncoder(
    "google/t5gemma-2-4b-4b", 
    "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
    device=DEVICE
)

def get_best_vector(text):
    """Holt das 'lauteste' Token aus dem Wort (ignoriert Stille/Null-Vektoren)."""
    # encode_query gibt (Batch, Seq, Dim)
    emb, tokens = engine.encode_query(text)
    vecs = emb[0] # Erste Sequenz im Batch
    
    # Wir suchen den Index mit der höchsten Magnitude (L2 Norm)
    norms = torch.norm(vecs, dim=1)
    max_val, max_idx = torch.max(norms, dim=0)
    
    if max_val.item() == 0:
        print(f"⚠️ Warnung: '{text}' ist komplett still (Gewicht 0). Stopword?")
        return None, None

    best_vec = vecs[max_idx]
    best_token = tokens[max_idx]
    
    return best_vec, best_token

def check_similarity(word1, word2):
    v1, t1 = get_best_vector(word1)
    v2, t2 = get_best_vector(word2)
    
    if v1 is None or v2 is None:
        return

    # Cosine berechnen
    sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
    
    print(f"⚔️  '{t1}' ({word1}) vs '{t2}' ({word2})")
    print(f"    -> Cosine Similarity: {sim:.4f}")
    
    # Bewertung
    if sim > 0.85: print("    -> Status: 💎 SUPER MATCH (Trash-Gate pass)")
    elif sim > 0.55: print("    -> Status: ✅ GOOD MATCH (Semantic-Gate pass)")
    elif sim > 0.45: print("    -> Status: ⚠️ WEAK (Knapp daneben)")
    else: print("    -> Status: ❌ NO MATCH")
    print("-" * 40)

print("\n🔬 Vektor-Diagnose (Fixed):")
print("===========================")

# 1. Der erfolgreiche Fall
check_similarity("Baby", "Kind")

# 2. Die Problemfälle (Atommüll)
check_similarity("Atommüll", "Abfall")
check_similarity("Atommüll", "Atomkraft") # Oft stärker, da gleicher Wortstamm
check_similarity("Atommüll", "Müll")

# 3. Der semantische Fall
check_similarity("Gefahr", "Problem")
check_similarity("Gefahr", "Risiko")
