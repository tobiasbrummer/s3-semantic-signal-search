import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

def test_model_resolution(model_name, device="cpu"):
    print(f"\n🚀 TESTE MODELL: {model_name}")
    print("-" * 50)
    
    try:
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
        # Wir laden nur den Encoder, um Speicher zu sparen
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            low_cpu_mem_usage=True
        ).to(device).eval()
        print(f"✅ Modell geladen in {time.time() - start_time:.1f}s")
    except Exception as e:
        print(f"❌ Fehler beim Laden von {model_name}: {e}")
        return

    # Die kritischen Paare aus der Diagnose
    pairs = [
        ("Gefahr durch Atommüll", "Der Mietvertrag wurde fristgerecht gekündigt.", "At", "igt"),
        ("Gefahr durch Atommüll", "Radioaktiver Abfall ist ein Problem der Atomkraft.", "At", "▁ein"),
        ("Baby Bettruhe", "Hier schläft ein Kind.", "Baby", "▁Kind")
    ]

    for q_text, d_text, q_sub, d_sub in pairs:
        # Encoding
        q_in = tokenizer(q_text, return_tensors="pt").to(device)
        d_in = tokenizer(d_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            q_emb = model.get_encoder()(input_ids=q_in.input_ids).last_hidden_state[0]
            d_emb = model.get_encoder()(input_ids=d_in.input_ids).last_hidden_state[0]
            
        q_tokens = tokenizer.convert_ids_to_tokens(q_in.input_ids[0])
        d_tokens = tokenizer.convert_ids_to_tokens(d_in.input_ids[0])
        
        # Indizes finden
        try:
            q_idx = [i for i, t in enumerate(q_tokens) if q_sub.lower() in t.lower()][0]
            d_idx = [i for i, t in enumerate(d_tokens) if d_sub.lower() in t.lower()][0]
            
            vec_q = q_emb[q_idx]
            vec_d = d_emb[d_idx]
            
            sim = F.cosine_similarity(vec_q.unsqueeze(0), vec_d.unsqueeze(0)).item()
            
            print(f"Match: '{q_tokens[q_idx]}' vs '{d_tokens[d_idx]}'")
            print(f"  -> Cosine Similarity: {sim:.4f}")
            if "At" in q_sub and "igt" in d_sub:
                status = "❌ HALLUZINATION" if sim > 0.7 else "✅ KORREKT GETRENNT"
                print(f"  -> Status: {status}")
        except Exception as e:
            print(f"  -> Fehler beim Token-Matching: {e}")
        print("-" * 30)

    # Speicher freigeben
    del model
    del tokenizer
    if device == "cuda": torch.cuda.empty_cache()

if __name__ == "__main__":
    if torch.backends.mps.is_available(): DEVICE = "mps"
    elif torch.cuda.is_available(): DEVICE = "cuda"
    else: DEVICE = "cpu"

    models = [
        "google/t5gemma-2-270m-270m",
        "google/t5gemma-2-1b-1b",
        "google/t5gemma-2-4b-4b"
    ]

    for m in models:
        test_model_resolution(m, DEVICE)
