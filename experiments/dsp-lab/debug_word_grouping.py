import torch
from frankenstein_engine_v2 import FrankensteinEngineV2

if __name__ == "__main__":
    if torch.backends.mps.is_available(): DEVICE = "mps"
    elif torch.cuda.is_available(): DEVICE = "cuda"
    else: DEVICE = "cpu"

    engine = FrankensteinEngineV2(device=DEVICE)
    
    # Der konkrete Satz aus dem PDF (Seite 2)
    test_text = "2. Bankgeheimnis und Bankauskunft (1) Bankgeheimnis"
    
    print(f"\n🔍 DEBUG WORD-POOLING FÜR: '{test_text}'")
    print("=" * 60)
    
    # Wir bilden den Normalisierungsschritt nach
    clean_text = engine._normalize_text(test_text)
    processed_text = " " + clean_text
    print(f"Normalisierter Text: '{processed_text}'")
    
    # Tokenisierung
    inputs = engine.tokenizer(processed_text, return_tensors="pt")
    tokens = engine.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    print(f"\nTokens vom Modell:")
    print(tokens)
    
    # Pooling-Logik simulieren mit Debug-Ausgabe
    print(f"\nGrouping-Prozess:")
    print("-" * 40)
    
    word_data = []
    curr_word = None
    SPACE_PREFIXES = ["Ġ", "▁", " ", " "]

    for i, tok in enumerate(tokens):
        if tok in [engine.tokenizer.bos_token, engine.tokenizer.eos_token, engine.tokenizer.pad_token, "[CLS]", "[SEP]"]:
            continue
        
        is_start = any(tok.startswith(p) for p in SPACE_PREFIXES) or curr_word is None
        
        if is_start:
            if curr_word:
                print(f"  ✅ Wort fertig: '{curr_word['text']}'")
                word_data.append(curr_word)
            
            clean_tok = tok
            for p in SPACE_PREFIXES: clean_tok = clean_tok.replace(p, "")
            print(f"  🆕 Start neues Wort mit Token: '{tok}' -> '{clean_tok}'")
            curr_word = {"text": clean_tok, "tokens": [tok]}
        else:
            clean_part = tok.replace("##", "")
            curr_word["text"] += clean_part
            curr_word["tokens"].append(tok)
            print(f"  ➕ Hänge Token an: '{tok}' -> Aktueller Text: '{curr_word['text']}'")

    if curr_word:
        print(f"  ✅ Wort fertig: '{curr_word['text']}'")
        word_data.append(curr_word)

    print("\nFinales Inventar für diesen Satz:")
    print([w['text'] for w in word_data])
