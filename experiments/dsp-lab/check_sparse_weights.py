import torch
from frankenstein_engine_v2 import FrankensteinEngineV2

if __name__ == "__main__":
    if torch.backends.mps.is_available(): DEVICE = "mps"
    elif torch.cuda.is_available(): DEVICE = "cuda"
    else: DEVICE = "cpu"

    engine = FrankensteinEngineV2(device=DEVICE)

    # 1. Gewichte in der Query prüfen
    query = "Bankgeheimnis und Zinsen"
    q_words = engine.encode(query, is_query=True)

    print(f"\n📊 SPARSE-GEWICHTE IN DER QUERY: '{query}'")
    print("-" * 50)
    for w in q_words:
        print(f"Wort: {w['text']:<15} | Gewicht: {w['weight']:.4f}")

    # 2. Gewichte in einem Beispielsatz (Dokument) prüfen
    doc_text = "Die Bank bewahrt das Bankgeheimnis und berechnet Zinsen für das Konto auch wenn es leer ist."
    d_words = engine.encode(doc_text)

    print(f"\n📊 SPARSE-GEWICHTE IM DOKUMENT:")
    print("-" * 50)
    targets = ["bankgeheimnis", "zinsen", "und", "auch"]
    for w in d_words:
        if w['text'].lower() in targets:
            print(f"Wort: {w['text']:<15} | Gewicht: {w['weight']:.4f}")
