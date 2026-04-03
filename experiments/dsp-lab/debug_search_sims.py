import torch
from frankenstein_engine_v2 import FrankensteinEngineV2, FrankensteinSearcherV2
import fitz
import numpy as np

if __name__ == "__main__":
    if torch.backends.mps.is_available(): DEVICE = "mps"
    elif torch.cuda.is_available(): DEVICE = "cuda"
    else: DEVICE = "cpu"

    engine = FrankensteinEngineV2(device=DEVICE)
    doc = fitz.open("contract.pdf")
    full_text = " ".join([page.get_text() for page in doc])
    
    # Wir indexieren das PDF
    words_doc = engine.encode(full_text)
    
    # Wir encodieren die Query
    query = "Bankgeheimnis"
    words_query = engine.encode(query, is_query=True)
    qw = words_query[0]
    
    print(f"\n🔍 ANALYSE MATCHES FÜR QUERY-WORT: '{qw['text']}'")
    print("=" * 60)
    
    # Wir suchen die Top 10 semantischen Treffer im Dokument
    sims = []
    for dw in words_doc:
        sim = torch.dot(qw["vec"].float(), dw["vec"].float()).item()
        sims.append((sim, dw))
    
    sims.sort(key=lambda x: x[0], reverse=True)
    
    for i, (sim, dw) in enumerate(sims[:10]):
        print(f"{i+1}. '{dw['text']}' | Sim: {sim:.4f} | Weight: {dw['weight']:.2f} | Z: {dw['z']:.2f}")
