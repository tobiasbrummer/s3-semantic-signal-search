import torch
from frankenstein_engine_v2 import FrankensteinEngineV2
import fitz

if __name__ == "__main__":
    if torch.backends.mps.is_available(): DEVICE = "mps"
    elif torch.cuda.is_available(): DEVICE = "cuda"
    else: DEVICE = "cpu"

    engine = FrankensteinEngineV2(device=DEVICE)
    doc = fitz.open("contract.pdf")
    text = " ".join([page.get_text() for page in doc])
    
    words = engine.encode(text)
    
    print(f"\n📊 WORT-INVENTAR (ERSTE 100 WÖRTER):")
    print("-" * 50)
    # Wir suchen speziell nach 'Bankgeheimnis'
    found = False
    for i, w in enumerate(words[:200]):
        if i < 100: print(f"'{w['text']}'", end=", ")
        if "bankgeheimnis" in w['text'].lower():
            print(f"\n\n🎯 GEFUNDEN: '{w['text']}' an Position {i} (Gewicht: {w['weight']:.2f}, Z: {w['z']:.2f})")
            found = True
    
    if not found:
        print("\n\n❌ 'Bankgeheimnis' wurde im gesamten Inventar nicht gefunden!")
