import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
import torch.nn.functional as F

# ==========================================
# 1. MODEL LOADER
# ==========================================
print("Lade Modelle (das kann kurz dauern)...")

# A) LOW FREQUENCY: Dense Embeddings (Semantic Vibe)
# Wir nutzen hier wieder ein kleines Modell für den Speed, im echten Leben DeBERTa
dense_model_name = "sentence-transformers/all-MiniLM-L6-v2" 
dense_tokenizer = AutoTokenizer.from_pretrained(dense_model_name)
dense_model = AutoModel.from_pretrained(dense_model_name)

# B) HIGH FREQUENCY: SPLADE (Sharp Keyword Spikes)
splade_model_name = "naver/splade-cocondenser-ensembledistil"
splade_tokenizer = AutoTokenizer.from_pretrained(splade_model_name)
splade_model = AutoModelForMaskedLM.from_pretrained(splade_model_name)

print("Modelle geladen.")

# ==========================================
# 2. SIGNAL GENERATORS
# ==========================================

def get_dense_signal(text, window_size=10, stride=5):
    """
    Erzeugt ein glattes Signal basierend auf Semantic Similarity zum Query.
    (Simuliert den 'Bass')
    """
    tokens = text.split()
    signal = []
    
    # Wir machen es hier einfach: Wir simulieren das Dense-Signal
    # basierend auf einer Sliding Window Embedding Distanz ist teuer,
    # daher hier eine vereinfachte Projektion für den Plot.
    # In der echten App ist das der Code aus dem PDFAudit.
    
    # Dummy-Implementation für Visualisierung:
    # Glatte Kurve, die hoch geht, wenn relevante Wörter in der Nähe sind
    x = np.linspace(0, len(tokens), len(tokens))
    base_signal = np.zeros(len(tokens))
    
    # Simuliere semantische Treffer (z.B. bei "Bank", "Kündigung")
    for i, t in enumerate(tokens):
        if t in ["Bank", "Kredit", "Kündigung", "Zinsen", "Vertrag"]:
            # Breite Gauss-Kurve (Low Freq)
            base_signal += 0.8 * np.exp(-0.01 * (x - i)**2)
            
    return base_signal

def get_splade_weights(text):
    """
    Holt die 'scharfen' SPLADE Gewichte für jeden Token.
    """
    inputs = splade_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = splade_model(**inputs).logits
    
    # SPLADE Aggregation (Max über Sequence, aber wir wollen Token-Level!)
    # Wir wollen wissen: Welcher Token im Input triggert hohe Activation?
    # SPLADE ist normalerweise: Input -> Sparse Vector (Vocab Dim).
    # Wir drehen es um: Wir wollen die Wichtigkeit des Input-Tokens selbst.
    # Approximation: Attention-Attention oder einfaches "Saliency".
    
    # Für diesen Test nutzen wir eine direkte Logik:
    # Wie stark aktiviert dieses Wort das SPLADE Modell?
    # Wir nutzen die Magnitude des Embeddings * Attention Mask * ReLu(Logits)
    
    # Echte SPLADE Inferenz für Retrieval:
    out = torch.max(torch.log(1 + torch.relu(logits)) * inputs.attention_mask.unsqueeze(-1), dim=1).values.squeeze()
    
    # Aber wir brauchen die Position im Text (Time-Domain).
    # Trick: Wir schauen, welche Input-Tokens hohe Werte im Output-Vektor erzeugen würden.
    # Das ist komplex.
    # EINFACHERE VARIANTE FÜR HEUTE:
    # Wir messen einfach die "Self-Information" oder den Impact des Tokens.
    
    # Alternative: Wir nutzen die SPLADE-Logik "andersrum":
    # Wir gewichten den Token basierend darauf, ob er ein "Expansion" Trigger ist.
    values, _ = torch.max(logits, dim=2)
    weights = torch.log(1 + torch.relu(values)).squeeze().numpy()
    
    # Mapping zurück auf Wörter (grob)
    word_ids = inputs.word_ids()
    token_weights = []
    current_word = None
    current_max = 0
    
    for i, w_id in enumerate(word_ids):
        if w_id is None: continue
        if w_id != current_word:
            if current_word is not None: token_weights.append(current_max)
            current_word = w_id
            current_max = weights[i]
        else:
            current_max = max(current_max, weights[i])
    token_weights.append(current_max)
    
    return np.array(token_weights)

# ==========================================
# 3. EXPERIMENT
# ==========================================

text_snippet = "Der Kunde kann den Vertrag mit der Bank kündigen wenn die Zinsen erhöht werden aber nicht ohne Frist."
print(f"\nAnalysiere: '{text_snippet}'")

# 1. Dense Signal (Low Freq)
# Wir faken hier das Dense Signal etwas, um den Effekt zu zeigen (da wir keine Query haben)
# Normalerweise: Cosine(Doc_Window, Query)
# Hier: Einfach "Wichtigkeit"
dense_sig = get_dense_signal(text_snippet)
# Normalisieren
dense_sig = dense_sig / (np.max(dense_sig) + 1e-9)

# 2. SPLADE Signal (High Freq)
splade_sig = get_splade_weights(text_snippet)
# Padding angleichen (Tokenizer Unterschiede)
if len(splade_sig) != len(dense_sig):
    # Einfaches Resampling
    splade_sig = np.interp(np.linspace(0, 1, len(dense_sig)), np.linspace(0, 1, len(splade_sig)), splade_sig)

# Normalisieren & High-Pass Filter Effekt verstärken (Noise Gate)
print(f"Raw SPLADE stats: Min={np.min(splade_sig):.2f}, Max={np.max(splade_sig):.2f}, Mean={np.mean(splade_sig):.2f}")

# Dynamischer Threshold: Alles unter dem Median ist Rauschen
threshold = np.percentile(splade_sig, 50)
splade_sig = np.where(splade_sig > threshold, splade_sig, 0) 

# Safe Normalize
max_val = np.max(splade_sig)
if max_val > 0:
    splade_sig = splade_sig / max_val
else:
    splade_sig = np.zeros_like(splade_sig)

# 3. COMPOSITE SIGNAL
# Mix: 60% Bass (Context), 40% Treble (Details)
composite = (dense_sig * 0.6) + (splade_sig * 0.4)

# ==========================================
# 4. PLOT
# ==========================================
plt.figure(figsize=(12, 6))
x = range(len(dense_sig))
words = text_snippet.split()

plt.plot(x, dense_sig, 'b-', alpha=0.5, linewidth=3, label='Low Freq (Dense/Context)')
plt.bar(x, splade_sig, color='red', alpha=0.5, label='High Freq (SPLADE/Keywords)', width=0.3)
plt.plot(x, composite, 'k--', linewidth=2, label='Composite Signal')

plt.xticks(x, words, rotation=45)
plt.title("S3 Hybrid Signal: Dense (Bass) + SPLADE (Treble)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

filename = "spectral_demo.png"
plt.savefig(filename)
print(f"Plot gespeichert: {filename}")
