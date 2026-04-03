import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import numpy as np
import fitz
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import re

class FrankensteinDSPLab:
    def __init__(self, device="cpu"):
        self.device = device
        print(f"🏗️ Initialisiere DSP-Lab auf {device}...")
        model_name = "ibm-granite/granite-embedding-278m-multilingual"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dense_model = AutoModel.from_pretrained(model_name).to(device).eval()
        sparse_name = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
        self.sparse_tokenizer = AutoTokenizer.from_pretrained(sparse_name)
        self.sparse_model = AutoModelForMaskedLM.from_pretrained(sparse_name).to(self.device).eval()

    def get_signal(self, text):
        clean_text = " " + " ".join(re.sub(r'[^\w\s]', ' ', text).split())
        inputs = self.tokenizer(clean_text, return_tensors="pt")
        with torch.no_grad():
            emb = self.dense_model(input_ids=inputs.input_ids.to(self.device)).last_hidden_state[0]
            s_in = self.sparse_tokenizer(clean_text, return_tensors="pt").to(self.device)
            s_weights = torch.max(torch.log(1 + torch.relu(self.sparse_model(s_in.input_ids).logits)), dim=-1)[0][0]

        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        signal_vecs, weights, words = [], [], []
        curr_v, curr_w, curr_t = [], [], ""
        SPACE_PREFIXES = ["Ġ", "▁", " ", " "]

        for i, tok in enumerate(tokens):
            if tok in [self.tokenizer.bos_token, self.tokenizer.eos_token, "[CLS]", "[SEP]"]:
                continue
            if any(tok.startswith(p) for p in SPACE_PREFIXES) or not curr_v:
                if curr_v:
                    v = torch.stack(curr_v).mean(dim=0)[:512]
                    signal_vecs.append(F.normalize(v.unsqueeze(0), p=2, dim=1)[0])
                    weights.append(max(curr_w))
                    words.append(curr_t)
                curr_v, curr_w = [emb[i]], [s_weights[min(i, len(s_weights)-1)].item()]
                curr_t = tok
                for p in SPACE_PREFIXES: curr_t = curr_t.replace(p, "")
            else:
                curr_v.append(emb[i])
                curr_w.append(s_weights[min(i, len(s_weights)-1)].item())
                curr_t += tok.replace("##", "")
        if curr_v:
            v = torch.stack(curr_v).mean(dim=0)[:512]
            signal_vecs.append(F.normalize(v.unsqueeze(0), p=2, dim=1)[0])
            weights.append(max(curr_w))
            words.append(curr_t)
            
        return torch.stack(signal_vecs).cpu().float(), np.array(weights), words

def plot_dsp_focused(signal, weights, words, title, filename, top_indices=None):
    """Plottet nur die Top 10 dominantesten Frequenz-Bänder."""
    num_words = len(words)
    x_words = np.arange(num_words)
    
    # 1. Dominante Dimensionen finden (falls nicht übergeben)
    if top_indices is None:
        # Wir nehmen die Dimensionen mit der höchsten absoluten Summe (Magnitude)
        magnitudes = torch.abs(signal).sum(dim=0)
        top_indices = torch.topk(magnitudes, 10).indices.numpy()
    
    focused_signal = signal[:, top_indices].numpy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), gridspec_kw={'height_ratios': [1, 1]})
    
    # --- OBEN: Top 10 Frequenz-Spuren ---
    ax1.set_facecolor('#050505')
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))
    
    # Frequenzen für die 10 Dimensionen berechnen
    all_plot_freqs = 2500 + (focused_signal * 500)
    f_min, f_max = all_plot_freqs.min(), all_plot_freqs.max()
    padding = (f_max - f_min) * 0.1 if f_max > f_min else 50
    
    for i, d_idx in enumerate(top_indices):
        freq_data = all_plot_freqs[:, i]
        ax1.plot(x_words, freq_data, color=colors[i], linewidth=2, alpha=0.8, label=f"Dim {d_idx}")
        ax1.scatter(x_words, freq_data, color=colors[i], s=30, zorder=5)
    
    ax1.set_title(f"Top 10 Semantic Frequencies: {title}", color='white', fontsize=14)
    ax1.set_ylabel("Frequency (Hz)", color='white')
    ax1.set_ylim(f_min - padding, f_max + padding)
    ax1.axhline(2500, color='white', linestyle='--', alpha=0.3)
    ax1.legend(loc='upper right', facecolor='#222222', labelcolor='white', fontsize=8)
    ax1.tick_params(axis='y', colors='white')

    # --- UNTEN: Top 10 Heatmap ---
    # Wir zeigen hier nur die 10 gewählten Dimensionen
    weighted_heatmap = focused_signal.T * weights[np.newaxis, :]
    im = ax2.imshow(weighted_heatmap, aspect='auto', cmap='magma', origin='lower', 
                    interpolation='nearest')
    
    # Sparse Kurve Overlay
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x_words, weights, color='white', linewidth=3, alpha=0.9, label="Sparse Gain")
    ax2_twin.set_ylabel("Sparse Weight (Gain)", color='white')
    ax2_twin.tick_params(axis='y', colors='white')

    ax2.set_title("Focused Spectral Heatmap (Top 10 Dimensions)", color='white')
    ax2.set_ylabel("Top Dimension Index", color='white')
    ax2.set_xlabel("Words", color='white')
    
    # Wort-Labels
    ax2.set_xticks(x_words)
    ax2.set_xticklabels(words, rotation=45, ha='right', fontsize=10, color='white')
    ax2.set_yticks(range(10))
    ax2.set_yticklabels([f"Dim {idx}" for idx in top_indices], color='white')
    
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, facecolor='#111111')
    print(f"📸 Fokus-Plot gespeichert: {filename}")
    plt.close()
    return top_indices

if __name__ == "__main__":
    if torch.backends.mps.is_available(): DEVICE = "mps"
    elif torch.cuda.is_available(): DEVICE = "cuda"
    else: DEVICE = "cpu"

    lab = FrankensteinDSPLab(DEVICE)
    
    doc_text = "Bankgeheimnis und Bankauskunft. Die Bank ist zur Verschwiegenheit verpflichtet."
    query_text = "Bankgeheimnis"
    
    d_sig, d_weights, d_words = lab.get_signal(doc_text)
    q_sig, q_weights, q_words = lab.get_signal(query_text)
    
    # Wir nehmen die Top 10 aus der QUERY und nutzen sie für beide Plots
    magnitudes_q = torch.abs(q_sig).sum(dim=0)
    top_10 = torch.topk(magnitudes_q, 10).indices.numpy()
    
    print(f"\n🎯 Dominante Frequenzen der Query: {top_10}")
    
    plot_dsp_focused(q_sig, q_weights, q_words, "Query Signal", "dsp_query_focused.png", top_indices=top_10)
    plot_dsp_focused(d_sig, d_weights, d_words, "Dokument Signal", "dsp_doc_focused.png", top_indices=top_10)
