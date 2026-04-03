import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModel
import random

# =========================================================
# 1. CORE LOGIC (UNSER KOMPRESSOR)
# =========================================================
class CompressorLogic:
    def _point_line_dist_fast(self, px, py, sx, sy, ex, ey):
        line_dx = ex - sx
        line_dy = ey - sy
        if line_dx == 0 and line_dy == 0: 
            return np.sqrt((px - sx)**2 + (py - sy)**2)
        return abs(line_dx * (py - sy) - line_dy * (px - sx)) / np.sqrt(line_dx**2 + line_dy**2)

    def _rdp_iterative(self, points, epsilon):
        stack = [(0, len(points)-1)]
        keep = {0, len(points)-1}
        xs, ys = points[:, 0], points[:, 1]
        while stack:
            start, end = stack.pop()
            if end - start <= 1: continue
            sx, sy = xs[start], ys[start]
            ex, ey = xs[end], ys[end]
            dmax = 0.0; index = start
            for i in range(start+1, end):
                d = self._point_line_dist_fast(xs[i], ys[i], sx, sy, ex, ey)
                if d > dmax: index = i; dmax = d
            if dmax > epsilon:
                keep.add(index); stack.append((start, index)); stack.append((index, end))
        return sorted(list(keep))

    def compress_dimension_debug(self, values, rel_eps, slope_tol, rel_dist, min_noise):
        v_min, v_max = np.min(values), np.max(values)
        rng = v_max - v_min
        
        abs_eps = max(rng * rel_eps, min_noise)
        abs_dist = max(rng * rel_dist, min_noise)
        
        # --- 1. SLOPE VARIANTE ---
        segs = []
        if rng >= 0: 
            points = np.column_stack((np.arange(len(values)), values))
            keep_idx = self._rdp_iterative(points, abs_eps)
            
            raw_segs = []
            for i in range(len(keep_idx)-1):
                s, e = keep_idx[i], keep_idx[i+1]
                l = e - s
                slope = (points[e,1] - points[s,1]) / l
                raw_segs.append({'s': points[s,1], 'm': slope, 'l': l, 'e': points[e,1]})
            
            if raw_segs:
                curr = raw_segs[0]
                for nxt in raw_segs[1:]:
                    if abs(curr['m'] - nxt['m']) < slope_tol:
                         pred_mid = curr['s'] + ((nxt['e'] - curr['s']) / (curr['l'] + nxt['l'])) * curr['l']
                         if abs(pred_mid - curr['e']) < abs_dist:
                            new_len = curr['l'] + nxt['l']
                            if new_len <= 255: 
                                curr = {'s': curr['s'], 'm': (nxt['e'] - curr['s']) / new_len, 'l': new_len, 'e': nxt['e']}
                                continue
                    segs.append(curr); curr = nxt
                segs.append(curr)

        rec_slope = np.zeros(len(values))
        cur = 0
        for s in segs:
            s_q = int(np.clip(np.round(((s['s'] - v_min)/rng)*255), 0, 255)) if rng > 0 else 0
            s_val = (s_q / 255.0) * rng + v_min
            line = s_val + (s['m'] * np.arange(s['l'] + 1))
            l_write = min(s['l'], len(values) - cur)
            rec_slope[cur:cur+l_write] = line[:l_write]
            if cur + s['l'] == len(values): rec_slope[-1] = line[l_write] if l_write < len(line) else line[-1]
            cur += s['l']
            
        size_seg = len(segs) * 4
        
        # --- 2. INT8 VARIANTE ---
        if rng == 0:
            rec_int8 = values
        else:
            q = np.clip(np.round(((values - v_min) / rng) * 255), 0, 255)
            rec_int8 = (q / 255.0) * rng + v_min
        
        size_int8 = len(values)

        return {
            'slope_data': rec_slope,
            'slope_bytes': size_seg,
            'slope_count': len(segs),
            'int8_data': rec_int8,
            'int8_bytes': size_int8,
            'range': rng,
            'abs_eps': abs_eps
        }

comp = CompressorLogic()

# =========================================================
# 2. STREAMLIT APP (JINA VERSION)
# =========================================================
st.set_page_config(layout="wide", page_title="Jina V3 Compressor Lab")

st.title("🧪 Jina Embeddings V3 Compressor Lab")

# --- SIDEBAR ---
st.sidebar.header("Kompressions-Parameter")
rel_eps = st.sidebar.slider("Relative Epsilon (%)", 0.0, 0.5, 0.08, 0.01)
slope_tol = st.sidebar.slider("Slope Tolerance", 0.0, 2.0, 0.40, 0.05)
rel_dist = st.sidebar.slider("Max Merge Dist (%)", 0.0, 0.5, 0.08, 0.01)
min_noise = st.sidebar.slider("Noise Floor (Abs)", 0.0, 0.2, 0.03, 0.01)

st.sidebar.markdown("---")
force_slope = st.sidebar.checkbox("👀 Erzwinge 'Slope' Ansicht", value=True)

# MODEL LOADING (JINA V3)
model_id = "jinaai/jina-embeddings-v3"

@st.cache_resource
def load_model():
    # trust_remote_code=True ist für Jina zwingend erforderlich
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    return tokenizer, model

with st.spinner(f"Lade {model_id} (das kann beim ersten Mal dauern)..."):
    tokenizer, model = load_model()

# --- INPUT ---
default_text = "Jina Embeddings v3 ist ein leistungsstarkes Modell mit 1024 Dimensionen, das Matryoshka-Representation unterstützt."
text = st.text_area("Eingabetext", default_text, height=70)

if text:
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
    
    # Jina V3: last_hidden_state hat Shape [Batch, Seq, 1024]
    embeddings = outputs.last_hidden_state[0].float().cpu().numpy()
    
    # Word Aggregation
    word_ids = inputs.word_ids(batch_index=0)
    unique_ids = sorted(list(set(w for w in word_ids if w is not None)))
    word_vecs = []
    labels = []
    for wid in unique_ids:
        idx = [i for i, w in enumerate(word_ids) if w == wid]
        word_vecs.append(np.mean(embeddings[idx], axis=0))
        span = inputs.word_to_chars(0, word_index=wid)
        labels.append(text[span.start:span.end])
    
    matrix = np.array(word_vecs)
    n_words, n_dims = matrix.shape

    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader(f"Viewer ({n_dims} Dims)")
        # Jina hat 1024 Dims, also passen wir den Slider an
        selected_dim = st.number_input(f"Wähle Dimension (0-{n_dims-1})", 0, n_dims-1, 42)
        
        vals = matrix[:, selected_dim]
        res = comp.compress_dimension_debug(vals, rel_eps, slope_tol, rel_dist, min_noise)
        
        orig_bytes = len(vals) * 4
        winner = "Slope" if res['slope_bytes'] < res['int8_bytes'] else "Int8"
        used_bytes = res['slope_bytes'] if winner == "Slope" else res['int8_bytes']
        
        dim_ratio = orig_bytes / used_bytes
        
        st.metric("Echte Ratio", f"{dim_ratio:.1f}x", f"Winner: {winner}")
        st.caption(f"Slope: {res['slope_bytes']} B ({res['slope_count']} Segs)")
        st.caption(f"Int8: {res['int8_bytes']} B")
        st.caption(f"Range: {res['range']:.4f}")

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=vals, mode='lines+markers', name='Original', line=dict(color='gray', width=3), opacity=0.3))
        
        if not force_slope or winner == "Int8":
             fig.add_trace(go.Scatter(y=res['int8_data'], mode='lines', name='Int8 Noise', line=dict(color='blue', width=1), visible='legendonly'))
        
        if force_slope or winner == "Slope":
            fig.add_trace(go.Scatter(y=res['slope_data'], mode='lines+markers', name='Slope Segments', line=dict(color='red', width=2)))

        fig.update_layout(title=f"Dimension {selected_dim} (Jina V3)", height=450, hovermode="x unified", xaxis=dict(tickmode='array', tickvals=list(range(len(labels))), ticktext=labels))
        st.plotly_chart(fig, use_container_width=True)

    # --- BENCHMARK ---
    st.divider()
    if st.button("🚀 Matrix Benchmark (1024 Dims)"):
        with st.spinner("Rechne..."):
            total_bytes = 0
            slope_wins = 0
            rec_matrix = np.zeros_like(matrix)
            
            for d in range(n_dims):
                r = comp.compress_dimension_debug(matrix[:, d], rel_eps, slope_tol, rel_dist, min_noise)
                if r['slope_bytes'] < r['int8_bytes']:
                    total_bytes += r['slope_bytes']
                    rec_matrix[:, d] = r['slope_data']
                    slope_wins += 1
                else:
                    total_bytes += r['int8_bytes']
                    rec_matrix[:, d] = r['int8_data']
            
            # Header Overhead: 4 Global + 9 pro Dim
            total_bytes += 4 + (n_dims * 9)
            orig_total = n_words * n_dims * 4
            ratio = orig_total / total_bytes
            
            t_orig = torch.tensor(matrix)
            t_rec = torch.tensor(rec_matrix)
            sim = torch.mean(F.cosine_similarity(t_orig, t_rec, dim=1)).item()
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Global Ratio", f"{ratio:.2f}x")
            c2.metric("Cos Sim", f"{sim:.5f}")
            c3.metric("Linear Dims", f"{slope_wins} / {n_dims}")
            
            if sim >= 0.99: st.success("Perfekt!")
            elif sim >= 0.97: st.info("Gut.")
            else: st.warning("Qualität kritisch.")
