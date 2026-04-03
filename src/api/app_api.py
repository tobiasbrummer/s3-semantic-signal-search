import streamlit as st
import requests
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F

# =========================================================
# 1. CORE LOGIC
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

    # --- MODE A: SMART SLOPE ---
    def compress_dimension_slope(self, values, rel_eps, slope_tol, rel_dist, min_noise):
        v_min, v_max = np.min(values), np.max(values)
        rng = v_max - v_min
        abs_eps = max(rng * rel_eps, min_noise)
        abs_dist = max(rng * rel_dist, min_noise)
        
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
        if rng == 0: rec_int8 = values
        else:
            q = np.clip(np.round(((values - v_min) / rng) * 255), 0, 255)
            rec_int8 = (q / 255.0) * rng + v_min
        return {'slope_data': rec_slope, 'slope_bytes': size_seg, 'int8_data': rec_int8, 'int8_bytes': len(values)}

    # --- MODE B: BLOCK ENVELOPE ---
    def compress_minmax_envelope(self, values, block_size=4):
        pad_len = (block_size - (len(values) % block_size)) % block_size
        padded = np.pad(values, (0, pad_len), mode='edge') if pad_len > 0 else values
        blocks = padded.reshape(-1, block_size)
        v_min, v_max = np.min(blocks, axis=1), np.max(blocks, axis=1)
        
        g_min, g_max = np.min(values), np.max(values)
        rng = g_max - g_min if g_max != g_min else 1.0

        q_min = np.clip(np.round(((v_min - g_min)/rng)*255), 0, 255)
        q_max = np.clip(np.round(((v_max - g_min)/rng)*255), 0, 255)
        rec_min = (q_min / 255.0) * rng + g_min
        rec_max = (q_max / 255.0) * rng + g_min
        
        rec_avg = (rec_min + rec_max) / 2.0
        rec_full = np.repeat(rec_avg, block_size)[:len(values)]
        env_upper = np.repeat(rec_max, block_size)[:len(values)]
        env_lower = np.repeat(rec_min, block_size)[:len(values)]
        
        return rec_full, env_lower, env_upper, len(v_min) * 2, rng

    # --- MODE D: JINA NATIVE BINARY ---
    def compress_binary(self, values):
        """
        Jina v3 Native Binary Quantization.
        Werte > 0 werden 1, sonst 0.
        Gepackt als Bits (8 Werte pro Byte).
        Ratio: 32x.
        """
        # 1. Binarisierung (True/False)
        bits = values > 0
        
        # 2. Packing (8 Bool -> 1 Byte)
        # Wir müssen sicherstellen, dass die Länge durch 8 teilbar ist oder gepaddet wird,
        # np.packbits macht das Padding am Ende automatisch mit 0.
        packed_bytes = np.packbits(bits)
        
        # 3. Speicherbedarf
        size_bytes = len(packed_bytes)
        
        # 4. Rekonstruktion für Visualisierung/Cosine Sim
        # In der Praxis nutzt man Hamming Distance.
        # Für Cosine Sim Simulation wandeln wir Bits zurück in floats:
        # 1 -> +1.0
        # 0 -> -1.0
        # (Das ist der Standard-Weg, um Binary Embeddings mit Cosine Sim zu vergleichen)
        
        # Unpacken
        unpacked_bits = np.unpackbits(packed_bytes)[:len(values)]
        
        # Mapping 0->-1, 1->1
        # Formel: 2*x - 1  (bei 0: -1, bei 1: 1)
        rec_vals = (unpacked_bits.astype(np.float32) * 2) - 1
        
        # Wir skalieren die Visualisierung auf die durchschnittliche Amplitude des Originals,
        # damit man es im Graphen besser sieht (rein optisch, für Mathe irrelevant)
        scale = np.mean(np.abs(values))
        rec_vals_visual = rec_vals * scale
        
        return rec_vals_visual, size_bytes, rec_vals

comp = CompressorLogic()

# =========================================================
# 2. STREAMLIT APP
# =========================================================
st.set_page_config(layout="wide", page_title="Embedding Compressor Lab")
st.title("🧪 Embedding Compression Workbench")

# --- SIDEBAR ---
api_url = st.sidebar.text_input("API URL", "http://localhost:8202/embeddings")

st.sidebar.markdown("---")
st.sidebar.header("Methode")
mode = st.sidebar.radio("Modus", ["Smart Slope (Linear)", "Block Envelope (Min/Max)", "Jina Native Binary (32x)"])

st.sidebar.markdown("---")
st.sidebar.header("Parameter")

params = {}
if mode == "Smart Slope (Linear)":
    params['rel_eps'] = st.sidebar.slider("Relative Epsilon (%)", 0.0, 0.5, 0.08, 0.01)
    params['slope_tol'] = st.sidebar.slider("Slope Tolerance", 0.0, 2.0, 0.40, 0.05)
    params['rel_dist'] = st.sidebar.slider("Max Merge Dist (%)", 0.0, 0.5, 0.08, 0.01)
    params['min_noise'] = st.sidebar.slider("Noise Floor (Abs)", 0.0, 0.2, 0.03, 0.01)
elif mode == "Block Envelope (Min/Max)":
    params['block_size'] = st.sidebar.slider("Block Größe", 2, 32, 4)
else:
    # JINA BINARY
    st.sidebar.info("Keine Parameter nötig.\nWendet Standard Binarisierung (>0) an.\nRatio fix: 32x")

def fetch_embeddings(text, url):
    try:
        payload = {"input": text, "model": "jina-v3"}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and len(data) > 0 and "embedding" in data[0]: return np.array(data[0]["embedding"])
        arr = np.array(data)
        if arr.ndim == 3: return arr[0]
        if arr.ndim == 2: return arr
        return None
    except Exception as e:
        st.error(f"API Fehler: {e}"); return None

default_text = "Jina v3 ist das erste Modell, das speziell für binäre Matryoshka-Embeddings trainiert wurde."
text = st.text_area("Eingabetext", default_text, height=70)

if text and api_url:
    embeddings = fetch_embeddings(text, api_url)
    if embeddings is not None:
        n_words, n_dims = embeddings.shape
        matrix = embeddings
        labels = [f"Tok {i}" for i in range(n_words)]

        col1, col2 = st.columns([3, 1])
        with col2:
            st.subheader("Stats")
            selected_dim = st.number_input("Dimension", 0, n_dims-1, 42)
            vals = matrix[:, selected_dim]
            orig_bytes = len(vals) * 4
            
            if mode == "Smart Slope (Linear)":
                res = comp.compress_dimension_slope(vals, params['rel_eps'], params['slope_tol'], params['rel_dist'], params['min_noise'])
                winner = "Slope" if res['slope_bytes'] < res['int8_bytes'] else "Int8"
                rec_vals = res['slope_data'] if winner == "Slope" else res['int8_data']
                used_bytes = res['slope_bytes'] if winner == "Slope" else res['int8_bytes']
                st.metric("Ratio", f"{orig_bytes/used_bytes:.1f}x", winner)
            
            elif mode == "Block Envelope (Min/Max)":
                rec_vals, env_low, env_high, used_bytes, _ = comp.compress_minmax_envelope(vals, params['block_size'])
                st.metric("Ratio", f"{orig_bytes/used_bytes:.1f}x", f"Blk-{params['block_size']}")
                
            else: # JINA BINARY
                rec_vals, used_bytes, _ = comp.compress_binary(vals)
                st.metric("Ratio", f"{orig_bytes/used_bytes:.1f}x", "Binary")
                st.caption("1 Bit pro Dimension")

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=vals, mode='lines', name='Original', line=dict(color='gray', width=2), opacity=0.4))
            
            if mode == "Block Envelope (Min/Max)":
                fig.add_trace(go.Scatter(x=np.concatenate([np.arange(len(env_high)), np.arange(len(env_high))[::-1]]), y=np.concatenate([env_high, env_low[::-1]]), fill='toself', fillcolor='rgba(255,0,0,0.2)', line=dict(color='rgba(0,0,0,0)'), name='Envelope'))
                fig.add_trace(go.Scatter(y=rec_vals, mode='lines', name='Reconstruction', line=dict(color='red', width=2)))
            
            elif mode == "Jina Native Binary (32x)":
                # Step-Line für Binary
                fig.add_trace(go.Scatter(y=rec_vals, mode='lines', line_shape='hv', name='Binary State (+/-)', line=dict(color='green', width=2)))
                # Zero Line
                fig.add_hline(y=0, line_dash="dash", line_color="black")
            
            else:
                fig.add_trace(go.Scatter(y=rec_vals, mode='lines+markers', name='Reconstruction', line=dict(color='red', width=2)))

            fig.update_layout(title=f"Dimension {selected_dim}", height=450, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        if st.button(f"🚀 Starte Matrix Benchmark ({mode})"):
            with st.spinner("Rechne..."):
                rec_matrix = np.zeros_like(matrix)
                total_bytes = 0
                
                for d in range(n_dims):
                    v = matrix[:, d]
                    if mode == "Smart Slope (Linear)":
                        r = comp.compress_dimension_slope(v, params['rel_eps'], params['slope_tol'], params['rel_dist'], params['min_noise'])
                        if r['slope_bytes'] < r['int8_bytes']: total_bytes += r['slope_bytes']; rec_matrix[:, d] = r['slope_data']
                        else: total_bytes += r['int8_bytes']; rec_matrix[:, d] = r['int8_data']
                    
                    elif mode == "Block Envelope (Min/Max)":
                        r_d, _, _, b, _ = comp.compress_minmax_envelope(v, params['block_size'])
                        rec_matrix[:, d] = r_d; total_bytes += b
                    
                    else: # Binary
                        _, b, r_raw = comp.compress_binary(v)
                        rec_matrix[:, d] = r_raw; total_bytes += b
                
                ratio = (n_words * n_dims * 4) / total_bytes
                
                # Bei Binary berechnen wir Cosine Sim zwischen Float-Original und (+1/-1)-Rekonstruktion.
                # Das ist mathematisch sehr nah an der Hamming Distance Performance.
                sim = torch.mean(F.cosine_similarity(torch.tensor(matrix), torch.tensor(rec_matrix), dim=1)).item()
                mse = F.mse_loss(torch.tensor(matrix), torch.tensor(rec_matrix)).item()
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Global Ratio", f"{ratio:.2f}x")
                c2.metric("Cos Sim", f"{sim:.5f}")
                c3.metric("MSE", f"{mse:.5f}")
                
                if mode == "Jina Native Binary (32x)":
                    if sim > 0.90: st.success("Für 32x Kompression ist das ein Top-Wert!")
                    st.info("Hinweis: In der Praxis nutzt man Hamming-Distance für Speed.")
