import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import fitz

def slerp(v0, v1, t):
    dot = torch.dot(v0, v1)
    if dot > 0.9995: return F.normalize((1.0 - t) * v0 + t * v1, p=2, dim=-1)
    theta_0 = torch.acos(torch.clamp(dot, -1.0, 1.0))
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    return (torch.sin(theta_0 - theta_t) / sin_theta_0) * v0 + (sin_theta_t / sin_theta_0) * v1

class DriftValidator:
    def __init__(self, model_name="ibm-granite/granite-embedding-278m-multilingual", device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def validate(self, pdf_path):
        doc = fitz.open(pdf_path)
        text = " ".join([page.get_text() for page in doc])
        inputs = self.tokenizer(text, return_tensors="pt")
        all_ids = inputs.input_ids[0]
        
        # Nahtstelle bei Token 256
        # Fenster 1: 0 bis 384
        # Fenster 2: 128 bis 512
        
        with torch.no_grad():
            emb1 = self.model(input_ids=all_ids[0:384].unsqueeze(0).to(self.device)).last_hidden_state[0]
            emb2 = self.model(input_ids=all_ids[128:512].unsqueeze(0).to(self.device)).last_hidden_state[0]
        
        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)
        
        # Token bei Position 256 absolut
        abs_idx = 256
        rel_idx1 = 256
        rel_idx2 = 256 - 128
        
        v1 = emb1[rel_idx1]
        v2 = emb2[rel_idx2]
        v_stitched = slerp(v1, v2, 0.5)
        
        token_str = self.tokenizer.convert_ids_to_tokens([all_ids[abs_idx]])[0]
        print(f"\n🔍 DRIFT-ANALYSE FÜR TOKEN AN POSITION {abs_idx} ('{token_str}')")
        print("=" * 70)
        
        sim_raw = torch.dot(v1, v2).item()
        print(f"Ähnlichkeit Fenster 1 vs Fenster 2 (Roh-Drift): {sim_raw:.4f}")
        print(f"Abstand V1 -> SLERP-Mitte: {torch.dot(v1, v_stitched).item():.4f}")
        print(f"Abstand V2 -> SLERP-Mitte: {torch.dot(v2, v_stitched).item():.4f}")

if __name__ == "__main__":
    if torch.backends.mps.is_available(): DEVICE = "mps"
    elif torch.cuda.is_available(): DEVICE = "cuda"
    else: DEVICE = "cpu"
    DriftValidator(device=DEVICE).validate("contract.pdf")