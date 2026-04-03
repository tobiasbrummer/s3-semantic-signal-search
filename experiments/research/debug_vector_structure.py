
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)

text = "A B C"
toks = model.tokenizer.tokenize(text)
ids = model.tokenizer(text)['input_ids']
vecs = model.encode([text], return_colbert_vecs=True)['colbert_vecs'][0]

print(f"Text: '{text}'")
print(f"Tokens ({len(toks)}): {toks}")
print(f"IDs ({len(ids)}): {ids}")
print(f"Vecs ({len(vecs)}): Shape {vecs.shape}")

# Vergleich der Normen (um CLS/SEP zu finden)
import numpy as np
norms = np.linalg.norm(vecs, axis=1)
print(f"Norms: {norms}")
