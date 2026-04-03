
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)
text = "Test"
out = model.encode([text], return_colbert_vecs=True)
vecs = out['colbert_vecs'][0]
toks = model.tokenizer(text)['input_ids']
print(f"Vecs: {len(vecs)}, Toks: {len(toks)}")
