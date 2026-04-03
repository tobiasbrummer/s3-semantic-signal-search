
import numpy as np
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)

# Testsequenz: A A B
# Tokens: A, A, B
# IDs: CLS, A, A, B, SEP
# Vecs: 4 Stück. 
# Hypothese 1 (CLS fehlt): A, A, B, SEP -> V0=A, V1=A -> Sim=1.0
# Hypothese 2 (SEP fehlt): CLS, A, A, B -> V1=A, V2=A -> Sim=1.0

text = "A A B"
vecs = model.encode([text], return_colbert_vecs=True)['colbert_vecs'][0]

# Cosine Sim zwischen Vektor 0 und 1
sim01 = np.dot(vecs[0], vecs[1])
# Cosine Sim zwischen Vektor 1 und 2
sim12 = np.dot(vecs[1], vecs[2])

print(f"Vecs Shape: {vecs.shape}")
print(f"Sim(0, 1): {sim01:.4f}")
print(f"Sim(1, 2): {sim12:.4f}")

if sim01 > 0.99:
    print(">> V0 und V1 sind identisch (A, A). CLS fehlt!")
elif sim12 > 0.99:
    print(">> V1 und V2 sind identisch (A, A). SEP fehlt (oder CLS ist da)!")
else:
    print(">> Unklar.")
