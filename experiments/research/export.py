# import numpy as np
import json

# Encoder
from spectral_plot import TokenEncoder
encoder = TokenEncoder()

with open("/var/home/t0bybr/containers/s3/research/code/S3_semantic_signal_search.md", "r") as f:
    document = f.read()

# Embeddings holen
embeddings = encoder.encode(document)

with open("embeddings.json", "w") as f:
    json.dump(embeddings.tolist(), f)

#   # Speichern als .npy (kompakt, schnell)
#   np.save("dokument_embeddings.npy", embeddings)

#   # Laden
#   loaded = np.load("dokument_embeddings.npy")

#   # Oder mit Metadaten als .npz
#   np.savez("dokument.npz",
#       embeddings=embeddings,
#       text=text,
#       n_tokens=len(embeddings)
#   )

  # Laden
#   data = np.load("dokument.npz", allow_pickle=True)
#   embeddings = data["embeddings"]
