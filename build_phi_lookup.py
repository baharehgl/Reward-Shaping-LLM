import os, pickle
import numpy as np
import pandas as pd
from llm_shaping import compute_potential  # this is your existing LLM function

n_steps = 25
data_dir = os.path.join(os.path.dirname(__file__),
                        "ydata-labeled-time-series-anomalies-v1_0", "A1Benchmark")

# 1) Gather all windows (quantized to 2 decimals)
lookup = {}
for fname in os.listdir(data_dir):
    path = os.path.join(data_dir, fname)
    df = pd.read_csv(path)
    vals = df['value'].values
    for t in range(n_steps, len(vals)):
        w = tuple(np.round(vals[t-n_steps:t], 2))
        lookup.setdefault(w, None)

# 2) Compute φ exactly once per unique window
for w in lookup:
    lookup[w] = compute_potential(w)

# 3) Save to disk
with open("phi_lookup.pkl", "wb") as f:
    pickle.dump(lookup, f)

print(f"✅ Precomputed φ for {len(lookup)} windows → phi_lookup.pkl")