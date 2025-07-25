# build_phi_lookup.py
import os, pickle, numpy as np, pandas as pd
from llm_shaping import compute_potential  # still uses live API here

n_steps    = 25
model_name = os.getenv("LLM_CHOICE", "gpt-3.5-turbo") #"gpt-4-0613", "llama-3"
lookup_fn  = f"phi_lookup_{model_name}.pkl"

data_dir = os.path.join(os.path.dirname(__file__),
                        "ydata-labeled-time-series-anomalies-v1_0",
                        "A1Benchmark")

# 1) collect windows
lookup = {}
for fname in os.listdir(data_dir):
    df   = pd.read_csv(os.path.join(data_dir, fname))
    vals = df['value'].values
    for t in range(n_steps, len(vals)):
        w = tuple(np.round(vals[t - n_steps:t], 2))
        lookup.setdefault(w, None)

# 2) call the live compute_potential once per w
for w in lookup:
    lookup[w] = compute_potential(w)

# 3) dump to model-specific file
with open(lookup_fn, "wb") as f:
    pickle.dump(lookup, f)

print(f"✅ Saved lookup for {model_name} → {lookup_fn}")