# debug_full.py
import os
import numpy as np
import tensorflow as tf
from llm_shaping import compute_potential, shaped_reward, llm_logs
from env import EnvTimeSeriesfromRepo

# 1) Clear logs up front
llm_logs.clear()
print(">>> [DEBUG] llm_logs cleared:", llm_logs)

# 2) Canary: one manual shaping call on zeros
n_steps = 25
zeros = (0.0,)*n_steps
phi_zero = compute_potential(zeros)
print(f">>> [DEBUG] φ(zeros) = {phi_zero}, logs now:", llm_logs)

# 3) Wrap your env to inject debug‐prints at each shaping call
def rewardfnc_debug(ts, tc, a):
    raw = shaped_reward(
        raw_reward=0.0,            # just to exercise compute_potential
        s=ts['value'][tc-n_steps:tc].values,
        s2=ts['value'][tc-n_steps+1:tc+1].values,
        gamma=0.5
    )
    print(f">>> [DEBUG] rewardfnc_debug called @ cursor={tc}, shaped={raw:.3f}")
    return [raw, raw]

# 4) Build env, assign the debug rewardfnc, and step through a few calls
data_dir = os.path.join(os.getcwd(), "ydata-labeled-time-series-anomalies-v1_0", "A1Benchmark")
env = EnvTimeSeriesfromRepo(data_dir)
env.rewardfnc = rewardfnc_debug
env.timeseries_curser = n_steps

# perform a handful of steps to populate llm_logs
for _ in range(5):
    action = np.random.choice([0,1])
    _, reward, done, _ = env.step(action)
    if done:
        break

print(">>> [DEBUG] After 5 steps, llm_logs has", len(llm_logs), "entries")

# 5) Write out the CSV
out_csv = "debug_llm_potentials.csv"
with open(out_csv, "w") as f:
    f.write("window,phi\n")
    for win, phi in llm_logs:
        f.write(f"\"{','.join(f'{v:.2f}' for v in win)}\",{phi}\n")

print(f">>> [DEBUG] Wrote {len(llm_logs)} rows to {out_csv}")
