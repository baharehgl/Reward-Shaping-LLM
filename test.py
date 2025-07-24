# debug_full.py
import os, numpy as np, tensorflow as tf
from llm_shaping import compute_potential, shaped_reward, llm_logs
from env import EnvTimeSeriesfromRepo

# 1) clear logs
llm_logs.clear()

# 2) canary shaping call
n_steps = 25
zeros = (0.0,)*n_steps
print(">>> [DEBUG] φ(zeros) =", compute_potential(zeros))
print(">>> [DEBUG] logs now:", llm_logs)

# 3) debug reward function
def rewardfnc_debug(ts, tc, a):
    # ts is a pandas DataFrame once reset() has been called
    raw  = shaped_reward(
        raw_reward=0.0,
        s=ts['value'][tc-n_steps:tc].values,
        s2=ts['value'][tc-n_steps+1:tc+1].values,
        gamma=0.5
    )
    print(f">>> [DEBUG] rewardfnc_debug @ cursor={tc} → {raw:.3f}")
    return [raw, raw]

# 4) build and reset your env
data_dir = os.path.join(os.getcwd(),
    "ydata-labeled-time-series-anomalies-v1_0", "A1Benchmark")
env = EnvTimeSeriesfromRepo(data_dir)
env.timeseries_curser_init = n_steps   # start at 25
env.rewardfnc = rewardfnc_debug

# **THIS IS THE KEY LINE**: initialize timeseries, timeseries_curser, etc.
state = env.reset()

# 5) step a few times
for i in range(5):
    action = np.random.choice([0,1])
    next_state, reward, done, _ = env.step(action)
    if done:
        break

print(">>> [DEBUG] After 5 steps, logs =", len(llm_logs))

# 6) write out CSV just like before…
out_csv = "debug_llm_potentials.csv"
with open(out_csv, "w") as f:
    f.write("window,phi\n")
    for win, phi in llm_logs:
        f.write(f"\"{','.join(f'{v:.2f}' for v in win)}\",{phi}\n")
print(f">>> [DEBUG] wrote {len(llm_logs)} rows to {out_csv}")