import os
import numpy as np
import pandas as pd


def predict_split(env_test, policy, out_csv, n_steps=25, log_every=100):
    rows = []
    total = len(env_test.repodirext)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    for idx, path in enumerate(env_test.repodirext):
        stem = os.path.splitext(os.path.basename(path))[0]
        pid, piece = stem.rsplit("_p", 1)
        raw = pd.read_csv(path)

        state = env_test.reset_to(idx)
        preds = []
        while True:
            probs = policy(state, 0)
            action = int(np.argmax(probs))
            preds.append(action)
            next_state, _, done, _ = env_test.step(action)
            if done:
                break
            state = next_state[action]

        n = len(raw)
        if len(preds) != n - n_steps:
            raise RuntimeError("%s: got %d predictions for %d rows (expected %d)"
                               % (stem, len(preds), n, n - n_steps))
        pred = np.concatenate([np.zeros(n_steps, dtype=int), np.array(preds, dtype=int)])
        scored = np.concatenate([np.zeros(n_steps, dtype=int), np.ones(len(preds), dtype=int)])
        rows.append(pd.DataFrame({"id": pid, "piece": int(piece), "datetime": raw["datetime"],
                                  "value": raw["value"], "is_anomaly": raw["is_anomaly"],
                                  "pred": pred, "scored": scored}))
        if (idx + 1) % log_every == 0 or idx + 1 == total:
            print("predicted %d of %d series" % (idx + 1, total), flush=True)
            pd.concat(rows).to_csv(out_csv, index=False)
    out = pd.concat(rows)
    out.to_csv(out_csv, index=False)
    print("saved", out_csv, "rows:", len(out))
    return out
