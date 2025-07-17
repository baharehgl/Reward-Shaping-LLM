# env_wadi.py  – compatible with myasp-wadi.py
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np


def default_state_fnc(ts, cursor):
    """Return the action-stacked window defined in make_state."""
    raise RuntimeError("statefnc must be set by the main script")


def default_reward_fnc(ts, cursor, action):
    return 1 if action == ts["anomaly"][cursor] else -1


class EnvTimeSeriesWaDi:
    def __init__(self, sensor_csv, label_csv, n_steps):
        self._load_sensor(sensor_csv)
        self._attach_labels(label_csv)

        self.timeseries_repo        = [self._ts]
        self.timeseries_curser_init = n_steps
        self.statefnc               = default_state_fnc
        self.rewardfnc              = default_reward_fnc
        self.action_space_n         = 2

    # ------------------------- helpers --------------------------------------
    def _load_sensor(self, path):
        df = pd.read_csv(path, decimal=".")
        df.columns = df.columns.str.strip()
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)
        if "Row" in df.columns:
            df = df.drop(columns=["Row"])
        self._sensor = df

    def _attach_labels(self, label_csv):
        lbl = pd.read_csv(label_csv, header=1, low_memory=False)
        raw = lbl["Attack LABLE (1:No Attack, -1:Attack)"].astype(int).values
        anomalies = np.where(raw == 1, 0, 1)
        L = min(len(self._sensor), len(anomalies))
        ts = self._sensor.iloc[:L].copy()
        ts["anomaly"] = anomalies[:L]
        ts["label"]   = -1
        self._ts = ts

    # ------------------------- RL interface ---------------------------------
    def reset(self):
        self.timeseries        = self.timeseries_repo[0]
        self.timeseries_curser = self.timeseries_curser_init

        # full_state has shape (2, 25, 124); keep variant 0
        full_state             = self.statefnc(self.timeseries,
                                               self.timeseries_curser)
        self.timeseries_states = full_state[0] \
            if isinstance(full_state, np.ndarray) and full_state.ndim > 2 \
            else full_state

        self.states_list = self._precompute_states()
        return self.timeseries_states

    def _precompute_states(self):
        out = []
        for c in range(self.timeseries_curser_init, len(self.timeseries)):
            s = self.statefnc(self.timeseries, c)
            s = s[0] if isinstance(s, np.ndarray) and s.ndim > 2 else s
            out.append(s)
        return out

    def step(self, action):
        r = self.rewardfnc(self.timeseries, self.timeseries_curser, action)
        self.timeseries_curser += 1
        done = int(self.timeseries_curser >= len(self.timeseries))

        if done:
            state = np.array([self.timeseries_states, self.timeseries_states])
        else:
            full_state = self.statefnc(self.timeseries, self.timeseries_curser)
            state      = full_state[action] \
                         if isinstance(full_state, np.ndarray) \
                         and full_state.ndim > 2 else full_state

        self.timeseries_states = state
        return state, r, done, {}
