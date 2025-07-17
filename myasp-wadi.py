
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

# ───────────────────────────────────────────────────────────────────────────────
# A) PRETRAIN VAE (EAGER MODE)
# ───────────────────────────────────────────────────────────────────────────────
from tensorflow.keras import layers, models, losses
from sklearn.preprocessing import RobustScaler

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(__file__)
WA_DI  = os.path.join(BASE, "WaDi")
SENSOR = os.path.join(WA_DI, "WADI_14days_new.csv")
LABEL  = os.path.join(WA_DI, "WADI_attackdataLABLE.csv")

# ─── 1) Load + clean sensor data ───────────────────────────────────────────────
df = pd.read_csv(SENSOR, decimal='.')
df.columns = df.columns.str.strip()

# Convert every column to numeric (coerce) and drop any row that has even one NaN
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna(axis=1, how='all')   # drop columns that are entirely NaN
df = df.dropna(axis=0, how='any')   # **drop any row that has any NaN**
df = df.reset_index(drop=True)

# Drop “Row” if present
if 'Row' in df.columns:
    df = df.drop(columns=['Row'])

print(f"[DATA] Kept {len(df.columns)} numeric sensor columns; {len(df)} time‐steps after dropping all‐NaN rows.")

# ─── 2) Load + align anomaly labels ─────────────────────────────────────────────
lbl_df       = pd.read_csv(LABEL, header=1, low_memory=False)
raw_lbl      = lbl_df["Attack LABLE (1:No Attack, -1:Attack)"].astype(int).values
anomaly_full = np.where(raw_lbl == 1, 0, 1)  # 1→No Attack→0,  -1→Attack→1

# Truncate both to the same minimum length
min_len = min(len(df), len(anomaly_full))
df = df.iloc[:min_len].reset_index(drop=True)
anomaly = anomaly_full[:min_len]
print(f"[DATA] Truncated series + labels to length = {min_len}")

# Build TS = sensors + “anomaly” + “label” (initially unlabeled = −1)
TS = df.copy()
TS["anomaly"] = anomaly
TS["label"]   = -1
print(f"[DATA] TS.shape = {TS.shape} (sensors + anomaly + label)")

feature_cols_full = df.columns.tolist()
n_features_full   = len(feature_cols_full)

# ─── 3) Sample up to MAX_VAE_SAMPLES “normal” windows ──────────────────────────
N_STEPS         = 25
MAX_VAE_SAMPLES = 200

# Find time‐indices c where anomaly[c] == 0 (and c ≥ N_STEPS)
normal_indices = np.where(TS["anomaly"].values[N_STEPS:] == 0)[0] + N_STEPS
sample_idx     = np.random.choice(
    normal_indices,
    size=min(MAX_VAE_SAMPLES, len(normal_indices)),
    replace=False
)
print(f"[VAE] Sampling {len(sample_idx)} normal windows of length {N_STEPS}")

# Build an array of shape (num_samples, 25, n_features_full)
temp = np.array([
    TS.iloc[i - N_STEPS + 1 : i + 1][feature_cols_full].values
    for i in sample_idx
], dtype="float32")  # (num_samples, 25, n_features_full)

# Drop any sensor column whose variance over these sampled windows is zero
sensor_var = temp.var(axis=(0,1))
keep_sensors = sensor_var > 0
feature_cols = [
    feature_cols_full[i]
    for i in range(n_features_full)
    if keep_sensors[i]
]
n_features = len(feature_cols)
print(f"[VAE] Dropped {n_features_full - n_features} zero‐var sensors; now n_features = {n_features}")

# Flatten each sampled window into shape (25 * n_features,)
windows = np.array([
    TS.iloc[i - N_STEPS + 1 : i + 1][feature_cols].values.flatten()
    for i in sample_idx
], dtype="float32")  # → (num_samples, 25*n_features)

INPUT_DIM = windows.shape[1]
print(f"[VAE] Flattened‐window INPUT_DIM = {INPUT_DIM}")

# Use RobustScaler (instead of StandardScaler) to avoid overflow when later transforming outlier windows
scaler = RobustScaler().fit(windows)
Xs     = scaler.transform(windows)
print(f"[VAE] Xs.shape = {Xs.shape}\n")

# ─── 4) Build & train the VAE ─────────────────────────────────────────────────
def build_vae(input_dim, hidden=64, latent=10):
    x_in = layers.Input((input_dim,))
    h    = layers.Dense(hidden, activation="relu")(x_in)
    z_mu = layers.Dense(latent)(h)
    z_lv = tf.clip_by_value(layers.Dense(latent)(h), -10.0, 10.0)
    z    = layers.Lambda(
        lambda t: t[0] + tf.exp(0.5 * t[1]) * tf.random.normal(tf.shape(t[0]))
    )([z_mu, z_lv])
    encoder = models.Model(x_in, [z_mu, z_lv, z], name="encoder")

    z_in  = layers.Input((latent,))
    dh    = layers.Dense(hidden, activation="relu")(z_in)
    x_out = layers.Dense(input_dim, activation="sigmoid")(dh)
    decoder = models.Model(z_in, x_out, name="decoder")

    recon = decoder(z)
    vae   = models.Model(x_in, recon, name="vae")
    rl = losses.mse(x_in, recon) * input_dim
    kl = -0.5 * tf.reduce_sum(1 + z_lv - tf.square(z_mu) - tf.exp(z_lv), axis=1)
    vae.add_loss(tf.reduce_mean(rl + kl))
    vae.compile(optimizer="adam")
    return vae, encoder, decoder

vae, encoder, decoder = build_vae(INPUT_DIM, hidden=128, latent=10)
vae.summary()

vae.fit(Xs, epochs=20, batch_size=32, verbose=1)
print("[VAE] Pretraining complete\n")

# ─── 5) Compute per‐step reconstruction error over entire TS (with RobustScaler) ──
print("[VAE] Computing per‐step reconstruction error…")
all_windows = np.array([
    TS.iloc[i - N_STEPS + 1 : i + 1][feature_cols].values.flatten()
    for i in range(N_STEPS - 1, len(TS))
], dtype="float32")  # → (len(TS)-(N_STEPS-1), 25*n_features)

# Scale them with the SAME RobustScaler, then clip to [-10, +10]
all_windows_scaled = scaler.transform(all_windows)
all_windows_scaled = np.clip(all_windows_scaled, -10.0, 10.0)

batch_size = 64
errs = []
for start in range(0, len(all_windows_scaled), batch_size):
    chunk = all_windows_scaled[start : start + batch_size]
    pred  = vae.predict(chunk, batch_size=chunk.shape[0], verbose=0)
    errs.append(np.mean((pred - chunk) ** 2, axis=1))

recon_err     = np.concatenate(errs, axis=0)
penalty_array = np.concatenate([np.zeros(N_STEPS - 1), recon_err])
print(f"[VAE] penalty_array ready (length = {len(penalty_array)})")
print(f"[VAE] stats: min={np.nanmin(penalty_array):.3e}, "
      f"max={np.nanmax(penalty_array):.3e}, mean={np.nanmean(penalty_array):.3e}\n")

# ───────────────────────────────────────────────────────────────────────────────
# B) RL TRAINING & VALIDATION (TF‐1 GRAPH MODE)
# ───────────────────────────────────────────────────────────────────────────────
tf.compat.v1.disable_eager_execution()
K = tf.compat.v1.keras.backend

from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── 6) Mini‐EnvWaDi ───────────────────────────────────────────────────────────
class EnvWaDi:
    """
    Environment for WADI.  Keeps:
      • self.timeseries: a pd.DataFrame with columns = feature_cols + ['anomaly','label']
      • self.t = current time index
      • statefnc, rewardfnc provided by caller
    reset() → returns state[s=0] of shape (25, n_features+1)
    step(action) → (two_states, [r0, r1], done, {})
    """
    def __init__(self, timeseries_df, statefnc, rewardfnc):
        # Make a fresh copy
        self.timeseries     = timeseries_df.copy().reset_index(drop=True)
        self.N              = len(self.timeseries)
        self.statefnc       = statefnc
        self.rewardfnc      = rewardfnc
        self.t0             = N_STEPS
        self.action_space_n = 2

    def reset(self):
        self.t = self.t0
        # Precompute a flattened “action=0” version of every future window (for AL+LP)
        self.states_list = []
        for c in range(self.t0, self.N):
            two_s = self.statefnc(self.timeseries, c)
            if two_s is not None:
                flat0 = two_s[0].flatten()
                self.states_list.append(flat0)
        # Return just the (action=0) slice at t=t0
        return self.statefnc(self.timeseries, self.t)[0]

    def step(self, action):
        r    = self.rewardfnc(self.timeseries, self.t, action)
        self.t += 1
        done = int(self.t >= self.N)
        if not done:
            two_s = self.statefnc(self.timeseries, self.t)
        else:
            # If done, return last‐possible window
            two_s = self.statefnc(self.timeseries, self.N - 1)
        return two_s, r, done, {}

# ─── 7) Q‐Network (TF‐1 style) ────────────────────────────────────────────────
class QNet:
    def __init__(self, scope):
        self.sc = scope

    def build(self):
        with tf.compat.v1.variable_scope(self.sc):
            self.S = tf.compat.v1.placeholder(
                tf.float32, [None, N_STEPS, n_features + 1], name="S"
            )
            self.T = tf.compat.v1.placeholder(
                tf.float32, [None, 2], name="T"
            )
            cell = tf.compat.v1.nn.rnn_cell.LSTMCell(64)
            seq, _ = tf.compat.v1.nn.static_rnn(
                cell, tf.compat.v1.unstack(self.S, N_STEPS, axis=1), dtype=tf.float32
            )
            self.Q    = layers.Dense(2)(seq[-1])
            self.loss = tf.reduce_mean(tf.square(self.Q - self.T))
            self.train = tf.compat.v1.train.AdamOptimizer(3e-4).minimize(self.loss)

    def predict(self, x, sess):
        return sess.run(self.Q, {self.S: x})

    def update(self, x, y, sess):
        sess.run(self.train, {self.S: x, self.T: y})

def copy_params(sess, src: QNet, dst: QNet):
    sv = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, src.sc
    )
    dv = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, dst.sc
    )
    for s_var, d_var in zip(
        sorted(sv, key=lambda v: v.name),
        sorted(dv, key=lambda v: v.name)
    ):
        sess.run(d_var.assign(s_var))

# ─── 8) State & Reward using penalty_array ────────────────────────────────────
def make_state(ts_df, c):
    if c < N_STEPS:
        return None
    W  = ts_df[feature_cols].iloc[c - N_STEPS + 1 : c + 1].values.astype("float32")
    a0 = np.concatenate([W, np.zeros((N_STEPS, 1), dtype="float32")], axis=1)
    a1 = np.concatenate([W, np.ones ((N_STEPS, 1), dtype="float32")], axis=1)
    return np.stack([a0, a1], axis=0)

def reward_fn(ts_df, c, a, coef):
    if c < N_STEPS:
        return [0.0, 0.0]
    pen = coef * float(penalty_array[c])
    lbl = ts_df["label"].iat[c]
    base = [TN, FP] if lbl == 0 else [FN, TP]
    return [base[0] + pen, base[1] + pen]

# ─── 9) Full train + equal-slice validate ─────────────────────────────────────
def train_and_validate(AL_budget):
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    K.set_session(sess)

    # Build Q networks
    ql = QNet("ql"); ql.build()
    qt = QNet("qt"); qt.build()
    sess.run(tf.compat.v1.global_variables_initializer())

    env = EnvWaDi(TS, make_state, lambda ts, cc, aa: [0.0, 0.0])
    from collections import namedtuple
    Transition = namedtuple("T", ["s", "r", "ns", "d"])
    memory, coef = [], 20.0

    # 9A) Warm-up IsolationForest on known-normal windows
    W0 = []
    env.reset()
    for ii in range(N_STEPS, len(env.timeseries)):
        if env.timeseries["label"].iat[ii] == 0:
            W0.append(
                env.timeseries[feature_cols]
                .iloc[ii - N_STEPS + 1 : ii + 1]
                .values.flatten()
            )
    if W0:
        from sklearn.ensemble import IsolationForest
        IsolationForest(contamination=0.01).fit(
            np.array(W0[:MAX_VAE_SAMPLES], dtype="float32")
        )

    # 9B) RL episodes
    for ep in range(1, EPISODES + 1):
        # Step(1): Label-Prop + Active-Learn
        env.reset()
        labs = np.array(env.timeseries["label"].iloc[N_STEPS :])
        if np.any(labs != -1):
            Warr = np.array(env.states_list)
            flat = Warr.reshape(Warr.shape[0], -1)
            lp   = LabelSpreading(kernel="knn", n_neighbors=10).fit(flat, labs)
            uncert = 1.0 - lp.label_distributions_.max(axis=1)
            idx    = np.argsort(-uncert)
            # first AL_budget get true anomaly
            for i in idx[:AL_budget]:
                env.timeseries["label"].iat[i + N_STEPS] = env.timeseries["anomaly"].iat[i + N_STEPS]
            # next NUM_LP get LP’s pseudo
            for i in idx[AL_budget : AL_budget + NUM_LP]:
                env.timeseries["label"].iat[i + N_STEPS] = lp.transduction_[i]

        # Step(2): Rollout with ε-greedy
        env.rewardfnc = lambda ts, cc, aa: reward_fn(ts, cc, aa, coef)
        s, done = env.reset(), False
        eps     = max(0.1, 1.0 - float(ep) / EPISODES)
        ep_reward = 0.0

        while not done:
            if random.random() < eps:
                a = random.choice([0, 1])
            else:
                two_s = make_state(env.timeseries, env.t)
                s0    = two_s[0]; s1 = two_s[1]
                q0    = ql.predict([s0], sess)[0]
                q1    = ql.predict([s1], sess)[0]
                a     = 1 if q1[1] > q0[0] else 0

            two_s, r, done, _ = env.step(a)
            ns = two_s[a] if not done else two_s[0]
            memory.append(Transition(s, r, ns, done))
            ep_reward += float(r[a])
            s = ns

        # Step(3): Replay updates (5 mini-batches)
        for _ in range(5):
            batch = random.sample(memory, min(BATCH_SIZE, len(memory)))
            S_batch, R_batch, NS_batch, _ = map(np.array, zip(*batch))
            qn     = qt.predict(NS_batch, sess)
            qn_max = np.max(qn, axis=1, keepdims=True)
            tgt    = R_batch + DISCOUNT * np.repeat(qn_max, 2, axis=1)
            ql.update(S_batch, tgt.astype("float32"), sess)

        copy_params(sess, ql, qt)

        # Step(4): Update dynamic reward coefficient
        coef = max(min(coef + 0.0005 * ep_reward, 5.0), 0.1)
        print(f"[train AL={AL_budget}] ep{ep:02d}/{EPISODES}  coef={coef:.2f}  reward={ep_reward:.2f}")

    # ─── 10) Equal-slice validation ───────────────────────────────────────────
    base_ts = TS.copy()
    seg     = len(base_ts) // K_SLICES
    outdir  = f"validation_AL{AL_budget}"
    os.makedirs(outdir, exist_ok=True)
    f1s, aus = [], []

    for i in range(K_SLICES):
        chunk_df = base_ts.iloc[i * seg : (i + 1) * seg].reset_index(drop=True)
        envv = EnvWaDi(chunk_df, make_state, lambda ts, cc, aa: [0.0, 0.0])
        s, done = envv.reset(), False
        P, G, V = [], [], []

        while not done:
            two_s = make_state(envv.timeseries, envv.t)
            s0    = two_s[0]; s1 = two_s[1]
            q0    = ql.predict([s0], sess)[0]
            q1    = ql.predict([s1], sess)[0]
            a     = 1 if q1[1] > q0[0] else 0

            P.append(a)
            G.append(envv.timeseries["anomaly"].iat[envv.t])
            V.append(s[-1][0])
            nxt, _, done, _ = envv.step(a)
            s = nxt[a] if not done else nxt[0]

        p, r, f1, _ = precision_recall_fscore_support(G, P, average="binary", zero_division=0)
        au          = average_precision_score(G, P)
        f1s.append(f1); aus.append(au)

        prefix = f"{outdir}/slice_{i}"
        np.savetxt(prefix + ".txt", [p, r, f1, au], fmt="%.6f")
        fig, ax = plt.subplots(4, sharex=True, figsize=(8,6))
        ax[0].plot(V);      ax[0].set_title("Time Series")
        ax[1].plot(P, "g"); ax[1].set_title("Predictions")
        ax[2].plot(G, "r"); ax[2].set_title("Ground Truth")
        ax[3].plot([au]*len(V), "m"); ax[3].set_title("AUPR")
        plt.tight_layout(); plt.savefig(prefix + ".png"); plt.close(fig)

        print(f"[val AL={AL_budget}] slice {i+1}/{K_SLICES}   F1={f1:.3f}   AU={au:.3f}")

    print(f"[val AL={AL_budget}] mean F1={np.mean(f1s):.3f}   mean AUPR={np.mean(aus):.3f}\n")


# ───────────────────────────────────────────────────────────────────────────────
# C) Driver loop
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Hyperparameters (you can lower EPISODES or BATCH_SIZE if it’s still slow)
    EPISODES   = 20     # RL episodes per AL‐budget
    BATCH_SIZE = 64      # minibatch size for replay
    DISCOUNT   = 0.5
    TN, TP, FP, FN = 1, 10, -1, -10
    NUM_LP     = 200     # number of pseudo‐labels each episode
    K_SLICES   = 5       # 3‐fold equal‐slice cross‐validation

    for AL in [1000, 5000, 10000]:
        print(f"\n=== ACTIVE LEARNING BUDGET: {AL} ===")
        train_and_validate(AL)
