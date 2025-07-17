# myasp-KPI.py

import os
import sys
import random

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import layers, models, losses

# ──────────── 1) HYPERPARAMETERS ────────────────────────────────────
n_steps                   = 25
n_input_dim               = 2
n_hidden_dim              = 128
validation_separate_ratio = 0.9

EPISODES      = 100
EPOCHES       = 5
VAL_EVERY     = 10
PATIENCE      = 2
NUM_AL        = 10
DISCOUNT      = 0.96

TN, TP, FP, FN = 1, 5, -1, -5
action_space   = [0, 1]

# ──────────── 2) VAE PRETRAINING ────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
NORMAL_DATA_DIR = os.path.join(current_dir, "normal-data")

latent_dim, inter_dim = 10, 64

def load_normal_data(path, n):
    windows = []
    for f in os.listdir(path):
        if not f.endswith('.csv'): continue
        vals = pd.read_csv(os.path.join(path, f))['value'].values
        if len(vals) < n: continue
        for i in range(len(vals)-n+1):
            windows.append(vals[i:i+n])
    return StandardScaler().fit_transform(np.array(windows))

class Sampling(layers.Layer):
    def call(self, inputs):
        m, lv = inputs
        eps = tf.keras.backend.random_normal(tf.shape(m))
        return m + tf.exp(0.5*lv)*eps

def build_and_train_vae():
    x_in = layers.Input((n_steps,))
    h = layers.Dense(inter_dim, activation='relu')(x_in)
    h = layers.Dense(inter_dim, activation='relu')(h)
    m = layers.Dense(latent_dim)(h)
    lv = layers.Dense(latent_dim)(h)
    lv = tf.clip_by_value(lv, -10.0, 10.0)
    z = Sampling()([m, lv])
    d = layers.Dense(inter_dim, activation='relu')(z)
    x_dec = layers.Dense(n_steps, activation='sigmoid')(d)

    vae = models.Model(x_in, x_dec)
    recon = losses.mse(x_in, x_dec) * n_steps
    kl    = -0.5 * tf.reduce_sum(1 + lv - tf.square(m) - tf.exp(lv), axis=-1)
    vae.add_loss(tf.reduce_mean(recon + kl))
    vae.compile(optimizer='adam')

    X = load_normal_data(NORMAL_DATA_DIR, n_steps)
    vae.fit(X, epochs=10, batch_size=32, verbose=1)
    return vae

vae_model = build_and_train_vae()


# ──────────── 3) KPI ENV + STATE/REWARD ─────────────────────────────
sys.path.append(current_dir)
from env_KPI import EnvKPI

def state_fn(ts, cur, prev=None, action=None):
    if cur < n_steps:
        return None
    if cur == n_steps:
        s = [[ts['value'].iat[i], 0] for i in range(n_steps)]
        s.pop(0); s.append([ts['value'].iat[cur], 1])
        return np.array(s, 'float32')
    s0 = np.concatenate((prev[1:], [[ts['value'].iat[cur], 0]]))
    s1 = np.concatenate((prev[1:], [[ts['value'].iat[cur], 1]]))
    return np.array([s0, s1], 'float32')

def reward_fn(ts, cur, action, lam):
    window = np.array([ts['value'].iloc[cur-n_steps:cur]])
    recon  = vae_model.predict(window)
    err    = np.mean((recon - window)**2)
    pen    = lam * err
    lbl    = ts['label'].iat[cur]
    if lbl == 0:
        return [TN + pen, FP + pen]
    else:
        return [FN + pen, TP + pen]

def reward_fn_test(ts, cur, action=0):
    an = ts['anomaly'].iat[cur]
    if an == 0:
        return [TN, FP]
    else:
        return [FN, TP]


# ──────────── 4) Q-NET + POLICY + ACTIVE LEARNING ───────────────────
class QNet:
    def __init__(self, lr, scope, sess):
        self.sess  = sess
        self.scope = scope
        with tf.compat.v1.variable_scope(scope):
            self.state  = tf.compat.v1.placeholder(tf.float32, [None, n_steps, n_input_dim], "state")
            self.target = tf.compat.v1.placeholder(tf.float32, [None, 2],               "target")
            cell = tf.compat.v1.nn.rnn_cell.LSTMCell(n_hidden_dim)
            outs, _ = tf.compat.v1.nn.dynamic_rnn(cell, self.state, dtype=tf.float32)
            last    = outs[:, -1, :]
            self.q   = layers.Dense(2)(last)
            self.loss= tf.reduce_mean(tf.square(self.q - self.target))
            self.train = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)

    def predict(self, s):
        return self.sess.run(self.q, {self.state: s})

    def update(self, s, t):
        self.sess.run(self.train, {self.state: s, self.target: t})

def make_policy(est):
    def policy_fn(obs, eps):
        A = np.ones(2)*(eps/2)
        q = est.predict([obs])[0]
        A[np.argmax(q)] += 1 - eps
        return A
    return policy_fn

class ActiveLearner:
    def __init__(self, env, N, est):
        self.env = env
        self.N   = N
        self.est = est

    def select(self):
        dists = []
        for st in self.env.states_list:
            q = self.est.predict([st])[0]
            dists.append(abs(q[0] - q[1]))
        idx = np.argsort(dists)
        return idx[:self.N].tolist()


# ──────────── 5) TRAINING LOOP w/ VALIDATION & EARLY STOP ───────────
def train_with_validation(train_csv, test_csv):
    # build environment
    env = EnvKPI(train_csv, test_csv)
    env.statefnc = state_fn

    total = env.datasetsize
    cut   = int(total * validation_separate_ratio)
    train_ids = list(range(cut))
    val_ids   = list(range(cut, total))

    # validation environment
    env_val = EnvKPI(train_csv, test_csv)
    env_val.statefnc  = state_fn
    env_val.rewardfnc = reward_fn_test

    # create TF session (VAE already in graph)
    sess = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(sess)

    q_net = QNet(3e-4, "q",   sess)
    tgt   = QNet(3e-4, "tgt", sess)
    sess.run(tf.compat.v1.global_variables_initializer())

    best_f1, no_imp, lam = 0.0, 0, 10.0
    reward_hist, lam_hist = [], []
    policy = make_policy(q_net)

    for ep in range(1, EPISODES + 1):
        # sample a KPI for training
        ki = random.choice(train_ids)
        env.rewardfnc = lambda ts, c, a: reward_fn(ts, c, a, lam)
        state = env.reset(to_idx=ki)
        env.states_list = env.get_states_list()

        # active learning
        al = ActiveLearner(env, NUM_AL, q_net)
        for idx in al.select():
            pos = idx + n_steps
            env.timeseries['label'].iat[pos] = env.timeseries['anomaly'].iat[pos]

        # collect memory over sliding windows
        memory = []
        for t, st in enumerate(env.states_list):
            eps = max(0.1, 1 - ep/EPISODES)
            a   = np.random.choice(2, p=policy(st, eps))
            r   = env.rewardfnc(env.timeseries, t + n_steps, a)
            memory.append((st, r, st, False))

        # train Q-net
        for _ in range(EPOCHES):
            batch = random.sample(memory, min(len(memory), 64))
            S, R, NS, _ = map(np.array, zip(*batch))
            qn = tgt.predict(NS)
            mx = np.max(qn, axis=1)
            tgt_vals = R + DISCOUNT * mx[:, None]
            q_net.update(S, tgt_vals)

        # sync target network
        if ep % 5 == 0:
            vq = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "q")
            vt = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "tgt")
            for x, y in zip(vq, vt):
                sess.run(y.assign(x))

        # logging & update lam
        total_r = sum(r[a] for (s, r, ns, d) in memory for a in [0])
        reward_hist.append(total_r)
        lam = max(0.1, min(10.0, lam + 0.001 * (0.0 - total_r)))
        lam_hist.append(lam)

        # validation
        if ep % VAL_EVERY == 0:
            f1s = []
            for vid in val_ids:
                env_val.reset(to_idx=vid)
                preds, truths = [], []
                for t, st in enumerate(env_val.get_states_list()):
                    a = np.argmax(policy(st, 0.0))
                    preds.append(a)
                    truths.append(env_val.timeseries['anomaly'].iat[t + n_steps])
                _, _, f1, _ = precision_recall_fscore_support(truths, preds, average='binary', zero_division=0)
                f1s.append(f1)
            val_f1 = np.mean(f1s)
            print(f"[VAL] Ep {ep}, F1 = {val_f1:.4f}")
            if val_f1 > best_f1:
                best_f1, no_imp = val_f1, 0
            else:
                no_imp += 1
                if no_imp >= PATIENCE:
                    print(f"Early stopping at episode {ep}")
                    break

    # save plots
    os.makedirs("exp", exist_ok=True)
    plt.figure(); plt.plot(reward_hist); plt.title("Reward"); plt.savefig("exp/reward.png"); plt.close()
    plt.figure(); plt.plot(lam_hist);    plt.title("Lambda"); plt.savefig("exp/lambda.png"); plt.close()

    print("Best validation F1:", best_f1)

if __name__ == "__main__":
    train_csv = os.path.join(current_dir, "KPI_data", "train", "phase2_train.csv")
    test_csv  = os.path.join(current_dir, "KPI_data", "test",  "phase2_ground_truth.csv")
    train_with_validation(train_csv, test_csv)
