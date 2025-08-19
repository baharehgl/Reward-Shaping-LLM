import matplotlib
matplotlib.use('Agg')
import pandas as pd
import itertools
import random
import sys
import time
from scipy import stats
import tensorflow as tf

import os, textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

tf.compat.v1.disable_eager_execution()

from mpl_toolkits.mplot3d import axes3d
from collections import deque, namedtuple
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Append current directory to sys.path so local modules can be imported.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the environment.
from env import EnvTimeSeriesfromRepo
from sklearn.svm import OneClassSVM
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

#from llm_shaping import shaped_reward, llm_logs
import importlib
import llm_shaping
importlib.reload(llm_shaping)
from llm_shaping import compute_potential, shaped_reward, llm_logs

# Canary tests: check φ(s) values before training
print(">>> φ(zeros)      =", compute_potential((0.0,)*25))
print(">>> φ(spike@last) =", compute_potential(tuple([0.0]*24 + [10.0])))
print("Last logs:", llm_logs[-2:])


compute_potential.cache_clear()
llm_logs.clear()

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

############################
# Macros and Hyperparameters.
DATAFIXED = 0  # whether target is fixed to a single time series
EPISODES = 3  # number of episodes (for demonstration)
DISCOUNT_FACTOR = 0.5  # reward discount factor
EPSILON = 0.5  # epsilon-greedy parameter
EPSILON_DECAY = 1.00  # epsilon decay

# Extrinsic reward values (heuristic):
TN_Value = 1  # True Negative
TP_Value = 5  # True Positive
FP_Value = -1  # False Positive
FN_Value = -5  # False Negative

NOT_ANOMALY = 0
ANOMALY = 1
action_space = [NOT_ANOMALY, ANOMALY]
action_space_n = len(action_space)

n_steps = 25  # sliding window length
n_input_dim = 2  # dimension of input to LSTM
n_hidden_dim = 128  # hidden dimension

validation_separate_ratio = 0.9

# Canary: should print exactly one LLM CALL
print(">>> TRAIN CANARY: φ(zeros) =", compute_potential((0.0,)*n_steps))
print(">>> TRAIN CANARY: after call llm_logs =", llm_logs)
llm_logs.clear()
print(f"[TRAIN] starting run, llm_logs cleared → length={len(llm_logs)}")

def plot_phi_histogram(experiment_dir):
    import pandas as pd
    import matplotlib.pyplot as plt
    csv_path = os.path.join(experiment_dir, "llm_potentials.csv")
    if not os.path.exists(csv_path):
        print("No llm_potentials.csv found in", experiment_dir)
        return
    df = pd.read_csv(csv_path)
    plt.figure()
    plt.hist(df['phi'], bins=50)
    plt.title("Distribution of LLM Potentials Φ(s)")
    plt.xlabel("Φ(s)")
    plt.ylabel("Count")
    out = os.path.join(experiment_dir, "plots", "phi_histogram.svg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, format="svg")
    plt.close()




########################### VAE Setup #####################
def load_normal_data(data_path, n_steps):
    all_files = [os.path.join(data_path, fname) for fname in os.listdir(data_path) if fname.endswith('.csv')]
    windows = []
    for file in all_files:
        df = pd.read_csv(file)
        if 'value' not in df.columns:
            continue
        values = df['value'].values
        if len(values) >= n_steps:
            for i in range(len(values) - n_steps + 1):
                window = values[i:i + n_steps]
                windows.append(window)
    windows = np.array(windows)
    scaler = StandardScaler()
    scaled_windows = scaler.fit_transform(windows)
    return scaled_windows


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_vae(original_dim, latent_dim=2, intermediate_dim=64):
    inputs = layers.Input(shape=(original_dim,))
    h = layers.Dense(intermediate_dim, activation='relu', kernel_initializer='he_normal')(inputs)
    h = layers.Dense(intermediate_dim, activation='relu', kernel_initializer='he_normal')(h)
    h = layers.Dense(intermediate_dim, activation='relu', kernel_initializer='he_normal')(h)
    z_mean = layers.Dense(latent_dim, kernel_initializer='he_normal')(h)
    z_log_var = layers.Dense(latent_dim, kernel_initializer='he_normal')(h)
    z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
    z = Sampling()([z_mean, z_log_var])
    decoder_h = layers.Dense(intermediate_dim, activation='relu', kernel_initializer='he_normal')
    h_decoded = decoder_h(z)
    decoder_mean = layers.Dense(original_dim, activation='sigmoid')
    x_decoded_mean = decoder_mean(h_decoded)
    encoder = models.Model(inputs, [z_mean, z_log_var, z])
    vae = models.Model(inputs, x_decoded_mean)
    reconstruction_loss = losses.mse(inputs, x_decoded_mean) * original_dim
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, clipnorm=1.0)
    vae.compile(optimizer=optimizer)
    return vae, encoder


original_dim = n_steps
latent_dim = 10
intermediate_dim = 64

vae, encoder = build_vae(original_dim, latent_dim, intermediate_dim)


#####################################################
# State and Reward Functions.



'''
def RNNBinaryStateFuc(timeseries, timeseries_curser, previous_state=[], action=None):
    if timeseries_curser == n_steps:
        state = []
        for i in range(timeseries_curser):
            state.append([timeseries['value'][i], 0])
        state.pop(0)
        state.append([timeseries['value'][timeseries_curser], 1])
        return np.array(state, dtype='float32')
    if timeseries_curser > n_steps:
        state0 = np.concatenate((previous_state[1:n_steps],
                                 [[timeseries['value'][timeseries_curser], 0]]))
        state1 = np.concatenate((previous_state[1:n_steps],
                                 [[timeseries['value'][timeseries_curser], 1]]))
        return np.array([state0, state1], dtype='float32')
    return None
'''
def RNNBinaryStateFuc(timeseries, timeseries_curser, previous_state=None, action=None):
    """
    Builds the next state (sliding window) for the RNN-based agent.

    - For timeseries_curser < n_steps: pads with zeros to length n_steps.
    - At timeseries_curser == n_steps: returns the first full window (shape [n_steps, 2]).
    - For timeseries_curser > n_steps: returns two candidate windows (state0, state1) of shape [n_steps, 2],
      stacked into a [2, n_steps, 2] array, where the last row flag is 0 or 1.
    """
    # Assume global n_steps is defined elsewhere
    # 1) Warm-up: not enough history → pad
    if timeseries_curser < n_steps:
        pad_len = n_steps - timeseries_curser
        pad = np.zeros((pad_len, 2), dtype='float32')
        hist = np.array([
            [timeseries['value'][i], 0]
            for i in range(timeseries_curser)
        ], dtype='float32')
        return np.vstack([pad, hist])

    # 2) Exactly full window: build a single [n_steps,2] array
    if timeseries_curser == n_steps:
        window = []
        for i in range(timeseries_curser):
            flag = 1 if i == timeseries_curser - 1 else 0
            window.append([timeseries['value'][i], flag])
        return np.array(window, dtype='float32')

    # 3) After warm-up: sliding window with two action-based variants
    # previous_state should be an array of shape [n_steps, 2]
    base = np.array(previous_state[1:n_steps], dtype='float32')
    # Append next value with flag 0 or 1
    state0 = np.vstack([base, [timeseries['value'][timeseries_curser], 0]])
    state1 = np.vstack([base, [timeseries['value'][timeseries_curser], 1]])
    # Return shape [2, n_steps, 2]
    return np.array([state0, state1], dtype='float32')


def RNNBinaryRewardFuc(timeseries, timeseries_curser, action=0, vae=None, dynamic_coef=1.0):
    if timeseries_curser >= n_steps:
        current_state = np.array([timeseries['value'][timeseries_curser - n_steps:timeseries_curser]])
        vae_reconstruction = vae.predict(current_state)
        reconstruction_error = np.mean(np.square(vae_reconstruction - current_state))
        #vae_penalty = - dynamic_coef * reconstruction_error
        vae_penalty = dynamic_coef * reconstruction_error
        if timeseries['label'][timeseries_curser] == 0:
            return [TN_Value + vae_penalty, FP_Value + vae_penalty]
        elif timeseries['label'][timeseries_curser] == 1:
            return [FN_Value + vae_penalty, TP_Value + vae_penalty]
        else:
            return [0, 0]
    else:
        return [0, 0]


def RNNBinaryRewardFucTest(timeseries, timeseries_curser, action=0):
    if timeseries_curser >= n_steps:
        if timeseries['anomaly'][timeseries_curser] == 0:
            return [TN_Value, FP_Value]
        elif timeseries['anomaly'][timeseries_curser] == 1:
            return [FN_Value, TP_Value]
    return [0, 0]


# ----------------------------
# Q-value Function Approximator.
class Q_Estimator_Nonlinear():
    def __init__(self, learning_rate=np.float32(0.01), scope="Q_Estimator_Nonlinear", summaries_dir=None):
        self.scope = scope
        self.summary_writer = None
        with tf.compat.v1.variable_scope(scope):
            self.state = tf.compat.v1.placeholder(shape=[None, n_steps, n_input_dim],
                                                  dtype=tf.float32, name="state")
            self.target = tf.compat.v1.placeholder(shape=[None, action_space_n],
                                                   dtype=tf.float32, name="target")
            self.weights = {'out': tf.Variable(tf.compat.v1.random_normal([n_hidden_dim, action_space_n]))}
            self.biases = {'out': tf.Variable(tf.compat.v1.random_normal([action_space_n]))}
            self.state_unstack = tf.unstack(self.state, n_steps, 1)
            lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(n_hidden_dim, forget_bias=1.0)
            self.outputs, self.states = tf.compat.v1.nn.static_rnn(lstm_cell, self.state_unstack, dtype=tf.float32)
            self.action_values = tf.matmul(self.outputs[-1], self.weights['out']) + self.biases['out']
            self.losses = tf.compat.v1.squared_difference(self.action_values, self.target)
            self.loss = tf.reduce_mean(self.losses)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            global_step_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="global_step")[0]
            self.train_op = self.optimizer.minimize(self.loss, global_step=global_step_var)
            self.summaries = tf.compat.v1.summary.merge([
                tf.compat.v1.summary.histogram("loss_hist", self.losses),
                tf.compat.v1.summary.scalar("loss", self.loss),
                tf.compat.v1.summary.histogram("q_values_hist", self.action_values),
                tf.compat.v1.summary.scalar("q_value", tf.reduce_max(self.action_values))
            ])
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.compat.v1.summary.FileWriter(summary_dir)

    def predict(self, state, sess=None):
        sess = sess or tf.compat.v1.get_default_session()
        return sess.run(self.action_values, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.compat.v1.get_default_session()
        feed_dict = {self.state: state, self.target: target}
        summaries, global_step, _ = sess.run([self.summaries,
                                              tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                                          scope="global_step")[0],
                                              self.train_op], feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return


def copy_model_parameters(sess, estimator1, estimator2):
    e1_params = sorted([t for t in tf.compat.v1.trainable_variables() if t.name.startswith(estimator1.scope)],
                       key=lambda v: v.name)
    e2_params = sorted([t for t in tf.compat.v1.trainable_variables() if t.name.startswith(estimator2.scope)],
                       key=lambda v: v.name)
    for e1_v, e2_v in zip(e1_params, e2_params):
        sess.run(e2_v.assign(e1_v))


# --- Modified make_epsilon_greedy_policy: pass sess explicitly.
def make_epsilon_greedy_policy(estimator, nA, sess):
    def policy_fn(observation, epsilon):
        A = np.ones(nA, dtype='float32') * epsilon / nA
        q_values = estimator.predict(state=[observation], sess=sess)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


# Proportional update for dynamic coefficient.
def update_dynamic_coef_proportional(current_coef, episode_reward, target_reward=100.0, alpha=0.01, min_coef=0.1,
                                     max_coef=10.0):
    new_coef = current_coef + alpha * (target_reward - episode_reward)
    return max(min(new_coef, max_coef), min_coef)


# --- Updated active_learning class ---
class active_learning(object):
    def __init__(self, env, N, strategy, estimator, already_selected):
        self.env = env
        self.N = N
        self.strategy = strategy
        self.estimator = estimator
        self.already_selected = already_selected

    def get_samples(self):
        states_list = self.env.states_list
        distances = []
        for state in states_list:
            q = self.estimator.predict(state=[state])[0]
            distances.append(abs(q[0] - q[1]))
        distances = np.array(distances)
        rank_ind = np.argsort(distances)
        rank_ind = [i for i in rank_ind if i < len(states_list) and i not in self.already_selected]
        return rank_ind[:self.N]

    def get_samples_by_score(self, threshold):
        states_list = self.env.states_list
        distances = []
        for state in states_list:
            q = self.estimator.predict(state=[state])[0]
            distances.append(abs(q[0] - q[1]))
        distances = np.array(distances)
        rank_ind = np.argsort(distances)
        rank_ind = [i for i in rank_ind if i < len(states_list) and i not in self.already_selected]
        return [t for t in rank_ind if distances[t] < threshold]

    def label(self, active_samples):
        for sample in active_samples:
            print('AL finds one of the most confused samples:')
            print(self.env.timeseries['value'].iloc[sample:sample + n_steps])
            print('Please label the last timestamp (0 for normal, 1 for anomaly):')
            label = input()
            self.env.timeseries.loc[sample + n_steps - 1, 'anomaly'] = float(label)
        return


class WarmUp(object):
    def warm_up_SVM(self, outliers_fraction, N):
        states_list = self.env.get_states_list()
        data = np.array(states_list).transpose(2, 0, 1).reshape(2, -1)[0].reshape(-1, n_steps)[:, -1].reshape(-1, 1)
        model = OneClassSVM(gamma='auto', nu=0.95 * outliers_fraction)
        model.fit(data)
        distances = model.decision_function(data)
        rank_ind = np.argsort(np.abs(distances))
        samples = rank_ind[0:N]
        return samples

    def warm_up_isolation_forest(self, outliers_fraction, X_train):
        from sklearn.ensemble import IsolationForest
        X_train_arr = np.array(X_train)
        data = X_train_arr[:, -1].reshape(-1, 1)
        clf = IsolationForest(contamination=outliers_fraction)
        clf.fit(data)
        return clf


def q_learning(env, sess, qlearn_estimator, target_estimator, num_episodes, num_epoches,
               replay_memory_size=500000, replay_memory_init_size=50000, experiment_dir='./log/',
               update_target_estimator_every=10000, discount_factor=0.99,
               epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=500000, batch_size=256,
               num_LabelPropagation=20, num_active_learning=5, test=0, vae_model=None):
    Transition = namedtuple("Transition", ["state", "reward", "next_state", "done"])
    replay_memory = []
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.compat.v1.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
        if test:
            return
    global_step_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="global_step")
    if len(global_step_list) == 0:
        raise ValueError("global_step variable not found!")
    total_t = sess.run(global_step_list[0])
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    # Pass sess explicitly:
    policy = make_epsilon_greedy_policy(qlearn_estimator, env.action_space_n, sess)
    num_label = 0
    print('Warm up starting...')
    outliers_fraction = 0.01
    data_train = []
    for num in range(env.datasetsize):
        env.reset()
        env.states_list = [s for s in env.states_list if s is not None]
        data_train.extend(env.states_list)

    model_warm = WarmUp().warm_up_isolation_forest(outliers_fraction, data_train)
    lp_model = LabelSpreading()
    for t in itertools.count():
        env.reset()
        env.states_list = [s for s in env.states_list if s is not None]
        data = np.array(env.states_list).transpose(2, 0, 1).reshape(2, -1)[0].reshape(-1, n_steps)[:, -1].reshape(-1, 1)

        anomaly_score = model_warm.decision_function(data)
        pred_score = [-1 * s + 0.5 for s in anomaly_score]
        warm_samples = np.argsort(pred_score)[:5]
        warm_samples = np.append(warm_samples, np.argsort(pred_score)[-5:])
        state_list = np.array(env.states_list).transpose(2, 0, 1)[0]
        labeled_index = [i - n_steps for i in range(n_steps, len(env.timeseries['label'])) if
                         env.timeseries['label'][i] != -1]
        for sample in warm_samples:
            if sample < len(env.states_list):
                state = env.states_list[sample]
                env.timeseries_curser = sample + n_steps
                action_probs = policy(state, epsilons[min(total_t, epsilon_decay_steps - 1)])
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                env.timeseries['label'][env.timeseries_curser] = env.timeseries['anomaly'][env.timeseries_curser]
                num_label += 1
                labeled_index.append(sample)
                next_state, reward, done, _ = env.step(action)
                replay_memory.append(Transition(state, reward, next_state, done))
        label_list = [env.timeseries['label'][i] for i in range(n_steps, len(env.timeseries['label']))]
        label_list = np.array(label_list)
        lp_model.fit(state_list, label_list)
        pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)
        certainty_index = np.argsort(pred_entropies)
        certainty_index = [i for i in certainty_index if i not in labeled_index]
        certainty_index = certainty_index[:num_LabelPropagation]
        for index in certainty_index:
            pseudo_label = lp_model.transduction_[index]
            env.timeseries['label'][index + n_steps] = pseudo_label
        if len(replay_memory) >= replay_memory_init_size:
            break

    dynamic_coef = 10.0
    episode_rewards = []
    coef_history = []

    for i_episode in range(num_episodes):
        #env.rewardfnc = lambda ts, tc, a: RNNBinaryRewardFuc(ts, tc, a, vae_model, dynamic_coef=dynamic_coef)
        episode_reward = 0.0

        if i_episode % 50 == 49:
            print("Save checkpoint in episode {}/{}".format(i_episode + 1, num_episodes))
            saver.save(sess, checkpoint_path)

        per_loop_time1 = time.time()
        state = env.reset()
        env.states_list = [s for s in env.states_list if s is not None]
        while env.datasetidx > env.datasetrng * validation_separate_ratio:
            state = env.reset()
            env.states_list = [s for s in env.states_list if s is not None]
            print('double reset')
        labeled_index = [i - n_steps for i in range(n_steps, len(env.timeseries['label'])) if
                         env.timeseries['label'][i] != -1]
        al = active_learning(env=env, N=num_active_learning, strategy='margin_sampling',
                             estimator=qlearn_estimator, already_selected=labeled_index)
        al_samples = al.get_samples()
        print('labeling samples: ' + str(al_samples) + ' in env ' + str(env.datasetidx))
        labeled_index.extend(al_samples)
        num_label += len(al_samples)
        state_list = np.array(env.states_list).transpose(2, 0, 1)[0]
        label_list = [env.timeseries['label'][i] for i in range(n_steps, len(env.timeseries['label']))]
        label_list = np.array(label_list)
        for new_sample in al_samples:
            label_list[new_sample] = env.timeseries['anomaly'][new_sample + n_steps]
            env.timeseries['label'][new_sample + n_steps] = env.timeseries['anomaly'][new_sample + n_steps]
        for sample in labeled_index:
            if sample < len(env.states_list):
                env.timeseries_curser = sample + n_steps
                epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
                state = env.states_list[sample]
                action_probs = policy(state, epsilon)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                #next_state, reward, done, _ = env.step(action)
                print(f">>> DEBUG: stepping at cursor={env.timeseries_curser}")
                next_state, reward, done, info = env.step(action)
                print(f">>> DEBUG: raw reward returned = {reward}")
                episode_reward += reward[action]
                if len(replay_memory) == replay_memory_size:
                    replay_memory.pop(0)
                replay_memory.append(Transition(state, reward, next_state, done))
        unlabeled_indices = [i for i, e in enumerate(label_list) if e == -1]
        label_list = np.array(label_list)
        lp_model.fit(state_list, label_list)
        pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)
        certainty_index = np.argsort(pred_entropies)
        certainty_index = [i for i in certainty_index if i in unlabeled_indices]
        certainty_index = certainty_index[:num_LabelPropagation]
        for index in certainty_index:
            pseudo_label = lp_model.transduction_[index]
            env.timeseries['label'][index + n_steps] = pseudo_label
        per_loop_time2 = time.time()
        for i_epoch in range(num_epoches):
            if qlearn_estimator.summary_writer:
                episode_summary = tf.compat.v1.Summary()
                qlearn_estimator.summary_writer.add_summary(episode_summary, total_t)
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, qlearn_estimator, target_estimator)
                print("\nCopied model parameters to target network.\n")
            samples = random.sample(replay_memory, batch_size)
            states_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))
            if discount_factor > 0:
                next_states_batch = np.squeeze(np.split(next_states_batch, 2, axis=1))
                next_states_batch0 = next_states_batch[0]
                next_states_batch1 = next_states_batch[1]
                q_values_next0 = target_estimator.predict(state=next_states_batch0, sess=sess)
                q_values_next1 = target_estimator.predict(state=next_states_batch1, sess=sess)
                targets_batch = reward_batch + (discount_factor *
                                                np.stack((np.amax(q_values_next0, axis=1),
                                                          np.amax(q_values_next1, axis=1)),
                                                         axis=-1))
            else:
                targets_batch = reward_batch
            qlearn_estimator.update(state=states_batch, target=targets_batch.astype(np.float32), sess=sess)
            total_t += 1
        per_loop_time_popu = per_loop_time2 - per_loop_time1
        per_loop_time_updt = time.time() - per_loop_time2
        print("Global step {} @ Episode {}/{}, time: {} + {}".format(total_t, i_episode + 1, num_episodes,
                                                                     per_loop_time_popu, per_loop_time_updt))
        dynamic_coef = update_dynamic_coef_proportional(dynamic_coef, episode_reward, target_reward=0.0, alpha=0.001,
                                                        min_coef=0.1, max_coef=10.0)
        print("Episode {}: total reward = {:.3f}, updated dynamic_coef = {:.3f}".format(i_episode, episode_reward,
                                                                                        dynamic_coef))
        episode_rewards.append(episode_reward)
        coef_history.append(dynamic_coef)
    return episode_rewards, coef_history


def q_learning_validator(env, estimator, num_episodes, record_dir=None, plot=1):
    from sklearn.metrics import precision_recall_fscore_support
    rec_file = open(record_dir + 'performance.txt', 'w')
    precision_all, recall_all, f1_all = [], [], []
    for i_episode in range(num_episodes):
        print("Episode {}/{}".format(i_episode + 1, num_episodes))
        policy = make_epsilon_greedy_policy(estimator, env.action_space_n, tf.compat.v1.get_default_session())
        state = env.reset()
        env.states_list = [s for s in env.states_list if s is not None]
        while env.datasetidx < env.datasetrng * validation_separate_ratio:
            state = env.reset()
            env.states_list = [s for s in env.states_list if s is not None]
            print('double reset')
        print('testing on: ' + str(env.repodirext[env.datasetidx]))
        predictions = []
        ground_truths = []
        ts_values = []
        for t in itertools.count():
            action_probs = policy(state, 0)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            predictions.append(action)
            # Retrieve ground truth from the environment.
            current_index = env.timeseries_curser
            if hasattr(env.timeseries['anomaly'], 'iloc'):
                ground_truth = env.timeseries['anomaly'].iloc[current_index]
            else:
                ground_truth = env.timeseries['anomaly'][current_index]
            ground_truths.append(ground_truth)
            ts_values.append(state[len(state) - 1][0])
            next_state, reward, done, _ = env.step(action)
            if done:
                break
            state = next_state[action]
        precision, recall, f1, _ = precision_recall_fscore_support(ground_truths, predictions, average='binary', zero_division=0)
        precision_all.append(precision)
        recall_all.append(recall)
        f1_all.append(f1)
        print("Episode {}: Precision:{}, Recall:{}, F1-score:{}".format(i_episode+1, precision, recall, f1))
        rec_file.write("Episode {}: Precision:{}, Recall:{}, F1-score:{}\n".format(i_episode+1, precision, recall, f1))
        if plot:
            '''
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(ts_values, label="Original", linewidth=1)
            ax.plot(predictions, label="Forecast", linewidth=1)  
            ax.scatter(np.where(ground_truths)[0],
                       np.array(ts_values)[np.where(ground_truths)[0]],
                       marker='x', label="True anomaly", c='tab:green')
            ax.scatter(np.where(predictions)[0],
                       np.array(ts_values)[np.where(predictions)[0]],
                       marker='o', label="Detected anomaly", c='tab:gray')
            ax.set_title("Your series name")
            ax.set_xlabel("timestamp")
            ax.set_ylabel("value")
            ax.legend(loc="upper left")
            plt.savefig(os.path.join(record_dir, "combined_plot.svg"), format="svg")
            plt.close()
            '''

            def _merge_indices_to_spans(idxs, gap=1):
                """Merge consecutive indices into (start,end) inclusive spans."""
                idxs = np.asarray(sorted(set(int(i) for i in idxs)), dtype=int)
                if idxs.size == 0:
                    return []
                spans, s = [], idxs[0]
                e = s
                for x in idxs[1:]:
                    if x <= e + gap:
                        e = x
                    else:
                        spans.append((s, e))
                        s = e = x
                spans.append((s, e))
                return spans

            def _draw_truth_spans(ax, spans):
                for s, e in spans:
                    ax.axvspan(s - 0.5, e + 0.5, color="#2ca02c", alpha=0.25)  # green translucent

            def _draw_detected_spans(ax, spans, ts=None):
                # light gray translucent blocks + optional dots at the signal value
                for s, e in spans:
                    ax.axvspan(s - 0.5, e + 0.5, color="#7f7f7f", alpha=0.25)
                if ts is not None and len(spans) > 0:
                    centers = [int((s + e) / 2) for s, e in spans]
                    centers = [c for c in centers if 0 <= c < len(ts)]
                    if centers:
                        ax.scatter(centers, np.asarray(ts)[centers], s=18, color="#7f7f7f", zorder=3)

            def _auto_zoom_windows(true_spans, N, length, pad=200):
                """Pick up to N windows around the largest true anomaly spans; fallback windows if needed."""
                wins = []
                if true_spans:
                    sorted_spans = sorted(true_spans, key=lambda se: se[1] - se[0], reverse=True)
                    for s, e in sorted_spans[:N]:
                        a = max(0, s - pad)
                        b = min(length, e + pad)
                        wins.append((a, b))
                while len(wins) < N:
                    start = 0 if len(wins) == 0 else max(0, length // 2 - 300)
                    wins.append((start, min(length, start + min(1500, length))))
                return wins[:N]

            # Prepare series
            ts = np.array(ts_values, dtype=float)  # original signal
            gts = np.array(ground_truths, dtype=int)  # 0/1 ground-truth anomalies
            preds = np.array(predictions, dtype=int)  # 0/1 detected anomalies

            true_idx = np.where(gts == 1)[0]
            det_idx = np.where(preds == 1)[0]

            true_spans = _merge_indices_to_spans(true_idx)
            det_spans = _merge_indices_to_spans(det_idx)

            # Short, readable series label (avoid long path in title)
            raw_label = str(env.repodirext[env.datasetidx]) if hasattr(env, "repodirext") else "series"
            series_name = os.path.splitext(os.path.basename(raw_label))[0]
            series_name = textwrap.shorten(series_name, width=40, placeholder="…")

            # Figure (2x2) with auto layout
            fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharey='row', constrained_layout=True)
            (ax_full_truth, ax_full_both), (ax_zoom1, ax_zoom2) = axs

            # ── Top-left: Full series, truth only
            ax_full_truth.plot(ts, lw=1.2, color="#1f77b4")
            _draw_truth_spans(ax_full_truth, true_spans)
            ax_full_truth.set_title(f"{series_name} (truth)", fontsize=11)
            ax_full_truth.set_xlabel("timestamp");
            ax_full_truth.set_ylabel("value")

            # ── Top-right: Full series, truth + detected
            ax_full_both.plot(ts, lw=1.0, color="#1f77b4")
            _draw_truth_spans(ax_full_both, true_spans)
            _draw_detected_spans(ax_full_both, det_spans, ts=ts)
            ax_full_both.set_title(f"{series_name} (truth + detected)", fontsize=11)
            ax_full_both.set_xlabel("timestamp");
            ax_full_both.set_ylabel("value")

            # ── Bottom row: two zoomed windows
            wins = _auto_zoom_windows(true_spans, N=2, length=len(ts), pad=200)
            for ax, (a, b) in zip((ax_zoom1, ax_zoom2), wins):
                ax.plot(np.arange(a, b), ts[a:b], lw=1.2, color="#1f77b4")
                # clip spans to window
                clip_truth = [(max(a, s), min(b - 1, e)) for s, e in true_spans if e >= a and s < b]
                clip_det = [(max(a, s), min(b - 1, e)) for s, e in det_spans if e >= a and s < b]
                _draw_truth_spans(ax, clip_truth)
                _draw_detected_spans(ax, clip_det, ts=ts)
                ax.set_xlim(a, b)
                ax.set_title(f"zoom {a}–{b}", fontsize=11)
                ax.set_xlabel("timestamp");
                ax.set_ylabel("value")

            # Legend above the plots (no overlap with titles)
            legend_elems = [
                Line2D([0], [0], color="#1f77b4", lw=2, label="Original signal"),
                Patch(facecolor="#2ca02c", edgecolor="none", alpha=0.25, label="True anomaly"),
                Patch(facecolor="#7f7f7f", edgecolor="none", alpha=0.25, label="Detected anomaly"),
            ]
            fig.legend(handles=legend_elems, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))

            # If you still see crowding on some systems, slightly reduce top area:
            fig.subplots_adjust(top=0.90)

            # Save
            out = os.path.join(record_dir, f"detections_episode_{i_episode}.svg")
            plt.savefig(out, format="svg")
            plt.close(fig)
            # ─────────────────────────────────────────────────────────────────────────────
            # ─────────────────────────────────────────────────────────────────

    rec_file.close()
    avg_f1 = np.mean(f1_all)
    print("Average F1-score over {} episodes: {}".format(num_episodes, avg_f1))
    return avg_f1


def save_plots(experiment_dir, episode_rewards, coef_history):
    plot_dir = os.path.join(experiment_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # reward curve
    plt.figure()
    plt.plot(episode_rewards, marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Training Reward Curve")
    plt.savefig(os.path.join(plot_dir, "reward_curve.svg"), format="svg")
    plt.close()

    # lambda curve
    plt.figure()
    plt.plot(coef_history, marker='s')
    plt.xlabel("Episode")
    plt.ylabel("Dynamic Coefficient")
    plt.title("Lambda Evolution")
    plt.savefig(os.path.join(plot_dir, "lambda_curve.svg"), format="svg")
    plt.close()




def train_wrapper(num_LP, num_AL, discount_factor):
    """
    Trains the RL agent with LLM-based potential shaping and saves metrics & plots.
    Smoke-tests one shaping step before running full training.
    Returns final validation metric (e.g., F1-score).
    """
    # 1) Prepare and train VAE on normal data
    data_directory = os.path.join(current_dir, "normal-data")
    x_train = load_normal_data(data_directory, n_steps)
    vae, _ = build_vae(original_dim, latent_dim, intermediate_dim)
    vae.fit(x_train, epochs=2, batch_size=32)
    vae.save('vae_model.h5')

    # We'll run on 100% of data by default
    percentage = [1.0]
    test = 0

    for j, pct in enumerate(percentage):
        # Build experiment directory name
        exp_name = f"A1_LP{num_LP}_AL{num_AL}_d{discount_factor:.2f}"
        exp_relative_dir = [exp_name]
        dataset_dirs = [os.path.join(current_dir, "ydata-labeled-time-series-anomalies-v1_0", "A1Benchmark")]

        for i, ds_path in enumerate(dataset_dirs):
            # Instantiate environment
            env = EnvTimeSeriesfromRepo(ds_path)

            # Clear previous LLM cache & logs
            compute_potential.cache_clear()
            llm_logs.clear()

            # Initialize cursor and state function
            env.timeseries_curser_init = n_steps
            _ = env.reset()
            env.statefnc = RNNBinaryStateFuc

            # Assign reward function with correct gamma
            def shaping_fn(ts, tc, a):
                phi0 = shaped_reward(
                    raw_reward=RNNBinaryRewardFuc(ts, tc, 0, vae, dynamic_coef=10.0)[0],
                    s=ts['value'][tc - n_steps:tc].values,
                    s2=ts['value'][tc - n_steps + 1:tc + 1].values,
                    gamma=discount_factor
                )
                phi1 = shaped_reward(
                    raw_reward=RNNBinaryRewardFuc(ts, tc, 1, vae, dynamic_coef=10.0)[1],
                    s=ts['value'][tc - n_steps:tc].values,
                    s2=ts['value'][tc - n_steps + 1:tc + 1].values,
                    gamma=discount_factor
                )
                return [phi0, phi1]

            env.rewardfnc = shaping_fn

            # SMOKE TEST: confirm shaping fires and logs
            compute_potential.cache_clear()
            llm_logs.clear()

            # Force a valid cursor step
            env.timeseries_curser = n_steps
            ts_copy = env.timeseries.copy()
            r0 = env.rewardfnc(ts_copy, n_steps, 0)
            r1 = env.rewardfnc(ts_copy, n_steps, 1)
            print("🚨 SMOKE TEST shaped rewards:", r0, r1)
            print("🚨 SMOKE TEST llm_logs entries:", llm_logs)
            assert len(llm_logs) >= 2, "LLM shaping did not fire!"

            # Prepare test env (no shaping)
            env_test = EnvTimeSeriesfromRepo(ds_path)
            env_test.timeseries_curser_init = n_steps
            env_test.statefnc = RNNBinaryStateFuc
            env_test.rewardfnc = RNNBinaryRewardFucTest

            # Dataset split
            if test == 1:
                env.datasetrng = env.datasetsize
            else:
                env.datasetrng = int(env.datasetsize * pct)

            experiment_dir = os.path.abspath(os.path.join("./exp", exp_relative_dir[i]))

            # Reset TF graph & load VAE
            tf.compat.v1.reset_default_graph()
            vae = load_model('vae_model.h5', custom_objects={'Sampling': Sampling}, compile=False)
            sess = tf.compat.v1.Session()
            from tensorflow.compat.v1.keras import backend as K
            K.set_session(sess)
            global_step = tf.Variable(0, name="global_step", trainable=False)

            # Build estimators
            qlearn_estimator = Q_Estimator_Nonlinear(scope="qlearn",
                                                    summaries_dir=experiment_dir,
                                                    learning_rate=0.0003)
            target_estimator = Q_Estimator_Nonlinear(scope="target")
            sess.run(tf.compat.v1.global_variables_initializer())

            # Run training & validation
            with sess.as_default():
                episode_rewards, coef_history = q_learning(
                    env, sess, qlearn_estimator, target_estimator,

                    num_episodes=3,
                    num_epoches=10,
                    experiment_dir=experiment_dir,
                    replay_memory_size=500000,
                    replay_memory_init_size=1500,
                    update_target_estimator_every=10,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    discount_factor=discount_factor,
                    batch_size=256,
                    num_LabelPropagation=num_LP,
                    num_active_learning=num_AL,
                    test=test,
                    vae_model=vae
                )
                final_metric = q_learning_validator(
                    env_test, qlearn_estimator,
                    int(env.datasetsize * (1 - validation_separate_ratio)),
                    experiment_dir
                )

            # Save plots and logs
            save_plots(experiment_dir, episode_rewards, coef_history)
            import csv
            os.makedirs(experiment_dir, exist_ok=True)
            with open(os.path.join(experiment_dir, "llm_potentials.csv"), "w") as f:
                writer = csv.writer(f)
                writer.writerow(["window", "phi"])
                for win, phi in llm_logs:
                    writer.writerow([" ".join(f"{v:.2f}" for v in win), phi])

            plot_phi_histogram(experiment_dir)

            # ─── LLM Diagnostics ────────────────────────────────────────────
            import pandas as pd

            # Load potentials CSV
            df = pd.read_csv(os.path.join(experiment_dir, "llm_potentials.csv"))
            phis = df["phi"].values

            # 1) Φ(s) time series
            plt.figure()
            plt.plot(phis, marker='.')
            plt.title("LLM Potential Φ(s) over time")
            plt.xlabel("window index")
            plt.ylabel("Φ(s)")
            plt.savefig(os.path.join(experiment_dir, "plots", "phi_timeseries.svg"), format="svg")
            plt.close()

            # 2) Φ(s) distribution
            plt.figure()
            plt.hist(phis, bins=50, alpha=0.7)
            plt.title("Histogram of LLM Potentials Φ(s)")
            plt.xlabel("Φ(s)")
            plt.ylabel("Count")
            plt.savefig(os.path.join(experiment_dir, "plots", "phi_distribution.svg"), format="svg")
            plt.close()
            # ─────────────────────────────────────────────────────────────────

            return final_metric


'''
def train_wrapper(num_LP, num_AL, discount_factor):
    data_directory = os.path.join(current_dir, "normal-data")
    x_train = load_normal_data(data_directory, n_steps)
    vae, _ = build_vae(original_dim, latent_dim, intermediate_dim)
    vae.fit(x_train, epochs=2, batch_size=32)
    vae.save('vae_model.h5')
    percentage = [1]
    test = 0
    for j in range(len(percentage)):
        exp_relative_dir = ['A1_LP_1500init_warmup_h128_b256_300ep_num_LP' + str(num_LP) +
                            '_num_AL' + str(num_AL) + '_d' + str(discount_factor)]
        dataset_dir = [os.path.join(current_dir, "ydata-labeled-time-series-anomalies-v1_0", "A1Benchmark")]
        for i in range(len(dataset_dir)):
            env = EnvTimeSeriesfromRepo(dataset_dir[i])
            compute_potential.cache_clear()
            llm_logs.clear()
            env.timeseries_curser_init = n_steps
            _ = env.reset()
            print(">>> TRAIN CANARY (fresh): φ(zeros) =", compute_potential((0.0,) * n_steps))
            print(">>> TRAIN CANARY log entries:", llm_logs)
            llm_logs.clear()
            env.statefnc = RNNBinaryStateFuc
            #env.rewardfnc = lambda ts, tc, a: RNNBinaryRewardFuc(ts, tc, a, vae, dynamic_coef=10.0)

            # new: wrap both actions in potential‐based shaping
            env.rewardfnc = lambda ts, tc, a: [
                shaped_reward(
                    raw_reward=RNNBinaryRewardFuc(ts, tc, 0, vae, dynamic_coef=10.0)[0],
                    s=ts['value'][tc - n_steps:tc].values,
                    s2=ts['value'][tc - n_steps + 1:tc + 1].values,
                    gamma=DISCOUNT_FACTOR
                ),
                shaped_reward(
                    raw_reward=RNNBinaryRewardFuc(ts, tc, 1, vae, dynamic_coef=10.0)[1],
                    s=ts['value'][tc - n_steps:tc].values,
                    s2=ts['value'][tc - n_steps + 1:tc + 1].values,
                    gamma=DISCOUNT_FACTOR
                )
            ]
            env.timeseries_curser_init = n_steps
            env.datasetfix = DATAFIXED
            env.datasetidx = 0
            env_test = env
            env_test.rewardfnc = RNNBinaryRewardFucTest
            if test == 1:
                env.datasetrng = env.datasetsize
            else:
                env.datasetrng = np.int32(env.datasetsize * float(percentage[j]))
            experiment_dir = os.path.abspath("./exp/{}".format(exp_relative_dir[i]))

            tf.compat.v1.reset_default_graph()
            vae = load_model('vae_model.h5', custom_objects={'Sampling': Sampling}, compile=False)
            sess = tf.compat.v1.Session()
            from tensorflow.compat.v1.keras import backend as K
            K.set_session(sess)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            qlearn_estimator = Q_Estimator_Nonlinear(scope="qlearn", summaries_dir=experiment_dir, learning_rate=0.0003)
            target_estimator = Q_Estimator_Nonlinear(scope="target")
            sess.run(tf.compat.v1.global_variables_initializer())

            with sess.as_default():
                episode_rewards, coef_history = q_learning(env, sess, qlearn_estimator, target_estimator,
                                                           num_episodes=3, num_epoches=10,
                                                           experiment_dir=experiment_dir,
                                                           replay_memory_size=500000,
                                                           replay_memory_init_size=1500,
                                                           update_target_estimator_every=10,
                                                           epsilon_start=1,
                                                           epsilon_end=0.1,
                                                           epsilon_decay_steps=500000,
                                                           discount_factor=discount_factor,
                                                           batch_size=256,
                                                           num_LabelPropagation=num_LP,
                                                           num_active_learning=num_AL,
                                                           test=test,
                                                           vae_model=vae)
                final_metric = q_learning_validator(env_test, qlearn_estimator,
                                                    int(env.datasetsize * (1 - validation_separate_ratio)),
                                                    experiment_dir)
            save_plots(experiment_dir, episode_rewards, coef_history)
            # 1) Save all Φ(s) values for histogram
            import csv
            with open(os.path.join(experiment_dir, "llm_potentials.csv"), "w") as f:
                w = csv.writer(f)
                w.writerow(["window", "phi"])
                for win, phi in llm_logs:
                    w.writerow([",".join(f"{v:.2f}" for v in win), phi])

            # 2) plot the φ histogram
            plot_phi_histogram(experiment_dir)

            # 3) Save reward and lambda curves as SVG
            plt.figure();
            plt.plot(episode_rewards);
            plt.title("Reward Curve")
            plt.savefig(os.path.join(experiment_dir, "reward_curve.svg"), format="svg");
            plt.close()
            plt.figure();
            plt.plot(coef_history);
            plt.title("Lambda Curve")
            plt.savefig(os.path.join(experiment_dir, "lambda_curve.svg"), format="svg");
            plt.close()
            return final_metric
'''

#train_wrapper(100, 1000, 0.92)
#train_wrapper(150, 5000, 0.94)
#train_wrapper(200, 10000, 0.96)
train_wrapper(200, 1000, 0.96)
#train_wrapper(200, 5000, 0.96)
#train_wrapper(200, 10000, 0.96)