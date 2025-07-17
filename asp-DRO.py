import os
import sys
import time
import random
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
from collections import deque, namedtuple
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.models import load_model
from mpl_toolkits.mplot3d import axes3d
from sklearn.svm import OneClassSVM
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

# Disable eager execution if necessary
tf.compat.v1.disable_eager_execution()

# ----------------------------------------------------------------
# Get the current script directory and build relative paths.
# ----------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))

# Adjust these paths based on your repository structure.
# In this example, the data folders are in the same directory as this script.
normal_data_directory = os.path.join(current_dir, 'normal-data')
anomaly_data_directory = os.path.join(current_dir, 'ydata-labeled-time-series-anomalies-v1_0', 'A1Benchmark')

# If you need to import modules from a subfolder (e.g., environment), adjust sys.path accordingly.
# For example, if the environment module is inside a subfolder named "environment" in your repository:
sys.path.append(current_dir)

# ----------------------------------------------------------------
# Import environment module from the sub-folder "environment"
# Adjust the import if your folder structure is different.
from environment.time_series_repo_ext import EnvTimeSeriesfromRepo

# Set which GPUs to use if needed (adjust or remove if not using GPUs)
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

# Macros and constants
DATAFIXED = 0  # whether target at a single time series dataset

EPISODES = 500  # number of episodes for training
DISCOUNT_FACTOR = 0.5  # reward discount factor [0,1]
EPSILON = 0.5  # epsilon-greedy method parameter for action selection
EPSILON_DECAY = 1.00  # epsilon-greedy method decay parameter

NOT_ANOMALY = 0
ANOMALY = 1

action_space = [NOT_ANOMALY, ANOMALY]
action_space_n = len(action_space)

n_steps = 25  # size of the slide window for SLIDE_WINDOW state and reward functions
n_input_dim = 2  # dimension of the input for a LSTM cell
n_hidden_dim = 128  # dimension of the hidden state in LSTM cell

# Reward Values
TP_Value = 5
TN_Value = 1
FP_Value = -1
FN_Value = -5

validation_separate_ratio = 0.9

# ----------------------------------------------------------------
# VAE-related functions and layers
# ----------------------------------------------------------------

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def load_normal_data(data_path):
    # Example assumes CSV files; adjust if your data format differs
    all_files = [os.path.join(data_path, fname) for fname in os.listdir(data_path) if fname.endswith('.csv')]
    data_list = [pd.read_csv(file) for file in all_files]
    data = pd.concat(data_list, axis=0, ignore_index=True)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.values)
    return scaled_data

def build_vae(original_dim, latent_dim=2, intermediate_dim=64):
    inputs = layers.Input(shape=(original_dim,))
    h = layers.Dense(intermediate_dim, activation='relu')(inputs)
    h = layers.Dense(intermediate_dim, activation='relu')(h)
    h = layers.Dense(intermediate_dim, activation='relu')(h)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)
    z = Sampling()([z_mean, z_log_var])

    decoder_h = layers.Dense(intermediate_dim, activation='relu')
    decoder_h = layers.Dense(intermediate_dim, activation='relu')
    decoder_h = layers.Dense(intermediate_dim, activation='relu')
    decoder_mean = layers.Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    encoder = models.Model(inputs, [z_mean, z_log_var, z])
    vae = models.Model(inputs, x_decoded_mean)

    reconstruction_loss = losses.binary_crossentropy(inputs, x_decoded_mean) * original_dim
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    return vae, encoder

# Constants for VAE
original_dim = 3  # Depends on your input feature dimension
latent_dim = 10
intermediate_dim = 64

vae, encoder = build_vae(original_dim, latent_dim, intermediate_dim)

# ----------------------------------------------------------------
# RL state and reward functions
# ----------------------------------------------------------------

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

def kl_divergence(p, q):
    """Compute the KL divergence KL(p || q)."""
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return np.sum(p * np.log(p / q))

def adaptive_scaling_factor_dro(preference_strength, tau_min=0.1, tau_max=5.0, rho=0.5, max_iter=10):
    """
    Learn tau dynamically using DRO (KL-constrained optimization).
    """
    tau = 1.0  # Initialize tau
    for _ in range(max_iter):
        kl_term = kl_divergence([preference_strength, 1 - preference_strength], [0.5, 0.5])
        grad = -np.log(1 + np.exp(-preference_strength / tau)) + rho - kl_term
        hess = preference_strength ** 2 * np.exp(-preference_strength / tau) / (tau ** 3 * (1 + np.exp(-preference_strength / tau)) ** 2)
        tau = tau - grad / (hess + 1e-8)
        tau = np.clip(tau, tau_min, tau_max)
    return tau

tau_values = []

def RNNBinaryRewardFuc(timeseries, timeseries_curser, action=0, vae=None, scale_factor=10):
    if timeseries_curser >= n_steps:
        current_state = np.array([timeseries['value'][timeseries_curser - n_steps:timeseries_curser]])
        vae_reconstruction = vae.predict(current_state)
        reconstruction_error = np.mean(np.square(vae_reconstruction - current_state))
        vae_penalty = -scale_factor * reconstruction_error
        preference_strength = np.clip(1 / (1 + np.exp(-reconstruction_error)), 0.05, 0.95)
        tau = adaptive_scaling_factor_dro(preference_strength)
        tau_values.append(tau)
        if timeseries['label'][timeseries_curser] == 0:
            return [(TN_Value + tau * (vae_penalty)), (FP_Value + tau * (vae_penalty))]
        if timeseries['label'][timeseries_curser] == 1:
            return [(FN_Value + tau * (vae_penalty)), (TP_Value + tau * (vae_penalty))]
    else:
        return [0, 0]

def RNNBinaryRewardFucTest(timeseries, timeseries_curser, action=0):
    if timeseries_curser >= n_steps:
        if timeseries['anomaly'][timeseries_curser] == 0:
            return [TN_Value, FP_Value]
        if timeseries['anomaly'][timeseries_curser] == 1:
            return [FN_Value, TP_Value]
    else:
        return [0, 0]

# ----------------------------------------------------------------
# Q-Learning: Neural Network Estimator and supporting functions
# ----------------------------------------------------------------

class Q_Estimator_Nonlinear():
    """
    Action-Value Function Approximator Q(s,a) using a TensorFlow RNN.
    """
    def __init__(self, learning_rate=np.float32(0.01), scope="Q_Estimator_Nonlinear", summaries_dir=None):
        self.scope = scope
        self.summary_writer = None
        with tf.compat.v1.variable_scope(scope):
            self.state = tf.compat.v1.placeholder(shape=[None, n_steps, n_input_dim],
                                                  dtype=tf.float32, name="state")
            self.target = tf.compat.v1.placeholder(shape=[None, action_space_n],
                                                   dtype=tf.float32, name="target")
            self.weights = {
                'out': tf.Variable(tf.compat.v1.random_normal([n_hidden_dim, action_space_n]))
            }
            self.biases = {
                'out': tf.Variable(tf.compat.v1.random_normal([action_space_n]))
            }
            self.state_unstack = tf.compat.v1.unstack(self.state, n_steps, 1)
            lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(n_hidden_dim, forget_bias=1.0)
            self.outputs, self.states = tf.compat.v1.nn.static_rnn(lstm_cell, self.state_unstack, dtype=tf.float32)
            self.action_values = tf.matmul(self.outputs[-1], self.weights['out']) + self.biases['out']
            self.losses = tf.compat.v1.squared_difference(self.action_values, self.target)
            self.loss = tf.reduce_mean(self.losses)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss,
                                                    global_step=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                                                              scope="global_step")[0])
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
    """
    Copies the model parameters of one estimator to another.
    """
    e1_params = [t for t in tf.compat.v1.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.compat.v1.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)
    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)
    sess.run(update_ops)

def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator.
    """
    def policy_fn(observation, epsilon):
        A = np.ones(nA, dtype='float32') * epsilon / nA
        q_values = estimator.predict(state=[observation])
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def q_learning(env,
               sess,
               qlearn_estimator,
               target_estimator,
               num_episodes,
               num_epoches,
               replay_memory_size=500000,
               replay_memory_init_size=50000,
               experiment_dir='./log/',
               update_target_estimator_every=10000,
               discount_factor=0.99,
               epsilon_start=1.0,
               epsilon_end=0.1,
               epsilon_decay_steps=500000,
               batch_size=512,
               num_LabelPropagation=20,
               num_active_learning=5,
               test=0,
               vae_model=None):
    """
    Q-Learning algorithm for off-policy TD control using function approximation.
    """
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
    total_t = sess.run(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="global_step")[0])
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    policy = make_epsilon_greedy_policy(qlearn_estimator, env.action_space_n)
    num_label = 0

    # Warm-up: Populate replay memory using a warm-up strategy (e.g., IsolationForest)
    popu_time = time.time()
    print('Warm up starting...')
    outliers_fraction = 0.01
    data_train = []
    for num in range(env.datasetsize):
        env.reset()
        data_train.extend(env.states_list)
    # Isolation Forest model
    model = WarmUp().warm_up_isolation_forest(outliers_fraction, data_train)
    lp_model = LabelSpreading()
    for t in itertools.count():
        env.reset()
        data = np.array(env.states_list).transpose(2, 0, 1).reshape(2, -1)[0].reshape(-1, n_steps)[:, -1].reshape(-1, 1)
        anomaly_score = model.decision_function(data)
        pred_score = [-1 * s + 0.5 for s in anomaly_score]
        warm_samples = np.argsort(pred_score)[:5]
        warm_samples = np.append(warm_samples, np.argsort(pred_score)[-5:])
        state_list = np.array(env.states_list).transpose(2, 0, 1)[0]
        label_list = [-1] * len(state_list)
        for sample in warm_samples:
            state = env.states_list[sample]
            env.timeseries_curser = sample + n_steps
            action_probs = policy(state, epsilons[min(total_t, epsilon_decay_steps - 1)])
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            env.timeseries['label'][env.timeseries_curser] = env.timeseries['anomaly'][env.timeseries_curser]
            num_label += 1
            label_list[sample] = int(env.timeseries['anomaly'][env.timeseries_curser])
            next_state, reward, done, _ = env.step(action)
            replay_memory.append(Transition(state, reward, next_state, done))
        unlabeled_indices = [i for i, e in enumerate(label_list) if e == -1]
        label_list = np.array(label_list)
        lp_model.fit(state_list, label_list)
        pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)
        certainty_index = np.argsort(pred_entropies)
        certainty_index = certainty_index[np.in1d(certainty_index, unlabeled_indices)][:num_LabelPropagation]
        for index in certainty_index:
            pseudo_label = lp_model.transduction_[index]
            env.timeseries['label'][index + n_steps] = pseudo_label
        if len(replay_memory) >= replay_memory_init_size:
            break

    # Main loop of training episodes
    for i_episode in range(num_episodes):
        if i_episode % 50 == 49:
            print("Save checkpoint in episode {}/{}".format(i_episode + 1, num_episodes))
            saver.save(tf.compat.v1.get_default_session(), checkpoint_path)
        per_loop_time1 = time.time()
        state = env.reset()
        while env.datasetidx > env.datasetrng * validation_separate_ratio:
            env.reset()
            print('double reset')
        labeled_index = [i for i, e in enumerate(env.timeseries['label']) if e != -1]
        labeled_index = [item for item in labeled_index if item >= 25]
        al = active_learning(env=env, N=num_active_learning, strategy='margin_sampling',
                             estimator=qlearn_estimator, already_selected=labeled_index)
        al_samples = al.get_samples()
        print('labeling samples: ' + str(al_samples) + ' in env ' + str(env.datasetidx))
        labeled_index.extend(al_samples)
        num_label += len(al_samples)
        state_list = np.array(env.states_list).transpose(2, 0, 1)[0]
        label_list = np.array(env.timeseries['label'][n_steps:])
        for new_sample in al_samples:
            label_list[new_sample] = env.timeseries['anomaly'][n_steps + new_sample]
            env.timeseries['label'][n_steps + new_sample] = env.timeseries['anomaly'][n_steps + new_sample]
        for samples in labeled_index:
            env.timeseries_curser = samples + n_steps
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
            state = env.states_list[samples]
            action_probs = policy(state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)
            replay_memory.append(Transition(state, reward, next_state, done))
        unlabeled_indices = [i for i, e in enumerate(label_list) if e == -1]
        label_list = np.array(label_list)
        lp_model.fit(state_list, label_list)
        pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)
        certainty_index = np.argsort(pred_entropies)
        certainty_index = certainty_index[np.in1d(certainty_index, unlabeled_indices)][:num_LabelPropagation]
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
                q_values_next0 = target_estimator.predict(state=next_states_batch0)
                q_values_next1 = target_estimator.predict(state=next_states_batch1)
                targets_batch = reward_batch + (discount_factor *
                                                np.stack((np.amax(q_values_next0, axis=1),
                                                          np.amax(q_values_next1, axis=1)),
                                                         axis=-1))
            else:
                targets_batch = reward_batch
            qlearn_estimator.update(state=states_batch, target=targets_batch.astype(np.float32))
            total_t += 1
        per_loop_time_popu = per_loop_time2 - per_loop_time1
        per_loop_time_updt = time.time() - per_loop_time2
        print("Global step {} @ Episode {}/{}, time: {} + {}".format(total_t, i_episode + 1, num_episodes, per_loop_time_popu, per_loop_time_updt))
    return

def evaluate_model(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    mcc = matthews_corrcoef(y_true, y_pred)
    balanced_acc = (recall + (tn / (tn + fp))) / 2
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    return {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "MCC": mcc,
        "Balanced Accuracy": balanced_acc,
        "FPR": fpr,
        "FNR": fnr
    }

def q_learning_validator(env, estimator, num_episodes, record_dir=None, plot=1):
    y_true_all = []
    y_pred_all = []
    for i_episode in range(num_episodes):
        policy = make_epsilon_greedy_policy(estimator, env.action_space_n)
        state = env.reset()
        while env.datasetidx < env.datasetrng * validation_separate_ratio:
            state = env.reset()
        for t in itertools.count():
            action_probs = policy(state, 0)
            action = np.argmax(action_probs)
            y_true_all.append(env.timeseries['anomaly'][env.timeseries_curser])
            y_pred_all.append(action)
            next_state, reward, done, _ = env.step(action)
            if done:
                break
            state = next_state[action]
    results = evaluate_model(y_true_all, y_pred_all)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    return results

# ----------------------------------------------------------------
# Active Learning and Warm-Up helper classes
# ----------------------------------------------------------------

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
            distance = abs(q[0] - q[1])
            distances.append(distance)
        distances = np.array(distances)
        if len(distances.shape) < 2:
            min_margin = abs(distances)
        else:
            sort_distances = np.sort(distances, 1)[:, -2:]
            min_margin = sort_distances[:, 1] - sort_distances[:, 0]
        rank_ind = np.argsort(min_margin)
        rank_ind = [i for i in rank_ind if i not in self.already_selected]
        active_samples = rank_ind[0:self.N]
        return active_samples

    def get_samples_by_score(self, threshold):
        states_list = self.env.states_list
        distances = []
        for state in states_list:
            q = self.estimator.predict(state=[state])[0]
            distance = abs(q[0] - q[1])
            distances.append(distance)
        distances = np.array(distances)
        if len(distances.shape) < 2:
            min_margin = abs(distances)
        else:
            sort_distances = np.sort(distances, 1)[:, -2:]
            min_margin = sort_distances[:, 1] - sort_distances[:, 0]
        rank_ind = np.argsort(min_margin)
        rank_ind = [i for i in rank_ind if i not in self.already_selected]
        active_samples = [t for t in rank_ind if distances[t] < threshold]
        return active_samples

    def label(self, active_samples):
        for sample in active_samples:
            print('AL finds one of the most confused samples:')
            print(self.env.timeseries['value'].iloc[sample:sample + n_steps])
            print('Please label the last timestamp based on your knowledge')
            print('0 for non-anomaly; 1 for anomaly')
            label = input()
            self.env.timeseries.loc[sample + n_steps - 1, 'anomaly'] = label
        return

class WarmUp(object):
    def warm_up_SVM(self, outliers_fraction, N):
        states_list = self.env.get_states_list()
        data = np.array(states_list).transpose(2, 0, 1).reshape(2, -1)[0].reshape(-1, n_steps)[:, -1].reshape(-1, 1)
        model = OneClassSVM(gamma='auto', nu=0.95 * outliers_fraction)
        model.fit(data)
        distances = model.decision_function(data)
        if len(distances.shape) < 2:
            min_margin = abs(distances)
        else:
            sort_distances = np.sort(distances, 1)[:, -2:]
            min_margin = sort_distances[:, 1] - sort_distances[:, 0]
        rank_ind = np.argsort(min_margin)
        samples = rank_ind[0:N]
        return samples

    def warm_up_isolation_forest(self, outliers_fraction, X_train):
        from sklearn.ensemble import IsolationForest
        data = np.array(X_train).transpose(2, 0, 1).reshape(2, -1)[0].reshape(-1, n_steps)[:, -1].reshape(-1, 1)
        clf = IsolationForest(contamination=outliers_fraction)
        clf.fit(data)
        return clf

# ----------------------------------------------------------------
# Train function that sets up the environment and runs training.
# ----------------------------------------------------------------

def train(num_LP, num_AL, discount_factor, learn_tau=True):
    """
    Train the RL agent with learned tau using DRO.
    """
    # VAE parameters
    original_dim = 3  # number of features
    latent_dim = 10
    intermediate_dim = 64

    # Load dataset from relative path
    x_train = load_normal_data(normal_data_directory)

    # Train the VAE
    vae, _ = build_vae(original_dim, latent_dim, intermediate_dim)
    vae.fit(x_train, epochs=50, batch_size=32)
    vae.save('vae_model.h5')
    vae = load_model('vae_model.h5', custom_objects={'Sampling': Sampling}, compile=False)

    # Experiment and dataset directories (relative paths)
    exp_relative_dir = ['RLVAL with DRO and Adaptive Scaling']
    dataset_dirs = [anomaly_data_directory]

    for i in range(len(dataset_dirs)):
        env = EnvTimeSeriesfromRepo(dataset_dirs[i])
        env.statefnc = RNNBinaryStateFuc
        if learn_tau:
            env.rewardfnc = lambda timeseries, timeseries_curser, action: RNNBinaryRewardFuc(
                timeseries, timeseries_curser, action, vae
            )
        else:
            env.rewardfnc = RNNBinaryRewardFuc  # Without adaptive tau
        env.timeseries_curser_init = n_steps
        env.datasetfix = DATAFIXED
        env.datasetidx = 0
        # Testing environment
        env_test = env
        env_test.rewardfnc = RNNBinaryRewardFucTest
        experiment_dir = os.path.abspath(os.path.join(current_dir, 'exp', exp_relative_dir[i]))
        tf.compat.v1.reset_default_graph()
        global_step = tf.Variable(0, name="global_step", trainable=False)
        qlearn_estimator = Q_Estimator_Nonlinear(scope="qlearn", summaries_dir=experiment_dir, learning_rate=0.0003)
        target_estimator = Q_Estimator_Nonlinear(scope="target")
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())
        with sess.as_default():
            q_learning(env,
                       sess=sess,
                       qlearn_estimator=qlearn_estimator,
                       target_estimator=target_estimator,
                       num_episodes=300,
                       num_epoches=10,
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
                       test=0,
                       vae_model=vae
                       )
            optimization_metric = q_learning_validator(env_test, qlearn_estimator,
                                                       int(env.datasetsize * (1 - validation_separate_ratio)),
                                                       experiment_dir)
        return optimization_metric

def plot_tau_evolution():
    plt.figure(figsize=(10, 5))
    plt.plot(tau_values, label="Tau over time", alpha=0.8)
    plt.xlabel("Training Steps")
    plt.ylabel("Tau Value")
    plt.title("Evolution of Tau during Training")
    plt.legend()
    plt.show()

# ----------------------------------------------------------------
# Run training and plot tau evolution.
# ----------------------------------------------------------------

plot_tau_evolution()

# Example training runs with different parameters
train(100, 30, 0.92, learn_tau=True)
train(150, 50, 0.94, learn_tau=True)
train(200, 100, 0.96, learn_tau=True)

# You can uncomment or modify additional training calls as needed.
