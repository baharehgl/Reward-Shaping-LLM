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

tf.compat.v1.disable_eager_execution()

from mpl_toolkits.mplot3d import axes3d
from collections import deque, namedtuple
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import your environment that reads CSV only and renames 'label' -> 'anomaly'
from env1 import EnvTimeSeriesfromRepo
from sklearn.svm import OneClassSVM
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

############################
# Macros and Hyperparameters.
DATAFIXED = 0  # We'll override with env.datasetfix=1 so it doesn't reload CSV each time
EPISODES = 10  # Reduced for quick smoke testing
DISCOUNT_FACTOR = 0.5
EPSILON = 0.5
EPSILON_DECAY = 1.00

TN_Value = 1
TP_Value = 5
FP_Value = -1
FN_Value = -5

NOT_ANOMALY = 0
ANOMALY = 1
action_space = [NOT_ANOMALY, ANOMALY]
action_space_n = len(action_space)

n_steps = 25
n_input_dim = 2
n_hidden_dim = 128

validation_separate_ratio = 0.9


########################### VAE Setup #####################
def load_normal_data(data_path, n_steps, sample_rows=None):
    """
    Loads CSV files from a directory and optionally only the first sample_rows of each file.
    """
    all_files = [os.path.join(data_path, fname) for fname in os.listdir(data_path) if fname.endswith('.csv')]
    windows = []
    for file in all_files:
        try:
            # Use sample_rows to load only a subset of rows (for testing)
            if sample_rows:
                df = pd.read_csv(file, nrows=sample_rows)
            else:
                df = pd.read_csv(file)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
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


def RNNBinaryRewardFuc(timeseries, timeseries_curser, action=0, vae=None, dynamic_coef=1.0):
    if timeseries_curser >= n_steps:
        current_state = np.array([timeseries['value'][timeseries_curser - n_steps:timeseries_curser]])
        vae_reconstruction = vae.predict(current_state)
        reconstruction_error = np.mean(np.square(vae_reconstruction - current_state))
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
            global_step_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="global_step")[
                0]
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


def make_epsilon_greedy_policy(estimator, nA, sess):
    def policy_fn(observation, epsilon):
        A = np.ones(nA, dtype='float32') * epsilon / nA
        q_values = estimator.predict(state=[observation], sess=sess)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def update_dynamic_coef_proportional(current_coef, episode_reward, target_reward=100.0, alpha=0.01, min_coef=0.1,
                                     max_coef=10.0):
    new_coef = current_coef + alpha * (target_reward - episode_reward)
    return max(min(new_coef, max_coef), min_coef)


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
        # Instead of waiting for input, we automatically assign a default label.
        for sample in active_samples:
            print('Active Learning found a confused sample:')
            print(self.env.timeseries['value'].iloc[sample:sample + n_steps])
            default_label = 0  # Set default label here (0 for normal, 1 for anomaly)
            print(f'Automatically labeling sample as: {default_label}')
            self.env.timeseries.loc[sample + n_steps - 1, 'anomaly'] = float(default_label)
        return


class WarmUp(object):
    def __init__(self, env):
        self.env = env

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


MAX_LABELPROP_SAMPLES = 10000


def fit_labelprop_subsample(lp_model, states_list, labels_list):
    labeled_mask = (labels_list != -1)
    labeled_count = np.count_nonzero(labeled_mask)
    if labeled_count < 1:
        print("No labeled samples found. Skipping label propagation.")
        return
    labeled_classes = np.unique(labels_list[labeled_mask])
    if len(labeled_classes) < 1:
        print("No labeled classes found. Skipping label propagation.")
        return

    n = len(states_list)
    if n > MAX_LABELPROP_SAMPLES:
        idxs = np.random.choice(n, MAX_LABELPROP_SAMPLES, replace=False)
        states_sub = [states_list[i] for i in idxs]
        labels_sub = labels_list[idxs]
        lp_model.fit(states_sub, labels_sub)
    else:
        lp_model.fit(states_list, labels_list)


###############################################
# Q-Learning
def q_learning(env, sess, qlearn_estimator, target_estimator, num_episodes, num_epoches,
               replay_memory_size=500000, replay_memory_init_size=1000,  # reduced for faster warm-up
               experiment_dir='./log/',
               update_target_estimator_every=10000, discount_factor=0.99,
               epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=500000, batch_size=256,
               num_LabelPropagation=20, num_active_learning=5, test=0, vae_model=None):
    from collections import namedtuple
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
    policy = make_epsilon_greedy_policy(qlearn_estimator, env.action_space_n, sess)
    num_label = 0

    print("=== Warm up starting... ===")
    warm_start_time = time.time()
    outliers_fraction = 0.01
    data_train = []
    for _ in range(env.datasetsize):
        env.reset()
        env.states_list = [s for s in env.states_list if s is not None]
        data_train.extend(env.states_list)
    warm_start = WarmUp(env)
    model_warm = warm_start.warm_up_isolation_forest(outliers_fraction, data_train)
    lp_model = LabelSpreading()

    t = 0
    while len(replay_memory) < replay_memory_init_size:
        loop_start = time.time()

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
                env.timeseries_curser = sample + n_steps
                action_probs = policy(env.states_list[sample], epsilons[min(total_t, epsilon_decay_steps - 1)])
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                env.timeseries.loc[env.timeseries_curser, 'label'] = env.timeseries.loc[
                    env.timeseries_curser, 'anomaly']
                num_label += 1
                labeled_index.append(sample)
                next_state, reward, done, _ = env.step(action)
                replay_memory.append(Transition(env.states_list[sample], reward, next_state, done))

        label_list = [env.timeseries['label'][i] for i in range(n_steps, len(env.timeseries['label']))]
        label_list = np.array(label_list)

        fit_labelprop_subsample(lp_model, state_list, label_list)
        if hasattr(lp_model, 'label_distributions_'):
            pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)
            certainty_index = np.argsort(pred_entropies)
            certainty_index = [i for i in certainty_index if i not in labeled_index]
            certainty_index = certainty_index[:num_LabelPropagation]
            for index in certainty_index:
                pseudo_label = lp_model.transduction_[index]
                env.timeseries.loc[index + n_steps, 'label'] = pseudo_label

        loop_end = time.time()
        print(f"[Warm-up iteration {t}] Took {loop_end - loop_start:.2f} seconds. Replay size={len(replay_memory)}")
        t += 1

    warm_end_time = time.time()
    print(f"=== Warm up finished in {warm_end_time - warm_start_time:.2f} seconds. ===")

    dynamic_coef = 10.0
    episode_rewards = []
    coef_history = []

    for i_episode in range(num_episodes):
        ep_start = time.time()

        env.rewardfnc = lambda ts, tc, a: RNNBinaryRewardFuc(ts, tc, a, vae_model, dynamic_coef=dynamic_coef)
        episode_reward = 0.0

        if i_episode % 50 == 49:
            print("Save checkpoint in episode {}/{}".format(i_episode + 1, num_episodes))
            saver.save(sess, checkpoint_path)

        per_loop_time1 = time.time()
        state = env.reset()
        env.states_list = [s for s in env.states_list if s is not None]

        # Prevent infinite double resets
        reset_counter = 0
        while env.datasetidx > env.datasetrng * validation_separate_ratio and reset_counter < 10:
            state = env.reset()
            env.states_list = [s for s in env.states_list if s is not None]
            print('double reset')
            reset_counter += 1

        labeled_index = [i - n_steps for i in range(n_steps, len(env.timeseries['label'])) if
                         env.timeseries['label'][i] != -1]

        al = active_learning(env=env, N=num_active_learning, strategy='margin_sampling',
                             estimator=qlearn_estimator, already_selected=labeled_index)
        al_samples = al.get_samples()
        print('Labeling samples: ' + str(al_samples) + ' in env dataset ' + str(env.datasetidx))
        # Automatically label the selected samples
        al.label(al_samples)
        labeled_index.extend(al_samples)
        num_label += len(al_samples)

        state_list = np.array(env.states_list).transpose(2, 0, 1)[0]
        label_list = [env.timeseries['label'][i] for i in range(n_steps, len(env.timeseries['label']))]
        label_list = np.array(label_list)

        for new_sample in al_samples:
            label_list[new_sample] = env.timeseries.loc[new_sample + n_steps, 'anomaly']
            env.timeseries.loc[new_sample + n_steps, 'label'] = env.timeseries.loc[new_sample + n_steps, 'anomaly']

        for sample in labeled_index:
            if sample < len(env.states_list):
                env.timeseries_curser = sample + n_steps
                epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
                state = env.states_list[sample]
                action_probs = policy(state, epsilon)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward[action]
                if len(replay_memory) == replay_memory_size:
                    replay_memory.pop(0)
                replay_memory.append(Transition(state, reward, next_state, done))

        unlabeled_indices = [i for i, e in enumerate(label_list) if e == -1]
        label_list = np.array(label_list)

        fit_labelprop_subsample(lp_model, state_list, label_list)
        if hasattr(lp_model, 'label_distributions_'):
            pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)
            certainty_index = np.argsort(pred_entropies)
            certainty_index = [i for i in certainty_index if i in unlabeled_indices]
            certainty_index = certainty_index[:num_LabelPropagation]
            for index in certainty_index:
                pseudo_label = lp_model.transduction_[index]
                env.timeseries.loc[index + n_steps, 'label'] = pseudo_label

        per_loop_time2 = time.time()

        # Train for num_epoches on the replay buffer
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
        print("Global step {} @ Episode {}/{}, time: {:.2f} + {:.2f}".format(
            total_t, i_episode + 1, num_episodes, per_loop_time_popu, per_loop_time_updt))

        dynamic_coef = update_dynamic_coef_proportional(dynamic_coef, episode_reward,
                                                        target_reward=0.0, alpha=0.001,
                                                        min_coef=0.1, max_coef=10.0)

        print("Episode {}: total reward = {:.3f}, updated dynamic_coef = {:.3f}".format(
            i_episode, episode_reward, dynamic_coef))

        episode_rewards.append(episode_reward)
        coef_history.append(dynamic_coef)

        ep_end = time.time()
        print(f"Episode {i_episode} took {ep_end - ep_start:.2f} seconds total.\n")

    return episode_rewards, coef_history


###############################################
# Validator
def q_learning_validator(env, estimator, num_episodes, record_dir=None, plot=1):
    from sklearn.metrics import precision_recall_fscore_support
    rec_file = open(os.path.join(record_dir, 'performance.txt'), 'w')
    precision_all, recall_all, f1_all = [], [], []

    for i_episode in range(num_episodes):
        print("Validator Episode {}/{}".format(i_episode + 1, num_episodes))
        policy = make_epsilon_greedy_policy(estimator, env.action_space_n, tf.compat.v1.get_default_session())
        state = env.reset()
        env.states_list = [s for s in env.states_list if s is not None]

        reset_counter = 0
        while env.datasetidx < env.datasetrng * validation_separate_ratio and reset_counter < 10:
            state = env.reset()
            env.states_list = [s for s in env.states_list if s is not None]
            print('double reset (validator)')
            reset_counter += 1

        print('Testing on: ' + str(env.repodirext[env.datasetidx]))
        predictions = []
        ground_truths = []
        ts_values = []

        for t in itertools.count():
            action_probs = policy(state, 0)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            predictions.append(action)
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

        precision, recall, f1, _ = precision_recall_fscore_support(ground_truths, predictions,
                                                                   average='binary', zero_division=0)
        precision_all.append(precision)
        recall_all.append(recall)
        f1_all.append(f1)
        print("Validator Episode {}: Precision:{:.3f}, Recall:{:.3f}, F1-score:{:.3f}".format(
            i_episode + 1, precision, recall, f1))
        rec_file.write("Episode {}: Precision:{}, Recall:{}, F1-score:{}\n".format(
            i_episode + 1, precision, recall, f1))

        if plot:
            f, axarr = plt.subplots(3, sharex=True)
            axarr[0].plot(ts_values)
            axarr[0].set_title('Time Series')
            axarr[1].plot(predictions, color='g')
            axarr[1].set_title('Predictions')
            axarr[2].plot(ground_truths, color='r')
            axarr[2].set_title('Ground Truth')
            plt.savefig(os.path.join(record_dir, f"validation_episode_{i_episode}.png"))
            plt.close(f)

    rec_file.close()
    avg_f1 = np.mean(f1_all)
    print("Average F1-score over {} validation episodes: {:.3f}".format(num_episodes, avg_f1))
    return avg_f1


def save_plots(experiment_dir, episode_rewards, coef_history):
    plot_dir = os.path.join(experiment_dir, "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plt.figure()
    plt.plot(episode_rewards, marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Training Reward Curve")
    plt.savefig(os.path.join(plot_dir, "reward_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(coef_history, marker='s', color='orange')
    plt.xlabel("Episode")
    plt.ylabel("Dynamic Coefficient")
    plt.title("Dynamic Coefficient Evolution")
    plt.savefig(os.path.join(plot_dir, "dynamic_coef_curve.png"))
    plt.close()


###############################################
# Train Wrapper
def train_wrapper_kpi(num_LP, num_AL, discount_factor, labeled_percentage):
    test = 0  # training mode
    # For testing, load only a subset of rows from each CSV by setting sample_rows
    train_data_directory = os.path.join(current_dir, "KPI_data", "train")
    x_train = load_normal_data(train_data_directory, n_steps, sample_rows=1000)

    vae, _ = build_vae(original_dim, latent_dim, intermediate_dim)
    vae.fit(x_train, epochs=2, batch_size=32)
    vae.save('vae_model.h5')

    percentage = [labeled_percentage]
    train_dataset_dir = os.path.join(current_dir, "KPI_data", "train")
    test_dataset_dir = os.path.join(current_dir, "KPI_data", "test")

    for j in range(len(percentage)):
        exp_relative_dir = [f'KPI_LP_{num_LP}_AL_{num_AL}_d{discount_factor}']
        env = EnvTimeSeriesfromRepo(train_dataset_dir)
        # Force datasetfix=1 so we do NOT reload CSV each reset
        env.datasetfix = 1

        env.statefnc = RNNBinaryStateFuc
        env.rewardfnc = lambda ts, tc, a: RNNBinaryRewardFuc(ts, tc, a, vae, dynamic_coef=10.0)
        env.timeseries_curser_init = n_steps
        env.datasetidx = 0

        env_test = EnvTimeSeriesfromRepo(test_dataset_dir)
        env_test.datasetfix = 1
        env_test.statefnc = RNNBinaryStateFuc
        env_test.rewardfnc = RNNBinaryRewardFucTest

        if test == 1:
            env.datasetrng = env.datasetsize
        else:
            calc_rng = int(env.datasetsize * float(percentage[j]))
            env.datasetrng = max(1, calc_rng)

        experiment_dir = os.path.abspath(f"./exp/{exp_relative_dir[0]}")
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
            # We do only a few episodes to test if it finishes quickly
            episode_rewards, coef_history = q_learning(
                env, sess, qlearn_estimator, target_estimator,
                num_episodes=EPISODES,  # 10 episodes
                num_epoches=10,
                experiment_dir=experiment_dir,
                replay_memory_size=500000,
                replay_memory_init_size=1000,  # 1k for faster warm-up
                update_target_estimator_every=10,
                epsilon_start=1,
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

        save_plots(experiment_dir, episode_rewards, coef_history)
        return final_metric


if __name__ == '__main__':
    # Run KPI experiment with 1% labeled data, 10 episodes total for testing purposes.
    print("Running KPI experiment with 1% labeled data, 10 episodes total...")
    final_metric_1p = train_wrapper_kpi(100, 30, 0.92, 0.01)
    print("Final metric for 1% labeled data:", final_metric_1p)
