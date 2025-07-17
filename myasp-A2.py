import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import random
import sys
import os
import time
from scipy import stats
import tensorflow as tf

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

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

############################
# Macros and Hyperparameters.
DATAFIXED = 0  # whether target is fixed to a single time series
EPISODES = 300  # number of episodes (for demonstration)
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
            self.state_unstack = tf.compat.v1.unstack(self.state, n_steps, 1)
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
        env.rewardfnc = lambda ts, tc, a: RNNBinaryRewardFuc(ts, tc, a, vae_model, dynamic_coef=dynamic_coef)
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
                next_state, reward, done, _ = env.step(action)
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
            f, axarr = plt.subplots(3, sharex=True)
            axarr[0].plot(ts_values)
            axarr[0].set_title('Time Series')
            axarr[1].plot(predictions, color='g')
            axarr[1].set_title('Predictions')
            axarr[2].plot(ground_truths, color='r')
            axarr[2].set_title('Ground Truth')
            plt.savefig(os.path.join(record_dir, "validation_episode_{}.png".format(i_episode)))
            plt.close(f)
    rec_file.close()
    avg_f1 = np.mean(f1_all)
    print("Average F1-score over {} episodes: {}".format(num_episodes, avg_f1))
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


def train_wrapper(num_LP, num_AL, discount_factor):
    data_directory = os.path.join(current_dir, "normal-data")
    x_train = load_normal_data(data_directory, n_steps)
    vae, _ = build_vae(original_dim, latent_dim, intermediate_dim)
    vae.fit(x_train, epochs=200, batch_size=32)
    vae.save('vae_model.h5')
    percentage = [1]
    test = 0
    for j in range(len(percentage)):
        # Change experiment name to reflect A2 dataset.
        exp_relative_dir = ['A2_LP_1500init_warmup_h128_b256_300ep_num_LP' + str(num_LP) +
                            '_num_AL' + str(num_AL) + '_d' + str(discount_factor)]
        # Change dataset directory to use the A2Benchmark folder.
        dataset_dir = [os.path.join(current_dir, "ydata-labeled-time-series-anomalies-v1_0", "A2Benchmark")]
        for i in range(len(dataset_dir)):
            env = EnvTimeSeriesfromRepo(dataset_dir[i])
            env.statefnc = RNNBinaryStateFuc
            env.rewardfnc = lambda ts, tc, a: RNNBinaryRewardFuc(ts, tc, a, vae, dynamic_coef=10.0)
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
                                                           num_episodes=300, num_epoches=10,
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
            return final_metric


train_wrapper(100, 30, 0.92)
train_wrapper(150, 50, 0.94)
train_wrapper(200, 100, 0.96)
