
import pandas as pd
import numpy as np
import random
import os
import sklearn.preprocessing

# Define constants.
NOT_ANOMALY = 0
ANOMALY = 1

REWARD_CORRECT = 1
REWARD_INCORRECT = -1

action_space = [NOT_ANOMALY, ANOMALY]


def defaultStateFuc(timeseries, timeseries_curser, previous_state=None, action=None):
    """
    Default state function: returns the value at the current time index.
    In your RL setup, you can replace this with a sliding-window state function.
    """
    return timeseries['value'][timeseries_curser]


def defaultRewardFuc(timeseries, timeseries_curser, action):
    """
    Default reward: returns REWARD_CORRECT if the action matches the anomaly label,
    and REWARD_INCORRECT otherwise.
    """
    if action == timeseries['anomaly'][timeseries_curser]:
        return REWARD_CORRECT
    else:
        return REWARD_INCORRECT


class EnvTimeSeriesfromRepo():
    def __init__(self, repodir='environment/time_series_repo/'):
        # Get all CSV file paths in the repository directory.
        self.repodir = repodir
        self.repodirext = []
        for subdir, dirs, files in os.walk(self.repodir):
            for file in files:
                if file.endswith('.csv'):
                    self.repodirext.append(os.path.join(subdir, file))

        # Check that CSV files were found.
        if len(self.repodirext) == 0:
            raise ValueError("No CSV files found in directory: {}".format(self.repodir))

        self.action_space_n = len(action_space)

        # Initialize variables.
        self.timeseries = None
        self.timeseries_curser = -1
        self.timeseries_curser_init = 0
        self.timeseries_states = None
        self.statefnc = defaultStateFuc
        self.rewardfnc = defaultRewardFuc

        self.datasetsize = len(self.repodirext)
        self.datasetfix = 0
        self.datasetidx = random.randint(0, self.datasetsize - 1)
        self.datasetrng = self.datasetsize

        self.timeseries_repo = []
        self.states_list = []

        # Read and preprocess each CSV file.
        for i in range(len(self.repodirext)):
            # Here we assume the Yahoo Benchmark format: column index 1 is "value", and index 2 is "anomaly".
            ts = pd.read_csv(self.repodirext[i], usecols=[1, 2],
                             header=0, names=['value', 'anomaly'])
            # Add a marker column for labeling.
            ts['label'] = -1
            ts = ts.astype(np.float32)

            # Scale the 'value' column to [0,1] using MinMaxScaler.
            scaler = sklearn.preprocessing.MinMaxScaler()
            ts['value'] = scaler.fit_transform(ts[['value']])
            self.timeseries_repo.append(ts)

    def reset(self):
        """
        Reset the environment: choose a new time series, reset the cursor, and compute the initial state.
        """
        if self.datasetfix == 0:
            self.datasetidx = (self.datasetidx + 1) % self.datasetrng

        print("Loading file: ", self.repodirext[self.datasetidx])
        self.timeseries = self.timeseries_repo[self.datasetidx]
        self.timeseries_curser = self.timeseries_curser_init
        self.timeseries_states = self.statefnc(self.timeseries, self.timeseries_curser)
        self.states_list = self.get_states_list()

        return self.timeseries_states

    def reset_to(self, id):
        """
        Reset the environment to a specific file by its index.
        """
        if id < 0 or id >= self.datasetrng:
            raise ValueError("Invalid dataset index: {}".format(id))
        self.datasetidx = id
        self.timeseries = self.timeseries_repo[self.datasetidx]
        self.timeseries_curser = self.timeseries_curser_init
        self.timeseries_states = self.statefnc(self.timeseries, self.timeseries_curser)
        self.states_list = self.get_states_list()

        return self.timeseries_states

    def reset_getall(self):
        """
        Load the entire dataset (including a timestamp column) and scale the values.
        """
        if self.datasetfix == 0:
            self.datasetidx = (self.datasetidx + 1) % self.datasetrng

        self.timeseries = pd.read_csv(self.repodirext[self.datasetidx],
                                      usecols=[0, 1, 2],
                                      header=0, names=['timestamp', 'value', 'anomaly'])
        self.timeseries = self.timeseries.astype(np.float32)
        self.timeseries_curser = self.timeseries_curser_init

        scaler = sklearn.preprocessing.MinMaxScaler()
        self.timeseries['value'] = scaler.fit_transform(self.timeseries[['value']])
        return self.timeseries

    def step(self, action):
        """
        Take a step in the environment.
        Returns a tuple: (state, reward, done, info)
        """
        # 1. Compute reward at the current cursor
        reward = self.rewardfnc(self.timeseries,
                                self.timeseries_curser,
                                action)

        # 2. Advance the cursor
        self.timeseries_curser += 1

        # 3. Check for terminal condition
        if self.timeseries_curser >= self.timeseries['value'].size:
            done = 1
            # at terminal: return the last state twice (as in your original)
            state = np.array([self.timeseries_states,
                              self.timeseries_states])
        else:
            done = 0

            # 4. Decide whether to call your RNN state‐fnc or hold the old state
            #    Use the length of your current window to infer n_steps:
            window_size = self.timeseries_states.shape[0]

            if self.timeseries_curser < window_size:
                # warm‐up period: keep the previous state
                next_state = self.timeseries_states
            else:
                # now we have enough history, compute the real next state
                next_state = self.statefnc(self.timeseries,
                                           self.timeseries_curser,
                                           self.timeseries_states,
                                           action)

            # 5. Only update if next_state is valid (not None)
            if next_state is not None:
                # if the returned state has an extra action dimension, pick the right slice
                if isinstance(next_state, np.ndarray) and \
                        next_state.ndim > np.array(self.timeseries_states).ndim:
                    self.timeseries_states = next_state[action]
                else:
                    self.timeseries_states = next_state

            # 6. Expose the raw `state` that the agent sees
            state = next_state

        # 7. Return exactly the same signature
        return state, reward, done, []




    def get_states_list(self):
        """
        Build and return a list of states for the current time series.
        """
        self.timeseries = self.timeseries_repo[self.datasetidx]
        self.timeseries_curser = self.timeseries_curser_init
        state_list = []
        for cursor in range(self.timeseries_curser_init, self.timeseries['value'].size):
            if len(state_list) == 0:
                state = self.statefnc(self.timeseries, cursor)
            else:
                state = self.statefnc(self.timeseries, cursor, state_list[-1])
                # If the state function returns multiple states (e.g., for binary branching), take the first one.
                if isinstance(state, np.ndarray) and state.ndim > 1:
                    state = state[0]
            state_list.append(state)
        return state_list
