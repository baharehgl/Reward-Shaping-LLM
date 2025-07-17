import os
import random
import pandas as pd
import numpy as np
import sklearn.preprocessing

# Define constants.
NOT_ANOMALY = 0
ANOMALY = 1

REWARD_CORRECT = 1
REWARD_INCORRECT = -1

action_space = [NOT_ANOMALY, ANOMALY]


def defaultStateFuc(timeseries, timeseries_curser, previous_state=None, action=None):
    """
    Default state function: returns the 'value' at the current time index.
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
        """
        This environment only looks for .csv files in repodir.
        It does NOT read .hdf, so PyTables is unnecessary.
        """
        self.repodir = repodir
        self.repodirext = []

        # Gather only .csv files, skip __MACOSX and hidden files.
        for subdir, dirs, files in os.walk(self.repodir):
            if '__MACOSX' in subdir:
                continue
            for file in files:
                if file.endswith('.csv') and not file.startswith('._'):
                    self.repodirext.append(os.path.join(subdir, file))

        if len(self.repodirext) == 0:
            raise ValueError(f"No CSV files found in directory: {self.repodir}")

        self.action_space_n = len(action_space)

        # Initialize variables.
        self.timeseries = None
        self.timeseries_curser = -1
        self.timeseries_curser_init = 0
        self.timeseries_states = None
        self.statefnc = defaultStateFuc
        self.rewardfnc = defaultRewardFuc

        self.timeseries_repo = []
        self.states_list = []

        # Process each CSV file.
        for path in self.repodirext:
            try:
                # Read the entire CSV so we can see all columns.
                df = pd.read_csv(path, encoding='latin1')

                # If your CSV has columns: [timestamp, value, label, KPI ID],
                # rename "label" -> "anomaly" if present.
                if 'label' in df.columns and 'anomaly' not in df.columns:
                    df.rename(columns={'label': 'anomaly'}, inplace=True)

                # If 'anomaly' still doesn't exist, create it as 0.
                if 'anomaly' not in df.columns:
                    df['anomaly'] = 0

                # Check that there's a 'value' column
                if 'value' not in df.columns:
                    # If not, assume second column is 'value'
                    cols = list(df.columns)
                    if len(cols) >= 2:
                        df.rename(columns={cols[1]: 'value'}, inplace=True)
                    else:
                        raise ValueError(f"File {path} does not contain a 'value' column or enough columns.")

                # If your code also needs a 'label' column for some reason, you can create it
                # but typically we only need 'anomaly' for RL logic. Let's just ensure 'label' exists:
                if 'label' not in df.columns:
                    # If you do want a 'label' column separate from 'anomaly', create it:
                    df['label'] = df['anomaly']  # or -1, depending on your usage.

                # Convert columns to numeric as needed
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df['anomaly'] = pd.to_numeric(df['anomaly'], errors='coerce').fillna(0).astype(np.int32)
                df.dropna(subset=['value'], inplace=True)
                df['value'] = df['value'].astype(np.float32)

                # Scale the 'value' column to [0,1]
                if df[['value']].shape[0] == 0:
                    print(f"Warning: file {path} has no valid data; skipping.")
                    continue
                scaler = sklearn.preprocessing.MinMaxScaler()
                df['value'] = scaler.fit_transform(df[['value']])

                self.timeseries_repo.append(df)
            except Exception as e:
                print(f"Error reading file: {path}\n{e}")
                continue

        if len(self.timeseries_repo) == 0:
            raise ValueError(f"No valid time series data found in directory: {self.repodir}")

        self.datasetsize = len(self.timeseries_repo)
        self.datasetfix = 0
        self.datasetidx = random.randint(0, self.datasetsize - 1)
        self.datasetrng = self.datasetsize

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
        If you need the entire dataset with columns [0,1,2], typically [timestamp, value, anomaly].
        Adjust if your CSV doesn't have these exact columns.
        """
        if self.datasetfix == 0:
            self.datasetidx = (self.datasetidx + 1) % self.datasetrng

        path = self.repodirext[self.datasetidx]
        df = pd.read_csv(path, encoding='latin1')

        # rename label->anomaly if needed
        if 'label' in df.columns and 'anomaly' not in df.columns:
            df.rename(columns={'label': 'anomaly'}, inplace=True)

        if 'anomaly' not in df.columns:
            df['anomaly'] = 0

        if 'value' not in df.columns:
            cols = list(df.columns)
            if len(cols) >= 2:
                df.rename(columns={cols[1]: 'value'}, inplace=True)
            else:
                raise ValueError(f"File {path} does not contain enough columns for 'value' and 'anomaly'")

        df = df.astype({'value': 'float32', 'anomaly': 'int32'})
        self.timeseries_curser = self.timeseries_curser_init

        scaler = sklearn.preprocessing.MinMaxScaler()
        df['value'] = scaler.fit_transform(df[['value']])
        self.timeseries = df
        return self.timeseries

    def step(self, action):
        """
        Take a step in the environment.
        Returns a tuple: (state, reward, done, info)
        """
        reward = self.rewardfnc(self.timeseries, self.timeseries_curser, action)
        self.timeseries_curser += 1

        if self.timeseries_curser >= self.timeseries['value'].size:
            done = 1
            state = np.array([self.timeseries_states, self.timeseries_states])
        else:
            done = 0
            state = self.statefnc(self.timeseries, self.timeseries_curser, self.timeseries_states, action)

        if isinstance(state, np.ndarray) and state.ndim > np.array(self.timeseries_states).ndim:
            self.timeseries_states = state[action]
        else:
            self.timeseries_states = state

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
                if isinstance(state, np.ndarray) and state.ndim > 1:
                    state = state[0]
            state_list.append(state)
        return state_list
