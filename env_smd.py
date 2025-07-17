import pandas as pd
import numpy as np
import os
import sklearn.preprocessing
import random

# Define constants.
NOT_ANOMALY = 0
ANOMALY = 1


# Default state function: returns the sensor value at the current time index.
def defaultStateFuc(timeseries, timeseries_curser, previous_state=None, action=None):
    return timeseries['value'][timeseries_curser]


# Default reward: returns +1 if action equals the ground truth anomaly value, else -1.
def defaultRewardFuc(timeseries, timeseries_curser, action):
    if action == timeseries['anomaly'][timeseries_curser]:
        return 1
    else:
        return -1


class EnvTimeSeriesfromRepo():
    def __init__(self, sensor_dir='SMD/ServerMachineDataset/test/', label_dir='SMD/ServerMachineDataset/test_label/'):
        """
        sensor_dir  : Directory that contains sensor CSV or TXT files.
        label_dir   : Directory that contains corresponding label files (one column of labels).
        Matching is done based on the file name without extension.
        """
        # List sensor files.
        self.sensor_files = []
        for subdir, dirs, files in os.walk(sensor_dir):
            for file in files:
                if file.endswith('.csv') or file.endswith('.txt'):
                    self.sensor_files.append(os.path.join(subdir, file))

        if len(self.sensor_files) == 0:
            raise ValueError("No sensor files found in directory: {}".format(sensor_dir))

        # List label files.
        self.label_files = {}
        for subdir, dirs, files in os.walk(label_dir):
            for file in files:
                if file.endswith('.csv') or file.endswith('.txt'):
                    # Use file base name (without extension) as key.
                    base_name = os.path.splitext(file)[0]
                    self.label_files[base_name] = os.path.join(subdir, file)

        if len(self.label_files) == 0:
            raise ValueError("No label files found in directory: {}".format(label_dir))

        self.action_space_n = 2
        self.statefnc = defaultStateFuc
        self.rewardfnc = defaultRewardFuc

        self.datasetsize = len(self.sensor_files)
        self.datasetfix = 0
        self.datasetidx = random.randint(0, self.datasetsize - 1)
        self.datasetrng = self.datasetsize

        self.timeseries = None  # Merged sensor and label data.
        self.timeseries_curser = -1
        self.timeseries_curser_init = 0
        self.timeseries_states = None
        self.states_list = []

        self.timeseries_repo = []
        # Pre-load and merge sensor and corresponding label files.
        for sensor_path in self.sensor_files:
            # Read sensor data: assume sensor values are in the first column.
            df_sensor = pd.read_csv(sensor_path, sep=",", header=None)
            df_sensor = df_sensor[[0]]
            df_sensor.columns = ['value']
            scaler = sklearn.preprocessing.MinMaxScaler()
            df_sensor['value'] = scaler.fit_transform(df_sensor[['value']])

            sensor_base = os.path.splitext(os.path.basename(sensor_path))[0]
            if sensor_base in self.label_files:
                label_path = self.label_files[sensor_base]
            else:
                raise ValueError("No matching label file found for sensor file: {}".format(sensor_path))

            # Read label file (assumed one column, no header)
            df_label = pd.read_csv(label_path, sep=",", header=None)
            df_label.columns = ['anomaly']
            # Trim both dataframes to minimum length.
            min_length = min(df_sensor.shape[0], df_label.shape[0])
            df_sensor = df_sensor.iloc[:min_length].reset_index(drop=True)
            df_label = df_label.iloc[:min_length].reset_index(drop=True)

            # Merge sensor and label data.
            df_merged = pd.concat([df_sensor, df_label], axis=1)
            # Add a "label" column for training (initially -1).
            df_merged['label'] = -1
            df_merged = df_merged.astype(np.float32)

            self.timeseries_repo.append(df_merged)

    def reset(self):
        """
        Reset the environment: select a new time series, reset cursor,
        compute the initial state, and update the states_list.
        """
        if self.datasetfix == 0:
            self.datasetidx = (self.datasetidx + 1) % self.datasetrng

        print("Loading sensor file:", self.sensor_files[self.datasetidx])
        base_name = os.path.splitext(os.path.basename(self.sensor_files[self.datasetidx]))[0]
        print("Using label file:", self.label_files.get(base_name, "Not found"))
        self.timeseries = self.timeseries_repo[self.datasetidx]
        self.timeseries_curser = self.timeseries_curser_init
        self.timeseries_states = self.statefnc(self.timeseries, self.timeseries_curser)
        self.states_list = self.get_states_list()
        return self.timeseries_states

    def reset_to(self, id):
        if id < 0 or id >= self.datasetrng:
            raise ValueError("Invalid dataset index: {}".format(id))
        self.datasetidx = id
        self.timeseries = self.timeseries_repo[self.datasetidx]
        self.timeseries_curser = self.timeseries_curser_init
        self.timeseries_states = self.statefnc(self.timeseries, self.timeseries_curser)
        self.states_list = self.get_states_list()
        return self.timeseries_states

    def get_states_list(self):
        self.timeseries = self.timeseries_repo[self.datasetidx]
        self.timeseries_curser = self.timeseries_curser_init
        state_list = []
        for cursor in range(self.timeseries_curser_init, self.timeseries.shape[0]):
            if len(state_list) == 0:
                state = self.statefnc(self.timeseries, cursor)
            else:
                state = self.statefnc(self.timeseries, cursor, state_list[-1])
                if isinstance(state, np.ndarray) and state.ndim > 1:
                    state = state[0]
            state_list.append(state)
        return state_list

    def step(self, action):
        """
        Execute one time step: obtain reward based on current state and action,
        then advance the time series cursor. Returns (state, reward, done, info).
        """
        # Compute reward using the environment's reward function.
        reward = self.rewardfnc(self.timeseries, self.timeseries_curser, action)
        self.timeseries_curser += 1
        if self.timeseries_curser >= self.timeseries.shape[0]:
            done = 1
            # If done, return the same state twice.
            state = np.array([self.timeseries_states, self.timeseries_states])
        else:
            done = 0
            state = self.statefnc(self.timeseries, self.timeseries_curser, self.timeseries_states, action)
        # Update stored state.
        if isinstance(state, np.ndarray) and state.ndim > np.array(self.timeseries_states).ndim:
            self.timeseries_states = state[action]
        else:
            self.timeseries_states = state
        return state, reward, done, {}
