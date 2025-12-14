import gym
from gym import spaces
import pandas as pd
import numpy as np
import os

class OptiFolioEnv(gym.Env):
    def __init__(self, csv_folder, window_size=20, initial_cash=1000):
        super().__init__()
        self.window_size = window_size
        self.initial_cash = initial_cash

        self.assets = []
        self.dfs = []
        for file in os.listdir(csv_folder):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(csv_folder, file), parse_dates=["Date"])
                df.set_index("Date", inplace=True)
                self.dfs.append(df)
                self.assets.append(file.split(".")[0])

        self.df = pd.concat(self.dfs, axis=1, keys=self.assets).dropna()
        self.n_assets = len(self.assets)
        self.n_features = len(self.dfs[0].columns)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, self.n_assets * self.n_features),
            dtype=np.float32
        )
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.portfolio_value = self.initial_cash
        self.weights = np.zeros(self.n_assets)
        self.cash = self.initial_cash
        self.shares = np.zeros(self.n_assets)
        return self._get_observation()

    def _get_observation(self):
        obs_window = self.df.iloc[self.current_step-self.window_size:self.current_step]
        return obs_window.values.reshape(self.window_size, -1).astype(np.float32)

    def step(self, action):
        action = action / np.sum(action) if np.sum(action) > 0 else np.ones_like(action)/len(action)
        prices = np.array([self.df.iloc[self.current_step][(asset, "Adj_Close")] for asset in self.assets])
        total_value = self.cash + np.sum(self.shares * prices)
        new_portfolio_value = total_value

        self.weights = action
        self.shares = (self.weights * total_value) / prices
        self.cash = 0

        new_total_value = np.sum(self.shares * prices)
        reward = new_total_value - total_value
        self.portfolio_value = new_total_value

        self.current_step += 1
        done = self.current_step >= len(self.df)
        return self._get_observation(), reward, done, {}
