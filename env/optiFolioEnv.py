import gymnasium as gym
from gymnasium import spaces
import random as rd
import numpy as np
import pandas as pd

class optiFolioEnv(gym.Env):
    def __init__(self, dataset_path, initial_amount=10_000, lookback=20, max_days=252, target_vol=0.02):
        super(optiFolioEnv, self).__init__()

        self.dataset = pd.read_csv(dataset_path)
        self.dataset.sort_index(inplace=True)

        self.initial_amount = initial_amount
        self.lookback = lookback
        self.max_days = max_days
        self.days_passed = 0
        self.portfolio_value = self.initial_amount
        self.current_step = self.lookback
        self.target_vol = target_vol  

        self.return_cols = [col for col in self.dataset.columns if col.startswith("norm_returns_")]
        self.num_assets = len(self.return_cols)
        self.num_features = self.dataset.shape[1] - 1  

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.lookback * self.num_features,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.num_assets,), dtype=np.float32
        )

        self.recent_returns = []

    def _get_observation(self):
        df_slice = self.dataset.iloc[self.current_step - self.lookback:self.current_step, 1:]
        obs = df_slice.values.astype(np.float32).flatten()
        return obs

    def step(self, action):
        weights = np.array(action, dtype=np.float32)
        weights = np.clip(weights, 0, 1)
        if weights.sum() > 0:
            weights /= weights.sum()
        else:
            weights = np.ones_like(weights) / len(weights)

        returns = self.dataset.iloc[self.current_step][self.return_cols].values.astype(np.float32)
        portfolio_return = np.dot(weights, returns)

        self.recent_returns.append(portfolio_return)
        if len(self.recent_returns) > self.lookback:
            self.recent_returns.pop(0)

        portfolio_vol = np.std(self.recent_returns) if len(self.recent_returns) > 1 else 0.0

        reward = np.log(1 + portfolio_return)  
        if portfolio_vol > self.target_vol:
            penalty = (portfolio_vol - self.target_vol)
            reward -= penalty 

        self.portfolio_value *= (1 + portfolio_return)

        self.current_step += 1
        self.days_passed += 1
        done = self.current_step >= len(self.dataset) or self.days_passed >= self.max_days

        obs = self._get_observation() if not done else np.zeros(self.lookback * self.num_features, dtype=np.float32)

        info = {"portfolio_value": self.portfolio_value, "weights": weights, "portfolio_vol": portfolio_vol}

        return obs, reward, done, info

    def reset(self):
        self.current_step = rd.randint(self.lookback, len(self.dataset) - self.max_days - 1)
        self.days_passed = 0
        self.portfolio_value = self.initial_amount
        self.recent_returns = []

        obs = self._get_observation()
        return obs, {"portfolio_value": self.portfolio_value}
