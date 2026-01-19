import gymnasium as gym
import numpy as np
import pandas as pd


class OptiFolioEnv(gym.Env):
    def __init__(self, dataframe: pd.DataFrame, return_data : pd.DataFrame, initial_amount=1000, lookback=30):
        super(OptiFolioEnv, self).__init__()

        self.initial_amount = initial_amount
        self.portfolio_value = self.initial_amount
        self.lookback = lookback
        self.n_assets = len(return_data.columns)
        self.n_features = len(dataframe.columns)
        
        self.feature_data = dataframe.values.astype(np.float32)
        self.return_data = return_data.values.astype(np.float32)
        self.n_steps = self.feature_data.shape[0]

        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(self.n_assets,), dtype=np.float32
        )
    
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.lookback, self.n_features),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.portfolio_value = self.initial_amount
        self.current_step = self.lookback
        
        observation = self.feature_data[self.current_step - self.lookback : self.current_step]
        info = {"portfolio_value": self.portfolio_value,}
        return observation, info
    
    def step(self, action): 
        weights = np.array(action, dtype=np.float32)
        weights = weights / (np.sum(weights) + 1e-8)
        
        current_returns = self.return_data[self.current_step]
        portfolio_return = np.dot(weights, current_returns)
        
        new_portfolio_value = self.portfolio_value * (1 + portfolio_return)

        reward = (new_portfolio_value - self.portfolio_value) / self.portfolio_value
        
        self.portfolio_value = new_portfolio_value
        self.current_step += 1
        
        terminated = self.current_step >= self.n_steps - 1
        truncated = False
        
        next_observation = self.feature_data[self.current_step - self.lookback : self.current_step]
        
        info = {
            "portfolio_value": self.portfolio_value,
            "weights": weights,
            "return": portfolio_return
        }

        return next_observation, reward, terminated, truncated, info