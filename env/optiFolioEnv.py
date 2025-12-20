import pandas as pd
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import os

class OptiFolioEnv(gym.Env):
    def __init__(self, csv_folder: str, window_size: int = 20, initial_amount: float = 1000.0, transaction_fee: float = 0.001):
        super(OptiFolioEnv, self).__init__()
        self.df = self.load_dataframes(os.path.normpath(csv_folder))
        self.window_size = window_size
        self.transaction_fee = transaction_fee
        self.portfolio_value = initial_amount
        self.price_columns = [col for col in self.df.columns if col.endswith("_Close")]
        self.n_assets = len(self.price_columns)

        self.current_step = self.window_size
        self.prev_weights = np.ones(self.n_assets) / self.n_assets

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low= -np.inf,
            high= np.inf,
            shape= (self.window_size, self.df.shape[1]),
            dtype= np.float32
        )

    def step(self, action):
        old_portfolio_value = self.portfolio_value

        action = np.clip(action, 0.0, 1.0)
        weights = action / (np.sum(action) + 1e-8)

        prices_now = self.df[self.price_columns].iloc[self.current_step].values
        prices_next = self.df[self.price_columns].iloc[self.current_step + 1].values

        returns = (prices_next - prices_now) / (prices_now + 1e-8)

        portfolio_return = np.dot(weights, returns)
        self.portfolio_value *= (1 + portfolio_return)

        turnover = np.sum(np.abs(weights - self.prev_weights))
        cost = turnover * self.portfolio_value * self.transaction_fee
        self.portfolio_value -= cost

        reward = np.log(self.portfolio_value / (old_portfolio_value + 1e-8))

        self.prev_weights = weights
        self.current_step += 1

        obs = self.df.iloc[
            self.current_step - self.window_size : self.current_step
        ].values.astype(np.float32)

        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        if self.portfolio_value <= 0:
            truncated = True

        info = {
            "portfolio_value": self.portfolio_value
        }
        
        return obs, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.portfolio_value = 1000.0
        self.prev_weights = np.ones(self.n_assets) / self.n_assets

        obs = self.df[self.price_columns].iloc[
            self.current_step - self.window_size : self.current_step
        ].values.astype(np.float32)

        return obs, {}

    def load_dataframes(self, csv_path: str) -> pd.DataFrame:
        all_dfs = []
        
        for file in os.listdir(csv_path):
            if file.endswith('.csv'):
                asset_name = os.path.splitext(file)[0]
                full_path = os.path.join(csv_path, file)
                
                temp_df = pd.read_csv(full_path)
                
                temp_df['Date'] = pd.to_datetime(temp_df['Date'], errors='coerce')
                temp_df.set_index('Date', inplace=True)
                
                temp_df = temp_df.add_prefix(f"{asset_name}_")
                all_dfs.append(temp_df)

        if not all_dfs:
            return pd.DataFrame()

        final_df = pd.concat(all_dfs, axis=1, join='outer')
        final_df.sort_index(inplace=True)
        
        final_df.ffill(inplace=True) 

        final_df.fillna(0, inplace=True) 
        
        return final_df