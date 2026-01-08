import gymnasium as gym
from gymnasium import spaces
import random as rd
import numpy as np
import pandas as pd

class optiFolioEnv(gym.Env):
    def __init__(
        self,
        dataset_path,
        initial_amount=10_000,
        lookback=20,
        max_days=31,
        target_vol=0.02
    ):
        """
        Custom Gym environment for portfolio optimization.
        """
        super(optiFolioEnv, self).__init__()

        # Load dataset
        self.dataset = pd.read_csv(dataset_path)
        self.dataset.sort_index(inplace=True)
        self.dataset.fillna(0, inplace=True)

        # Environment parameters
        self.initial_amount = initial_amount
        self.lookback = lookback
        self.max_days = max_days
        self.target_vol = target_vol

        # Internal state
        self.days_passed = 0
        self.portfolio_value = self.initial_amount
        self.current_step = self.lookback

        # Identify return columns (used for portfolio returns)
        self.return_cols = [
            col for col in self.dataset.columns if col.startswith("norm_returns_")
        ]
        self.num_assets = len(self.return_cols)

        # Number of features used in observations (excluding date/index column)
        self.num_features = self.dataset.shape[1] - 1

        # Observation space: flattened lookback window + target volatility
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback * self.num_features + 1,),
            dtype=np.float32
        )

        # Action space: portfolio weights for each asset
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.num_assets,),
            dtype=np.float32
        )

        # Store recent portfolio returns for volatility estimation
        self.recent_returns = []

        # Used to calculate volatility
        self.lambda_vol = 0.94
        self.portfolio_var = 0.0

    def _get_observation(self):
        """
        Build the observation vector using a rolling lookback window
        and append the target volatility.
        """
        df_slice = self.dataset.iloc[
            self.current_step - self.lookback : self.current_step, 1:
        ]

        obs = df_slice.values.astype(np.float32).flatten()

        # Append target volatility as an additional feature
        obs = np.append(obs, self.target_vol).astype(np.float32)

        return obs

    def step(self, action):
        """
        Execute one environment step using portfolio weights as action.
        """
        # Convert action to valid portfolio weights
        weights = np.array(action, dtype=np.float32)
        weights = np.clip(weights, 0, 1)

        if weights.sum() > 0:
            weights /= weights.sum()
        else:
            weights = np.ones_like(weights) / len(weights)

        # Get asset returns at current timestep
        returns = self.dataset.iloc[self.current_step][
            self.return_cols
        ].values.astype(np.float32)

        # Compute portfolio return
        portfolio_return = np.dot(weights, returns)

        # Store recent returns for volatility calculation
        self.recent_returns.append(portfolio_return)
        if len(self.recent_returns) > self.lookback:
            self.recent_returns.pop(0)

        # Compute EWMA portfolio volatility
        self.portfolio_var = (
            self.lambda_vol * self.portfolio_var +
            (1 - self.lambda_vol) * portfolio_return**2
        )

        portfolio_vol = np.sqrt(self.portfolio_var) * np.sqrt(252)

        # Reward: return penalized by excess volatility
        alpha = 2.0 
        excess_vol = max(0.0, portfolio_vol - self.target_vol)
        reward = portfolio_return - alpha * excess_vol

        # Update portfolio value
        self.portfolio_value *= (1 + portfolio_return)

        # Advance time
        self.current_step += 1
        self.days_passed += 1

        # Episode termination conditions
        terminated = self.current_step >= len(self.dataset)
        truncated = self.days_passed >= self.max_days
        done = terminated or truncated

        # Build next observation
        if not done:
            obs = self._get_observation()
        else:
            obs = np.zeros(
                self.lookback * self.num_features + 1, dtype=np.float32
            )

        # Additional diagnostic information
        info = {
            "portfolio_value": self.portfolio_value,
            "weights": weights,
            "portfolio_vol": portfolio_vol
        }

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to a random starting point.
        """
        if seed is not None:
            rd.seed(seed)

        # Random start to avoid overfitting to a fixed period
        self.current_step = rd.randint(
            self.lookback, len(self.dataset) - self.max_days - 1
        )

        self.days_passed = 0
        self.portfolio_value = self.initial_amount
        self.recent_returns = []

        obs = self._get_observation()
        info = {"portfolio_value": self.portfolio_value}

        return obs, info
