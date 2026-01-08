import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random as rd


class optiFolioEnv(gym.Env):
    """
    Custom Gymnasium environment for portfolio optimization with PPO.
    """

    def __init__(
        self,
        dataset_path,
        initial_amount=10_000,
        lookback=20,
        max_days=252,
        target_vol=0.02
    ):
        super().__init__()

        # Load and clean dataset
        self.dataset = pd.read_csv(dataset_path)
        self.dataset.sort_index(inplace=True)
        self.dataset.fillna(0.0, inplace=True)

        # Environment parameters
        self.initial_amount = initial_amount
        self.lookback = lookback
        self.max_days = max_days
        self.target_vol = target_vol

        # Internal state
        self.current_step = lookback
        self.days_passed = 0
        self.portfolio_value = initial_amount

        # Asset returns columns
        self.return_cols = [
            col for col in self.dataset.columns
            if col.startswith("norm_returns_")
        ]
        self.num_assets = len(self.return_cols)

        # Number of features used in observations
        # (excluding date/index column)
        self.num_features = self.dataset.shape[1] - 1

        # Observation space
        # Flattened lookback window + target volatility
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback * self.num_features + 1,),
            dtype=np.float32
        )

        # Action space
        # PPO outputs free values, we normalize later
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_assets,),
            dtype=np.float32
        )

        # Volatility estimation (EWMA)
        self.lambda_vol = 0.94
        self.portfolio_var = 0.0

        # Previous weights (for turnover penalty)
        self.prev_weights = None

    # Build observation
    def _get_observation(self):
        """
        Build observation using rolling lookback window
        and append target volatility.
        """
        df_slice = self.dataset.iloc[
            self.current_step - self.lookback : self.current_step,
            1:
        ]

        obs = df_slice.values.astype(np.float32).flatten()

        # Append target volatility as a feature
        obs = np.append(obs, self.target_vol).astype(np.float32)

        # Numerical safety
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return obs

    # Environment step
    def step(self, action):
        """
        Execute one step in the environment.
        """

        # Action processing
        action = np.clip(action, -1.0, 1.0)

        # Softmax-style normalization to get valid weights
        weights = np.exp(action)
        weights /= weights.sum()

        # Get asset returns
        returns = self.dataset.iloc[self.current_step][
            self.return_cols
        ].values.astype(np.float32)

        # Portfolio return
        portfolio_return = np.dot(weights, returns)

        # Safety clip
        portfolio_return = np.clip(portfolio_return, -0.999, None)

        # Update portfolio value
        self.portfolio_value *= (1.0 + portfolio_return)

        # Volatility (EWMA)
        self.portfolio_var = (
            self.lambda_vol * self.portfolio_var +
            (1.0 - self.lambda_vol) * portfolio_return**2
        )

        portfolio_vol = np.sqrt(self.portfolio_var) * np.sqrt(252)

        # Reward design (V2)

        # Log-return
        r = np.log(1.0 + portfolio_return)

        # Alpha bonus (reward good trades more)
        alpha = 0.5
        alpha_bonus = alpha * max(0.0, r)

        # Volatility penalty (only above target)
        beta = 0.3
        vol_excess = max(0.0, portfolio_vol - self.target_vol)
        vol_penalty = beta * min(vol_excess / self.target_vol, 1.0)

        # Turnover penalty (avoid over-trading)
        delta = 0.05
        if self.prev_weights is None:
            turnover_penalty = 0.0
        else:
            turnover_penalty = delta * np.sum(
                np.abs(weights - self.prev_weights)
            )

        # Final reward
        reward = (
            r
            + alpha_bonus
            - vol_penalty
            - turnover_penalty
        )

        # Clip reward for PPO stability
        reward = float(np.clip(reward, -2.0, 2.0))

        # Store previous weights
        self.prev_weights = weights.copy()

        # Advance time
        self.current_step += 1
        self.days_passed += 1

        # Termination conditions
        terminated = self.current_step >= len(self.dataset)
        truncated = self.days_passed >= self.max_days
        done = terminated or truncated

        # Next observation
        if not done:
            obs = self._get_observation()
        else:
            obs = np.zeros(
                self.lookback * self.num_features + 1,
                dtype=np.float32
            )

        # Safety checks
        assert np.isfinite(obs).all(), "NaN in observation"
        assert np.isfinite(reward), "NaN in reward"

        # Extra info
        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_vol": portfolio_vol,
            "weights": weights
        }

        return obs, reward, terminated, truncated, info

    # Reset environment
    def reset(self, *, seed=None, options=None):
        """
        Reset environment to a random starting point.
        """
        if seed is not None:
            rd.seed(seed)

        self.current_step = rd.randint(
            self.lookback,
            len(self.dataset) - self.max_days - 1
        )

        self.days_passed = 0
        self.portfolio_value = self.initial_amount
        self.portfolio_var = 0.0
        self.prev_weights = None

        obs = self._get_observation()
        info = {"portfolio_value": self.portfolio_value}

        return obs, info
