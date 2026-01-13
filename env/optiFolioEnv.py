import gymnasium as gym

import pandas as pd 

class OptiFolioEnv(gym.Env):
    """
    Custom Gymnasium environment for portfolio optimization.
    """
    def __init__(self, dataframe : pd.DataFrame, initial_amount = 1_000):
        super(OptiFolioEnv).__init__()
        self.action_space = gym.spaces.Discrete() 


    def reset(self, *, seed = None, options = None):
        pass

    def step(self,action):
        pass

        