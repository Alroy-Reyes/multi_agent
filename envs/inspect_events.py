import os
import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.logger import TBXLoggerCallback
from ray.tune import run
import gymnasium as gym  # Make sure you are using the correct package
import numpy as np
from gymnasium.spaces import Discrete, Box  # Correct imports for gymnasium

# Define your custom environment
class SimpleEnv(gym.Env):
    def __init__(self):
        super(SimpleEnv, self).__init__()
        self.action_space = Discrete(2)  # Example action space with 2 possible actions
        self.observation_space = Box(low=-1, high=1, shape=(4,), dtype=np.float32)  # Example observation space

    def reset(self, *, seed=None, options=None):
        # Reset logic here (example with a simple state)
        self.state = np.zeros(4)  # Reset state to an array of zeros
        info = {}  # Optional information
        return self.state, info

    def step(self, action):
        # Transition logic after the agent takes an action
        self.state = np.zeros(4)  # Example new state
        reward = 1.0  # Set a simple reward for the action
        done = False  # Whether the episode is done
        truncated = False  # Make sure this is a boolean (True or False)
        
        info = {}  # Additional information
        
        # Return in the correct format: (state, reward, done, info, truncated)
        return self.state, reward, done, info, truncated

# Register the environment
ray.tune.register_env("simple_env", lambda config: SimpleEnv())

# PPO Configuration
config = PPOConfig().environment("simple_env", disable_env_checking=True).framework("torch")


# Create PPO agent
ppo = PPO(config=config)

# Start Training
for i in range(10):
    result = ppo.train()
    print(f"Iteration: {i}, Reward: {result['episode_reward_mean']}")
