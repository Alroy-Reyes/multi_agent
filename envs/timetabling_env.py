from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from gymnasium import spaces
import numpy as np


class TimetablingEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "timetabling_v0"}

    def __init__(self):
        super().__init__()

        self.possible_agents = [f"agent_{i}" for i in range(3)]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}

        self.action_spaces = {agent: spaces.Discrete(5) for agent in self.possible_agents}
        self.observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
            for agent in self.possible_agents
        }

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._agent_selector = iter(self.agents)
        self.agent_selection = self.agents[0]
        self.has_reset = True

        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.observations = {
            agent: np.random.rand(10).astype(np.float32)
            for agent in self.agents
        }

    def observe(self, agent):
        return self.observations[agent]

    def step(self, action):
        agent = self.agent_selection

        # Random reward for testing
        reward = np.random.rand()
        self.rewards[agent] = reward
        self._cumulative_rewards[agent] += reward

        self.terminations[agent] = True
        self.truncations[agent] = False
        self.dones[agent] = True

        # Select next agent
        self.agent_selection = next(self._agent_selector, None)
        if self.agent_selection is None:
            self.agents = []

    def render(self):
        print("Rendering... (not implemented)")

    def close(self):
        pass


def env():
    return wrappers.CaptureStdoutWrapper(TimetablingEnv())
