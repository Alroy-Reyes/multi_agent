from pettingzoo import AECEnv
from gymnasium import spaces
import numpy as np

class TimetablingEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "timetabling_v0"}

    def __init__(self):
        super().__init__()
        self.possible_agents = [f"agent_{i}" for i in range(3)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.agents)}
        self.action_spaces = {agent: spaces.Discrete(10) for agent in self.agents}
        self.observation_spaces = {
            agent: spaces.Box(0, 1, shape=(10,), dtype=np.float32)
            for agent in self.agents
        }

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._agent_selector = iter(self.agents)
        self.current_agent = next(self._agent_selector)
        self.dones = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.observations = {
            agent: np.random.rand(10).astype(np.float32)
            for agent in self.agents
        }

    def observe(self, agent):
        return self.observations[agent]

    def step(self, action):
        agent = self.current_agent
        self.rewards[agent] = np.random.random()
        self.dones[agent] = True
        self.infos[agent] = {}
        self.current_agent = next(self._agent_selector, None)
        if self.current_agent is None:
            self.agents = []

    def render(self):
        print("Render not implemented.")
