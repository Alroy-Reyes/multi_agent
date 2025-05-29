from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces
import numpy as np


class TimetablingEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "timetabling_env_v0"}

    def __init__(self):
        super().__init__()
        self._num_agents = 3
        self.num_timeslots = 10

        self.possible_agents = [f"agent_{i}" for i in range(self._num_agents)]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}

        self.action_spaces = {
            agent: spaces.Discrete(self.num_timeslots)
            for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(self.num_timeslots,), dtype=np.float32)
            for agent in self.possible_agents
        }

        self.reset()

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.teacher_availability = np.random.randint(0, 2, size=(self._num_agents, self.num_timeslots)).astype(np.float32)
        self.teacher_preferences = np.random.rand(self._num_agents, self.num_timeslots).astype(np.float32)
        self.room_available = np.random.randint(0, 2, size=self.num_timeslots).astype(np.float32)

        self.observations = {
            agent: self.teacher_availability[self.agent_name_mapping[agent]]
            for agent in self.agents
        }

    def observe(self, agent):
        obs=np.array(self.observations[agent], dtype=np.float32)
        print(f"[DEBUG] Observation for {agent}: {obs}")
        return obs

    def step(self, action):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self.agent_selection=self._agent_selector.next()
            return

        idx = self.agent_name_mapping[agent]
        timeslot = action
        reward = 0.0

        # ❌ Hard Constraint: Teacher unavailable
        if self.teacher_availability[idx][timeslot] == 0:
            reward -= 1.0

        # ❌ Hard Constraint: Room unavailable
        if self.room_available[timeslot] == 0:
            reward -= 1.0

        # ✅ Soft Constraint: Teacher prefers this timeslot
        preference_score = self.teacher_preferences[idx][timeslot]
        reward += 0.2 * preference_score

        self.rewards[agent] = reward
        self._cumulative_rewards[agent] += reward
        self.terminations[agent] = True

        self.agent_selection = self._agent_selector.next()

    def render(self):
        print(f"Current agent: {self.agent_selection}")
        for agent in self.possible_agents:
            print(f"{agent} reward: {self.rewards.get(agent, 0)}")

    def close(self):
        pass


def env():
    from pettingzoo.utils import wrappers
    return wrappers.CaptureStdoutWrapper(TimetablingEnv())
