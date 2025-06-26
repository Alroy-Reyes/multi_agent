from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces
import numpy as np


class TimetablingEnv(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "timetabling_env_v4_with_negotiation",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        num_sahas=4,
        num_cmas=2,
        num_teachers=5,
        num_subjects=12,
        num_timeslots=5,
        buildings_room_info=None,
        max_classes_per_teacher=3,
    ):
        super().__init__()

        # Configuration
        self.num_sahas = num_sahas
        self.num_cmas = num_cmas
        self.num_teachers = num_teachers
        self.num_subjects = num_subjects
        self.num_timeslots = num_timeslots
        self.max_classes = max_classes_per_teacher

        # Room info
        self.buildings_room_info = buildings_room_info or {0: ["lecture", "lab"], 1: ["lecture", "lecture"]}

        # Agent lists
        self.saha_agents = [f"saha_{i}" for i in range(self.num_sahas)]
        self.cma_agents = [f"cma_{i}" for i in range(self.num_cmas)]
        self.possible_agents = self.saha_agents + self.cma_agents
        self.agent_name_mapping = {a: i for i, a in enumerate(self.possible_agents)}

        # Action/observation spaces
        self.saha_action_space = spaces.Discrete(self.num_teachers * self.num_timeslots)
        self.saha_observation_space = spaces.Box(0.0, 1.0, (self.num_teachers,), np.float32)

        self.cma_action_spaces = {}
        self.cma_observation_spaces = {}
        for idx, bldg in enumerate(self.buildings_room_info):
            rooms = len(self.buildings_room_info[bldg])
            self.cma_action_spaces[f"cma_{idx}"] = spaces.Discrete(rooms * self.num_timeslots)
            self.cma_observation_spaces[f"cma_{idx}"] = spaces.Box(
                0.0, 1.0, (rooms * self.num_timeslots,), np.float32
            )

        self.action_spaces = {**{a: self.saha_action_space for a in self.saha_agents},
                              **self.cma_action_spaces}
        self.observation_spaces = {**{a: self.saha_observation_space for a in self.saha_agents},
                                   **self.cma_observation_spaces}

        # Internal state
        self.subject_assignments = np.full(self.num_subjects, -1, dtype=int)
        # Corrected comprehension without line continuation escape
        self.buildings_room_schedule = {
            b: np.full(
                (len(self.buildings_room_info[b]), self.num_timeslots),
                -1,
                dtype=int
            )
            for b in self.buildings_room_info
        }
        self.teacher_classes = {f"teacher_{i}": 0 for i in range(self.num_teachers)}

        # Runtime
        self.saha_teacher_scores = {}
        self.cma_observations = {}
        self.observations = {}
        self.current_reward = 0.0
        self.cma_placed = set()

    def reset(self, *, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        # Reset metrics
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        # Reset schedules
        self.subject_assignments.fill(-1)
        for b in self.buildings_room_schedule:
            self.buildings_room_schedule[b].fill(-1)
        self.teacher_classes = {t: 0 for t in self.teacher_classes}

        # New random preferences
        self.saha_teacher_scores = {
            a: np.random.rand(self.num_teachers).astype(np.float32)
            for a in self.saha_agents
        }

        self.update_cma_observations()
        self.cma_placed.clear()
        self.update_observations()
        self.current_reward = 0.0

    def observe(self, agent):
        return self.observations.get(
            agent,
            np.zeros_like(self.observation_spaces[agent].sample())
        )

    def step(self, action):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_done_step(None)
            return

        # Delegate to helper methods
        if agent.startswith("saha"):
            self._step_saha(agent, action)
        else:
            self._step_cma(agent, action)

        # Record reward
        self.rewards[agent] = self.current_reward
        self._cumulative_rewards[agent] += self.current_reward
        self.update_observations()

        # Advance agent selection
        if self.agents:
            self.agent_selection = self._agent_selector.next()
        if all(self.terminations[a] or self.truncations[a] for a in self.agents):
            self.agents = []

    def _step_saha(self, agent, action):
        idx = int(agent.split("_")[1])
        per = self.num_subjects // self.num_sahas
        start, end = idx * per, (idx + 1) * per if idx < self.num_sahas - 1 else self.num_subjects
        candidates = [s for s in range(start, end) if self.subject_assignments[s] == -1]
        if not candidates:
            self._terminate(agent)
            return
        subj = candidates[0]
        tid, timeslot = divmod(action, self.num_timeslots)
        teacher = f"teacher_{tid}"
        if self.teacher_classes[teacher] < self.max_classes:
            self.subject_assignments[subj] = tid
            self.teacher_classes[teacher] += 1
            self.current_reward = 1.0
        else:
            self.current_reward = -1.0

    def _step_cma(self, agent, action):
        idx = int(agent.split("_")[1])
        b = list(self.buildings_room_info.keys())[idx]
        room, timeslot = divmod(action, self.num_timeslots)
        per = self.num_subjects // self.num_cmas
        start, end = idx * per, (idx + 1) * per if idx < self.num_cmas - 1 else self.num_subjects
        candidates = [s for s in range(start, end) if self.subject_assignments[s] != -1 and s not in self.cma_placed]
        if not candidates:
            self._terminate(agent)
            return
        subj = candidates[0]
        if self.buildings_room_schedule[b][room][timeslot] == -1:
            self.buildings_room_schedule[b][room][timeslot] = subj
            self.cma_placed.add(subj)
            self.current_reward = 1.0
        else:
            result = self.resolve_conflict(idx, subj, room, timeslot)
            self.current_reward = result
            if result < 0:
                self._terminate(agent)
        self.update_cma_observations()

    def _terminate(self, agent):
        self.terminations[agent] = True
        self.agents.remove(agent)
        self._agent_selector = agent_selector(self.agents)
        if self.agents:
            self.agent_selection = self._agent_selector.reset()

    def resolve_conflict(self, cma_idx, subject, room, timeslot):
        b = list(self.buildings_room_info.keys())[cma_idx]
        slots = self.buildings_room_schedule[b]
        print(f"Conflict at CMA_{cma_idx}: room {room}, slot {timeslot} booked.")
        # Try each SAHA
        for saha in self.saha_agents:
            if self.negotiation_possible(saha, subject, cma_idx):
                # Find any free slot
                for r in range(slots.shape[0]):
                    for t in range(self.num_timeslots):
                        if slots[r][t] == -1:
                            slots[r][t] = subject
                            print(f"Reassigned subject {subject} to room {r}, slot {t}.")
                            return 2.0
        print("Negotiation failed: no feasible reassignment.")
        return -2.0

    def negotiation_possible(self, saha_agent, subject, cma_idx):
        tid = self.subject_assignments[subject]
        teacher = f"teacher_{tid}"
        # Block if teacher at capacity
        if self.teacher_classes.get(teacher, 0) >= self.max_classes:
            print(f"Negotiation block: {teacher} at capacity.")
            return False
        # Block if building is full
        b = list(self.buildings_room_info.keys())[cma_idx]
        if not np.any(self.buildings_room_schedule[b] == -1):
            print("Negotiation block: no free slots in building.")
            return False
        return True

    def update_cma_observations(self):
        for i, b in enumerate(self.buildings_room_info):
            self.cma_observations[f"cma_{i}"] = (
                self.buildings_room_schedule[b] == -1
            ).astype(np.float32).flatten()

    def update_observations(self):
        for a in self.possible_agents:
            if a.startswith("saha"):
                self.observations[a] = self.saha_teacher_scores[a]
            else:
                self.observations[a] = self.cma_observations[a]

    def render(self):
        pass

    def close(self):
        pass
