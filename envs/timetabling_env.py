# envs/timetabling_env.py

from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces
import numpy as np


class TimetablingEnv(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "timetabling_env_v2",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        num_sahas=2,
        num_cmas=2,
        num_teachers=5,
        num_subjects=6,
        num_timeslots=5,
        buildings_room_info=None,
        max_classes_per_teacher=3,
    ):
        super().__init__()

        # --- CONFIGURATION ---
        self.num_sahas = num_sahas
        self.num_cmas = num_cmas
        self.num_teachers = num_teachers
        self.num_subjects = num_subjects
        self.num_timeslots = num_timeslots
        self.max_classes = max_classes_per_teacher

        # Default to two buildings if none provided
        self.buildings_room_info = (
            buildings_room_info
            if buildings_room_info is not None
            else {0: ["lecture", "lab"], 1: ["lecture", "lecture"]}
        )

        # --- AGENT SETUP ---
        self.saha_agents = [f"saha_{i}" for i in range(self.num_sahas)]
        self.cma_agents = [f"cma_{i}" for i in range(self.num_cmas)]
        self.possible_agents = self.saha_agents + self.cma_agents
        self.agent_name_mapping = {
            agent: i for i, agent in enumerate(self.possible_agents)
        }

        # --- ACTION SPACES ---
        # SAHA: pick a teacher index ∈ [0, num_teachers)
        self.saha_action_space = spaces.Discrete(self.num_teachers)
        # CMA: pick one of (room × timeslot) pairs
        self.cma_action_spaces = {}
        for i, building_id in enumerate(self.buildings_room_info):
            n_rooms = len(self.buildings_room_info[building_id])
            self.cma_action_spaces[f"cma_{i}"] = spaces.Discrete(
                n_rooms * self.num_timeslots
            )

        # --- OBSERVATION SPACES ---
        # SAHA observes a vector length=num_teachers of “scores”
        self.saha_observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_teachers,), dtype=np.float32
        )
        # CMA observes a flattened binary map: (room × timeslot) free/occupied
        self.cma_observation_spaces = {}
        for i, building_id in enumerate(self.buildings_room_info):
            n_rooms = len(self.buildings_room_info[building_id])
            self.cma_observation_spaces[f"cma_{i}"] = spaces.Box(
                low=0,
                high=1,
                shape=(n_rooms * self.num_timeslots,),
                dtype=np.float32,
            )

        # Combine into per-agent dicts
        self.action_spaces = {}
        self.observation_spaces = {}
        for agent in self.possible_agents:
            if agent.startswith("saha"):
                self.action_spaces[agent] = self.saha_action_space
                self.observation_spaces[agent] = self.saha_observation_space
            else:
                self.action_spaces[agent] = self.cma_action_spaces[agent]
                self.observation_spaces[agent] = self.cma_observation_spaces[agent]

        # --- INTERNAL STATE ---
        # subject_assignments[s] = teacher_index or –1 if unassigned
        self.subject_assignments = np.full(self.num_subjects, -1, dtype=int)
        # building_id → 2D array (n_rooms × num_timeslots), each entry = subject_index or –1
        self.buildings_room_schedule = {}
        # track classes assigned to each “teacher_i”
        self.teacher_classes = {f"teacher_{i}": 0 for i in range(self.num_teachers)}

        # These will be initialized/reset in reset()
        self.saha_teacher_scores = {}
        self.cma_observations = {}
        self.observations = {}
        self.printed_agents = set()

        # Keep track of which subjects each CMA has already placed
        # so that they don’t keep placing the same subject over and over.
        self.cma_placed = set()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, *, seed=None, options=None):
        # --- Reset AEC plumbing ---
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        # Reset per-agent reward/termination structures
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # --- Reset subject assignments & teacher loads ---
        self.subject_assignments[:] = -1
        self.teacher_classes = {f"teacher_{i}": 0 for i in range(self.num_teachers)}

        # --- Reset room schedules for every building ---
        self.buildings_room_schedule.clear()
        for bldg_id, room_types in self.buildings_room_info.items():
            n_rooms = len(room_types)
            self.buildings_room_schedule[bldg_id] = np.full(
                (n_rooms, self.num_timeslots), fill_value=-1, dtype=int
            )

        # --- Build random “teacher scores” for SAHAs to observe ---
        self.saha_teacher_scores = {
            agent: np.random.rand(self.num_teachers).astype(np.float32)
            for agent in self.saha_agents
        }

        # --- Build initial CMA observations (all rooms empty) ---
        self.update_cma_observations()

        # Clear the per-episode “which subjects have CMA-placed” set
        self.cma_placed.clear()

        # Build the initial combined observations dictionary
        self.printed_agents.clear()
        self.update_observations()

        # **DO NOT RETURN ANYTHING** here. The PettingZoo AEC wrapper
        # will immediately call `env.observe(env.agent_selection)` next.

    def update_cma_observations(self):
        """Flatten each building’s (room x timeslot) free/occupied map into a 1D array."""
        self.cma_observations = {}
        for i, bldg_id in enumerate(self.buildings_room_info):
            schedule = self.buildings_room_schedule[bldg_id]
            flat_avail = (schedule == -1).astype(np.float32).flatten()
            self.cma_observations[f"cma_{i}"] = flat_avail

    def update_observations(self):
        """Assemble a single `self.observations` dict that maps each agent → its own vector."""
        self.observations = {}
        for agent in self.possible_agents:
            if agent.startswith("saha"):
                self.observations[agent] = self.saha_teacher_scores[agent]
            else:
                self.observations[agent] = self.cma_observations[agent]

    def observe(self, agent):
        """Called by PettingZooEnv wrapper to fetch the current observation for `agent`."""
        return self.observations.get(
            agent,
            np.zeros(self.observation_space(agent).shape, dtype=np.float32),
        )

    def step(self, action):
        """
        AEC-style step:
          - Only called for `agent = self.agent_selection`.
          - We do NOT return anything; the wrapper will collect (obs, reward, done, info).
        """
        agent = self.agent_selection

        # If this agent is already marked done, skip it immediately.
        if self.terminations[agent] or self.truncations[agent]:
            self._was_done_step(None)
            return

        reward = 0.0

        if agent.startswith("saha"):
            # -------- SAHA logic: assign a teacher to the next unassigned subject in this SAHA’s block --------
            saha_idx = int(agent.split("_")[1])
            subjects_per_saha = self.num_subjects // self.num_sahas
            start_subj = saha_idx * subjects_per_saha
            end_subj = (
                (saha_idx + 1) * subjects_per_saha
                if saha_idx < self.num_sahas - 1
                else self.num_subjects
            )

            # Find lowest‐index unassigned subject in this SAHA’s range
            subj_candidates = [
                s
                for s in range(start_subj, end_subj)
                if self.subject_assignments[s] == -1
            ]

            if not subj_candidates:
                # No more subjects to assign → terminate SAHA
                self.terminations[agent] = True
                # Remove from the AEC cycle:
                self.agents.remove(agent)
                self._agent_selector = agent_selector(self.agents)
                if self.agents:
                    self.agent_selection = self._agent_selector.reset()
                return
            else:
                subject = subj_candidates[0]
                teacher_idx = action  # action = teacher index

                teacher_agent_name = f"teacher_{teacher_idx}"
                # Only assign if that teacher still has capacity
                if self.teacher_classes[teacher_agent_name] < self.max_classes:
                    self.subject_assignments[subject] = teacher_idx
                    self.teacher_classes[teacher_agent_name] += 1
                    reward = 1.0  # success
                else:
                    reward = -1.0  # penalty (teacher at max load)

                # Note: We do not immediately kill the SAHA even if the teacher just filled up.
                # The SAHA only terminates once all its subjects are assigned.

        else:
            # -------- CMA logic: assign a room & timeslot to an already‐assigned subject in this CMA’s block --------
            cma_idx = int(agent.split("_")[1])
            bldg_id = list(self.buildings_room_info.keys())[cma_idx]
            n_rooms = len(self.buildings_room_info[bldg_id])

            subjects_per_cma = self.num_subjects // self.num_cmas
            start_subj = cma_idx * subjects_per_cma
            end_subj = (
                (cma_idx + 1) * subjects_per_cma
                if cma_idx < self.num_cmas - 1
                else self.num_subjects
            )

            # Only consider those subjects in our block that have been assigned a teacher
            # AND that the CMA has not yet placed in a room.
            subj_candidates = [
                s
                for s in range(start_subj, end_subj)
                if (self.subject_assignments[s] != -1 and s not in self.cma_placed)
            ]

            if not subj_candidates:
                # Nothing left to place → terminate CMA
                self.terminations[agent] = True
                self.agents.remove(agent)
                self._agent_selector = agent_selector(self.agents)
                if self.agents:
                    self.agent_selection = self._agent_selector.reset()
                return
            else:
                subject = subj_candidates[0]
                # Decode action into (room, timeslot)
                room = action // self.num_timeslots
                timeslot = action % self.num_timeslots

                # If that room-timeslot is free, place it
                if self.buildings_room_schedule[bldg_id][room][timeslot] == -1:
                    self.buildings_room_schedule[bldg_id][room][timeslot] = subject
                    reward = 1.0
                    # Mark this subject as “already placed” so CMA won’t reuse it
                    self.cma_placed.add(subject)
                else:
                    reward = -1.0  # penalty for double‐booking

                # Refresh CMA observations after any change
                self.update_cma_observations()

        # Store and accumulate this agent’s reward
        self.rewards[agent] = reward
        self._cumulative_rewards[agent] += reward

        # Refresh every agent’s observation vector
        self.update_observations()

        # Advance to the next agent in the AEC cycle (if any remain)
        if self.agents:
            self.agent_selection = self._agent_selector.next()

        # If all agents are done, clear self.agents so AEC sees “episode over”
        if all(self.terminations[a] or self.truncations[a] for a in self.agents):
            self.agents = []

        # Do NOT return anything: the wrapper will gather (obs, rewards, dones, infos).

    def _was_done_step(self, action):
        """
        Called when an agent was already marked done before its turn. We simply
        remove it and move on to the next agent.
        """
        agent = self.agent_selection
        self._cumulative_rewards[agent] += self.rewards.get(agent, 0.0)
        if agent in self.agents:
            self.agents.remove(agent)
            self._agent_selector = agent_selector(self.agents)
            if self.agents:
                self.agent_selection = self._agent_selector.next()

    def render(self):
        # No-op for “silent” mode. You can implement a human-facing snapshot if you like.
        pass

    def close(self):
        pass
