from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces
import numpy as np
import re

class TimetablingEnv(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "timetabling_env_v5_with_negotiation_and_campuses",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        num_sahas: int = 4,
        num_cmas: int | None = None,
        num_teachers: int = 5,
        num_subjects: int = 12,
        num_timeslots: int = 5,
        num_days: int = 5,                           # <-- new
        room_codes: list[str] | None = None,
        subject_codes: list[str] | None = None,
        subject_campuses: list[list[str]] | None = None,
        max_classes_per_teacher: int = 3,
    ):
        super().__init__()

        # ─── CONFIG ───────────────────────────────────────────────────────────────
        self.num_sahas     = num_sahas
        self.num_teachers  = num_teachers
        self.num_subjects  = num_subjects
        self.num_timeslots = num_timeslots
        self.num_days      = num_days               # <-- store days
        self.max_classes   = max_classes_per_teacher

        # ─── BUILDING / ROOM SETUP ───────────────────────────────────────────────
        self.room_codes = room_codes or []
        self.building_keys: list[str] = []
        self.buildings_room_info: dict[str, list[str]] = {}
        for code in self.room_codes:
            bldg = code[0]
            if bldg not in self.buildings_room_info:
                self.building_keys.append(bldg)
                self.buildings_room_info[bldg] = []
            self.buildings_room_info[bldg].append(code)

        inferred_cmas = len(self.building_keys)
        if num_cmas is None:
            self.num_cmas = inferred_cmas
        else:
            if num_cmas != inferred_cmas:
                raise ValueError(
                    f"num_cmas override ({num_cmas}) != distinct buildings ({inferred_cmas})"
                )
            self.num_cmas = num_cmas

        # ─── SUBJECT / PRIORITY / CAMPUS SETUP ────────────────────────────────────
        self.subject_codes = subject_codes or [f"Subject_{i}" for i in range(self.num_subjects)]
        self.subject_track = []
        self.subject_level = []
        for code in self.subject_codes:
            m = re.match(r"([A-Za-z]+)(\d*)", code)
            track = m.group(1)
            level = int(m.group(2)) if m.group(2).isdigit() else 0
            self.subject_track.append(track)
            self.subject_level.append(level)
        self.max_level = max(self.subject_level, default=0)

        if subject_campuses is None:
            self.subject_campuses = [
                list(self.building_keys) for _ in range(self.num_subjects)
            ]
        else:
            self.subject_campuses = subject_campuses

        self.qualification = {
            f"teacher_{i}": set(self.subject_codes)
            for i in range(self.num_teachers)
        }

        # ─── AGENT LISTS ──────────────────────────────────────────────────────────
        self.saha_agents = [f"saha_{i}" for i in range(self.num_sahas)]
        self.cma_agents  = [f"cma_{b}"  for b in self.building_keys]
        self.possible_agents = self.saha_agents + self.cma_agents

        # ─── ACTION / OBSERVATION SPACES ─────────────────────────────────────────
        # Sahas choose (teacher × timeslot) as before
        self.saha_action_space      = spaces.Discrete(self.num_teachers * self.num_timeslots)
        self.saha_observation_space = spaces.Box(
            low=np.float32(0.0),
            high=np.float32(1.0),
            shape=(self.num_teachers,),
            dtype=np.float32
        )

        # CMAs now choose among (room × day × timeslot)
        self.cma_action_spaces      = {}
        self.cma_observation_spaces = {}
        for b in self.building_keys:
            n_rooms = len(self.buildings_room_info[b])
            size    = n_rooms * self.num_days * self.num_timeslots
            self.cma_action_spaces[f"cma_{b}"]      = spaces.Discrete(size)
            self.cma_observation_spaces[f"cma_{b}"] = spaces.Box(
                low=np.float32(0.0),
                high=np.float32(1.0),
                shape=(size,),
                dtype=np.float32
            )

        self.action_spaces      = {
            **{a: self.saha_action_space for a in self.saha_agents},
            **self.cma_action_spaces,
        }
        self.observation_spaces = {
            **{a: self.saha_observation_space for a in self.saha_agents},
            **self.cma_observation_spaces,
        }

        # initialize dynamic state
        self._init_state()

    def _init_state(self):
        # one teacher assignment per subject
        self.subject_assignments = np.full(self.num_subjects, -1, dtype=int)

        # now a 3-D schedule: (n_rooms, num_days, num_timeslots)
        self.buildings_room_schedule = {
            b: np.full(
                (len(self.buildings_room_info[b]), self.num_days, self.num_timeslots),
                -1,
                dtype=int
            )
            for b in self.building_keys
        }

        self.teacher_classes     = {f"teacher_{i}": 0 for i in range(self.num_teachers)}
        self.conflict_count      = 0
        self.negotiation_success = 0
        self.saha_teacher_scores = {}
        self.cma_observations    = {}
        self.observations        = {}
        self.current_reward      = 0.0
        self.cma_placed          = set()

    def reset(self, *, seed=None, options=None):
        self.agents            = self.possible_agents[:]
        self._agent_selector   = agent_selector(self.agents)
        self.agent_selection   = self._agent_selector.reset()
        self.rewards           = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations      = {a: False for a in self.agents}
        self.truncations       = {a: False for a in self.agents}
        self.infos             = {a: {} for a in self.agents}

        self._init_state()

        # random initial teacher‐scores for Sahas
        for a in self.saha_agents:
            self.saha_teacher_scores[a] = np.random.rand(self.num_teachers).astype(np.float32)

        self._update_cma_observations()
        self._update_all_observations()

        # cast all observations
        for a in self.possible_agents:
            self.observations[a] = np.array(self.observations[a], dtype=np.float32)

        # sanity‐check bounds
        for a in self.possible_agents:
            low  = self.observation_spaces[a].low
            high = self.observation_spaces[a].high
            if not np.allclose(self.observations[a], np.clip(self.observations[a], low, high)):
                raise ValueError(f"Observation for agent {a} out of bounds!")

    def observe(self, agent):
        return self.observations[agent]

    def step(self, action):
        agent = self.agent_selection
        if action is not None:
            if agent.startswith("saha"):
                self._step_saha(agent, action)
            else:
                self._step_cma(agent, action)

            # assign reward to current agent
            self.rewards[agent] = self.current_reward
            self._cumulative_rewards[agent] += self.current_reward

        # update all obs for next agent
        self._update_cma_observations()
        self._update_all_observations()

        # handle terminations
        if self.terminations.get(agent) or self.truncations.get(agent):
            self.agents.remove(agent)
            self._agent_selector = agent_selector(self.agents)
        if self.agents:
            self.agent_selection = self._agent_selector.next()

        # attach error_rate info
        for a in self.agents:
            self.infos[a]["error_rate"] = self.calculate_error_rate()

    def calculate_error_rate(self):
        total = self.num_subjects
        errors = total - len(self.cma_placed)
        return errors / total if total > 0 else 0.0

    def subject_priority(self, subj_idx: int) -> int:
        return (self.max_level + 1) - self.subject_level[subj_idx]

    def _step_saha(self, agent: str, action: int):
        idx   = int(agent.split("_")[1])
        per   = self.num_subjects // self.num_sahas
        start = idx * per
        end   = (idx + 1) * per if idx < self.num_sahas - 1 else self.num_subjects

        candidates = sorted(
            [s for s in range(start, end) if self.subject_assignments[s] == -1],
            key=self.subject_priority, reverse=True
        )
        if not candidates:
            self.current_reward       = 0.0
            self.terminations[agent]  = True
            return

        subj = candidates[0]
        tid, _ = divmod(action, self.num_timeslots)
        teacher_key = f"teacher_{tid}"
        code        = self.subject_codes[subj]
        valid_buildings = self.subject_campuses[subj]

        if code not in self.qualification[teacher_key] or not valid_buildings:
            self.current_reward = -1.0
            return

        self.subject_assignments[subj]      = tid
        self.teacher_classes[teacher_key]  += 1
        self.current_reward                 = 1.0

    def _step_cma(self, agent: str, action: int):
        bkey = agent.split("_", 1)[1]
        total_slots = self.num_days * self.num_timeslots
        room_idx    = action // total_slots
        rest        = action % total_slots
        day_idx, ts = divmod(rest, self.num_timeslots)

        # if all placed, terminate everyone
        if len(self.cma_placed) == self.num_subjects:
            for a in list(self.agents):
                self.terminations[a] = True
            return

        candidates = [
            s for s in range(self.num_subjects)
            if (
                self.subject_assignments[s] != -1
                and s not in self.cma_placed
                and bkey in self.subject_campuses[s]
            )
        ]
        if not candidates:
            self.current_reward       = 0.0
            self.terminations[agent]  = True
            return

        subj = sorted(candidates, key=self.subject_priority, reverse=True)[0]
        schedule = self.buildings_room_schedule[bkey]

        if schedule[room_idx, day_idx, ts] == -1:
            schedule[room_idx, day_idx, ts] = subj
            self.cma_placed.add(subj)
            self.current_reward = 1.0
        else:
            self.current_reward = self._resolve_conflict(subj)

    def _resolve_conflict(self, subject: int) -> float:
        self.conflict_count += 1
        if self._negotiation_possible(subject):
            self.negotiation_success += 1
            return 2.0
        return -2.0

    def _negotiation_possible(self, subject: int) -> bool:
        old_tid     = self.subject_assignments[subject]
        old_teacher = f"teacher_{old_tid}"
        if self.teacher_classes[old_teacher] >= self.max_classes:
            for alt in range(self.num_teachers):
                alt_key = f"teacher_{alt}"
                if self.teacher_classes[alt_key] < self.max_classes:
                    self.subject_assignments[subject] = alt
                    self.teacher_classes[old_teacher]  -= 1
                    self.teacher_classes[alt_key]      += 1
                    return True
            return False
        return True

    def _update_cma_observations(self):
        for b in self.building_keys:
            free = (self.buildings_room_schedule[b] == -1).astype(np.float32)
            self.cma_observations[f"cma_{b}"] = free.flatten()

    def _update_all_observations(self):
        for a in self.possible_agents:
            if a in self.saha_agents:
                self.observations[a] = self.saha_teacher_scores[a]
            else:
                self.observations[a] = self.cma_observations[a]

    def render(self):
        pass

    def close(self):
        pass
