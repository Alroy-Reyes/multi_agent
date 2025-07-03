from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces
import numpy as np

class TimetablingEnv(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "timetabling_env_v5_with_negotiation",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        num_sahas: int = 4,
        num_cmas: int | None = None,
        num_teachers: int = 5,
        num_subjects: int = 12,
        num_timeslots: int = 5,
        room_codes: list[str] | None = None,
        max_classes_per_teacher: int = 3,
    ):
        super().__init__()

        # Config
        self.num_sahas     = num_sahas
        self.num_teachers  = num_teachers
        self.num_subjects  = num_subjects
        self.num_timeslots = num_timeslots
        self.max_classes   = max_classes_per_teacher

        # Flat list of all rooms
        self.room_codes = room_codes or []

        # Group rooms by building prefix
        self.building_keys = []
        self.buildings_room_info: dict[str, list[str]] = {}
        for code in self.room_codes:
            bldg = code[0]
            if bldg not in self.buildings_room_info:
                self.building_keys.append(bldg)
                self.buildings_room_info[bldg] = []
            self.buildings_room_info[bldg].append(code)

        # CMA count from distinct buildings
        inferred_cmas = len(self.building_keys)
        if num_cmas is None:
            self.num_cmas = inferred_cmas
        else:
            if num_cmas != inferred_cmas:
                raise ValueError(
                    f"num_cmas override ({num_cmas}) != buildings ({inferred_cmas})"
                )
            self.num_cmas = num_cmas

        # Agents
        self.saha_agents     = [f"saha_{i}" for i in range(self.num_sahas)]
        self.cma_agents      = [f"cma_{b}" for b in self.building_keys]
        self.possible_agents = self.saha_agents + self.cma_agents

        # Spaces
        self.saha_action_space      = spaces.Discrete(self.num_teachers * self.num_timeslots)
        self.saha_observation_space = spaces.Box(0.0, 1.0, (self.num_teachers,), np.float32)

        self.cma_action_spaces      = {}
        self.cma_observation_spaces = {}
        for b in self.building_keys:
            n_rooms = len(self.buildings_room_info[b])
            self.cma_action_spaces[f"cma_{b}"]      = spaces.Discrete(n_rooms * self.num_timeslots)
            self.cma_observation_spaces[f"cma_{b}"] = spaces.Box(0.0, 1.0, (n_rooms * self.num_timeslots,), np.float32)

        self.action_spaces      = {
            **{a: self.saha_action_space for a in self.saha_agents},
            **self.cma_action_spaces,
        }
        self.observation_spaces = {
            **{a: self.saha_observation_space for a in self.saha_agents},
            **self.cma_observation_spaces,
        }

        # Internal state
        self._init_state()

    def _init_state(self):
        # subject → teacher
        self.subject_assignments = np.full(self.num_subjects, -1, dtype=int)
        # building → (room × timeslot)
        self.buildings_room_schedule = {
            b: np.full((len(self.buildings_room_info[b]), self.num_timeslots), -1, dtype=int)
            for b in self.building_keys
        }
        # teacher workloads
        self.teacher_classes = {f"teacher_{i}": 0 for i in range(self.num_teachers)}
        # negotiation stats
        self.conflict_count      = 0
        self.negotiation_success = 0
        # placeholders
        self.saha_teacher_scores = {}
        self.cma_observations     = {}
        self.observations         = {}
        self.current_reward       = 0.0
        self.cma_placed           = set()

    def reset(self, *, seed=None, options=None):
        # PettingZoo boilerplate
        self.agents          = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards             = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations        = {a: False for a in self.agents}
        self.truncations         = {a: False for a in self.agents}
        self.infos               = {a: {}    for a in self.agents}

        # reset schedules & counters
        self._init_state()

        # randomize SAHA utilities
        for a in self.saha_agents:
            self.saha_teacher_scores[a] = np.random.rand(self.num_teachers).astype(np.float32)

        # initial observations
        self._update_cma_observations()
        self._update_all_observations()

    def observe(self, agent):
        return self.observations[agent]

    def step(self, action):
        agent = self.agent_selection

        if action is not None:
            if agent.startswith("saha"):
                self._step_saha(agent, action)
            else:
                self._step_cma(agent, action)

            # record reward
            self.rewards[agent] = self.current_reward
            self._cumulative_rewards[agent] += self.current_reward

        # refresh observations
        self._update_cma_observations()
        self._update_all_observations()

        # handle done
        if self.terminations.get(agent, False) or self.truncations.get(agent, False):
            self.agents.remove(agent)
            self._agent_selector = agent_selector(self.agents)

        # next agent
        if self.agents:
            self.agent_selection = self._agent_selector.next()

    def _step_saha(self, agent, action):
        idx = int(agent.split("_")[1])
        per = self.num_subjects // self.num_sahas
        start = idx * per
        end   = (idx + 1) * per if idx < self.num_sahas - 1 else self.num_subjects

        # pick next unassigned subject
        candidates = [s for s in range(start, end) if self.subject_assignments[s] == -1]
        if not candidates:
            self.current_reward = 0.0
            self.terminations[agent] = True
            return

        subj = candidates[0]
        tid, _ = divmod(action, self.num_timeslots)
        tkey = f"teacher_{tid}"
        if self.teacher_classes[tkey] < self.max_classes:
            self.subject_assignments[subj] = tid
            self.teacher_classes[tkey]   += 1
            self.current_reward = 1.0
        else:
            self.current_reward = -1.0

    def _step_cma(self, agent, action):
        bkey = agent.split("_", 1)[1]

        # **Option A**: place any SAHA-assigned subject not yet in a room
        candidates = [
            s for s in range(self.num_subjects)
            if self.subject_assignments[s] != -1 and s not in self.cma_placed
        ]
        if not candidates:
            self.current_reward = 0.0
            self.terminations[agent] = True
            return

        subj = candidates[0]
        room_idx, timeslot = divmod(action, self.num_timeslots)

        if self.buildings_room_schedule[bkey][room_idx][timeslot] == -1:
            self.buildings_room_schedule[bkey][room_idx][timeslot] = subj
            self.cma_placed.add(subj)
            self.current_reward = 1.0
        else:
            self.current_reward = self._resolve_conflict(subj)

    def _resolve_conflict(self, subject):
        self.conflict_count += 1
        if self._negotiation_possible(subject):
            self.negotiation_success += 1
            return 2.0
        else:
            return -2.0

    def _negotiation_possible(self, subject):
        old_tid     = self.subject_assignments[subject]
        old_teacher = f"teacher_{old_tid}"
        # if that teacher is at capacity, try to bump to an alternate
        if self.teacher_classes[old_teacher] >= self.max_classes:
            for alt in range(self.num_teachers):
                alt_teacher = f"teacher_{alt}"
                if self.teacher_classes[alt_teacher] < self.max_classes:
                    # reassign subject
                    self.subject_assignments[subject] = alt
                    self.teacher_classes[old_teacher]    -= 1
                    self.teacher_classes[alt_teacher]   += 1
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


# ───────────────────────────────────────────────
# Quick smoke-test: run once, print out a human-readable schedule
# ───────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Define your “real” lists:
    subject_names  = [f"Subject_{i}"    for i in range(12)]
    teacher_names  = [f"Teacher_{i}"    for i in range(5)]
    timeslot_labels = ["07:30–09:00","09:15–10:45","11:00–12:30","12:45–14:15","14:30–16:00"]
    room_codes     = ["A1","A2","B1","B2","C1"]

    # 2) Pass them into the env
    env = TimetablingEnv(
        num_sahas=4,
        num_teachers=len(teacher_names),
        num_subjects=len(subject_names),
        num_timeslots=len(timeslot_labels),
        room_codes=room_codes,
        max_classes_per_teacher=3,
    )
    # Hack: stick them on the env so we can lookup by index
    env.subject_names   = subject_names
    env.teacher_names   = teacher_names
    env.timeslot_labels = timeslot_labels

    env.reset()
    while env.agents:
        a   = env.agent_selection
        obs = env.observe(a)
        act = env.action_spaces[a].sample() if obs is not None else None
        env.step(act)

    # 3) Print Subject→Teacher in human form
    print("Assignments (subject → teacher):")
    for idx, tid in enumerate(env.subject_assignments):
        name = env.subject_names[idx]
        tname = (
            "UNASSIGNED" if tid<0 else env.teacher_names[tid]
        )
        print(f"  {name:12s} → {tname}")

    # 4) Print per-building schedule with room, timeslot and subject/teacher names
    for b in env.building_keys:
        print(f"\nBuilding {b}:")
        rooms = env.buildings_room_info[b]
        sched = env.buildings_room_schedule[b]
        for room_idx, room in enumerate(rooms):
            for ts_idx in range(env.num_timeslots):
                subj_idx = sched[room_idx, ts_idx]
                if subj_idx < 0:
                    continue  # empty
                subj_name = env.subject_names[subj_idx]
                teacher   = env.subject_assignments[subj_idx]
                teacher_name = env.teacher_names[teacher]
                timeslot = env.timeslot_labels[ts_idx]
                print(f"  Room {room:3s} | {timeslot} | {subj_name:12s} | {teacher_name}")
