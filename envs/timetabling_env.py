#timetabling_env.py

from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces
import numpy as np
import re
from functools import lru_cache

class ParallelTimetablingEnv(ParallelEnv):
    """
    Parallel version where all agents act simultaneously.
    This improves training efficiency and better models real-world scheduling.
    """
    metadata = {
        "render_modes": ["human"],
        "name": "parallel_timetabling_env_v6",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        num_sahas: int = 4,
        num_cmas: int | None = None,
        num_teachers: int = 5,
        num_subjects: int = 12,
        num_timeslots: int = 5,
        num_days: int = 5,
        room_codes: list[str] | None = None,
        subject_codes: list[str] | None = None,
        subject_campuses: list[list[str]] | None = None,
        max_classes_per_teacher: int = 3,
        enable_communication: bool = True,  # New: agents can share info
    ):
        super().__init__()

        # ─── CONFIG ───────────────────────────────────────────────────────────────
        self.num_sahas = num_sahas
        self.num_teachers = num_teachers
        self.num_subjects = num_subjects
        self.num_timeslots = num_timeslots
        self.num_days = num_days
        self.max_classes = max_classes_per_teacher
        self.enable_communication = enable_communication

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

        # ─── AGENTS ────────────────────────────────────────────────────────────────
        self.saha_agents = [f"saha_{i}" for i in range(self.num_sahas)]
        self.cma_agents = [f"cma_{b}" for b in self.building_keys]
        self.possible_agents = self.saha_agents + self.cma_agents
        self.agents = self.possible_agents[:]

        # ─── ACTION / OBSERVATION SPACES ─────────────────────────────────────────
        # Enhanced observation space with communication channel
        comm_size = 10 if enable_communication else 0
        
        # SAHAs: teacher selection + priority info + communication
        self.saha_observation_size = (
            self.num_teachers +  # teacher availability
            self.num_subjects +  # subject assignment status
            comm_size  # communication channel
        )
        self.saha_action_space = spaces.Discrete(self.num_teachers + 1)  # +1 for "wait"
        self.saha_observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.saha_observation_size,),
            dtype=np.float32
        )

        # CMAs: room scheduling + coordination info
        self.cma_action_spaces = {}
        self.cma_observation_spaces = {}
        for b in self.building_keys:
            n_rooms = len(self.buildings_room_info[b])
            size = n_rooms * self.num_days * self.num_timeslots
            # +1 for "wait" action to allow coordination
            self.cma_action_spaces[f"cma_{b}"] = spaces.Discrete(size + 1)
            obs_size = size + self.num_subjects + comm_size
            self.cma_observation_spaces[f"cma_{b}"] = spaces.Box(
                low=0.0, high=1.0,
                shape=(obs_size,),
                dtype=np.float32
            )

        self.action_spaces = {
            **{a: self.saha_action_space for a in self.saha_agents},
            **self.cma_action_spaces,
        }
        self.observation_spaces = {
            **{a: self.saha_observation_space for a in self.saha_agents},
            **self.cma_observation_spaces,
        }

        # Parallel execution tracking
        self.action_buffer = {}
        self.communication_buffer = np.zeros(comm_size, dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        """Reset for parallel environment"""
        if seed is not None:
            np.random.seed(seed)
            
        self.agents = self.possible_agents[:]
        
        # Initialize state
        self.subject_assignments = np.full(self.num_subjects, -1, dtype=int)
        self.buildings_room_schedule = {
            b: np.full(
                (len(self.buildings_room_info[b]), self.num_days, self.num_timeslots),
                -1, dtype=int
            )
            for b in self.building_keys
        }
        self.teacher_classes = {f"teacher_{i}": 0 for i in range(self.num_teachers)}
        self.teacher_schedules = {
            f"teacher_{i}": np.zeros((self.num_days, self.num_timeslots), dtype=bool)
            for i in range(self.num_teachers)
        }
        
        self.conflict_count = 0
        self.negotiation_success = 0
        self.cma_placed = set()
        self.timestep = 0
        self.max_timesteps = 100
        
        # Reset communication
        self.communication_buffer.fill(0)
        self.action_buffer.clear()
        
        # Generate initial observations for all agents
        observations = self._get_all_observations()
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(self, actions):
        """
        Parallel step: all agents act simultaneously.
        Actions are resolved in a coordinated manner.
        """
        self.timestep += 1
        
        # Store all actions first (simultaneous decision making)
        self.action_buffer = actions.copy()
        
        # Phase 1: Process SAHA actions (teacher assignments)
        saha_rewards = self._process_saha_actions_parallel(actions)
        
        # Phase 2: Update communication buffer based on SAHA decisions
        if self.enable_communication:
            self._update_communication()
        
        # Phase 3: Process CMA actions (room assignments)
        cma_rewards = self._process_cma_actions_parallel(actions)
        
        # Combine rewards
        rewards = {**saha_rewards, **cma_rewards}
        
        # Check termination conditions
        terminations = {}
        truncations = {}
        
        all_assigned = all(s >= 0 for s in self.subject_assignments)
        all_placed = len(self.cma_placed) == self.num_subjects
        time_limit = self.timestep >= self.max_timesteps
        
        for agent in self.agents:
            terminations[agent] = all_assigned and all_placed
            truncations[agent] = time_limit and not terminations[agent]
        
        # Generate new observations
        observations = self._get_all_observations()
        
        # Info for metrics
        infos = {}
        for agent in self.agents:
            infos[agent] = {
                "error_rate": self.calculate_error_rate(),
                "conflict_count": self.conflict_count,
                "negotiation_success": self.negotiation_success,
                "timestep": self.timestep,
            }
        
        return observations, rewards, terminations, truncations, infos
    
    def _process_saha_actions_parallel(self, actions):
        """Process all SAHA actions simultaneously with conflict resolution"""
        rewards = {}
        teacher_requests = {}  # Track which subjects want which teachers
        
        # Collect all SAHA intentions
        for agent in self.saha_agents:
            if agent not in actions:
                rewards[agent] = 0.0
                continue
                
            action = actions[agent]
            if action >= self.num_teachers:  # "wait" action
                rewards[agent] = -0.1  # Small penalty for waiting
                continue
            
            idx = int(agent.split("_")[1])
            per = self.num_subjects // self.num_sahas
            start = idx * per
            end = (idx + 1) * per if idx < self.num_sahas - 1 else self.num_subjects
            
            # Find highest priority unassigned subject
            candidates = sorted(
                [s for s in range(start, end) if self.subject_assignments[s] == -1],
                key=lambda x: self.subject_priority(x),
                reverse=True
            )
            
            if candidates:
                subj = candidates[0]
                teacher_key = f"teacher_{action}"
                
                if teacher_key not in teacher_requests:
                    teacher_requests[teacher_key] = []
                teacher_requests[teacher_key].append((agent, subj))
            else:
                rewards[agent] = 0.0
        
        # Resolve conflicts: multiple subjects wanting same teacher
        for teacher_key, requests in teacher_requests.items():
            remaining_capacity = self.max_classes - self.teacher_classes[teacher_key]
            
            if remaining_capacity <= 0:
                # Teacher at capacity - all requests fail
                for agent, subj in requests:
                    rewards[agent] = -1.0
            elif len(requests) <= remaining_capacity:
                # Can accommodate all requests
                for agent, subj in requests:
                    teacher_idx = int(teacher_key.split("_")[1])
                    self.subject_assignments[subj] = teacher_idx
                    self.teacher_classes[teacher_key] += 1
                    rewards[agent] = 1.0
            else:
                # Competition: prioritize based on subject importance
                sorted_requests = sorted(
                    requests,
                    key=lambda x: self.subject_priority(x[1]),
                    reverse=True
                )
                
                for i, (agent, subj) in enumerate(sorted_requests):
                    if i < remaining_capacity:
                        teacher_idx = int(teacher_key.split("_")[1])
                        self.subject_assignments[subj] = teacher_idx
                        self.teacher_classes[teacher_key] += 1
                        rewards[agent] = 1.0
                    else:
                        rewards[agent] = -0.5  # Failed due to competition
        
        return rewards
    
    def _process_cma_actions_parallel(self, actions):
        """Process all CMA actions simultaneously with conflict resolution"""
        rewards = {}
        slot_requests = {}  # Track which subjects want which slots
        
        # Collect all CMA intentions
        for agent in self.cma_agents:
            if agent not in actions:
                rewards[agent] = 0.0
                continue
                
            bkey = agent.split("_", 1)[1]
            action = actions[agent]
            
            n_rooms = len(self.buildings_room_info[bkey])
            max_action = n_rooms * self.num_days * self.num_timeslots
            
            if action >= max_action:  # "wait" action
                rewards[agent] = -0.1
                continue
            
            # Decode action to room/day/timeslot
            total_slots = self.num_days * self.num_timeslots
            room_idx = action // total_slots
            rest = action % total_slots
            day_idx, ts = divmod(rest, self.num_timeslots)
            
            # Find subjects this CMA can schedule
            candidates = [
                s for s in range(self.num_subjects)
                if (self.subject_assignments[s] != -1 and
                    s not in self.cma_placed and
                    bkey in self.subject_campuses[s])
            ]
            
            if candidates:
                # Choose highest priority subject
                subj = sorted(candidates, key=lambda x: self.subject_priority(x), reverse=True)[0]
                slot_key = (bkey, room_idx, day_idx, ts)
                
                if slot_key not in slot_requests:
                    slot_requests[slot_key] = []
                slot_requests[slot_key].append((agent, subj))
            else:
                rewards[agent] = 0.0
        
        # Resolve slot conflicts
        for (bkey, room_idx, day_idx, ts), requests in slot_requests.items():
            schedule = self.buildings_room_schedule[bkey]
            
            if schedule[room_idx, day_idx, ts] == -1:
                if len(requests) == 1:
                    # No conflict
                    agent, subj = requests[0]
                    schedule[room_idx, day_idx, ts] = subj
                    self.cma_placed.add(subj)
                    
                    # Update teacher schedule
                    teacher_idx = self.subject_assignments[subj]
                    self.teacher_schedules[f"teacher_{teacher_idx}"][day_idx, ts] = True
                    
                    rewards[agent] = 1.0
                else:
                    # Multiple agents want same slot - auction/priority
                    winner = max(requests, key=lambda x: self.subject_priority(x[1]))
                    for agent, subj in requests:
                        if (agent, subj) == winner:
                            schedule[room_idx, day_idx, ts] = subj
                            self.cma_placed.add(subj)
                            
                            teacher_idx = self.subject_assignments[subj]
                            self.teacher_schedules[f"teacher_{teacher_idx}"][day_idx, ts] = True
                            
                            rewards[agent] = 0.8  # Slightly lower for competition
                        else:
                            rewards[agent] = -0.3  # Lost competition
                            self.conflict_count += 1
            else:
                # Slot already occupied
                for agent, subj in requests:
                    rewards[agent] = -1.0
                    self.conflict_count += 1
        
        return rewards
    
    def _update_communication(self):
        """Update shared communication buffer for coordination"""
        # Share key scheduling state
        self.communication_buffer[0] = len(self.cma_placed) / max(1, self.num_subjects)
        self.communication_buffer[1] = self.conflict_count / max(1, self.timestep)
        self.communication_buffer[2] = sum(self.teacher_classes.values()) / max(1, self.num_subjects)
        
        # Share workload distribution
        for i, tc in enumerate(list(self.teacher_classes.values())[:7]):
            self.communication_buffer[3 + i] = tc / max(1, self.max_classes)
    
    def _get_all_observations(self):
        """Generate observations for all agents"""
        observations = {}
        
        # SAHA observations
        for agent in self.saha_agents:
            obs = self._get_saha_observation(agent)
            observations[agent] = obs
        
        # CMA observations  
        for agent in self.cma_agents:
            obs = self._get_cma_observation(agent)
            observations[agent] = obs
        
        return observations
    
    def _get_saha_observation(self, agent):
        """Enhanced SAHA observation with coordination info"""
        obs = []
        
        # Teacher availability (normalized by remaining capacity)
        for i in range(self.num_teachers):
            teacher_key = f"teacher_{i}"
            remaining = max(0, self.max_classes - self.teacher_classes[teacher_key])
            obs.append(remaining / self.max_classes)
        
        # Subject assignment status
        for i in range(self.num_subjects):
            obs.append(1.0 if self.subject_assignments[i] >= 0 else 0.0)
        
        # Communication channel
        if self.enable_communication:
            obs.extend(self.communication_buffer)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_cma_observation(self, agent):
        """Enhanced CMA observation with coordination info"""
        bkey = agent.split("_", 1)[1]
        obs = []
        
        # Room availability
        schedule = self.buildings_room_schedule[bkey]
        free = (schedule == -1).astype(np.float32)
        obs.extend(free.flatten())
        
        # Subject placement status
        for i in range(self.num_subjects):
            if i in self.cma_placed:
                obs.append(1.0)
            elif self.subject_assignments[i] >= 0 and bkey in self.subject_campuses[i]:
                obs.append(0.5)  # Ready to place
            else:
                obs.append(0.0)
        
        # Communication channel
        if self.enable_communication:
            obs.extend(self.communication_buffer)
        
        return np.array(obs, dtype=np.float32)
    
    def subject_priority(self, subj_idx: int) -> int:
        """Calculate subject priority"""
        return (self.max_level + 1) - self.subject_level[subj_idx]
    
    def calculate_error_rate(self):
        """Calculate scheduling error rate"""
        total = self.num_subjects
        errors = total - len(self.cma_placed)
        return errors / total if total > 0 else 0.0
    
    def render(self):
        """Optional rendering"""
        pass
    
    def close(self):
        """Cleanup"""
        pass

    @lru_cache(maxsize=256)
    def _cached_priority(self, subj_idx: int) -> int:
        """Cached priority calculation for efficiency"""
        return self.subject_priority(subj_idx)