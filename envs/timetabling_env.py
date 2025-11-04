"""
COMPLETE FIXED TIMETABLING ENVIRONMENT - v14.7 AGGRESSIVE CONFLICT PENALTIES

Version 14.7: CRITICAL FIX - Aggressive Conflict Penalties for RL Learning
========================================================================================
ALL FIXES APPLIED + NEW AGGRESSIVE PENALTIES:
✅ FIX #1-12: All previous fixes
✅ FIX #13: AGGRESSIVE conflict penalties (conflicts now HURT more than placements help)
✅ FIX #14: Exponential global conflict penalty (per step)
✅ FIX #15: Quadratic completion pressure (forces 90%+ placement)
✅ FIX #16: Curriculum learning (progressive difficulty)
========================================================================================

REWARD REBALANCING:
Before: Placement +150, Conflict -20 → Model learns "conflicts are acceptable"
After:  Placement +150, Conflict -200 → Model learns "avoid conflicts at all costs!"

EXPECTED BEHAVIOR:
- Conflicts should decrease IMMEDIATELY (iteration 10-50)
- Student scores should rise to 40+ by iteration 50
- Overall scores should reach 75+ by iteration 200
- Zero conflicts achievable by iteration 500
========================================================================================
"""

from __future__ import annotations

from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces
import numpy as np
import re
from functools import lru_cache
from typing import List, Dict, Tuple, Union, Set
from collections import defaultdict
import math


class ParallelTimetablingEnv(ParallelEnv):
    """
    COMPLETE FIXED parallel multi-agent timetabling environment
    
    Version 14.7: AGGRESSIVE CONFLICT PENALTIES for effective RL learning
    """

    metadata = {
        "render_modes": ["human"],
        "name": "parallel_timetabling_env_v14_7_aggressive_penalties",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        num_teachers: int,
        num_subjects: int,
        num_timeslots: int,
        num_days: int,
        room_codes: List[str],
        subject_codes: List[str] | None = None,
        subject_campuses: List[List[str]] | None = None,
        subject_areas: List[str] | None = None,
        subject_allowed_rooms: List[List[str]] | None = None,
        subject_section_idx: List[int] | None = None,
        section_labels: List[str] | None = None,
        area_teacher_indices: Dict[str, List[int]] | None = None,
        max_classes_per_teacher: int = 3,
        teacher_max_classes_map: Dict[str, int] | None = None,
        subject_teacher_idx: List[int] | None = None,
        strict_teacher_match: bool = False,
        r_teacher_match: float = 40.0,
        r_teacher_mismatch: float = -1.0,
        r_area_match: float = 25.0,
        r_area_mismatch: float = -1.5,
        r_workload_balance: float = 15.0,
        base_success_reward: float = 150.0,
        wait_penalty: float = 20.0,
        r_conflict_penalty: float = -200.0,          # NEW: AGGRESSIVE
        r_duplicate_penalty: float = -300.0,         # NEW: EVEN WORSE
        r_section_conflict_penalty: float = -250.0,  # NEW: CRITICAL
        include_focus_scalar: bool = True,
        include_focus_tor_scalar: bool = False,
        include_section_features: bool = True,
        include_workload_features: bool = True,
        enable_communication: bool = True,
        use_action_masks: bool = True,
        max_timesteps: int = 400,
        subject_allowed_timeslots: List[List[int]] | None = None,
        timeslot_definitions: List[Dict] | None = None,
        enable_repair_pass: bool = False,
        subject_required_placements: Dict[int, int] | None = None,
        subject_modalities: List[str] | None = None,
        subject_modality_idx: List[int] | None = None,
        modality_labels: List[str] | None = None,
        virtual_rooms: List[str] | None = None,
        r_online_bonus: float = 8.0,
        enable_progressive_difficulty: bool = True,
        difficulty_ramp_steps: int = 100,
        enable_milestone_rewards: bool = True,
        enable_curriculum_learning: bool = True,     # NEW: Curriculum
    ):
        super().__init__()

        # Basic parameters
        self.num_teachers = int(num_teachers)
        self.num_subjects = int(num_subjects)
        self.num_timeslots = int(num_timeslots)
        self.num_days = int(num_days)
        self.max_classes = int(max_classes_per_teacher)
        self.enable_communication = bool(enable_communication)
        self.max_timesteps = int(max_timesteps)
        self.include_focus_scalar = bool(include_focus_scalar)
        self.include_focus_tor_scalar = bool(include_focus_tor_scalar)
        self.include_section_features = bool(include_section_features)
        self.include_workload_features = bool(include_workload_features)
        self.use_action_masks = bool(use_action_masks)
        self.comm_size = 14 if self.enable_communication else 0
        self.enable_repair_pass = bool(enable_repair_pass)
        self.enable_milestone_rewards = bool(enable_milestone_rewards)
        self.enable_curriculum_learning = bool(enable_curriculum_learning)

        # Progressive difficulty
        self.enable_progressive_difficulty = enable_progressive_difficulty
        self.difficulty_ramp_steps = difficulty_ramp_steps

        # Reward parameters (AGGRESSIVE REBALANCING)
        self.r_teacher_match = float(r_teacher_match)
        self.r_teacher_mismatch = float(r_teacher_mismatch)
        self.r_area_match = float(r_area_match)
        self.r_area_mismatch = float(r_area_mismatch)
        self.r_workload_balance = float(r_workload_balance)
        self.base_success_reward = float(base_success_reward)
        self.wait_penalty = float(wait_penalty)
        self.r_online_bonus = float(r_online_bonus)
        
        # NEW: AGGRESSIVE CONFLICT PENALTIES
        self.r_conflict_penalty = float(r_conflict_penalty)              # -200
        self.r_duplicate_penalty = float(r_duplicate_penalty)            # -300
        self.r_section_conflict_penalty = float(r_section_conflict_penalty)  # -250
        
        self.reward_scale = 0.01
        
        # Curriculum learning tracking
        self.global_step_counter = 0
        self.training_phase = 0

        # Per-placement teacher tracking
        self.placement_teachers = {}
        self.subject_day_usage = {}
        self.subject_placement_count = {}

        # Modality setup
        self.modality_labels = modality_labels or ['Face-to-Face', 'Online', 'Hybrid']
        self.modality_to_idx = {m: i for i, m in enumerate(self.modality_labels)}
        
        if subject_modalities is None:
            self.subject_modalities = ['Face-to-Face'] * self.num_subjects
        else:
            self.subject_modalities = list(subject_modalities)
        
        if subject_modality_idx is None:
            self.subject_modality_idx = [self.modality_to_idx.get(m, 0) 
                                         for m in self.subject_modalities]
        else:
            self.subject_modality_idx = list(subject_modality_idx)
        
        self.virtual_rooms = set(virtual_rooms or [])

        # Room setup
        self.room_codes = list(room_codes)
        self.physical_rooms = set(r for r in self.room_codes if r not in self.virtual_rooms)
        
        self.building_keys: List[str] = []
        self.buildings_room_info: Dict[str, List[str]] = {}
        for code in self.room_codes:
            b = code[0]
            if b not in self.buildings_room_info:
                self.building_keys.append(b)
                self.buildings_room_info[b] = []
            self.buildings_room_info[b].append(code)

        self.room_code_to_pair: Dict[str, Tuple[str, int]] = {}
        for b in self.building_keys:
            for ridx, rcode in enumerate(self.buildings_room_info[b]):
                self.room_code_to_pair[rcode] = (b, ridx)

        # Global slot setup
        self.global_slots: List[Tuple[str, int, int, int]] = []
        for b in self.building_keys:
            n_rooms = len(self.buildings_room_info[b])
            for r_idx in range(n_rooms):
                for d in range(self.num_days):
                    for ts in range(self.num_timeslots):
                        self.global_slots.append((b, r_idx, d, ts))
        self.global_slot_count = len(self.global_slots)
        self.slot_choices = self.global_slot_count

        self.slot_summary_size = len(self.building_keys) * self.num_days * self.num_timeslots

        # Subject setup
        self.subject_codes = subject_codes or [f"Subject_{i}" for i in range(self.num_subjects)]
        if subject_areas is None:
            def _area_from_code(s: str) -> str:
                s = str(s or "").strip()
                if not s:
                    return "X"
                first_word = re.split(r"[^A-Za-z]+", s)[0]
                return first_word if first_word else "X"
            self.subject_areas = [_area_from_code(c) for c in self.subject_codes]
        else:
            self.subject_areas = list(subject_areas)
        
        self.areas = sorted(list(dict.fromkeys(self.subject_areas)))
        self.area_to_idx = {a: i for i, a in enumerate(self.areas)}
        self.subject_area_idx = [self.area_to_idx.get(a, 0) for a in self.subject_areas]

        # Teacher area assignments
        if area_teacher_indices is None:
            teachers_per_area = max(1, self.num_teachers // len(self.areas))
            self.area_teacher_indices = {}
            for i, area in enumerate(self.areas):
                start_idx = i * teachers_per_area
                end_idx = min((i + 1) * teachers_per_area, self.num_teachers)
                self.area_teacher_indices[area] = list(range(start_idx, end_idx))
        else:
            self.area_teacher_indices = dict(area_teacher_indices)

        for area in self.areas:
            if area not in self.area_teacher_indices or not self.area_teacher_indices[area]:
                raise ValueError(f"Area '{area}' has no assigned teachers")
        
        self.teacher_to_area = {}
        for area, teacher_list in self.area_teacher_indices.items():
            for teacher_idx in teacher_list:
                if teacher_idx in self.teacher_to_area:
                    raise ValueError(f"Teacher {teacher_idx} assigned to multiple areas")
                self.teacher_to_area[teacher_idx] = area

        self.max_teachers_per_area = max(len(teachers) for teachers in self.area_teacher_indices.values())

        # Subject constraints
        if subject_campuses is None:
            self.subject_campuses = [list(self.building_keys) for _ in range(self.num_subjects)]
        else:
            self.subject_campuses = subject_campuses

        if subject_allowed_rooms is None:
            self.subject_allowed_rooms = [[] for _ in range(self.num_subjects)]
        else:
            assert len(subject_allowed_rooms) == self.num_subjects
            self.subject_allowed_rooms = subject_allowed_rooms

        self.subject_allowed_pairs: List[List[Tuple[str, int]]] = []
        for s in range(self.num_subjects):
            pairs: List[Tuple[str, int]] = []
            rooms = self.subject_allowed_rooms[s]
            if rooms:
                for rc in rooms:
                    if rc in self.room_code_to_pair:
                        pairs.append(self.room_code_to_pair[rc])
            else:
                for b in self.subject_campuses[s]:
                    for ridx in range(len(self.buildings_room_info[b])):
                        pairs.append((b, ridx))
            self.subject_allowed_pairs.append(pairs)

        # Section setup
        self.subject_section_idx = list(subject_section_idx or [0] * self.num_subjects)
        self.section_labels = list(section_labels or ["GEN"])
        self.num_sections = len(self.section_labels)

        # Teacher preferences
        self.subject_teacher_idx = (subject_teacher_idx or [-1] * self.num_subjects)
        if len(self.subject_teacher_idx) != self.num_subjects:
            self.subject_teacher_idx = (
                self.subject_teacher_idx[: self.num_subjects] +
                [-1] * max(0, self.num_subjects - len(self.subject_teacher_idx))
            )
        self.teacher_max_classes_map = teacher_max_classes_map or {}
        self.strict_teacher_match = bool(strict_teacher_match)

        # Timeslot constraints
        if subject_allowed_timeslots is None:
            self.subject_allowed_timeslots = [list(range(self.num_timeslots)) 
                                              for _ in range(self.num_subjects)]
        else:
            assert len(subject_allowed_timeslots) == self.num_subjects
            self.subject_allowed_timeslots = [list(slots) for slots in subject_allowed_timeslots]
        
        for i, allowed in enumerate(self.subject_allowed_timeslots):
            if not allowed:
                raise ValueError(f"Subject {i} ({self.subject_codes[i]}) has no allowed timeslots!")

        self.timeslot_definitions = timeslot_definitions or []
        self.timeslot_durations = {}
        for slot_def in self.timeslot_definitions:
            idx = slot_def.get('index', -1)
            duration = slot_def.get('duration', 90)
            if idx >= 0:
                self.timeslot_durations[idx] = duration
        
        # Placement requirements
        if subject_required_placements is not None:
            self.subject_required_placements = dict(subject_required_placements)
        else:
            self.subject_required_placements = {}
            for s in range(self.num_subjects):
                if (hasattr(self, 'subject_allowed_timeslots') and 
                    s < len(self.subject_allowed_timeslots) and 
                    self.subject_allowed_timeslots[s]):
                    first_slot = self.subject_allowed_timeslots[s][0]
                    duration = self.timeslot_durations.get(first_slot, 90)
                    self.subject_required_placements[s] = 2 if duration == 90 else 1
                else:
                    self.subject_required_placements[s] = 1

        # Agent setup
        self.saha_agents = [f"saha_{a}" for a in self.areas]
        self.possible_agents = self.saha_agents[:]
        self.agents = self.possible_agents[:]

        # Action and observation spaces
        self.saha_action_space = spaces.MultiDiscrete(
            np.array([self.max_teachers_per_area + 1, self.slot_choices + 1], dtype=np.int64)
        )

        extra = (1 if self.include_focus_scalar else 0) + (1 if self.include_focus_tor_scalar else 0)
        sec_feat_dim = (self.num_days + 2) if self.include_section_features else 0
        workload_feat_dim = self.max_teachers_per_area + 2 if self.include_workload_features else 0
        modality_feat_dim = len(self.modality_labels) + 1

        self.saha_obs_core_size = (
            self.max_teachers_per_area +
            self.slot_summary_size +
            self.num_subjects +
            extra +
            sec_feat_dim +
            workload_feat_dim +
            modality_feat_dim +
            self.comm_size
        )

        if self.use_action_masks:
            self.saha_observation_space = spaces.Dict({
                "obs": spaces.Box(low=0.0, high=1.0, shape=(self.saha_obs_core_size,), dtype=np.float32),
                "teacher_mask": spaces.Box(low=0, high=1, shape=(self.max_teachers_per_area + 1,), dtype=np.float32),
                "slot_mask": spaces.Box(low=0, high=1, shape=(self.slot_choices + 1,), dtype=np.float32),
            })
        else:
            self.saha_observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(self.saha_obs_core_size,), dtype=np.float32
            )

        self.action_spaces = {a: self.saha_action_space for a in self.saha_agents}
        self.observation_spaces = {a: self.saha_observation_space for a in self.saha_agents}
        self.communication_buffer = np.zeros(self.comm_size, dtype=np.float32)

        print(f"\n{'='*50}")
        print(f"TIMETABLING ENVIRONMENT v14.7 - AGGRESSIVE PENALTIES")
        print(f"{'='*50}")
        print(f"\n=== ALL FIXES APPLIED ===")
        print(f"  ✅ FIX #1-12: All previous fixes")
        print(f"  ✅ FIX #13: AGGRESSIVE conflict penalties")
        print(f"  ✅ FIX #14: Exponential global penalties")
        print(f"  ✅ FIX #15: Quadratic completion pressure")
        print(f"  ✅ FIX #16: Curriculum learning")
        print(f"\n=== REWARD REBALANCING ===")
        print(f"  • Base success: {self.base_success_reward}")
        print(f"  • Conflict penalty: {self.r_conflict_penalty} (was -20)")
        print(f"  • Duplicate penalty: {self.r_duplicate_penalty} (was -20)")
        print(f"  • Section conflict: {self.r_section_conflict_penalty} (was -15)")
        print(f"  • Wait penalty: {self.wait_penalty}")
        print(f"\n=== EXPECTED BEHAVIOR ===")
        print(f"  • Conflicts decrease IMMEDIATELY (iter 10-50)")
        print(f"  • Student scores rise to 40+ by iter 50")
        print(f"  • Zero conflicts achievable by iter 500")
        print(f"{'='*50}\n")

    def _get_difficulty_factor(self) -> float:
        """Get current difficulty factor for progressive training"""
        if not self.enable_progressive_difficulty:
            return 1.0
        return min(1.0, self.timestep / self.difficulty_ramp_steps)

    def _get_curriculum_multiplier(self) -> float:
        """
        Get curriculum learning multiplier for penalty scaling
        
        Phase 1 (0-10k steps / ~100 episodes): Learning basics, lighter penalties (0.5x)
        Phase 2 (10k-30k steps / ~100-300 episodes): Normal penalties (1.0x)
        Phase 3 (30k+ steps / 300+ episodes): Perfection mode, heavier penalties (1.5x)
        """
        if not self.enable_curriculum_learning:
            return 1.0
        
        if self.global_step_counter < 10000:
            return 0.5  # Learning phase - be gentle
        elif self.global_step_counter < 30000:
            return 1.0  # Normal phase
        else:
            return 1.5  # Perfection phase - be strict
    
    def _is_virtual_room(self, room_code: str) -> bool:
        """Check if room is virtual"""
        return room_code in self.virtual_rooms

    def _validate_modality_room_match(self, subject_idx: int, room_code: str) -> Tuple[bool, str]:
        """Validate room matches subject modality"""
        modality = self.subject_modalities[subject_idx]
        is_virtual = self._is_virtual_room(room_code)
        
        if modality == 'Online':
            if not is_virtual:
                return False, "online_needs_virtual"
            return True, "ok"
        elif modality == 'Face-to-Face':
            if is_virtual:
                return False, "f2f_needs_physical"
            return True, "ok"
        elif modality == 'Hybrid':
            if is_virtual:
                return False, "hybrid_needs_physical"
            return True, "ok"
        
        return True, "ok"

    def _max_for(self, teacher_key: str) -> int:
        per_teacher = int(self.teacher_max_classes_map.get(teacher_key, self.max_classes))
        return int(min(per_teacher, self.max_classes))

    def _agent_area_idx(self, agent: str) -> int:
        return self.area_to_idx.get(agent.split("_", 1)[1], 0)

    def _agent_area_name(self, agent: str) -> str:
        return agent.split("_", 1)[1]

    def _saha_focus_subject(self, agent: str) -> int:
        area_idx = self._agent_area_idx(agent)
        candidates = [s for s in range(self.num_subjects)
                      if self.subject_area_idx[s] == area_idx and s not in self.placed_subjects]
        if not candidates:
            return -1
        candidates.sort(key=lambda s: (self.fail_count[s] // 10, -self.subject_priority(s)))
        return candidates[0]

    def _area_teacher_idx_to_global(self, agent: str, area_teacher_idx: int) -> int:
        area_name = self._agent_area_name(agent)
        area_teachers = self.area_teacher_indices.get(area_name, [])
        if 0 <= area_teacher_idx < len(area_teachers):
            return area_teachers[area_teacher_idx]
        return -1

    def _global_teacher_idx_to_area(self, agent: str, global_teacher_idx: int) -> int:
        area_name = self._agent_area_name(agent)
        area_teachers = self.area_teacher_indices.get(area_name, [])
        try:
            return area_teachers.index(global_teacher_idx)
        except ValueError:
            return -1

    def _calculate_area_workload_balance(self, agent: str) -> float:
        area_name = self._agent_area_name(agent)
        area_teachers = self.area_teacher_indices.get(area_name, [])
        if len(area_teachers) <= 1:
            return 1.0
        loads = [self.teacher_classes[f"teacher_{t}"] for t in area_teachers]
        if not loads:
            return 1.0
        std_dev = float(np.std(loads))
        if not np.isfinite(std_dev):
            return 1.0
        balance = 1.0 / (1.0 + std_dev)
        return float(np.clip(balance, 0.0, 1.0))

    def _get_placement_count(self, subject_idx: int, section_idx: int) -> int:
        """How many times has this subject been placed"""
        return self.subject_placement_count.get((subject_idx, section_idx), 0)

    def _is_subject_fully_placed(self, subject_idx: int) -> bool:
        """Check if subject has reached required placement count"""
        sec_idx = self.subject_section_idx[subject_idx]
        current = self._get_placement_count(subject_idx, sec_idx)
        required = self.subject_required_placements.get(subject_idx, 1)
        return current >= required
    
    def _is_placement_complete(self, subject_idx: int, day_idx: int, timeslot_idx: int) -> bool:
        """Verify a placement is COMPLETE with all required information"""
        if subject_idx not in self.subject_assignments:
            return False
        
        teacher_idx = self.subject_assignments[subject_idx]
        sec_idx = self.subject_section_idx[subject_idx]
        area_idx = self.subject_area_idx[subject_idx]
        
        placement_found_in_room = False
        for building_key, schedule in self.buildings_room_schedule.items():
            for room_idx in range(schedule.shape[0]):
                if schedule[room_idx, day_idx, timeslot_idx] == subject_idx:
                    placement_found_in_room = True
                    break
            if placement_found_in_room:
                break
        
        if not placement_found_in_room:
            return False
        
        teacher_key = (area_idx, teacher_idx)
        if teacher_key not in self.teacher_schedules:
            return False
        
        if not self.teacher_schedules[teacher_key][day_idx, timeslot_idx]:
            return False
        
        section_key = (sec_idx, area_idx)
        if section_key not in self.section_schedules:
            return False
        
        if not self.section_schedules[section_key][day_idx, timeslot_idx]:
            return False
        
        return True

    def get_complete_placements(self) -> int:
        """Count ONLY complete placements that will be exported"""
        complete_count = 0
        
        for subject_idx in range(self.num_subjects):
            complete_placement_count = 0
            
            for day_idx in range(self.num_days):
                for ts_idx in range(self.num_timeslots):
                    if self._is_placement_complete(subject_idx, day_idx, ts_idx):
                        complete_placement_count += 1
            
            required = self.subject_required_placements.get(subject_idx, 1)
            if complete_placement_count >= required:
                complete_count += 1
        
        return complete_count

    def get_complete_placement_rate(self) -> float:
        """Get the percentage of subjects with complete placements"""
        complete = self.get_complete_placements()
        return (complete / self.num_subjects) * 100 if self.num_subjects > 0 else 0.0

    def _can_place_on_day(self, subject_idx: int, day_idx: int, timeslot_idx: int) -> Tuple[bool, str]:
        """Check if subject can be placed on specified day"""
        sec_idx = self.subject_section_idx[subject_idx]
        current_count = self._get_placement_count(subject_idx, sec_idx)
        required_count = self.subject_required_placements.get(subject_idx, 1)
        
        if current_count >= required_count:
            return False, "max_placements_reached"
        
        used_days = self.subject_day_usage.get(subject_idx, set())
        
        if not used_days:
            return True, "first_placement"
        
        if day_idx in used_days:
            return False, "day_duplicate"
        
        return True, "ok"
  
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.agents = self.possible_agents[:]

        # Initialize state
        self.subject_assignments = np.full(self.num_subjects, -1, dtype=int)
        self.placement_teachers = {}
        self.subject_placement_count = {}
        self.placed_subjects = set()
        self.subject_day_usage = {}
        self.subject_day_placements = defaultdict(set)
        
        self.teacher_classes = {f"teacher_{i}": 0 for i in range(self.num_teachers)}
        self.teacher_schedules = {
            f"teacher_{i}": np.zeros((self.num_days, self.num_timeslots), dtype=bool)
            for i in range(self.num_teachers)
        }
        self.section_schedules = {
            f"section_{i}": np.zeros((self.num_days, self.num_timeslots), dtype=bool)
            for i in range(self.num_sections)
        }

        self.buildings_room_schedule = {
            b: np.full((len(self.buildings_room_info[b]), self.num_days, self.num_timeslots), -1, dtype=int)
            for b in self.building_keys
        }

        self.timestep = 0
        self.conflict_count = 0
        self.fail_count = np.zeros(self.num_subjects, dtype=np.int32)
        self.fail_stats = {
            "no_intent": 0,
            "campus_block": 0,
            "teacher_cap": 0,
            "room_busy": 0,
            "teacher_busy": 0,
            "room_not_allowed": 0,
            "strict_mismatch": 0,
            "section_busy": 0,
            "area_mismatch": 0,
            "already_placed": 0,
            "timeslot_mismatch": 0,
            "section_duplicate": 0,
            "online_needs_virtual": 0,
            "f2f_needs_physical": 0,
            "hybrid_needs_physical": 0,
            "section_conflict_resolved": 0,
            "day_duplicate": 0,
        }
        
        self.modality_stats = {m: {'attempted': 0, 'placed': 0, 'total_placements': 0} 
                              for m in self.modality_labels}
        
        self.modality_subjects_placed = {m: set() for m in self.modality_labels}
        self.modality_subjects_attempted = {m: set() for m in self.modality_labels}
        
        self._milestone_flags = {
            '50': False,
            '70': False,
            '85': False,
            '90': False,
            '95': False,
            '100': False
        }
        
        if self.comm_size:
            self.communication_buffer.fill(0)

        observations = self._get_all_observations()
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _decode_action(self, a: Union[np.ndarray, List[int], Tuple[int, int], int]) -> Tuple[int, int, bool]:
        if isinstance(a, (list, tuple, np.ndarray)) and len(a) == 2:
            t_idx = int(a[0]); s_idx = int(a[1])
            is_wait = (t_idx >= self.max_teachers_per_area) or (s_idx >= self.slot_choices)
            return t_idx, s_idx, is_wait
        return self.max_teachers_per_area, self.slot_choices, True

    def step(self, actions):
        """
        COMPLETE FIXED step function with AGGRESSIVE conflict penalties
        """
        self.timestep += 1
        self.global_step_counter += 1
        difficulty = self._get_difficulty_factor()
        curriculum_mult = self._get_curriculum_multiplier()

        rewards: Dict[str, float] = {a: 0.0 for a in self.agents}
        terminations, truncations = {}, {}

        intents: List[Tuple[str, int, int, str, int, int, int]] = []
        
        attempted_subjects_this_step = set()

        # ============================================================
        # PHASE 1: COLLECT AND VALIDATE INTENTS
        # ============================================================
        for agent in self.saha_agents:
            act = actions.get(agent, np.array([self.max_teachers_per_area, self.slot_choices], dtype=np.int64))
            area_teacher_idx, slot_idx, is_wait = self._decode_action(act)

            if is_wait:
                rewards[agent] -= self.wait_penalty
                self.fail_stats["no_intent"] += 1
                continue

            if slot_idx < 0 or slot_idx >= self.slot_choices:
                rewards[agent] -= 5.0
                self.fail_stats["no_intent"] += 1
                continue
            
            focus = self._saha_focus_subject(agent)
            if focus < 0:
                self.fail_stats["no_intent"] += 1
                continue

            sec_idx = self.subject_section_idx[focus]
            
            current_count = self._get_placement_count(focus, sec_idx)
            required_count = self.subject_required_placements.get(focus, 1)
            
            if current_count >= required_count:
                rewards[agent] -= 10.0 * difficulty
                self.fail_stats["already_placed"] += 1
                continue
            
            if focus in self.placed_subjects:
                rewards[agent] -= 10.0 * difficulty
                self.fail_stats["already_placed"] += 1
                continue
            
            if focus in attempted_subjects_this_step:
                rewards[agent] -= 10.0 * difficulty
                self.fail_stats["section_duplicate"] += 1
                continue
            
            attempted_subjects_this_step.add(focus)

            bkey, room_idx, day_idx, ts = self.global_slots[slot_idx]
            room_code = self.buildings_room_info[bkey][room_idx]

            modality = self.subject_modalities[focus]
            
            if focus not in self.modality_subjects_attempted[modality]:
                self.modality_stats[modality]['attempted'] += 1
                self.modality_subjects_attempted[modality].add(focus)
            
            is_valid, reason = self._validate_modality_room_match(focus, room_code)
            if not is_valid:
                rewards[agent] -= 2.0 * difficulty
                self.fail_count[focus] += 1
                self.fail_stats[reason] = self.fail_stats.get(reason, 0) + 1
                continue

            if ts not in self.subject_allowed_timeslots[focus]:
                rewards[agent] -= 1.5 * difficulty
                self.fail_count[focus] += 1
                self.fail_stats["timeslot_mismatch"] += 1
                continue

            if bkey not in self.subject_campuses[focus]:
                rewards[agent] -= 0.5 * difficulty
                self.fail_count[focus] += 1
                self.fail_stats["campus_block"] += 1
                continue

            global_teacher_idx = self._area_teacher_idx_to_global(agent, area_teacher_idx)
            if global_teacher_idx < 0:
                rewards[agent] -= 0.5
                self.fail_stats["no_intent"] += 1
                self.fail_count[focus] += 1
                continue

            tkey = f"teacher_{global_teacher_idx}"

            if self.teacher_classes[tkey] >= self._max_for(tkey):
                rewards[agent] -= 0.5 * difficulty
                self.fail_count[focus] += 1
                self.fail_stats["teacher_cap"] += 1
                continue

            tor = self.subject_teacher_idx[focus]
            tor_valid = 0 <= tor < self.num_teachers
            if self.strict_teacher_match and tor_valid and global_teacher_idx != tor:
                rewards[agent] += self.r_teacher_mismatch
                self.fail_count[focus] += 1
                self.fail_stats["strict_mismatch"] += 1
                continue

            schedule = self.buildings_room_schedule[bkey]
            if schedule[room_idx, day_idx, ts] != -1:
                rewards[agent] -= 0.5 * difficulty
                self.fail_count[focus] += 1
                self.fail_stats["room_busy"] += 1
                continue

            if self.teacher_schedules[tkey][day_idx, ts]:
                rewards[agent] -= 0.5 * difficulty
                self.fail_count[focus] += 1
                self.fail_stats["teacher_busy"] += 1
                continue

            skey = f"section_{sec_idx}"
            if self.section_schedules[skey][day_idx, ts]:
                rewards[agent] -= 0.5 * difficulty
                self.fail_count[focus] += 1
                self.fail_stats["section_busy"] += 1
                continue

            allowed_rooms = self.subject_allowed_rooms[focus]
            if allowed_rooms and (room_code not in allowed_rooms):
                rewards[agent] -= 1.0 * difficulty
                self.fail_count[focus] += 1
                self.fail_stats["room_not_allowed"] += 1
                continue

            intents.append((agent, focus, global_teacher_idx, bkey, room_idx, day_idx, ts))

        # ============================================================
        # PHASE 2: CONFLICT RESOLUTION (3-STAGE) WITH AGGRESSIVE PENALTIES
        # ============================================================
        
        teacher_time_map: Dict[Tuple[int, int, int], List[Tuple[str, int, int, str, int, int, int]]] = {}
        for it in intents:
            _, subj, t_idx, _, _, d, ts = it
            teacher_time_map.setdefault((t_idx, d, ts), []).append(it)

        kept_after_teacher: List[Tuple[str, int, int, str, int, int, int]] = []
        for _, group in teacher_time_map.items():
            if len(group) == 1:
                kept_after_teacher.append(group[0])
            else:
                winner = max(group, key=lambda x: self.subject_priority(x[1]))
                kept_after_teacher.append(winner)
                for it in group:
                    if it is not winner:
                        # AGGRESSIVE PENALTY
                        penalty = self.r_conflict_penalty * curriculum_mult
                        rewards[it[0]] += penalty
                        self.conflict_count += 1

        room_slot_map: Dict[Tuple[str, int, int, int], List[Tuple[str, int, int, str, int, int, int]]] = {}
        for it in kept_after_teacher:
            _, _, _, bkey, r_idx, d, ts = it
            room_slot_map.setdefault((bkey, r_idx, d, ts), []).append(it)

        kept_after_room: List[Tuple[str, int, int, str, int, int, int]] = []
        for _, group in room_slot_map.items():
            if len(group) == 1:
                kept_after_room.append(group[0])
            else:
                winner = max(group, key=lambda x: self.subject_priority(x[1]))
                kept_after_room.append(winner)
                for it in group:
                    if it is not winner:
                        # AGGRESSIVE PENALTY
                        penalty = self.r_conflict_penalty * curriculum_mult
                        rewards[it[0]] += penalty
                        self.conflict_count += 1

        section_time_map: Dict[Tuple[int, int, int], List[Tuple[str, int, int, str, int, int, int]]] = {}
        for it in kept_after_room:
            agent, subj, _, _, _, d, ts = it
            sec_idx = self.subject_section_idx[subj]
            section_time_map.setdefault((sec_idx, d, ts), []).append(it)

        final_winners: List[Tuple[str, int, int, str, int, int, int]] = []
        for _, group in section_time_map.items():
            if len(group) == 1:
                final_winners.append(group[0])
            else:
                winner = max(group, key=lambda x: self.subject_priority(x[1]))
                final_winners.append(winner)
                for it in group:
                    if it is not winner:
                        # EVEN MORE AGGRESSIVE FOR SECTION CONFLICTS
                        penalty = self.r_section_conflict_penalty * curriculum_mult
                        rewards[it[0]] += penalty
                        self.conflict_count += 1
                        self.fail_stats["section_conflict_resolved"] += 1

        # ============================================================
        # PHASE 3: ATOMIC PLACEMENT
        # ============================================================
        
        placed_subjects_this_step = set()
        placed_teacher_times_this_step = set()
        placed_room_slots_this_step = set()
        placed_section_times_this_step = set()
        
        step_placement_counts = {}
        successful_placements = 0

        for agent, subj, global_t_idx, bkey, r_idx, d, ts in final_winners:
            sec_idx = self.subject_section_idx[subj]
            tkey = f"teacher_{global_t_idx}"
            skey = f"section_{sec_idx}"
            
            key = (subj, sec_idx)
            current_count = self._get_placement_count(subj, sec_idx)
            step_placements = step_placement_counts.get(key, 0)
            total_count = current_count + step_placements
            required_count = self.subject_required_placements.get(subj, 1)
            
            if total_count >= required_count:
                # AGGRESSIVE DUPLICATE PENALTY
                penalty = self.r_duplicate_penalty * curriculum_mult
                rewards[agent] += penalty
                self.fail_stats["section_duplicate"] += 1
                continue
            
            can_place, reason = self._can_place_on_day(subj, d, ts)
            if not can_place:
                if reason == "day_duplicate":
                    # AGGRESSIVE PENALTY
                    penalty = self.r_duplicate_penalty * curriculum_mult
                    rewards[agent] += penalty
                    self.fail_stats["day_duplicate"] += 1
                elif reason == "max_placements_reached":
                    rewards[agent] -= 20.0
                    self.fail_stats["max_placements"] += 1
                else:
                    rewards[agent] -= 10.0
                self.fail_count[subj] += 1
                continue
            
            if (global_t_idx, d, ts) in placed_teacher_times_this_step:
                rewards[agent] -= 15.0 * curriculum_mult
                self.conflict_count += 1
                continue
            
            if (bkey, r_idx, d, ts) in placed_room_slots_this_step:
                rewards[agent] -= 15.0 * curriculum_mult
                self.conflict_count += 1
                continue
            
            if (sec_idx, d, ts) in placed_section_times_this_step:
                rewards[agent] -= 15.0 * curriculum_mult
                self.conflict_count += 1
                continue
            
            if subj in placed_subjects_this_step:
                rewards[agent] -= 20.0 * curriculum_mult
                self.conflict_count += 1
                continue
            
            if subj in self.placed_subjects:
                rewards[agent] -= 20.0 * curriculum_mult
                self.fail_stats["already_placed"] += 1
                continue

            schedule = self.buildings_room_schedule[bkey]
            
            if schedule[r_idx, d, ts] != -1:
                rewards[agent] -= 0.5
                self.conflict_count += 1
                continue
            
            if self.teacher_schedules[tkey][d, ts]:
                rewards[agent] -= 0.5
                self.conflict_count += 1
                continue
            
            if self.section_schedules[skey][d, ts]:
                rewards[agent] -= 0.5
                self.conflict_count += 1
                continue

            # PERFORM PLACEMENT
            pre_balance = self._calculate_area_workload_balance(agent)
            room_code = self.buildings_room_info[bkey][r_idx]
            
            schedule[r_idx, d, ts] = subj
            self.teacher_schedules[tkey][d, ts] = True
            self.section_schedules[skey][d, ts] = True
            placed_teacher_times_this_step.add((global_t_idx, d, ts))
            placed_room_slots_this_step.add((bkey, r_idx, d, ts))
            placed_section_times_this_step.add((sec_idx, d, ts))
            placed_subjects_this_step.add(subj)
            self.teacher_classes[tkey] += 1
            self.subject_assignments[subj] = global_t_idx
            self.placement_teachers[(subj, d, ts)] = global_t_idx
            
            key = (subj, sec_idx)
            current = self.subject_placement_count.get(key, 0)
            self.subject_placement_count[key] = current + 1
            step_placement_counts[key] = step_placement_counts.get(key, 0) + 1
            
            if subj not in self.subject_day_usage:
                self.subject_day_usage[subj] = set()
            self.subject_day_usage[subj].add(d)
            
            if self._is_subject_fully_placed(subj):
                self.placed_subjects.add(subj)
            
            self.fail_count[subj] = 0
            successful_placements += 1

            modality = self.subject_modalities[subj]
            if subj not in self.modality_subjects_placed[modality]:
                self.modality_stats[modality]['placed'] += 1
                self.modality_subjects_placed[modality].add(subj)
            self.modality_stats[modality]['total_placements'] += 1

            # Calculate rewards
            r = self.base_success_reward
            
            is_virtual = self._is_virtual_room(room_code)
            if modality == 'Online' and is_virtual:
                r += self.r_online_bonus
            elif modality == 'Hybrid':
                r += self.r_online_bonus * 0.5

            tor = self.subject_teacher_idx[subj]
            tor_valid = 0 <= tor < self.num_teachers
            
            if tor_valid:
                if global_t_idx == tor:
                    r += self.r_teacher_match
                else:
                    tor_key = f"teacher_{tor}"
                    tor_full = self.teacher_classes.get(tor_key, 0) >= self._max_for(tor_key)
                    r += (0.0 if tor_full else self.r_teacher_mismatch)

            subject_area = self.subject_areas[subj]
            teacher_area = self.teacher_to_area.get(global_t_idx, None)
            if teacher_area == subject_area:
                r += self.r_area_match
            else:
                r += self.r_area_mismatch
                self.fail_stats["area_mismatch"] += 1

            post_balance = self._calculate_area_workload_balance(agent)
            balance_delta = post_balance - pre_balance
            if np.isfinite(balance_delta):
                if balance_delta > 0:
                    r += self.r_workload_balance * balance_delta
                elif balance_delta < -0.05:
                    r -= self.r_workload_balance * 0.3 * abs(balance_delta)

            if self._is_subject_fully_placed(subj):
                r += 5.0

            if not np.isfinite(r):
                r = 0.0

            rewards[agent] += r

        # ============================================================
        # NEW: EXPONENTIAL GLOBAL CONFLICT PENALTY (PER STEP)
        # ============================================================
        if self.conflict_count > 0:
            # Exponential penalty that never caps - always provides gradient
            conflict_penalty = -50 * (1 - math.exp(-0.02 * self.conflict_count))
            conflict_penalty *= curriculum_mult
            
            # Apply to ALL agents (shared responsibility)
            for agent in self.agents:
                rewards[agent] += conflict_penalty
            
            # Debug output (periodic)
            if self.timestep % 50 == 0 and self.conflict_count > 10:
                print(f"   ⚠️ Step {self.timestep}: {self.conflict_count} conflicts → {conflict_penalty:.1f} global penalty")

        # ============================================================
        # MILESTONE REWARDS (REBALANCED)
        # ============================================================
        if self.enable_milestone_rewards:
            placement_rate = len(self.placed_subjects) / self.num_subjects
            
            if placement_rate >= 0.50 and not self._milestone_flags['50']:
                milestone_bonus = 30.0
                for a in self.agents:
                    rewards[a] += milestone_bonus
                self._milestone_flags['50'] = True
                print(f"  🎯 50% MILESTONE @ step {self.timestep}! (+{milestone_bonus})")
            
            if placement_rate >= 0.70 and not self._milestone_flags['70']:
                milestone_bonus = 50.0
                for a in self.agents:
                    rewards[a] += milestone_bonus
                self._milestone_flags['70'] = True
                print(f"  🎯 70% MILESTONE @ step {self.timestep}! (+{milestone_bonus})")
            
            if placement_rate >= 0.85 and not self._milestone_flags.get('85', False):
                milestone_bonus = 100.0
                for a in self.agents:
                    rewards[a] += milestone_bonus
                self._milestone_flags['85'] = True
                print(f"  🎯🎯 85% MILESTONE @ step {self.timestep}! (+{milestone_bonus})")
            
            if placement_rate >= 0.90 and not self._milestone_flags['90']:
                milestone_bonus = 200.0
                for a in self.agents:
                    rewards[a] += milestone_bonus
                self._milestone_flags['90'] = True
                print(f"  🎯🎯 90% MILESTONE @ step {self.timestep}! (+{milestone_bonus})")
            
            if placement_rate >= 0.95 and not self._milestone_flags['95']:
                milestone_bonus = 400.0
                for a in self.agents:
                    rewards[a] += milestone_bonus
                self._milestone_flags['95'] = True
                print(f"  🎯🎯🎯 95% MILESTONE @ step {self.timestep}! (+{milestone_bonus})")
            
            if placement_rate >= 1.0 and not self._milestone_flags['100']:
                milestone_bonus = 1000.0
                for a in self.agents:
                    rewards[a] += milestone_bonus
                self._milestone_flags['100'] = True
                print(f"  🎯🎯🎯🎯 100% COMPLETE @ step {self.timestep}! (+{milestone_bonus})")

        # ============================================================
        # PHASE 4: TERMINATION AND EPISODE-END BONUSES
        # ============================================================
        all_placed = (len(self.placed_subjects) == self.num_subjects)
        time_limit = (self.timestep >= self.max_timesteps)
        
        if self.enable_repair_pass and time_limit and not all_placed:
            self._repair_fill()
        
        all_placed = (len(self.placed_subjects) == self.num_subjects)

        for agent in self.agents:
            terminations[agent] = all_placed
            truncations[agent] = (time_limit and not all_placed)

        # Episode-end bonuses with QUADRATIC completion pressure
        if all_placed or time_limit:
            duplicate_penalty = 0.0
            for subj in range(self.num_subjects):
                sec_idx = self.subject_section_idx[subj]
                current = self._get_placement_count(subj, sec_idx)
                required = self.subject_required_placements.get(subj, 1)
                if current > required:
                    excess = current - required
                    # AGGRESSIVE duplicate penalty
                    duplicate_penalty += excess * self.r_duplicate_penalty * curriculum_mult
            
            if duplicate_penalty < 0:  # It's negative
                for a in self.agents:
                    rewards[a] += duplicate_penalty
            
            valid_placements = 0
            for subj in range(self.num_subjects):
                sec_idx = self.subject_section_idx[subj]
                current = self._get_placement_count(subj, sec_idx)
                required = self.subject_required_placements.get(subj, 1)
                if current == required:
                    valid_placements += 1
            
            completion_rate = valid_placements / max(1, self.num_subjects)
            
            completion_bonus = 100.0 * (completion_rate ** 3)
            
            placed_tors = 0
            total_placed_with_tor = 0
            for subj in self.placed_subjects:
                tor = self.subject_teacher_idx[subj]
                if 0 <= tor < self.num_teachers:
                    total_placed_with_tor += 1
                    for d in range(self.num_days):
                        for ts in range(self.num_timeslots):
                            if self.placement_teachers.get((subj, d, ts), -1) == tor:
                                placed_tors += 1
                                break
                        else:
                            continue
                        break
            
            tor_satisfaction_bonus = 0.0
            if total_placed_with_tor > 0:
                tor_rate = placed_tors / total_placed_with_tor
                tor_satisfaction_bonus = 30.0 * tor_rate
            
            total_balance = sum(self._calculate_area_workload_balance(agent) for agent in self.agents)
            avg_balance = total_balance / len(self.agents) if self.agents else 0.0
            balance_bonus = 10.0 * avg_balance
            
            # QUADRATIC COMPLETION PRESSURE (more aggressive)
            completion_pressure = 0.0
            
            if completion_rate < 0.90:
                gap = 0.90 - completion_rate
                completion_pressure = -5000.0 * (gap ** 2)  # Quadratic!
                if self.timestep <= 10 or self.timestep % 100 == 0:
                    print(f"  ⚠️ Below 90%: {completion_rate*100:.1f}% → penalty {completion_pressure:.1f}")
            
            elif completion_rate < 0.95:
                gap = 0.95 - completion_rate
                completion_pressure = -2000.0 * (gap ** 2)  # Quadratic!
                if self.timestep % 100 == 0:
                    print(f"  ⚠️ Below 95%: {completion_rate*100:.1f}% → penalty {completion_pressure:.1f}")
            
            elif completion_rate < 1.0:
                gap = 1.0 - completion_rate
                completion_pressure = -1000.0 * (gap ** 2)  # Quadratic!
                if self.timestep % 100 == 0:
                    print(f"  ⚠️ Below 100%: {completion_rate*100:.1f}% → penalty {completion_pressure:.1f}")
            
            else:
                completion_pressure = 10000.0
                print(f"  🎉🎉🎉 100% COMPLETION! Bonus: +{completion_pressure:.1f}")
            
            total_bonus = (
                completion_bonus + 
                tor_satisfaction_bonus + 
                balance_bonus + 
                completion_pressure
            )
            
            for a in self.agents:
                rewards[a] += total_bonus
            
            if self.timestep <= 10 or time_limit:
                print(f"\n  📊 Episode End Bonuses:")
                print(f"     Completion: {completion_rate*100:.1f}%")
                print(f"     Completion bonus: +{completion_bonus:.1f}")
                print(f"     TOR satisfaction: +{tor_satisfaction_bonus:.1f}")
                print(f"     Balance: +{balance_bonus:.1f}")
                print(f"     Pressure: {completion_pressure:+.1f}")
                print(f"     Duplicate penalty: {duplicate_penalty:+.1f}")
                print(f"     Total bonus: {total_bonus:+.1f}\n")

        observations = self._get_all_observations()
        infos = {
            agent: {
                "error_rate": self.calculate_error_rate(),
                "conflict_count": self.conflict_count,
                "timestep": self.timestep,
                "fail_stats": dict(self.fail_stats),
                "workload_balance": self._calculate_area_workload_balance(agent),
                "modality_stats": dict(self.modality_stats),
                "difficulty": difficulty,
                "curriculum_phase": curriculum_mult,
                "placement_rate": len(self.placed_subjects) / self.num_subjects,
            } for agent in self.agents
        }

        # Apply reward scaling and clipping
        for agent in rewards:
            rewards[agent] = rewards[agent] * self.reward_scale
            rewards[agent] = np.clip(rewards[agent], -100.0, 100.0)
            
            if not np.isfinite(rewards[agent]):
                rewards[agent] = 0.0
        
        return observations, rewards, terminations, truncations, infos

    def _repair_fill(self):
        """Enhanced repair with duplicate prevention"""
        repair_placed_subjects = set()
        
        for subj in range(self.num_subjects):
            if subj in self.placed_subjects or subj in repair_placed_subjects:
                continue

            sec_idx = self.subject_section_idx[subj]
            current_count = self._get_placement_count(subj, sec_idx)
            required_count = self.subject_required_placements.get(subj, 1)
            
            if current_count >= required_count:
                continue

            subject_area = self.subject_areas[subj]
            area_teachers = self.area_teacher_indices.get(subject_area, [])
            tor = self.subject_teacher_idx[subj]
            area_teachers_by_load = sorted(area_teachers, key=lambda t: self.teacher_classes[f"teacher_{t}"])
            
            teacher_order = []
            if 0 <= tor < self.num_teachers:
                teacher_order.append(tor)
                area_teachers_by_load = [t for t in area_teachers_by_load if t != tor]
            teacher_order.extend(area_teachers_by_load)
            other_teachers = [t for t in range(self.num_teachers) if t not in area_teachers and t != tor]
            other_teachers_by_load = sorted(other_teachers, key=lambda t: self.teacher_classes[f"teacher_{t}"])
            teacher_order.extend(other_teachers_by_load)

            allowed_timeslots = set(self.subject_allowed_timeslots[subj])

            placed = False
            for t_idx in teacher_order:
                tkey = f"teacher_{t_idx}"
                if self.teacher_classes[tkey] >= self._max_for(tkey):
                    continue

                pairs = self.subject_allowed_pairs[subj]
                if not pairs:
                    continue

                found = False
                for bkey, ridx in pairs:
                    room_code = self.buildings_room_info[bkey][ridx]
                    
                    is_valid, _ = self._validate_modality_room_match(subj, room_code)
                    if not is_valid:
                        continue
                    
                    sched = self.buildings_room_schedule[bkey]
                    for d in range(self.num_days):
                        if d in self.subject_day_placements.get(subj, set()):
                            continue
                        
                        for ts in range(self.num_timeslots):
                            if ts not in allowed_timeslots:
                                continue
                            
                            can_place, _ = self._can_place_on_day(subj, d, ts)
                            if not can_place:
                                continue
                            
                            if sched[ridx, d, ts] != -1:
                                continue
                            if self.teacher_schedules[tkey][d, ts]:
                                continue
                            if self.section_schedules[f"section_{sec_idx}"][d, ts]:
                                continue
                            
                            if subj in self.placed_subjects or subj in repair_placed_subjects:
                                found = True
                                placed = True
                                break
                            
                            sched[ridx, d, ts] = subj
                            self.teacher_schedules[tkey][d, ts] = True
                            self.section_schedules[f"section_{sec_idx}"][d, ts] = True
                            self.teacher_classes[tkey] += 1
                            self.subject_assignments[subj] = t_idx
                            self.placement_teachers[(subj, d, ts)] = t_idx
                            
                            key = (subj, sec_idx)
                            current = self.subject_placement_count.get(key, 0)
                            self.subject_placement_count[key] = current + 1
                            
                            if subj not in self.subject_day_usage:
                                self.subject_day_usage[subj] = set()
                            self.subject_day_usage[subj].add(d)
                            
                            self.subject_day_placements[subj].add(d)
                            
                            if self._is_subject_fully_placed(subj):
                                self.placed_subjects.add(subj)
                            
                            repair_placed_subjects.add(subj)
                            
                            found = True
                            placed = True
                            break
                        if found: break
                    if found: break
                if placed:
                    break

    def _update_communication(self):
        """Update communication buffer for agents"""
        if not self.comm_size:
            return
        placed_frac = len(self.placed_subjects) / max(1, self.num_subjects)
        self.communication_buffer[0] = np.float32(np.clip(placed_frac, 0.0, 1.0))
        conflict_pressure = self.conflict_count / max(1, self.num_subjects)
        self.communication_buffer[1] = np.float32(np.clip(conflict_pressure, 0.0, 1.0))
        
        for i in range(7):
            val = 0.0
            if i < len(self.areas):
                area = self.areas[i]
                area_teachers = self.area_teacher_indices[area]
                if area_teachers:
                    loads = [self.teacher_classes[f"teacher_{t}"] for t in area_teachers]
                    if loads and max(loads) > 0:
                        std_dev = float(np.std(loads))
                        if np.isfinite(std_dev):
                            balance_score = 1.0 / (1.0 + std_dev)
                            val = float(np.clip(balance_score, 0.0, 1.0))
            idx = 2 + i
            if idx < self.comm_size:
                self.communication_buffer[idx] = np.float32(val)

    def _build_masks_for(self, agent: str, focus: int) -> Tuple[np.ndarray, np.ndarray]:
        """Teacher-slot consistency in action masking"""
        area_name = self._agent_area_name(agent)
        area_teachers = self.area_teacher_indices.get(area_name, [])
        
        tmask = np.zeros(self.max_teachers_per_area + 1, dtype=np.float32)
        
        if focus in self.placed_subjects:
            tmask[-1] = 1.0
            smask = np.zeros(self.slot_choices + 1, dtype=np.float32)
            smask[-1] = 1.0
            return tmask, smask

        available_teachers = []
        
        for i, global_t_idx in enumerate(area_teachers):
            if i >= self.max_teachers_per_area:
                break
            tk = f"teacher_{global_t_idx}"
            if self.teacher_classes[tk] < self._max_for(tk):
                available_teachers.append((i, global_t_idx))
                tmask[i] = 1.0
        
        tmask[-1] = 1.0
        
        slot_to_valid_teachers: Dict[int, Set[int]] = {}
        smask = np.zeros(self.slot_choices + 1, dtype=np.float32)
        
        if 0 <= focus < self.num_subjects:
            sec_idx = self.subject_section_idx[focus]
            skey = f"section_{sec_idx}"
            allowed_pairs = set(self.subject_allowed_pairs[focus])
            allowed_blds = set(self.subject_campuses[focus])
            allowed_timeslots = set(self.subject_allowed_timeslots[focus])
            
            modality = self.subject_modalities[focus]
            
            current_count = self._get_placement_count(focus, sec_idx)
            required_count = self.subject_required_placements.get(focus, 1)
            
            if current_count >= required_count:
                smask[-1] = 1.0
                return tmask, smask
            
            for idx, (b, ridx, d, ts) in enumerate(self.global_slots):
                if d in self.subject_day_placements.get(focus, set()):
                    continue
                
                if b not in allowed_blds:
                    continue
                
                if ts not in allowed_timeslots:
                    continue
                
                room_code = self.buildings_room_info[b][ridx]
                is_virtual = self._is_virtual_room(room_code)
                
                if modality == 'Online' and not is_virtual:
                    continue
                if modality == 'Face-to-Face' and is_virtual:
                    continue
                if modality == 'Hybrid' and is_virtual:
                    continue
                
                if self.section_schedules[skey][d, ts]:
                    continue
                
                if (b, ridx) not in allowed_pairs:
                    continue
                
                if self.buildings_room_schedule[b][ridx, d, ts] != -1:
                    continue
                
                free_teachers_at_slot = set()
                for area_teacher_idx, global_teacher_idx in available_teachers:
                    tkey = f"teacher_{global_teacher_idx}"
                    if not self.teacher_schedules[tkey][d, ts]:
                        free_teachers_at_slot.add(area_teacher_idx)
                
                if free_teachers_at_slot:
                    slot_to_valid_teachers[idx] = free_teachers_at_slot
                    smask[idx] = 1.0
            
            if slot_to_valid_teachers:
                teachers_with_slots = set()
                for valid_teachers in slot_to_valid_teachers.values():
                    teachers_with_slots.update(valid_teachers)
                
                for i, (area_idx, global_idx) in enumerate(available_teachers):
                    if area_idx not in teachers_with_slots:
                        tmask[i] = 0.0
        
        smask[-1] = 1.0
        
        if not np.all(np.isfinite(tmask)):
            tmask = np.zeros(self.max_teachers_per_area + 1, dtype=np.float32)
            tmask[-1] = 1.0
        if not np.all(np.isfinite(smask)):
            smask = np.zeros(self.slot_choices + 1, dtype=np.float32)
            smask[-1] = 1.0
        
        return tmask, smask

    def _get_all_observations(self):
        """Get observations for all agents"""
        if self.enable_communication:
            self._update_communication()
        return {agent: self._get_saha_observation(agent) for agent in self.saha_agents}

    def _get_saha_observation(self, agent):
        """Get observation for a single SAHA agent"""
        core: List[float] = []
        area_name = self._agent_area_name(agent)
        area_teachers = self.area_teacher_indices.get(area_name, [])

        for i in range(self.max_teachers_per_area):
            if i < len(area_teachers):
                global_t_idx = area_teachers[i]
                tk = f"teacher_{global_t_idx}"
                cap = max(1, self._max_for(tk))
                remaining = max(0, cap - self.teacher_classes[tk])
                val = remaining / cap
                core.append(float(np.clip(val, 0.0, 1.0)))
            else:
                core.append(0.0)

        for b in self.building_keys:
            sched = self.buildings_room_schedule[b]
            free = (sched == -1)
            for d in range(self.num_days):
                for ts in range(self.num_timeslots):
                    if free.shape[0] > 0:
                        frac_free = float(free[:, d, ts].mean())
                        core.append(float(np.clip(frac_free, 0.0, 1.0)))
                    else:
                        core.append(0.0)

        for s in range(self.num_subjects):
            core.append(1.0 if s in self.placed_subjects else 0.0)

        focus = -1
        if self.include_focus_scalar:
            focus = self._saha_focus_subject(agent)
            if focus < 0:
                core.append(0.0)
            else:
                val = (focus + 1) / max(1, self.num_subjects)
                core.append(float(np.clip(val, 0.0, 1.0)))
                
        if self.include_focus_tor_scalar:
            if focus < 0:
                focus = self._saha_focus_subject(agent)
            tor = self.subject_teacher_idx[focus] if 0 <= focus < self.num_subjects else -1
            area_tor_idx = self._global_teacher_idx_to_area(agent, tor) if tor >= 0 else -1
            if area_tor_idx < 0:
                core.append(0.0)
            else:
                val = (area_tor_idx + 1) / max(1, self.max_teachers_per_area)
                core.append(float(np.clip(val, 0.0, 1.0)))

        if self.include_section_features:
            if focus < 0:
                core.append(0.0)
                core.extend([0.0] * self.num_days)
                core.append(0.0)
            else:
                sec_idx = self.subject_section_idx[focus]
                sec_id_scalar = (sec_idx + 1) / max(1, self.num_sections)
                core.append(float(np.clip(sec_id_scalar, 0.0, 1.0)))

                pairs = self.subject_allowed_pairs[focus]
                day_avail: List[float] = []
                for d in range(self.num_days):
                    ts_ok = 0
                    for ts in range(self.num_timeslots):
                        if self.section_schedules[f"section_{sec_idx}"][d, ts]:
                            continue
                        any_free = False
                        for (b, ridx) in pairs:
                            if self.buildings_room_schedule[b][ridx, d, ts] == -1:
                                any_free = True
                                break
                        if any_free:
                            ts_ok += 1
                    day_val = ts_ok / max(1, self.num_timeslots)
                    day_avail.append(float(np.clip(day_val, 0.0, 1.0)))
                core.extend(day_avail)

                total_ts = self.num_days * self.num_timeslots
                feasible_slots = 0
                for d in range(self.num_days):
                    for ts in range(self.num_timeslots):
                        if self.section_schedules[f"section_{sec_idx}"][d, ts]:
                            continue
                        ok = False
                        for (b, ridx) in pairs:
                            if self.buildings_room_schedule[b][ridx, d, ts] == -1:
                                ok = True
                                break
                        if ok:
                            feasible_slots += 1
                feasible_frac = feasible_slots / max(1, total_ts)
                core.append(float(np.clip(feasible_frac, 0.0, 1.0)))

        if self.include_workload_features:
            for i in range(self.max_teachers_per_area):
                if i < len(area_teachers):
                    global_t_idx = area_teachers[i]
                    tk = f"teacher_{global_t_idx}"
                    load = self.teacher_classes[tk]
                    cap = max(1, self._max_for(tk))
                    normalized_load = load / cap
                    core.append(float(np.clip(normalized_load, 0.0, 1.0)))
                else:
                    core.append(0.0)
            
            balance_score = self._calculate_area_workload_balance(agent)
            core.append(float(np.clip(balance_score, 0.0, 1.0)))
            
            if len(area_teachers) > 1:
                loads = [self.teacher_classes[f"teacher_{t}"] for t in area_teachers]
                if loads and max(loads) > min(loads):
                    std_dev = float(np.std(loads))
                    max_possible_std = float(max(loads) - min(loads))
                    if max_possible_std > 0:
                        normalized_std = 1.0 - (std_dev / max_possible_std)
                        core.append(float(np.clip(normalized_std, 0.0, 1.0)))
                    else:
                        core.append(1.0)
                else:
                    core.append(1.0)
            else:
                core.append(1.0)

        if focus >= 0:
            modality_idx = self.subject_modality_idx[focus]
            modality_onehot = [0.0] * len(self.modality_labels)
            modality_onehot[modality_idx] = 1.0
            core.extend(modality_onehot)
            
            if self.subject_modalities[focus] == 'Online':
                virtual_free = 0
                for vroom in self.virtual_rooms:
                    if vroom in self.room_code_to_pair:
                        b, ridx = self.room_code_to_pair[vroom]
                        sched = self.buildings_room_schedule[b]
                        if np.any(sched[ridx, :, :] == -1):
                            virtual_free += 1
                if len(self.virtual_rooms) > 0:
                    virtual_availability = virtual_free / len(self.virtual_rooms)
                    core.append(float(np.clip(virtual_availability, 0.0, 1.0)))
                else:
                    core.append(0.0)
            else:
                core.append(0.0)
        else:
            core.extend([0.0] * (len(self.modality_labels) + 1))

        if self.enable_communication and self.comm_size:
            core.extend(self.communication_buffer.tolist())

        core_arr = np.array(core, dtype=np.float32)
        
        if not np.all(np.isfinite(core_arr)):
            nan_mask = ~np.isfinite(core_arr)
            core_arr[nan_mask] = 0.0
        
        if core_arr.shape[0] != self.saha_obs_core_size:
            raise RuntimeError(
                f"Observation core length mismatch: got {core_arr.shape[0]} vs {self.saha_obs_core_size}"
            )

        if not self.use_action_masks:
            return np.clip(core_arr, 0.0, 1.0).astype(np.float32)

        if focus < 0:
            tmask = np.zeros(self.max_teachers_per_area + 1, dtype=np.float32); tmask[-1] = 1.0
            smask = np.zeros(self.slot_choices + 1, dtype=np.float32); smask[-1] = 1.0
        else:
            tmask, smask = self._build_masks_for(agent, focus)

        return {
            "obs": np.clip(core_arr, 0.0, 1.0).astype(np.float32),
            "teacher_mask": tmask.astype(np.float32),
            "slot_mask": smask.astype(np.float32),
        }

    def subject_priority(self, s: int) -> float:
        """Calculate priority for a subject"""
        p = 0.0
        if 0 <= self.subject_teacher_idx[s] < self.num_teachers:
            p += 10.0
        if self.subject_allowed_rooms[s]:
            p += 5.0
        p += min(5.0, (self.fail_count[s] // 10) * 1.0)
        p += s * 1e-4
        return p

    def calculate_error_rate(self):
        """Calculate placement error rate"""
        total = self.num_subjects
        errors = total - len(self.placed_subjects)
        return errors / total if total > 0 else 0.0

    def validate_schedule(self):
        """Comprehensive schedule validation"""
        conflicts = {
            'teacher_conflicts': [],
            'section_conflicts': [],
            'room_conflicts': [],
            'duplicate_subjects': [],
            'modality_violations': [],
            'summary': {}
        }
        
        print("\n" + "="*80)
        print("SCHEDULE VALIDATION")
        print("="*80)
        
        placement_map = defaultdict(list)
        
        for b in self.building_keys:
            b_sched = self.buildings_room_schedule[b]
            for r_idx in range(len(self.buildings_room_info[b])):
                for d in range(self.num_days):
                    for ts in range(self.num_timeslots):
                        subj = b_sched[r_idx, d, ts]
                        
                        if subj >= 0:
                            sec_idx = self.subject_section_idx[subj]
                            teacher = self.placement_teachers.get((subj, d, ts), -1)
                            room_code = self.buildings_room_info[b][r_idx]
                            
                            placement_map[(subj, sec_idx)].append({
                                'day': d,
                                'timeslot': ts,
                                'building': b,
                                'room_idx': r_idx,
                                'room_code': room_code,
                                'teacher': teacher
                            })
        
        total_placements = sum(len(placements) for placements in placement_map.values())
        print(f"   Found {total_placements} total placements")
        print(f"   Covering {len(placement_map)} unique (subject, section) pairs")
        
        duplicate_count = 0
        
        for (subj, sec_idx), placements in placement_map.items():
            actual_count = len(placements)
            required_count = self.subject_required_placements.get(subj, 1)
            
            if actual_count > required_count:
                excess = actual_count - required_count
                duplicate_count += excess
                
                conflicts['duplicate_subjects'].append({
                    'subject': self.subject_codes[subj],
                    'section': self.section_labels[sec_idx],
                    'required': required_count,
                    'actual': actual_count,
                    'excess': excess,
                    'placements': placements
                })
        
        if duplicate_count == 0:
            print("   ✅ No duplicates")
        else:
            print(f"\n   ❌ TOTAL DUPLICATES: {duplicate_count}")
        
        teacher_conflict_count = 0
        teacher_time_map = defaultdict(list)
        
        for (subj, sec_idx), placements in placement_map.items():
            for p in placements:
                teacher = p['teacher']
                if teacher >= 0:
                    key = (teacher, p['day'], p['timeslot'])
                    teacher_time_map[key].append({'subject': subj, 'section': sec_idx})
        
        for (teacher, day, ts), assignments in teacher_time_map.items():
            if len(assignments) > 1:
                teacher_conflict_count += (len(assignments) - 1)
                conflicts['teacher_conflicts'].append({
                    'teacher': teacher,
                    'day': day,
                    'timeslot': ts,
                    'count': len(assignments),
                    'assignments': assignments
                })
        
        if teacher_conflict_count == 0:
            print("   ✅ No teacher conflicts")
        else:
            print(f"\n   ❌ TEACHER CONFLICTS: {teacher_conflict_count}")
        
        total_conflicts = duplicate_count + teacher_conflict_count
        
        fully_placed_count = sum(
            1 for (subj, sec_idx), placements in placement_map.items()
            if len(placements) >= self.subject_required_placements.get(subj, 1)
        )
        
        conflicts['summary'] = {
            'total_conflicts': total_conflicts,
            'duplicate_placements': duplicate_count,
            'teacher_conflicts': teacher_conflict_count,
            'placement_rate': len(placement_map) / self.num_subjects * 100,
            'fully_placed': fully_placed_count,
            'fully_placed_rate': fully_placed_count / self.num_subjects * 100
        }
        
        print(f"\nSUMMARY:")
        print(f"Total conflicts: {total_conflicts}")
        print(f"Fully placed: {fully_placed_count}/{self.num_subjects} ({conflicts['summary']['fully_placed_rate']:.1f}%)")
        print("="*80 + "\n")
        
        return conflicts

    def render(self):
        """Render environment status"""
        print(f"\nStep: {self.timestep}/{self.max_timesteps}")
        print(f"Placed: {len(self.placed_subjects)}/{self.num_subjects}")
        print(f"Curriculum phase: {self._get_curriculum_multiplier():.1f}x")

    def close(self):
        pass

    @lru_cache(maxsize=256)
    def _cached_priority(self, subj_idx: int) -> float:
        return self.subject_priority(subj_idx)