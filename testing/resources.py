# training/train_ppo_pbt_workload_balanced.py
import sys, os, json, time, re, random
from typing import Optional
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

from ray import init
from ray.air import RunConfig
from ray.tune import CLIReporter, Tuner
from ray.air.config import CheckpointConfig
from ray.tune.logger import TBXLoggerCallback
from ray.tune.registry import register_env
from ray.tune.schedulers import PopulationBasedTraining
from ray import tune

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from torch.utils.tensorboard import SummaryWriter

# RLlib model bits
import torch
import torch.nn as nn
from gymnasium import spaces as gym_spaces
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog

# Import ENV and area resolver (assuming these are in your project)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.timetabling_env import ParallelTimetablingEnv
from area_subject_resolver import assign_area

# Windows-friendly Ray spill dirs
SPILL_DIR = "C:/ray_spill"; TEMP_DIR = "C:/ray_temp"
os.makedirs(SPILL_DIR, exist_ok=True); os.makedirs(TEMP_DIR, exist_ok=True)
os.environ["RAY_object_spilling_config"] = json.dumps(
    {"type": "filesystem","params": {"directory_path": SPILL_DIR}}
)

# Your existing SahaMaskedTwoHead model (keeping the same)
from collections import OrderedDict

class SahaMaskedTwoHead(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        hidden_sizes = model_config.get("custom_model_config", {}).get("hidden_sizes", [256, 256])

        nvec = getattr(action_space, "nvec", None)
        assert nvec is not None and len(nvec) == 2, "Expect MultiDiscrete with 2 heads"
        self.n_teacher = int(nvec[0])
        self.n_slot = int(nvec[1])

        layers = []
        layers += [nn.LazyLinear(hidden_sizes[0]), nn.ReLU()]
        for i in range(len(hidden_sizes) - 1):
            layers += [nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)

        final_dim = hidden_sizes[-1]
        self.teacher_head = nn.Linear(final_dim, self.n_teacher)
        self.slot_head = nn.Linear(final_dim, self.n_slot)
        self.value_branch = nn.Linear(final_dim, 1)

        self._logits_dim = self.n_teacher + self.n_slot
        self._value_out = None

    def _to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(x, dtype=torch.float32)

    def _extract_core_and_masks(self, obs):
        tmask = smask = None

        if isinstance(obs, (dict, OrderedDict)):
            core_val = obs.get("obs", None)

            if core_val is None:
                parts = []
                for k, v in obs.items():
                    if k in ("teacher_mask", "slot_mask"):
                        continue
                    if isinstance(v, (np.ndarray, torch.Tensor)) and np.asarray(v).ndim >= 1:
                        parts.append(self._to_tensor(v).float())
                if len(parts) == 0:
                    core_val = next(iter(obs.values()))
                    core = self._to_tensor(core_val).float()
                else:
                    parts = [p if p.ndim > 1 else p.unsqueeze(0) for p in parts]
                    core = torch.cat(parts, dim=-1).float()
            else:
                core = self._to_tensor(core_val).float()

            if "teacher_mask" in obs:
                tmask = self._to_tensor(obs["teacher_mask"]).float()
            if "slot_mask" in obs:
                smask = self._to_tensor(obs["slot_mask"]).float()

            def _ensure_2d(m):
                if m is None: return None
                if m.ndim == 1: m = m.unsqueeze(0)
                return m
            tmask = _ensure_2d(tmask)
            smask = _ensure_2d(smask)

        else:
            core = self._to_tensor(obs).float()
            if core.ndim == 1:
                core = core.unsqueeze(0)

        return core, tmask, smask

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        core, tmask, smask = self._extract_core_and_masks(obs)

        h = self.backbone(core)
        t_logits = self.teacher_head(h)
        s_logits = self.slot_head(h)

        if tmask is not None and smask is not None:
            big_neg = torch.finfo(t_logits.dtype).min / 2
            t_logits = t_logits.masked_fill(tmask < 0.5, big_neg)
            s_logits = s_logits.masked_fill(smask < 0.5, big_neg)

        logits = torch.cat([t_logits, s_logits], dim=1)
        self._value_out = self.value_branch(h).squeeze(-1)
        return logits, state

    def value_function(self):
        return self._value_out

ModelCatalog.register_custom_model("saha_masked_head", SahaMaskedTwoHead)

# ======================================================================
# CSV LOAD/CLEAN - Complete data preprocessing
# ======================================================================
csv_path = os.path.join(os.path.dirname(__file__), "../standardized_schedule_math_only.csv")
df = pd.read_csv(csv_path)

for col in ["Subject","Faculty","Room","Campus","Day","Time","Section"]:
    if col in df.columns:
        df[col] = df[col].fillna("").astype(str).str.replace(r"\s+"," ",regex=True).str.strip()

orig_subjects  = df["Subject"].tolist()
orig_teachers  = df["Faculty"].tolist()
orig_sections  = df["Section"].tolist() if "Section" in df.columns else ["GEN"] * len(df)

# ======================================================================
# DAYS
# ======================================================================
DAY_TOKENS_ORDER = ["MH","TF","TTH","T","H","F","M","W"]
DAY_REGEX = r"TTH|MH|TF|M|T|W|H|F"
def tokenize_days_exact(cell: str) -> list[str]:
    return re.findall(DAY_REGEX, (str(cell) or "").upper())
day_lists = df["Day"].apply(tokenize_days_exact)
present_days = {d for lst in day_lists for d in lst}
day_labels = [d for d in DAY_TOKENS_ORDER if d in present_days]
num_days   = len(day_labels)

# ======================================================================
# EXPAND ROWS PER DAY
# ======================================================================
df_rooms           = df[df["Room"] != ""]
room_to_campus     = df_rooms.set_index("Room")["Campus"].to_dict()
building_to_campus = {r[0]: c for r, c in room_to_campus.items()}
teacher_to_campus  = (df[["Faculty","Campus"]].drop_duplicates("Faculty")
                        .set_index("Faculty")["Campus"].to_dict())

expanded_subjects, expanded_teachers, expanded_campuses, expanded_sections = [], [], [], []
for subj, teach, sect, days in zip(orig_subjects, orig_teachers, orig_sections, day_lists):
    allowed = [b for b, c in building_to_campus.items() if c == teacher_to_campus.get(teach)]
    for _ in days:
        expanded_subjects.append(subj)
        expanded_teachers.append(teach)
        expanded_campuses.append(allowed)
        expanded_sections.append(sect)

subject_names   = [s if (isinstance(s,str) and s.strip() and s.lower()!="nan") else f"SUBJ_{i:04d}"
                   for i,s in enumerate(expanded_subjects)]
teacher_names    = expanded_teachers
subject_campuses = expanded_campuses

# Teacher IDs
teacher_id_to_name = list(pd.unique(pd.Series(teacher_names)))
teacher_name_to_idx = {name:i for i,name in enumerate(teacher_id_to_name)}
num_subjects = len(subject_names)
num_teachers = len(teacher_id_to_name)

# TOR per subject
subject_teacher_idx = [teacher_name_to_idx.get(t, -1) for t in teacher_names]

# Teacher caps (clamp to 4)
MAX_CAP = 4
_cap_counts = Counter(subject_teacher_idx)
teacher_max_classes_map = {f"teacher_{i}": int(min(MAX_CAP, _cap_counts.get(i, 0))) for i in range(num_teachers)}

# ======================================================================
# TIMESLOTS
# ======================================================================
def _norm_hhmm(s: str) -> str:
    s = s.replace(":", "").replace(".", "").strip()
    if len(s) == 3: s = "0" + s
    return s
def _norm_range(tok: str) -> str:
    a, b = tok.split("-"); return f"{_norm_hhmm(a)}-{_norm_hhmm(b)}"

time_tokens = (df["Time"].astype(str)
    .str.replace(r"\s+"," ",regex=True)
    .str.findall(r"(?:[01]?\d|2[0-3])[:.]?\d{2}-(?:[01]?\d|2[0-3])[:.]?\d{2}")
    .explode().dropna().map(_norm_range).drop_duplicates().tolist()
)
timeslot_labels = sorted(time_tokens)
num_timeslots   = len(timeslot_labels)

# ======================================================================
# ROOMS & SUBJECT→ALLOWED ROOMS
# ======================================================================
room_codes = sorted(df_rooms["Room"].unique().tolist())

df_sr = df[["Subject","Room"]].copy().astype(str).applymap(str.strip)
df_sr = df_sr[(df_sr["Subject"]!="") & (df_sr["Room"]!="")]
rooms_by_subject = df_sr.groupby("Subject")["Room"].apply(lambda s: sorted(set(s.tolist()))).to_dict()
subject_allowed_rooms = [rooms_by_subject.get(subj, []) for subj in subject_names]

# If a subject has explicit rooms, restrict buildings to those rooms' buildings
def _bkey(room_code: str) -> str: return room_code[0] if room_code else ""
subject_campuses_fixed = []
for i in range(num_subjects):
    rooms = subject_allowed_rooms[i]
    if rooms:
        buildings = sorted({_bkey(r) for r in rooms if r})
        subject_campuses_fixed.append(buildings)
    else:
        subject_campuses_fixed.append(subject_campuses[i])
subject_campuses = subject_campuses_fixed

# ======================================================================
# AREAS (CSV-first + fallback)
# ======================================================================
subject_areas = [assign_area(subj) for subj in subject_names]
area_labels   = sorted(list(dict.fromkeys(subject_areas)))

# ======================================================================
# SECTIONS
# ======================================================================
section_labels = list(pd.unique(pd.Series(expanded_sections)))
section_name_to_idx = {name: i for i, name in enumerate(section_labels)}
subject_section_idx = [section_name_to_idx.get(s, 0) for s in expanded_sections]

# ======================================================================
# AREA-BASED TEACHER GROUPING
# ======================================================================
print("Creating area-based teacher constraints...")

# Map teachers to areas based on the subjects they're assigned as TOR
teacher_to_areas = defaultdict(set)
for subj_idx, teacher_idx in enumerate(subject_teacher_idx):
    if teacher_idx >= 0:  # Valid teacher assignment
        area = subject_areas[subj_idx]
        teacher_to_areas[teacher_idx].add(area)

# Create area-to-teachers mapping
area_to_teachers = defaultdict(list)
for teacher_idx, areas in teacher_to_areas.items():
    # If teacher teaches multiple areas, assign to primary area (most subjects)
    if areas:
        # Count subjects per area for this teacher
        area_counts = defaultdict(int)
        for subj_idx, t_idx in enumerate(subject_teacher_idx):
            if t_idx == teacher_idx:
                area_counts[subject_areas[subj_idx]] += 1
        primary_area = max(area_counts.keys(), key=lambda a: area_counts[a])
        area_to_teachers[primary_area].append(teacher_idx)

# Handle teachers not assigned to any subject (assign to smallest area)
unassigned_teachers = []
for teacher_idx in range(num_teachers):
    if teacher_idx not in teacher_to_areas or not teacher_to_areas[teacher_idx]:
        unassigned_teachers.append(teacher_idx)

# Distribute unassigned teachers evenly across areas
for i, teacher_idx in enumerate(unassigned_teachers):
    area = area_labels[i % len(area_labels)]
    area_to_teachers[area].append(teacher_idx)

# Ensure every area has at least one teacher
for area in area_labels:
    if not area_to_teachers[area]:
        # Assign the first available teacher
        if unassigned_teachers:
            teacher_idx = unassigned_teachers.pop(0)
            area_to_teachers[area].append(teacher_idx)
        else:
            # Borrow from largest area
            largest_area = max(area_to_teachers.keys(), key=lambda a: len(area_to_teachers[a]))
            if len(area_to_teachers[largest_area]) > 1:
                teacher_idx = area_to_teachers[largest_area].pop()
                area_to_teachers[area].append(teacher_idx)

# Create area-to-teacher-indices mapping for environment
area_teacher_indices = {}
for area in area_labels:
    area_teacher_indices[area] = sorted(area_to_teachers[area])

print(f"Teacher distribution across areas:")
for area in area_labels:
    teachers = area_teacher_indices[area]
    teacher_names_for_area = [teacher_id_to_name[i] for i in teachers]
    print(f"  {area}: {len(teachers)} teachers - {teacher_names_for_area[:3]}{'...' if len(teachers) > 3 else ''}")

# Complete callback class for episode tracking and metrics
class ParallelScheduleCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.writer = SummaryWriter(log_dir="C:/ray_logs/pbt_tensorboard")
        self.episode_counter = 0
        
    def on_train_result(self, *, algorithm, result, **kwargs):
        it = result.get("training_iteration", 0)
        ts = result.get("timesteps_total", 0)
        rew = result.get("episode_reward_mean", 0.0)
        self.writer.add_scalar("Reward/EpisodeRewardMean", rew, it)
        self.writer.add_scalar("Timesteps/Total", ts, it)
        for k, v in result.get("custom_metrics", {}).items():
            if isinstance(v, (int,float)):
                self.writer.add_scalar(f"Custom/{k}", v, it)
        learner_info = result.get("info", {}).get("learner", {})
        for pid, data in learner_info.items():
            stats = data.get("learner_stats", {})
            for name in ("policy_loss","vf_loss","entropy"):
                val = stats.get(name)
                if val is not None:
                    self.writer.add_scalar(f"{pid}/{name}", val, it)
        if "time_this_iter_s" in result and result["time_this_iter_s"] > 0:
            throughput = ts / result["time_this_iter_s"]
            self.writer.add_scalar("Performance/Throughput", throughput, it)
            
    def _unwrap_env(self, base_env):
        env_like = None
        if hasattr(base_env, "get_sub_environments"):
            try: env_like = base_env.get_sub_environments()[0]
            except Exception: env_like = None
        if env_like is None and hasattr(base_env, "vector_env") and hasattr(base_env.vector_env, "envs"):
            try: env_like = base_env.vector_env.envs[0]
            except Exception: env_like = None
        if env_like is None: env_like = base_env
        return getattr(env_like, "par_env", env_like)
        
    def on_episode_end(self, *, worker, base_env, episode, **kwargs):
        self.episode_counter += 1
        if worker.worker_index != 1: return
        if self.episode_counter % 10 != 0: return
        env = self._unwrap_env(base_env)

        assigns = np.array(env.subject_assignments, dtype=int)
        num_assigned = int((assigns >= 0).sum())
        counts = list(env.teacher_classes.values())

        full_load = sum(1 for i,c in enumerate(counts) if c >= env._max_for(f"teacher_{i}"))
        under_load = sum(1 for c in counts if c < env.max_classes)

        tor_arr = np.array(env.subject_teacher_idx, dtype=int)
        has_tor = tor_arr >= 0
        placed_mask = np.zeros(env.num_subjects, dtype=bool)
        for s in env.placed_subjects: placed_mask[s] = True
        valid_placed = has_tor & placed_mask
        match_rate_placed = float((assigns[valid_placed] == tor_arr[valid_placed]).mean()) if valid_placed.any() else 0.0
        coverage = float(valid_placed.sum()) / max(1, has_tor.sum())

        # Area-appropriate assignment analysis
        area_appropriate = 0
        total_area_assignments = 0
        for subj_idx in env.placed_subjects:
            assigned_teacher = assigns[subj_idx]
            if assigned_teacher >= 0:
                subj_area = env.subject_areas[subj_idx]
                teacher_area_teachers = env.area_teacher_indices.get(subj_area, [])
                total_area_assignments += 1
                if assigned_teacher in teacher_area_teachers:
                    area_appropriate += 1
        
        area_match_rate = area_appropriate / max(1, total_area_assignments)

        # Workload balance analysis
        area_balance_scores = []
        for agent in env.agents:
            balance_score = env._calculate_area_workload_balance(agent)
            area_balance_scores.append(balance_score)
        avg_workload_balance = np.mean(area_balance_scores) if area_balance_scores else 0.0
        
        # Individual teacher workload analysis
        workload_std = np.std(counts) if counts else 0.0
        workload_range = max(counts) - min(counts) if counts else 0

        print(f"\n=== Episode {self.episode_counter} Summary (worker#1) ===")
        print(f"Timesteps: {env.timestep} | Placed: {len(env.placed_subjects)}/{env.num_subjects} "
              f"| Conflicts: {env.conflict_count} | Faculty full/partial: {full_load}/{under_load}")
        print(f"Teacher match rate (placed-only): {match_rate_placed:.3f} | Coverage: {coverage:.3f}")
        print(f"Area-appropriate assignments: {area_match_rate:.3f} ({area_appropriate}/{total_area_assignments})")
        print(f"Workload balance - Avg area score: {avg_workload_balance:.3f} | Std: {workload_std:.2f} | Range: {workload_range}")
        print("Fail stats:", env.fail_stats)
        print("=" * 40)

        for k, v in env.fail_stats.items():
            self.writer.add_scalar(f"Failures/{k}", v, self.episode_counter)

        # Track workload balance metrics
        self.writer.add_scalar("WorkloadBalance/AvgAreaScore", avg_workload_balance, self.episode_counter)
        self.writer.add_scalar("WorkloadBalance/TeacherStd", workload_std, self.episode_counter)
        self.writer.add_scalar("WorkloadBalance/TeacherRange", workload_range, self.episode_counter)

        episode.custom_metrics["teacher_match_rate_placed"] = match_rate_placed
        episode.custom_metrics["area_match_rate"] = area_match_rate
        episode.custom_metrics["avg_workload_balance"] = avg_workload_balance
        episode.custom_metrics["teacher_workload_std"] = workload_std
        episode.custom_metrics["teacher_workload_range"] = workload_range
        episode.custom_metrics["match_coverage"] = coverage
        episode.custom_metrics["parallel_timesteps"] = env.timestep
        episode.custom_metrics["conflict_rate"] = (env.conflict_count / env.timestep if env.timestep > 0 else 0.0)
        episode.custom_metrics["assignment_rate"] = (num_assigned / env.num_subjects if env.num_subjects > 0 else 0.0)
        if counts:
            workload_std = np.std(counts)
            episode.custom_metrics["workload_balance"] = 1.0 / (1.0 + workload_std)
            
    def __del__(self):
        if hasattr(self, "writer"): self.writer.close()

# Your existing environment factory
def make_parallel_env(config=None):
    # Use config parameters if provided, otherwise use defaults
    env_config = config or {}
    
    env = ParallelTimetablingEnv(
        num_teachers=num_teachers,
        num_subjects=num_subjects,
        num_days=num_days,
        num_timeslots=num_timeslots,
        room_codes=room_codes,
        subject_codes=subject_names,
        subject_campuses=subject_campuses,
        subject_areas=subject_areas,
        subject_allowed_rooms=subject_allowed_rooms,
        subject_section_idx=subject_section_idx,
        section_labels=section_labels,
        max_classes_per_teacher=4,
        teacher_max_classes_map=teacher_max_classes_map,
        subject_teacher_idx=subject_teacher_idx,
        area_teacher_indices=area_teacher_indices,
        
        # These can be tuned via config
        strict_teacher_match=env_config.get("strict_teacher_match", False),
        r_teacher_match=env_config.get("r_teacher_match", 2.0),
        r_teacher_mismatch=env_config.get("r_teacher_mismatch", -0.5),
        r_area_match=env_config.get("r_area_match", 1.5),
        r_area_mismatch=env_config.get("r_area_mismatch", -1.0),
        r_workload_balance=env_config.get("r_workload_balance", 0.8),
        base_success_reward=env_config.get("base_success_reward", 0.05),
        
        include_focus_scalar=True,
        include_focus_tor_scalar=False,
        include_section_features=True,
        include_workload_features=True,
        enable_communication=True,
        use_action_masks=True,
        max_timesteps=env_config.get("max_timesteps", 400),
        enable_repair_pass=False,
    )
    
    # Add labels for printing/export
    env.subject_names = subject_names
    env.teacher_names = teacher_names
    env.teacher_labels = teacher_id_to_name
    env.day_labels = day_labels
    env.timeslot_labels = timeslot_labels
    env.room_to_campus = room_to_campus
    env.area_labels = area_labels
    env.section_labels = section_labels
    env.subject_section_idx = subject_section_idx
    
    return ParallelPettingZooEnv(env)

def main():
    init(ignore_reinit_error=True, include_dashboard=False, _temp_dir=TEMP_DIR)
    register_env("parallel_timetabling_env", make_parallel_env)

    # Test environment to get spaces
    dummy = make_parallel_env()
    obs_dict, _ = dummy.reset()
    raw = dummy.par_env
    first_saha = [a for a in raw.agents if a.startswith("saha_")][0]

    print("Setting up PBT hyperparameter tuning...")
    print("Obs space:", raw.observation_spaces[first_saha])
    print("Action heads:", raw.action_spaces[first_saha].nvec.tolist())

    # PBT configuration function to ensure valid hyperparameter combinations
    def explore(config):
        # Ensure batch sizes are compatible
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        
        # Ensure we run at least one SGD iteration
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
            
        # Ensure entropy coefficient schedule is valid
        if "entropy_coeff" in config:
            config["entropy_coeff_schedule"] = [
                (0, config["entropy_coeff"]), 
                (150_000, config["entropy_coeff"] * 0.25), 
                (300_000, 0.0)
            ]
            
        return config

    # Define PBT scheduler
    pbt = PopulationBasedTraining(
        time_attr="timesteps_total",  # Use timesteps instead of time for RL
        perturbation_interval=50000,   # Every 50k timesteps
        resample_probability=0.25,
        
        # Define which hyperparameters to mutate and how
        hyperparam_mutations={
            # Learning rate - critical for RL
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            
            # PPO clipping parameter
            "clip_param": lambda: random.uniform(0.1, 0.3),
            
            # Batch sizes - important for stability
            "train_batch_size": lambda: random.choice([512, 1024, 1536, 2048, 3072]),
            "sgd_minibatch_size": lambda: random.choice([128, 256, 512, 1024]),
            
            # Training iterations per update
            "num_sgd_iter": lambda: random.randint(3, 15),
            
            # Entropy coefficient for exploration
            "entropy_coeff": lambda: random.uniform(0.0, 0.05),
            
            # Discount factor
            "gamma": lambda: random.uniform(0.95, 0.999),
            
            # Value function coefficient
            "vf_loss_coeff": lambda: random.uniform(0.25, 1.0),
            
            # Environment-specific rewards (these will be passed to env)
            "r_teacher_match": lambda: random.uniform(1.0, 3.0),
            "r_area_match": lambda: random.uniform(0.5, 2.0),
            "r_workload_balance": lambda: random.uniform(0.2, 1.5),
        },
        
        custom_explore_fn=explore,
    )

    # Define the policy configuration
    def get_policy_config(config):
        return {
            "model": {
                "custom_model": "saha_masked_head",
                "custom_model_config": {"hidden_sizes": [256, 256]},
            },
            "lr": config.get("lr", 1e-4),
        }

    # Base configuration that will be used as starting point for all trials
    base_config = {
        # Environment
        "env": "parallel_timetabling_env",
        "env_config": {
            "enable_communication": True,
            # These will be overridden by PBT mutations
            "r_teacher_match": 2.0,
            "r_area_match": 1.5, 
            "r_workload_balance": 0.8,
            "max_timesteps": 400,
        },
        
        # Framework
        "framework": "torch",
        "disable_env_checking": True,
        
        # Rollout configuration
        "num_rollout_workers": 2,  # Reduced for PBT (multiple trials)
        "rollout_fragment_length": 64,
        "batch_mode": "complete_episodes",
        "enable_connectors": False,
        "compress_observations": True,
        "num_envs_per_worker": 1,
        
        # Resources - CPU only (no GPU required)
        "num_gpus": 0,  # Set to 0 for CPU-only training
        "num_gpus_per_worker": 0,  # Explicitly disable GPU for workers
        "num_cpus_for_local_worker": 1,
        
        # Multi-agent setup
        "multiagent": {
            "policies": {
                "saha_policy": (
                    None,
                    raw.observation_spaces[first_saha],
                    raw.action_spaces[first_saha],
                    get_policy_config({}),
                )
            },
            "policy_mapping_fn": lambda agent_id, episode, **kwargs: "saha_policy",
            "policies_to_train": ["saha_policy"],
            "count_steps_by": "env_steps",
        },
        
        # Evaluation
        "evaluation_interval": 10,  # Less frequent for PBT
        "evaluation_duration": 2,
        "evaluation_duration_unit": "episodes",
        "evaluation_num_workers": 1,
        "evaluation_config": {
            "explore": False,
            "seed": 12345,
        },
        
        # Debugging
        "seed": 42,
        "callbacks": ParallelScheduleCallback,
        "_enable_new_api_stack": False,
        "_disable_preprocessor_api": False,
        
        # Hyperparameters to be tuned by PBT (starting values)
        "lr": 1e-4,
        "gamma": 0.98,
        "clip_param": 0.2,
        "train_batch_size": 1536,
        "sgd_minibatch_size": 512,
        "num_sgd_iter": 10,
        "vf_clip_param": 50000.0,
        "entropy_coeff": 0.02,
        "vf_loss_coeff": 0.5,
        "grad_clip": 10.0,
        
        # Environment reward parameters (will be passed to env_config)
        "r_teacher_match": 2.0,
        "r_area_match": 1.5,
        "r_workload_balance": 0.8,
    }

    # Enhanced reporter for PBT
    reporter = CLIReporter(
        parameter_columns=[
            "lr", "clip_param", "train_batch_size", "sgd_minibatch_size", 
            "entropy_coeff", "r_teacher_match", "r_area_match"
        ],
        metric_columns=[
            "episode_reward_mean",
            "timesteps_total",
            "custom_metrics/teacher_match_rate_placed_mean",
            "custom_metrics/area_match_rate_mean", 
            "custom_metrics/avg_workload_balance_mean",
            "custom_metrics/assignment_rate_mean",
            "training_iteration",
        ],
        max_report_frequency=30,
        sort_by_metric=True,
        metric="episode_reward_mean",
        mode="max",
    )

    # Run configuration
    run_cfg = RunConfig(
        stop={"timesteps_total": 100000},  # Stop after 1M timesteps
        local_dir="C:/ray_logs",
        name="PPO_PBT_Timetabling_Hyperparameter_Search",
        checkpoint_config=CheckpointConfig(
            checkpoint_frequency=20, 
            checkpoint_at_end=True, 
            num_to_keep=3
        ),
        callbacks=[TBXLoggerCallback()],
        progress_reporter=reporter,
        verbose=1,
        log_to_file="ppo_pbt_hyperparameter_search.log",
    )

    # Create and run the tuner with PBT
    tuner = Tuner(
        "PPO",
        param_space=base_config,
        tune_config=tune.TuneConfig(
            scheduler=pbt,
            num_samples=4,  # Reduced from 8 for CPU-only training
            metric="episode_reward_mean",
            mode="max",
            reuse_actors=True,  # Reuse actors for efficiency
        ),
        run_config=run_cfg,
    )

    print(f"Starting PBT with {4} parallel trials...")
    print("This will automatically tune hyperparameters during training!")
    print("Note: Running on CPU only - consider fewer trials for better performance")
    
    results = tuner.fit()

    print("\n" + "="*50)
    print("PBT HYPERPARAMETER SEARCH COMPLETED!")
    print("="*50)
    
    try:
        # Get the best trial
        best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
        
        print(f"\nBest trial achieved reward: {best_result.metrics.get('episode_reward_mean', 'N/A')}")
        print(f"Best trial assignment rate: {best_result.metrics.get('custom_metrics/assignment_rate_mean', 'N/A')}")
        print(f"Best trial teacher match rate: {best_result.metrics.get('custom_metrics/teacher_match_rate_placed_mean', 'N/A')}")
        
        print("\n--- BEST HYPERPARAMETERS ---")
        best_config = best_result.config
        important_params = [
            "lr", "clip_param", "gamma", "train_batch_size", "sgd_minibatch_size", 
            "num_sgd_iter", "entropy_coeff", "vf_loss_coeff",
            "r_teacher_match", "r_area_match", "r_workload_balance"
        ]
        
        for param in important_params:
            if param in best_config:
                print(f"  {param}: {best_config[param]}")
        
        # Save the best configuration to a file
        best_params_file = "C:/ray_logs/best_hyperparameters.json"
        with open(best_params_file, 'w') as f:
            json.dump({k: best_config[k] for k in important_params if k in best_config}, f, indent=2)
        print(f"\nBest hyperparameters saved to: {best_params_file}")
        
        # Print checkpoint path for the best trial
        print(f"Best trial checkpoint: {best_result.checkpoint}")
        
    except Exception as e:
        print(f"Error retrieving best result: {e}")
        print("Training completed but results analysis failed.")
    
    print("\nYou can now use these hyperparameters in your regular training script!")

if __name__ == "__main__":
    main()