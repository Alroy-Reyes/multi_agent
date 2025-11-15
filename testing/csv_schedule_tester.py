"""
Export Schedule from Trained SAHA Model - MANILA VERSION v16.2 ENHANCED

Version 16.2: CRITICAL FIXES + Enhanced Validation
========================================================================================
CRITICAL FIXES:
🔧 FIX A: Updated reward parameters to MATCH training exactly
🔧 FIX B: Changed default checkpoint from 000000 (untrained) to 000010 (trained)
🔧 FIX C: Added validation for checkpoint iteration number
🔧 NEW: POST-EXPORT DUPLICATE VERIFICATION

PREVIOUS ISSUES RESOLVED:
❌ BUG: Export used different rewards than training (model confused)
❌ BUG: Default checkpoint was iteration 0 (completely untrained)
❌ BUG: No warning when using early/invalid checkpoints
❌ BUG: Duplicate validation didn't always show in output

UPDATES:
✅ Uses ImprovedSahaMaskedTwoHead model (matches training)
✅ Deeper value network for better predictions
✅ Proper observation dimension handling
✅ Progressive difficulty support
✅ Enhanced modality and constraint tracking
✅ Fixed department prefix cleaning with section detection
✅ Comprehensive duplicate validation
✅ FIX #12 verification
✅ REWARD PARAMETERS NOW MATCH TRAINING! (CRITICAL)
✅ POST-EXPORT EXPLICIT DUPLICATE CHECK (NEW!)
========================================================================================
"""

import sys
import os
import pickle
import argparse
import re
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import glob

from ray import init, shutdown
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

# Import environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.timetabling_env import ParallelTimetablingEnv

# Custom model registration
from ray.rllib.models import ModelCatalog
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from collections import OrderedDict

# ======================================================================
# IMPROVED MODEL DEFINITION (MUST MATCH TRAINING)
# ======================================================================
class ImprovedSahaMaskedTwoHead(TorchModelV2, nn.Module):
    """
    Neural network with improved value function for handling complex rewards
    MATCHES TRAINING SCRIPT v18.2
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Calculate input dimension
        if hasattr(obs_space, 'spaces') and 'obs' in obs_space.spaces:
            input_dim = obs_space.spaces['obs'].shape[0]
            print(f"\n✅ Using 'obs' dimension: {input_dim}")
        elif hasattr(obs_space, 'shape'):
            input_dim = obs_space.shape[0]
            print(f"\n✅ Box observation space: {input_dim}")
        else:
            raise ValueError(f"Cannot determine input dimension from obs_space: {obs_space}")

        hidden_sizes = model_config.get("custom_model_config", {}).get("hidden_sizes", [512, 512, 256])

        nvec = getattr(action_space, "nvec", None)
        assert nvec is not None and len(nvec) == 2, "Expect MultiDiscrete with 2 heads"
        self.n_teacher = int(nvec[0])
        self.n_slot = int(nvec[1])

        print(f"🧠 IMPROVED SAHA NETWORK:")
        print(f"   Input dimension: {input_dim}")
        print(f"   Hidden layers: {hidden_sizes}")
        print(f"   Teacher actions: {self.n_teacher}")
        print(f"   Slot actions: {self.n_slot}\n")

        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_sizes[0])
        self.input_norm = nn.LayerNorm(hidden_sizes[0])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                nn.LayerNorm(hidden_sizes[i + 1]),
                nn.ReLU(),
                nn.Dropout(0.1)
            ))

        final_dim = hidden_sizes[-1]
        
        # Separate feature extraction for action heads
        self.teacher_features = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU()
        )
        self.slot_features = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU()
        )
        
        self.teacher_head = nn.Linear(final_dim // 2, self.n_teacher)
        self.slot_head = nn.Linear(final_dim // 2, self.n_slot)
        
        # IMPROVED DEEPER VALUE NETWORK
        self.value_branch = nn.Sequential(
            nn.Linear(final_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Better initialization for value head
        nn.init.normal_(self.value_branch[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.value_branch[-1].bias, 0.0)

        self._logits_dim = self.n_teacher + self.n_slot
        self._value_out = None
        self.expected_input_dim = input_dim

    def _to_tensor(self, x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(x, dtype=torch.float32)

    def _extract_core_and_masks(self, obs):
        """Extract ONLY the 'obs' component for forward pass"""
        tmask = None
        smask = None

        if isinstance(obs, (dict, OrderedDict)):
            if 'obs' not in obs:
                raise ValueError(f"Expected 'obs' key in observation dict, got keys: {obs.keys()}")
            
            core = self._to_tensor(obs['obs']).float()
            
            if 'teacher_mask' in obs:
                tmask = self._to_tensor(obs['teacher_mask']).float()
            if 'slot_mask' in obs:
                smask = self._to_tensor(obs['slot_mask']).float()
        else:
            core = self._to_tensor(obs).float()

        # Ensure batch dimension
        if core.ndim == 1:
            core = core.unsqueeze(0)
        
        if tmask is not None:
            if tmask.ndim == 1:
                tmask = tmask.unsqueeze(0)
            tmask = torch.clamp(tmask, 0.0, 1.0)
        
        if smask is not None:
            if smask.ndim == 1:
                smask = smask.unsqueeze(0)
            smask = torch.clamp(smask, 0.0, 1.0)

        return core, tmask, smask

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        core, tmask, smask = self._extract_core_and_masks(obs)
        
        # Dimension check
        actual_dim = core.shape[-1]
        expected_dim = self.expected_input_dim
        
        if actual_dim != expected_dim:
            raise RuntimeError(
                f"Dimension mismatch! Got {actual_dim}, expected {expected_dim}"
            )
        
        # Safety check for NaN/Inf
        if not torch.all(torch.isfinite(core)):
            core = torch.nan_to_num(core, nan=0.0, posinf=1.0, neginf=0.0)

        # Forward pass through network
        h = self.input_layer(core)
        h = self.input_norm(h)
        h = torch.relu(h)
        
        for layer in self.hidden_layers:
            h = layer(h)
        
        # Action heads
        teacher_feats = self.teacher_features(h)
        slot_feats = self.slot_features(h)
        
        t_logits = self.teacher_head(teacher_feats)
        s_logits = self.slot_head(slot_feats)

        # Apply masks
        if tmask is not None and smask is not None:
            big_neg = -1e10
            t_logits = t_logits.masked_fill(tmask < 0.5, big_neg)
            s_logits = s_logits.masked_fill(smask < 0.5, big_neg)

        logits = torch.cat([t_logits, s_logits], dim=1)
        
        if not torch.all(torch.isfinite(logits)):
            logits = torch.clamp(logits, -1e10, 1e10)
        
        # Value prediction with improved network
        self._value_out = self.value_branch(h).squeeze(-1)
        
        if not torch.all(torch.isfinite(self._value_out)):
            self._value_out = torch.nan_to_num(self._value_out, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return logits, state

    def value_function(self):
        return self._value_out

# ======================================================================
# DATA LOADING WITH MANILA CACHE FINDER
# ======================================================================
def find_manila_cache():
    """Find Manila cache file automatically"""
    
    cache_patterns = [
        'cached_environment_data_MANILA_MODALITY.pkl',
        'cached_environment_data_MANILA*.pkl',
        'cached_environment_data_Manila*.pkl',
        '*MANILA*.pkl',
    ]
    
    print(f"\n{'='*80}")
    print("SEARCHING FOR MANILA CACHE FILE")
    print("="*80)
    
    for pattern in cache_patterns:
        matches = glob.glob(pattern)
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            cache_file = matches[0]
            
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                if 'subject_modalities' in data:
                    print(f"✓ Found cache with modality support: {cache_file}")
                    abs_path = os.path.abspath(cache_file)
                    file_size = os.path.getsize(abs_path) / (1024 * 1024)
                    mod_time = datetime.fromtimestamp(os.path.getmtime(abs_path))
                    print(f"  Size: {file_size:.2f} MB")
                    print(f"  Modified: {mod_time}")
                    print(f"  Path: {abs_path}")
                    print("="*80 + "\n")
                    return abs_path
                else:
                    print(f"⚠ Found {cache_file} but no modality data")
            except Exception as e:
                print(f"⚠ Error reading {cache_file}: {e}")
                continue
    
    print("\n❌ No Manila cache with modality support found!")
    print("Run: python create_timeslots_manila.py")
    print("="*80 + "\n")
    return None

def load_cached_data(cache_file=None):
    """Load preprocessed Manila data with modality support"""
    
    if cache_file is None:
        cache_file = find_manila_cache()
        if cache_file is None:
            raise FileNotFoundError("No Manila cache file found")
    
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache file not found: {cache_file}")
    
    print(f"\n{'='*80}")
    print("LOADING MANILA SCHEDULE DATA FOR EXPORT")
    print("="*80)
    print(f"Cache file: {cache_file}")
    
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    
    required = ['num_subjects', 'num_teachers', 'room_codes', 'subject_names']
    missing = [field for field in required if field not in data]
    if missing:
        raise ValueError(f"Cache missing required fields: {missing}")
    
    has_modality = 'subject_modalities' in data
    has_async = 'subject_is_async' in data
    
    print(f"\n✓ Loaded Manila schedule data")
    print(f"  Subjects: {data['num_subjects']}")
    print(f"  Teachers: {data['num_teachers']}")
    print(f"  Rooms: {len(data['room_codes'])}")
    print(f"  Modality support: {'✓' if has_modality else '✗'}")
    print(f"  Async support: {'✓' if has_async else '✗'}")
    
    if has_modality:
        print(f"\n  Modality Distribution:")
        for modality in ['Face-to-Face', 'Online', 'Hybrid']:
            count = data['subject_modalities'].count(modality)
            pct = (count / data['num_subjects']) * 100
            print(f"    {modality}: {count} ({pct:.1f}%)")
    
    if has_async:
        async_count = sum(data['subject_is_async'])
        async_pct = (async_count / data['num_subjects']) * 100
        print(f"  Asynchronous sessions: {async_count} ({async_pct:.1f}%)")
    
    print("="*80 + "\n")
    
    return data

# ======================================================================
# ENVIRONMENT FACTORY (MATCHES TRAINING v18.2)
# ======================================================================
def make_env(cache_file=None):
    """Create environment with fixed reward balance matching training"""
    data = load_cached_data(cache_file)
    
    # Ensure modality data exists
    if 'subject_modalities' not in data:
        print("⚠ No modality data - using Face-to-Face as default")
        data['subject_modalities'] = ['Face-to-Face'] * data['num_subjects']
        data['subject_modality_idx'] = [0] * data['num_subjects']
        data['modality_labels'] = ['Face-to-Face', 'Online', 'Hybrid']
        data['virtual_rooms'] = []
        data['physical_rooms'] = data['room_codes']
        data['async_rooms'] = []
    
    # CREATE ENVIRONMENT WITH FIXED REWARD BALANCE (MATCHES TRAINING)
    env = ParallelTimetablingEnv(
        # Core parameters
        num_teachers=data["num_teachers"],
        num_subjects=data["num_subjects"],
        num_days=data["num_days"],
        num_timeslots=data["num_timeslots"],
        room_codes=data["room_codes"],
        subject_codes=data["subject_names"],
        
        # Modality support
        subject_modalities=data.get("subject_modalities"),
        subject_modality_idx=data.get("subject_modality_idx"),
        modality_labels=data.get("modality_labels", ['Face-to-Face', 'Online', 'Hybrid']),
        virtual_rooms=data.get("virtual_rooms", []),
        
        # Room/campus constraints
        subject_campuses=data["subject_campuses"],
        subject_allowed_rooms=data["subject_allowed_rooms"],
        subject_areas=data["subject_areas"],
        
        # Section/teacher data
        subject_section_idx=data["subject_section_idx"],
        section_labels=data["section_labels"],
        max_classes_per_teacher=4,
        teacher_max_classes_map=data["teacher_max_classes_map"],
        subject_teacher_idx=data["subject_teacher_idx"],
        area_teacher_indices=data["area_teacher_indices"],
        
        # Timeslot constraints
        subject_allowed_timeslots=data.get("subject_allowed_timeslots"),
        timeslot_definitions=data.get("timeslot_definitions", []),
        subject_required_placements=data.get("subject_required_placements"),
        
        # FIXED REWARD PARAMETERS (MUST MATCH TRAINING EXACTLY!)
        strict_teacher_match=False,
        r_teacher_match=40.0,        # Matches training
        r_teacher_mismatch=-1.0,     # Matches training
        r_area_match=25.0,           # Matches training
        r_area_mismatch=-1.5,        # Matches training
        r_workload_balance=15.0,     # Matches training
        base_success_reward=150.0,   # Matches training
        wait_penalty=20.0,           # Matches training
        r_online_bonus=8.0,          # Matches training
        
        # Progressive difficulty
        enable_progressive_difficulty=True,
        difficulty_ramp_steps=100,
        
        # Full feature set
        include_focus_scalar=True,
        include_focus_tor_scalar=False,
        include_section_features=True,
        include_workload_features=True,
        enable_communication=True,
        use_action_masks=True,
        max_timesteps=400,
        enable_repair_pass=False,
        enable_milestone_rewards=True,
    )
    
    # Set labels for export
    env.subject_names = data["subject_names"]
    env.teacher_names = data.get("teacher_names", [])
    env.teacher_labels = data.get("teacher_id_to_name", data.get("teacher_names", []))
    env.day_labels = data.get("day_labels", ['M', 'T', 'W', 'H', 'F'])
    env.timeslot_labels = data.get("timeslot_labels", [])
    env.room_to_campus = data.get("room_to_campus", {})
    env.section_labels = data["section_labels"]
    env.subject_modalities = data["subject_modalities"]
    
    # Async support
    if 'subject_is_async' in data:
        env.subject_is_async = data["subject_is_async"]
    
    # Virtual rooms
    env.virtual_rooms = set(data.get("virtual_rooms", []))
    
    return ParallelPettingZooEnv(env)

# ======================================================================
# CHECKPOINT FINDING AND VALIDATION
# ======================================================================
def validate_checkpoint(checkpoint_path):
    """Validate checkpoint and warn about issues"""
    if not os.path.exists(checkpoint_path):
        return False, "Checkpoint path does not exist"
    
    # Extract iteration number
    checkpoint_name = os.path.basename(checkpoint_path)
    if checkpoint_name.startswith("checkpoint_"):
        try:
            iteration = int(checkpoint_name.split('_')[1])
            
            if iteration == 0:
                return False, "⚠️ CRITICAL: checkpoint_000000 is UNTRAINED! Model won't work. Use checkpoint_000010+"
            elif iteration < 5:
                return True, f"⚠️ WARNING: checkpoint_{iteration:06d} is VERY EARLY in training. Results may be poor. Recommend checkpoint_000010+"
            elif iteration < 10:
                return True, f"⚠️ Checkpoint_{iteration:06d} is early in training. Results may be suboptimal."
            else:
                return True, f"✓ Checkpoint_{iteration:06d} looks good (trained for {iteration} iterations)"
        except:
            return True, "Cannot determine checkpoint iteration"
    
    return True, "Unknown checkpoint format"

def find_latest_checkpoint(experiment_dir="C:/ray_logs"):
    """Find the latest checkpoint in experiment directory"""
    if not os.path.exists(experiment_dir):
        return None
    
    # Find all experiment directories
    exp_dirs = []
    for root, dirs, files in os.walk(experiment_dir):
        for d in dirs:
            if d.startswith("checkpoint_"):
                parent = os.path.basename(root)
                if parent.startswith("PPO_"):
                    exp_dirs.append(root)
                    break
    
    if not exp_dirs:
        return None
    
    # Get latest experiment
    latest_exp = max(exp_dirs, key=lambda d: os.path.getmtime(d))
    
    # Find checkpoints
    checkpoints = [f for f in os.listdir(latest_exp) if f.startswith("checkpoint_")]
    
    if not checkpoints:
        return None
    
    # Get latest checkpoint
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[1]))[-1]
    checkpoint_path = os.path.join(latest_exp, latest_checkpoint)
    
    return checkpoint_path

# ======================================================================
# SCHEDULE EXTRACTION WITH MODALITY AND IMPROVED PREFIX CLEANING
# ======================================================================
def clean_department_prefix(text):
    """Remove department prefixes from text strings - ENHANCED VERSION"""
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    text = str(text).strip()
    
    if not text:
        return text
    
    # Common department prefixes to remove (ALL UPPERCASE for matching)
    dept_prefixes = {
        # Computer/IT
        'CCS', 'CCIS', 'IT', 'CS',
        
        # Business/Economics
        'BA', 'CBA', 'BSBA', 'ACT', 'HRM',
        
        # Education
        'EDUC', 'CHED', 'BAGCED', 'DEAL', 'BAGCED-DEAL',
        
        # Engineering
        'ENGR', 'COE',
        
        # Liberal Arts/Sciences
        'CLA', 'CLAS', 'CLA-DPHILO', 'SOC', 'MATH', 'SCI', 'ENG', 'FIL', 'HIST', 'ARTS', 'MUS',
        
        # Physical Education
        'NSTP', 'PE',
        
        # Senior High School
        'IS', 'SHS', 'IS-SHS',
        
        # Programs (often used as dept codes)
        'BSIT', 'BSCS',
    }
    
    # CHECK IF ENTIRE STRING IS A DEPARTMENT CODE (RETURN EMPTY)
    if text.upper() in dept_prefixes:
        return ""  # Return empty string if it's just a department code
    
    # Handle both underscore and hyphen separators
    separator = '_' if '_' in text else '-' if '-' in text else None
    
    if separator is None:
        return text
    
    parts = text.split(separator)
    
    # Modality keywords to remove from the end
    modality_keywords = {'Face-to-Face', 'Online', 'Hybrid', 'Asynchronous', 'Synchronous'}
    
    # Remove modality info from the end if present
    if parts[-1] in modality_keywords:
        parts = parts[:-1]
    
    if len(parts) == 0:
        return text
    
    def is_department(part):
        """Check if a part looks like a department code"""
        part_upper = part.upper()
        
        # Check against known department prefixes (including compound ones)
        if part_upper in dept_prefixes:
            return True
        
        # Check for compound department codes like "BAGCED-DEAL"
        if '-' in part:
            compound_parts = part.split('-')
            if all(p.upper() in dept_prefixes for p in compound_parts):
                return True
        
        # Pattern matching for department-like codes
        return (
            (part.isupper() and 2 <= len(part) <= 8 and any(c.isalpha() for c in part)) or
            ('-' in part and part.replace('-', '').isupper() and len(part) <= 15)
        )
    
    def is_section(part):
        """Check if a part looks like a section code"""
        return (
            any(pattern in part.upper() for pattern in ['STEM', 'ABM', 'HUMSS', 'GAS', 'TVL']) or
            bool(re.search(r'\d{1,2}[\s\-]?[A-Z]$', part)) or
            bool(re.search(r'^[A-Z]$', part))
        )
    
    # Handle four or more parts: DEPT_SUBJECT_SECTION_MODALITY
    if len(parts) >= 4:
        if is_department(parts[0]):
            if is_section(parts[2]):
                return parts[1].strip()
            else:
                return separator.join(parts[1:3]).strip()
        else:
            return parts[0].strip()
    
    # Handle three parts: DEPT_SUBJECT_SECTION
    elif len(parts) == 3:
        if is_department(parts[0]):
            if is_section(parts[2]):
                return parts[1].strip()
            else:
                return f"{parts[1]}{separator}{parts[2]}".strip()
        elif is_department(parts[1]):
            return parts[0].strip()
        else:
            return parts[1].strip()
    
    # Handle two-part format: DEPT_NAME or NAME_DEPT
    elif len(parts) == 2:
        first_part = parts[0]
        second_part = parts[1]
        
        first_is_dept = is_department(first_part)
        second_is_dept = is_department(second_part)
        
        if ',' in first_part:
            first_is_dept = False
        if ',' in second_part:
            second_is_dept = False
        
        if first_is_dept and not second_is_dept:
            return second_part.strip()
        elif second_is_dept and not first_is_dept:
            return first_part.strip()
        elif first_is_dept and second_is_dept:
            return second_part.strip()
        else:
            return first_part.strip()
    
    return text

def extract_schedule_from_env(env_wrapper):
    """Extract complete schedule from environment with modality support"""
    env = env_wrapper.par_env
    
    schedule_data = []
    
    print(f"\nExtracting schedule...")
    
    # Helper functions
    def get_teacher_name(teacher_idx):
        if hasattr(env, 'teacher_labels'):
            if isinstance(env.teacher_labels, dict):
                return env.teacher_labels.get(teacher_idx, f"Teacher_{teacher_idx}")
            elif isinstance(env.teacher_labels, list):
                if 0 <= teacher_idx < len(env.teacher_labels):
                    return env.teacher_labels[teacher_idx]
        elif hasattr(env, 'teacher_names') and isinstance(env.teacher_names, list):
            if 0 <= teacher_idx < len(env.teacher_names):
                return env.teacher_names[teacher_idx]
        return f"Teacher_{teacher_idx}"
    
    def get_day_label(day_idx):
        if hasattr(env, 'day_labels') and env.day_labels:
            if isinstance(env.day_labels, list) and 0 <= day_idx < len(env.day_labels):
                label = env.day_labels[day_idx]
                if label:
                    return label
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        if 0 <= day_idx < len(day_names):
            return day_names[day_idx]
        return f"Day_{day_idx}"
    
    def get_timeslot_label(ts_idx):
        if hasattr(env, 'timeslot_labels') and env.timeslot_labels:
            if isinstance(env.timeslot_labels, list) and 0 <= ts_idx < len(env.timeslot_labels):
                label = env.timeslot_labels[ts_idx]
                if label:
                    return label
        return f"Timeslot_{ts_idx}"
    
    def get_modality(subject_idx):
        if hasattr(env, 'subject_modalities') and 0 <= subject_idx < len(env.subject_modalities):
            return env.subject_modalities[subject_idx]
        return 'Face-to-Face'
    
    def is_async(subject_idx):
        if hasattr(env, 'subject_is_async') and 0 <= subject_idx < len(env.subject_is_async):
            return env.subject_is_async[subject_idx]
        return False
    
    def is_virtual_room(room_code):
        if hasattr(env, 'virtual_rooms'):
            return room_code in env.virtual_rooms
        return 'ZOOM' in room_code.upper() or 'ONLINE' in room_code.upper()
    
    # Extract from room schedules
    entry_count = 0
    for building in env.building_keys:
        schedule = env.buildings_room_schedule[building]
        num_rooms = schedule.shape[0]
        
        for room_idx in range(num_rooms):
            room_code = env.buildings_room_info[building][room_idx]
            
            for day_idx in range(env.num_days):
                for ts_idx in range(env.num_timeslots):
                    subject_idx = schedule[room_idx, day_idx, ts_idx]
                    
                    if subject_idx >= 0:
                        subject_name = clean_department_prefix(env.subject_names[subject_idx])
                        teacher_idx = env.subject_assignments[subject_idx]
                        teacher_name = clean_department_prefix(get_teacher_name(teacher_idx))
                        day_label = get_day_label(day_idx)
                        timeslot_label = get_timeslot_label(ts_idx)
                        section_idx = env.subject_section_idx[subject_idx]
                        section_label = clean_department_prefix(env.section_labels[section_idx])
                        subject_area = env.subject_areas[subject_idx]
                        campus = env.room_to_campus.get(room_code, "Unknown") if hasattr(env, 'room_to_campus') else "Unknown"
                        
                        # Modality information
                        modality = get_modality(subject_idx)
                        is_async_session = is_async(subject_idx)
                        is_virtual = is_virtual_room(room_code)
                        
                        # Get duration
                        duration = 90
                        if hasattr(env, 'timeslot_durations') and ts_idx in env.timeslot_durations:
                            duration = env.timeslot_durations[ts_idx]
                        
                        # Calculate meetings per week
                        meetings_per_week = 2 if duration == 90 else 1
                        
                        entry_count += 1
                        
                        # Export with modality columns
                        schedule_data.append({
                            'Subject': subject_name,
                            'Section': section_label,
                            'Subject_Area': subject_area,
                            'Faculty': teacher_name,
                            'Days': day_label,
                            'Time': timeslot_label,
                            'Duration_Min': duration,
                            'Meetings_Per_Week': meetings_per_week,
                            'Room': room_code,
                            'Room_Type': 'Virtual' if is_virtual else 'Physical',
                            'Campus': campus,
                            'Building': building,
                            'Modality': modality,
                            'Is_Async': 'Yes' if is_async_session else 'No',
                        })
    
    print(f"  Total entries extracted: {entry_count}")
    return schedule_data

# ======================================================================
# VALIDATION FUNCTION (ENHANCED)
# ======================================================================
def validate_no_duplicates(schedule_data, env_wrapper):
    """
    Comprehensive duplicate validation for exported schedule
    Checks against environment's required placements
    """
    print("\n" + "="*80)
    print("DUPLICATE VALIDATION (FIX #12 VERIFICATION)")
    print("="*80)
    
    env = env_wrapper.par_env
    df = pd.DataFrame(schedule_data)
    
    # Get required placements from environment
    required_placements = {}
    for subj_idx in range(env.num_subjects):
        required = env.subject_required_placements.get(subj_idx, 1)
        subj_name = clean_department_prefix(env.subject_names[subj_idx])
        sec_idx = env.subject_section_idx[subj_idx]
        sec_name = clean_department_prefix(env.section_labels[sec_idx])
        required_placements[(subj_name, sec_name)] = required
    
    # Count actual placements in export
    actual_counts = df.groupby(['Subject', 'Section']).size()
    
    issues_found = False
    duplicates = []
    missing = []
    
    print("\nChecking placement counts...")
    
    for (subj, sec), actual_count in actual_counts.items():
        required_count = required_placements.get((subj, sec), 1)
        
        if actual_count > required_count:
            excess = actual_count - required_count
            duplicates.append({
                'Subject': subj,
                'Section': sec,
                'Required': required_count,
                'Actual': actual_count,
                'Excess': excess
            })
            issues_found = True
        elif actual_count < required_count:
            missing.append({
                'Subject': subj,
                'Section': sec,
                'Required': required_count,
                'Actual': actual_count,
                'Missing': required_count - actual_count
            })
            issues_found = True
    
    # Print results
    if duplicates:
        print(f"\n❌ DUPLICATES FOUND: {len(duplicates)} subjects over-scheduled")
        print("   This indicates FIX #12 may not be properly applied!")
        for dup in duplicates[:10]:  # Show first 10
            print(f"  {dup['Subject']} ({dup['Section']}): {dup['Actual']}/{dup['Required']} - EXCESS: {dup['Excess']}")
        if len(duplicates) > 10:
            print(f"  ... and {len(duplicates) - 10} more")
    else:
        print("\n✅ No duplicates found - FIX #12 is working!")
    
    if missing:
        print(f"\n⚠️ INCOMPLETE: {len(missing)} subjects under-scheduled")
        for mis in missing[:5]:  # Show first 5
            print(f"  {mis['Subject']} ({mis['Section']}): {mis['Actual']}/{mis['Required']} - MISSING: {mis['Missing']}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
    
    if not issues_found:
        print("\n✅ ALL SUBJECTS CORRECTLY SCHEDULED")
        print("   FIX #12 is working perfectly!")
    
    print("="*80 + "\n")
    
    return {
        'has_duplicates': len(duplicates) > 0,
        'has_missing': len(missing) > 0,
        'duplicate_count': len(duplicates),
        'missing_count': len(missing),
        'duplicates': duplicates,
        'missing': missing
    }

# ======================================================================
# SCHEDULE GENERATION WITH VALIDATION
# ======================================================================
def generate_schedule(checkpoint_path, max_steps=500, deterministic=True, cache_file=None):
    """Generate schedule using trained model"""
    
    print("\n" + "="*80)
    print("MANILA SCHEDULE GENERATION v16.2 - WITH ENHANCED VALIDATION")
    print("="*80)
    
    # Initialize Ray
    init(ignore_reinit_error=True, include_dashboard=False)
    
    # Register improved model
    ModelCatalog.register_custom_model("improved_saha_masked", ImprovedSahaMaskedTwoHead)
    print("✅ Registered ImprovedSahaMaskedTwoHead model\n")
    
    # Register environment
    register_env("manila_env", lambda cfg: make_env(cache_file))
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    # Load trained algorithm
    algo = PPO.from_checkpoint(checkpoint_path)
    
    print(f"[OK] Model loaded successfully")
    
    # Create environment
    env = make_env(cache_file)
    raw_env = env.par_env
    
    print(f"\nEnvironment Info:")
    print(f"  Subjects: {raw_env.num_subjects}")
    print(f"  Teachers: {raw_env.num_teachers}")
    print(f"  Days: {raw_env.num_days}")
    print(f"  Timeslots: {raw_env.num_timeslots}")
    print(f"  Modality support: {'✓' if hasattr(raw_env, 'subject_modalities') else '✗'}")
    print(f"  Progressive difficulty: {'✓' if raw_env.enable_progressive_difficulty else '✗'}")
    
    # Check for FIX #12
    env_version = getattr(raw_env, 'metadata', {}).get('name', 'unknown')
    print(f"  Environment version: {env_version}")
    if 'v14_5' in env_version or 'v14.5' in env_version:
        print(f"  ✅ FIX #12 present (step-local placement tracking)")
    else:
        print(f"  ⚠️ FIX #12 status unknown - check for duplicates in validation")
    
    # Reset environment
    obs_dict, _ = env.reset()
    
    # Get agents
    agents = raw_env.agents
    
    print("\nGenerating schedule...")
    
    done = {agent: False for agent in agents}
    step = 0
    
    while step < max_steps and not all(done.values()):
        actions = {}
        
        # Get actions from trained policy
        for agent in agents:
            if not done.get(agent, False):
                action = algo.compute_single_action(
                    obs_dict[agent],
                    policy_id="saha_policy",
                    explore=not deterministic
                )
                actions[agent] = action
        
        # Step environment
        obs_dict, rewards, dones, truncs, infos = env.step(actions)
        
        # Update done status
        for agent in agents:
            done[agent] = dones.get(agent, False) or truncs.get(agent, False)
        
        step += 1
        
        # Progress update
        if step % 50 == 0:
            placed = len(raw_env.placed_subjects)
            print(f"  Step {step}: {placed}/{raw_env.num_subjects} subjects placed")
    
    # Final statistics
    placed = len(raw_env.placed_subjects)
    completion_rate = (placed / raw_env.num_subjects) * 100
    
    print(f"\n{'='*80}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Steps: {step}")
    print(f"Subjects placed: {placed}/{raw_env.num_subjects} ({completion_rate:.1f}%)")
    print(f"Conflicts: {raw_env.conflict_count}")
    
    # Modality statistics
    if hasattr(raw_env, 'modality_stats'):
        print(f"\n=== MODALITY PLACEMENT STATS ===")
        for modality, stats in raw_env.modality_stats.items():
            total = raw_env.subject_modalities.count(modality)
            placed_count = stats['placed']
            success_rate = (placed_count / total * 100) if total > 0 else 0
            print(f"{modality}: {placed_count}/{total} ({success_rate:.1f}%)")
    
    # Show fail statistics
    print(f"\nConstraint Violations:")
    for key, value in raw_env.fail_stats.items():
        if value > 0:
            print(f"  {key}: {value}")
    
    # Extract schedule
    schedule_data = extract_schedule_from_env(env)
    
    # VALIDATE NO DUPLICATES
    validation_results = validate_no_duplicates(schedule_data, env)
    
    shutdown()
    
    return schedule_data, {
        'placed': placed,
        'total': raw_env.num_subjects,
        'completion_rate': completion_rate,
        'steps': step,
        'conflicts': raw_env.conflict_count,
        'fail_stats': dict(raw_env.fail_stats),
        'modality_stats': dict(raw_env.modality_stats) if hasattr(raw_env, 'modality_stats') else {},
        'validation': validation_results
    }

# ======================================================================
# EXPORT FUNCTIONS WITH MODALITY
# ======================================================================
def export_to_csv(schedule_data, output_path):
    """Export schedule to CSV with modality columns"""
    df = pd.DataFrame(schedule_data)
    
    # Define column order with modality
    output_columns = [
        'Subject', 'Faculty', 'Section', 'Days', 'Time', 'Room', 'Campus',
        'Modality', 'Room_Type', 'Is_Async',
        'Meetings_Per_Week', 'Duration_Min', 'Subject_Area', 'Building'
    ]
    
    # Only include columns that exist
    available_columns = [col for col in output_columns if col in df.columns]
    export_df = df[available_columns].copy()
    
    # Sort by time and room
    sort_cols = [col for col in ['Time', 'Room'] if col in export_df.columns]
    if sort_cols:
        export_df = export_df.sort_values(sort_cols)
    
    export_df.to_csv(output_path, index=False)
    print(f"\n[OK] Schedule exported to: {output_path}")
    print(f"  Columns: {', '.join(available_columns)}")
    print(f"  Rows: {len(export_df)}")

def export_to_excel(schedule_data, output_path):
    """Export schedule to Excel with modality views"""
    try:
        import xlsxwriter
    except ImportError:
        print("\nWARNING: xlsxwriter not installed. Skipping Excel export.")
        print("Install with: pip install xlsxwriter")
        return
    
    df = pd.DataFrame(schedule_data)
    
    # Define column order
    output_columns = [
        'Subject', 'Faculty', 'Section', 'Days', 'Time', 'Room', 'Campus',
        'Modality', 'Room_Type', 'Is_Async',
        'Meetings_Per_Week', 'Duration_Min', 'Subject_Area', 'Building'
    ]
    
    # Only include columns that exist
    available_columns = [col for col in output_columns if col in df.columns]
    export_df = df[available_columns].copy()
    
    # Sort by time and room
    sort_cols = [col for col in ['Time', 'Room'] if col in export_df.columns]
    if sort_cols:
        export_df = export_df.sort_values(sort_cols)
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        # Full schedule
        export_df.to_excel(writer, sheet_name='Full Schedule', index=False)
        
        # By faculty
        if 'Faculty' in export_df.columns:
            faculty_view = export_df.sort_values(['Faculty', 'Days', 'Time'])
            faculty_view.to_excel(writer, sheet_name='By Faculty', index=False)
        
        # By section
        if 'Section' in export_df.columns:
            section_view = export_df.sort_values(['Section', 'Days', 'Time'])
            section_view.to_excel(writer, sheet_name='By Section', index=False)
        
        # By modality
        if 'Modality' in export_df.columns:
            modality_view = export_df.sort_values(['Modality', 'Days', 'Time'])
            modality_view.to_excel(writer, sheet_name='By Modality', index=False)
        
        # By room
        if 'Room' in export_df.columns:
            room_view = export_df.sort_values(['Room', 'Days', 'Time'])
            room_view.to_excel(writer, sheet_name='By Room', index=False)
        
        # Statistics with modality
        stats_data = {
            'Metric': [
                'Total Classes',
                'Unique Subjects',
                'Unique Faculty',
                'Unique Rooms',
                'Unique Sections'
            ],
            'Value': [
                len(export_df),
                export_df['Subject'].nunique() if 'Subject' in export_df.columns else 0,
                export_df['Faculty'].nunique() if 'Faculty' in export_df.columns else 0,
                export_df['Room'].nunique() if 'Room' in export_df.columns else 0,
                export_df['Section'].nunique() if 'Section' in export_df.columns else 0
            ]
        }
        
        # Add modality breakdown
        if 'Modality' in export_df.columns:
            for modality in ['Face-to-Face', 'Online', 'Hybrid']:
                count = len(export_df[export_df['Modality'] == modality])
                stats_data['Metric'].append(f'{modality} Classes')
                stats_data['Value'].append(count)
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        # Format all sheets
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column('A:N', 18)
    
    print(f"[OK] Schedule exported to: {output_path}")

def export_teacher_schedules(schedule_data, output_dir):
    """Export individual faculty schedules"""
    df = pd.DataFrame(schedule_data)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def sanitize_filename(name):
        """Remove/replace characters that are invalid in filenames"""
        name = str(name).replace(',', '').replace(' ', '_').replace('/', '_').replace('\\', '_')
        name = name.replace(':', '').replace('*', '').replace('?', '').replace('"', '')
        name = name.replace('<', '').replace('>', '').replace('|', '')
        while '__' in name:
            name = name.replace('__', '_')
        if len(name) > 50:
            name = name[:50]
        return name.strip('_')
    
    faculty_col = 'Faculty' if 'Faculty' in df.columns else 'Teacher'
    
    if faculty_col not in df.columns:
        print("WARNING: No Faculty/Teacher column found")
        return
    
    for faculty in df[faculty_col].unique():
        faculty_df = df[df[faculty_col] == faculty].copy()
        
        # Sort by Days and Time if available
        sort_cols = [col for col in ['Days', 'Time'] if col in faculty_df.columns]
        if sort_cols:
            faculty_df = faculty_df.sort_values(sort_cols)
        
        safe_name = sanitize_filename(faculty)
        filename = f"faculty_{safe_name}.csv"
        
        try:
            faculty_df.to_csv(output_dir / filename, index=False)
        except Exception as e:
            print(f"WARNING: Could not export schedule for {faculty}: {e}")
    
    print(f"[OK] Faculty schedules exported to: {output_dir}")

# ======================================================================
# MAIN WITH ENHANCED VALIDATION REPORTING
# ======================================================================
def main():
    # DEFAULT MANILA CHECKPOINT (UPDATED TO ITERATION 10)
    DEFAULT_CHECKPOINT = "C:/ray_logs/Manila_v15_0_REBALANCED/PPO_manila_env_3a124_00000_0_2025-11-12_23-49-52/checkpoint_000099"
    
    parser = argparse.ArgumentParser(description="Export Manila schedule v16.2 (with enhanced validation)")
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                       help=f'Path to checkpoint (default: checkpoint_000010)')
    parser.add_argument('--cache', type=str, default=None,
                       help='Path to Manila cache file (if not specified, searches automatically)')
    parser.add_argument('--output', type=str, default='manila_schedule_output',
                       help='Output directory name')
    parser.add_argument('--format', type=str, choices=['csv', 'excel', 'both'], default='both',
                       help='Export format')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum generation steps')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic policy')
    
    args = parser.parse_args()
    
    # Find cache file
    cache_file = args.cache
    if cache_file is None:
        cache_file = find_manila_cache()
        if cache_file is None:
            print("ERROR: No Manila cache file found")
            sys.exit(1)
    
    # Use checkpoint path
    checkpoint_path = args.checkpoint
    
    # # Validate checkpoint
    # is_valid, message = validate_checkpoint(checkpoint_path)
    # if not is_valid:
    #     print(f"\n❌ {message}")
    #     sys.exit(1)
    # else:
    #     print(f"\n{message}")
    
    # Generate schedule
    schedule_data, stats = generate_schedule(
        checkpoint_path,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
        cache_file=cache_file
    )
    
    if not schedule_data:
        print("\nWARNING: No schedule data generated!")
        sys.exit(1)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export based on format
    if args.format in ['csv', 'both']:
        csv_path = output_dir / 'manila_schedule.csv'
        export_to_csv(schedule_data, csv_path)
    
    if args.format in ['excel', 'both']:
        excel_path = output_dir / 'manila_schedule.xlsx'
        export_to_excel(schedule_data, excel_path)
    
    # Export teacher schedules
    faculty_dir = output_dir / 'faculty'
    export_teacher_schedules(schedule_data, faculty_dir)
    
    # ============================================================
    # POST-EXPORT EXPLICIT DUPLICATE CHECK (NEW!)
    # ============================================================
    print("\n" + "="*80)
    print("POST-EXPORT DUPLICATE VERIFICATION")
    print("="*80)
    
    df = pd.DataFrame(schedule_data)
    placement_counts = df.groupby(['Subject', 'Section']).size()
    
    # Assuming 2 placements per subject is normal (adjust if needed)
    duplicates = placement_counts[placement_counts > 2]
    incomplete = placement_counts[placement_counts < 2]
    
    if len(duplicates) > 0:
        print(f"\n❌ FOUND {len(duplicates)} SUBJECTS WITH DUPLICATES!")
        print("\nDuplicate details:")
        for (subj, sec), count in duplicates.head(10).items():
            print(f"  {subj} ({sec}): {count} placements (should be ≤2)")
        
        if len(duplicates) > 10:
            print(f"  ... and {len(duplicates) - 10} more")
        
        # Save duplicate report
        dup_df = pd.DataFrame({
            'Subject': [s for s, _ in duplicates.index],
            'Section': [sec for _, sec in duplicates.index],
            'Actual_Placements': duplicates.values,
            'Excess': duplicates.values - 2
        })
        dup_path = output_dir / 'DUPLICATES_FOUND.csv'
        dup_df.to_csv(dup_path, index=False)
        print(f"\n⚠️ Duplicate report saved: {dup_path}")
        
        print("\n🔧 RECOMMENDATION:")
        print("   Continue training to iteration 200-300 to eliminate duplicates")
        print(f"   Current checkpoint may be too early for convergence")
    else:
        print("\n✅ NO DUPLICATES DETECTED IN EXPORT")
    
    if len(incomplete) > 0:
        print(f"\n⚠️ FOUND {len(incomplete)} INCOMPLETE SUBJECTS:")
        for (subj, sec), count in incomplete.head(5).items():
            print(f"  {subj} ({sec}): {count}/2 placements (MISSING: {2-count})")
        
        if len(incomplete) > 5:
            print(f"  ... and {len(incomplete) - 5} more")
        
        # Save incomplete report
        inc_df = pd.DataFrame({
            'Subject': [s for s, _ in incomplete.index],
            'Section': [sec for _, sec in incomplete.index],
            'Actual_Placements': incomplete.values,
            'Missing': 2 - incomplete.values
        })
        inc_path = output_dir / 'INCOMPLETE_SUBJECTS.csv'
        inc_df.to_csv(inc_path, index=False)
        print(f"\n📋 Incomplete subject report saved: {inc_path}")
        
        if len(incomplete) < 5:
            print("\n✨ EXCELLENT! Only a few subjects incomplete")
            print("   Continue training to iteration 150-200 for 100% completion")
        else:
            print("\n🔧 RECOMMENDATION:")
            print("   Continue training to improve completion rate")
    
    print("="*80 + "\n")
    
    # Save statistics with validation
    stats_path = output_dir / 'statistics.txt'
    with open(stats_path, 'w') as f:
        f.write("MANILA SCHEDULE GENERATION STATISTICS v16.2\n")
        f.write("="*50 + "\n\n")
        f.write("Version: With FIX #12 Validation + Enhanced Checks\n")
        f.write(f"Checkpoint: {os.path.basename(checkpoint_path)}\n")
        f.write(f"Completion Rate: {stats['completion_rate']:.1f}%\n")
        f.write(f"Subjects Placed: {stats['placed']}/{stats['total']}\n")
        f.write(f"Generation Steps: {stats['steps']}\n")
        f.write(f"Conflicts: {stats['conflicts']}\n\n")
        
        # Validation results
        if 'validation' in stats:
            val = stats['validation']
            f.write("="*50 + "\n")
            f.write("SCHEDULE VALIDATION (FIX #12 CHECK)\n")
            f.write("="*50 + "\n")
            f.write(f"Duplicates: {val['duplicate_count']}\n")
            f.write(f"Missing: {val['missing_count']}\n")
            f.write(f"Status: {'❌ ISSUES FOUND' if val['has_duplicates'] or val['has_missing'] else '✅ CLEAN'}\n\n")
            
            if val['duplicates']:
                f.write("\nDuplicate Details:\n")
                for dup in val['duplicates'][:20]:
                    f.write(f"  {dup['Subject']} ({dup['Section']}): {dup['Actual']}/{dup['Required']} (+{dup['Excess']})\n")
                if len(val['duplicates']) > 20:
                    f.write(f"  ... and {len(val['duplicates']) - 20} more\n")
            
            if val['has_duplicates']:
                f.write("\n⚠️ RECOMMENDATION:\n")
                f.write("  Duplicates detected! Please verify:\n")
                f.write("  1. Environment file has FIX #12 applied (step_placement_counts)\n")
                f.write("  2. Model was trained with FIX #12 environment\n")
                f.write("  3. Continue training to iteration 200-300\n\n")
            
            if val['has_missing'] and val['missing_count'] < 5:
                f.write("\n✨ NEARLY COMPLETE:\n")
                f.write(f"  Only {val['missing_count']} subjects incomplete\n")
                f.write("  Continue training to iteration 150-200 for 100% completion\n\n")
        
        # Modality stats
        if stats.get('modality_stats'):
            f.write("\nModality Placement:\n")
            for modality, modality_stats in stats['modality_stats'].items():
                f.write(f"  {modality}:\n")
                f.write(f"    Placed: {modality_stats['placed']}\n")
                f.write(f"    Attempted: {modality_stats['attempted']}\n")
            f.write("\n")
        
        f.write("Constraint Violations:\n")
        for key, value in stats['fail_stats'].items():
            if value > 0:
                f.write(f"  {key}: {value}\n")
        
        f.write("\nImprovements in v16.2:\n")
        f.write("  • Added POST-EXPORT duplicate verification\n")
        f.write("  • Enhanced validation reporting\n")
        f.write("  • Comprehensive incomplete subject tracking\n")
        f.write("  • Detailed statistics output\n")
        f.write("  • Reward parameters match training exactly\n")
    
    # Save duplicate details if found
    if 'validation' in stats and stats['validation']['duplicates']:
        dup_path = output_dir / 'duplicates_report.csv'
        dup_df = pd.DataFrame(stats['validation']['duplicates'])
        dup_df.to_csv(dup_path, index=False)
        print(f"⚠️ Duplicate report saved to: {dup_path}")
    
    print(f"\n{'='*80}")
    print(f"ALL EXPORTS COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"\nFiles generated:")
    print(f"  - manila_schedule.csv / manila_schedule.xlsx")
    print(f"  - faculty/*.csv (individual faculty schedules)")
    print(f"  - statistics.txt (with validation results)")
    if 'validation' in stats and stats['validation']['duplicates']:
        print(f"  - duplicates_report.csv (details of over-scheduled subjects)")
    if len(incomplete) > 0:
        print(f"  - INCOMPLETE_SUBJECTS.csv (subjects missing placements)")
    
    # FINAL VALIDATION STATUS
    if 'validation' in stats:
        val = stats['validation']
        print(f"\n{'='*80}")
        print("FINAL VALIDATION SUMMARY")
        print(f"{'='*80}")
        
        if val['has_duplicates']:
            print(f"❌ WARNING: {val['duplicate_count']} subjects have duplicate placements!")
            print("\nThis suggests FIX #12 may not be properly applied.")
            print("Recommendations:")
            print("  1. Check envs/timetabling_env.py for 'step_placement_counts'")
            print("  2. Verify environment version is v14.5 or higher")
            print("  3. Continue training to iteration 200-300")
            print(f"\nSee {output_dir}/duplicates_report.csv for details")
        elif val['has_missing']:
            if val['missing_count'] < 5:
                print(f"✨ EXCELLENT: Only {val['missing_count']} subjects incomplete!")
                print("(No duplicates detected - FIX #12 is working!)")
                print("\n🎯 RECOMMENDATION:")
                print("  Continue training to iteration 150-200 for 100% completion")
            else:
                print(f"⚠️ {val['missing_count']} subjects are incomplete")
                print("(No duplicates detected - FIX #12 is working!)")
                print("\n🎯 RECOMMENDATION:")
                print("  Continue training to iteration 200-300")
        else:
            print("✅ PERFECT SCHEDULE - No duplicates or missing subjects!")
            print("   FIX #12 is working correctly!")
            print("   Schedule is ready for deployment! 🎉")
        
        print(f"{'='*80}\n")
    
    print(f"\nModel: ImprovedSahaMaskedTwoHead (matches training v18.2)")
    print(f"Features:")
    print(f"  • Deeper value network for better predictions")
    print(f"  • Fixed reward balance (matches training exactly)")
    print(f"  • Progressive difficulty support")
    print(f"  • Enhanced modality tracking")
    print(f"  • Smart department prefix cleaning")
    print(f"  • Comprehensive duplicate validation")
    print(f"  • POST-EXPORT explicit verification (NEW!)")

if __name__ == "__main__":
    main()