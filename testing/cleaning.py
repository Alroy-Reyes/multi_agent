"""
Export Schedule from Trained SAHA Model
Generates a complete timetable using the trained PPO model

Features:
- Automatic department prefix removal (CCS_NAME -> NAME)
- Column names match standard format (Faculty, Days, Time, etc.)
- Exports to CSV and Excel with multiple views
- Individual faculty schedule files
"""

import sys
import os
import pickle
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

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
# MODEL DEFINITION (must match training)
# ======================================================================
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

# ======================================================================
# DATA LOADING
# ======================================================================
def load_cached_data():
    """Load preprocessed data with real timeslot constraints"""
    cache_file = 'cached_environment_data_with_real_timeslots.pkl'
    
    if not os.path.exists(cache_file):
        print(f"ERROR: {cache_file} not found.")
        cache_file = 'cached_environment_data_CLEAN.pkl'
        
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"No cache files found. Run preprocessing first.")
    
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    
    return data

# ======================================================================
# ENVIRONMENT FACTORY
# ======================================================================
def make_env():
    """Create environment with all constraints"""
    data = load_cached_data()
    
    env = ParallelTimetablingEnv(
        num_teachers=data["num_teachers"],
        num_subjects=data["num_subjects"],
        num_days=data["num_days"],
        num_timeslots=data["num_timeslots"],
        room_codes=data["room_codes"],
        subject_codes=data["subject_names"],
        subject_campuses=data["subject_campuses"],
        subject_areas=data["subject_areas"],
        subject_allowed_rooms=data["subject_allowed_rooms"],
        subject_section_idx=data["subject_section_idx"],
        section_labels=data["section_labels"],
        max_classes_per_teacher=4,
        teacher_max_classes_map=data["teacher_max_classes_map"],
        subject_teacher_idx=data["subject_teacher_idx"],
        area_teacher_indices=data["area_teacher_indices"],
        subject_allowed_timeslots=data.get("subject_allowed_timeslots"),
        timeslot_definitions=data.get("timeslot_definitions", []),
        strict_teacher_match=False,
        r_teacher_match=2.0,
        r_teacher_mismatch=-0.5,
        r_area_match=1.5,
        r_area_mismatch=-1.0,
        r_workload_balance=0.8,
        base_success_reward=0.05,
        include_focus_scalar=True,
        include_focus_tor_scalar=False,
        include_section_features=True,
        include_workload_features=True,
        enable_communication=True,
        use_action_masks=True,
        max_timesteps=400,
        enable_repair_pass=True,  # Enable repair for export
    )
    
    # Set labels for export
    env.subject_names = data["subject_names"]
    env.teacher_names = data.get("teacher_names", [])
    env.teacher_labels = data["teacher_id_to_name"]
    env.day_labels = data["day_labels"]
    env.timeslot_labels = data["timeslot_labels"]
    env.room_to_campus = data["room_to_campus"]
    env.area_labels = data["area_labels"]
    env.section_labels = data["section_labels"]
    env.subject_section_idx = data["subject_section_idx"]
    
    if 'timeslot_definitions' in data:
        env.timeslot_definitions = data["timeslot_definitions"]
    
    return ParallelPettingZooEnv(env)

# ======================================================================
# CHECKPOINT FINDING
# ======================================================================
def find_latest_checkpoint(experiment_dir="C:/ray_logs"):
    """Find the latest checkpoint in experiment directory"""
    if not os.path.exists(experiment_dir):
        return None
    
    # Find PPO experiment directories
    exp_dirs = [d for d in os.listdir(experiment_dir) 
                if d.startswith("PPO_") and os.path.isdir(os.path.join(experiment_dir, d))]
    
    if not exp_dirs:
        return None
    
    # Get latest experiment
    latest_exp = max(exp_dirs, key=lambda d: os.path.getmtime(os.path.join(experiment_dir, d)))
    exp_path = os.path.join(experiment_dir, latest_exp)
    
    # Find trial directory
    trial_dirs = [d for d in os.listdir(exp_path) 
                  if os.path.isdir(os.path.join(exp_path, d))]
    
    if not trial_dirs:
        return None
    
    trial_path = os.path.join(exp_path, trial_dirs[0])
    
    # Find checkpoints
    checkpoints = [f for f in os.listdir(trial_path) if f.startswith("checkpoint_")]
    
    if not checkpoints:
        return None
    
    # Get latest checkpoint
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[1]))[-1]
    checkpoint_path = os.path.join(trial_path, latest_checkpoint)
    
    return checkpoint_path

# ======================================================================
# SCHEDULE EXTRACTION
# ======================================================================
def clean_department_prefix(text):
    """
    Remove department prefixes from text strings
    Handles formats like:
    - 'DEPT_SUBJECT_SECTION' -> 'SUBJECT'
    - 'DEPT-CODE_NAME' -> 'NAME' 
    - 'NAME_DEPT-CODE' -> 'NAME'
    """
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    text = str(text).strip()
    
    # Handle three-part format: DEPT_SUBJECT_SECTION
    if text.count('_') == 2:
        parts = text.split('_')
        return parts[1].strip()
    
    # Handle two-part format: DEPT_NAME or NAME_DEPT
    elif text.count('_') == 1:
        parts = text.split('_')
        first_part = parts[0]
        second_part = parts[1]
        
        # Check if first part is a department code
        first_is_dept = (
            ('-' in first_part and first_part[0].isupper()) or
            (first_part.isupper() and 2 <= len(first_part) <= 6 and first_part.isalpha())
        )
        
        # Check if second part is a department code
        second_is_dept = (
            ('-' in second_part and second_part[0].isupper()) or
            (second_part.isupper() and 2 <= len(second_part) <= 6 and second_part.isalpha())
        )
        
        # Additional check: if has comma, it's NOT a dept
        if ',' in first_part:
            first_is_dept = False
        if ',' in second_part:
            second_is_dept = False
        
        # Decide which to keep
        if first_is_dept and not second_is_dept:
            return second_part.strip()
        elif second_is_dept and not first_is_dept:
            return first_part.strip()
        elif first_is_dept and second_is_dept:
            return second_part.strip()
    
    return text

def extract_schedule_from_env(env_wrapper):
    """Extract complete schedule from environment"""
    env = env_wrapper.par_env
    
    schedule_data = []
    
    # Debug: Check if day_labels exist
    print(f"\nExtracting schedule...")
    print(f"  env.day_labels exists: {hasattr(env, 'day_labels')}")
    if hasattr(env, 'day_labels'):
        print(f"  env.day_labels value: {env.day_labels}")
        print(f"  env.day_labels length: {len(env.day_labels) if env.day_labels else 0}")
    
    # Handle teacher_labels - could be dict or list
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
    
    # Handle day labels with fallback
    def get_day_label(day_idx):
        if hasattr(env, 'day_labels') and env.day_labels:
            if isinstance(env.day_labels, list) and 0 <= day_idx < len(env.day_labels):
                label = env.day_labels[day_idx]
                if label:  # Not None or empty string
                    return label
        # Fallback to standard day names
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        if 0 <= day_idx < len(day_names):
            return day_names[day_idx]
        return f"Day_{day_idx}"
    
    # Handle timeslot labels with fallback
    def get_timeslot_label(ts_idx):
        if hasattr(env, 'timeslot_labels') and env.timeslot_labels:
            if isinstance(env.timeslot_labels, list) and 0 <= ts_idx < len(env.timeslot_labels):
                label = env.timeslot_labels[ts_idx]
                if label:
                    return label
        return f"Timeslot_{ts_idx}"
    
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
                    
                    if subject_idx >= 0:  # Valid placement
                        subject_name = clean_department_prefix(env.subject_names[subject_idx])
                        teacher_idx = env.subject_assignments[subject_idx]
                        teacher_name = clean_department_prefix(get_teacher_name(teacher_idx))
                        day_label = get_day_label(day_idx)
                        timeslot_label = get_timeslot_label(ts_idx)
                        section_idx = env.subject_section_idx[subject_idx]
                        section_label = clean_department_prefix(env.section_labels[section_idx])
                        subject_area = env.subject_areas[subject_idx]
                        campus = env.room_to_campus.get(room_code, "Unknown") if hasattr(env, 'room_to_campus') else "Unknown"
                        
                        # Get duration if available
                        duration = 90
                        if hasattr(env, 'timeslot_durations') and ts_idx in env.timeslot_durations:
                            duration = env.timeslot_durations[ts_idx]
                        
                        # Calculate meetings per week
                        meetings_per_week = 2 if duration == 90 else 1
                        
                        # Debug first few entries
                        if entry_count < 3:
                            print(f"  Entry {entry_count}: day_idx={day_idx}, day_label='{day_label}'")
                        entry_count += 1
                        
                        # Use cleaned column names matching the cleaner script
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
                            'Campus': campus,
                            'Building': building,
                        })
    
    print(f"  Total entries extracted: {entry_count}")
    return schedule_data

# ======================================================================
# SCHEDULE GENERATION
# ======================================================================
def generate_schedule(checkpoint_path, max_steps=500, deterministic=True):
    """Generate schedule using trained model"""
    
    print("\n" + "="*80)
    print("SCHEDULE GENERATION WITH SAHA MODEL")
    print("="*80)
    
    # Initialize Ray
    init(ignore_reinit_error=True, include_dashboard=False)
    
    # Register model
    ModelCatalog.register_custom_model("saha_masked_head", SahaMaskedTwoHead)
    
    # Register environment
    register_env("day_pattern_env", lambda cfg: make_env())
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    # Load trained algorithm
    algo = PPO.from_checkpoint(checkpoint_path)
    
    print(f"[OK] Model loaded successfully")
    
    # Create environment
    env = make_env()
    raw_env = env.par_env
    
    print(f"\nEnvironment Info:")
    print(f"  Subjects: {raw_env.num_subjects}")
    print(f"  Teachers: {raw_env.num_teachers}")
    print(f"  Days: {raw_env.num_days}")
    print(f"  Timeslots: {raw_env.num_timeslots}")
    
    # Reset environment
    obs_dict, _ = env.reset()
    
    # Get agents from wrapped environment
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
    
    # Debug label information
    print(f"\nLabel Information:")
    print(f"  Day labels: {raw_env.day_labels if hasattr(raw_env, 'day_labels') else 'NOT SET'}")
    print(f"  Timeslot labels: {raw_env.timeslot_labels[:3] if hasattr(raw_env, 'timeslot_labels') and raw_env.timeslot_labels else 'NOT SET'}...")
    
    # Show fail statistics
    print(f"\nConstraint Violations:")
    for key, value in raw_env.fail_stats.items():
        if value > 0:
            print(f"  {key}: {value}")
    
    # Extract schedule
    schedule_data = extract_schedule_from_env(env)
    
    # Debug: Check if schedule_data has days
    if schedule_data:
        print(f"\nSchedule extraction check:")
        print(f"  Total entries: {len(schedule_data)}")
        print(f"  First entry sample: {schedule_data[0]}")
        day_values = [entry.get('Day') for entry in schedule_data[:10]]
        print(f"  First 10 day values: {day_values}")
    
    shutdown()
    
    return schedule_data, {
        'placed': placed,
        'total': raw_env.num_subjects,
        'completion_rate': completion_rate,
        'steps': step,
        'conflicts': raw_env.conflict_count,
        'fail_stats': dict(raw_env.fail_stats)
    }

# ======================================================================
# EXPORT FUNCTIONS
# ======================================================================
def export_to_csv(schedule_data, output_path):
    """Export schedule to CSV with cleaned column names"""
    df = pd.DataFrame(schedule_data)
    
    # Define column order matching cleaner script
    output_columns = [
        'Subject', 'Faculty', 'Section', 'Days', 'Time', 'Room', 'Campus',
        'Meetings_Per_Week', 'Duration_Min', 'Subject_Area', 'Building'
    ]
    
    # Only include columns that exist
    available_columns = [col for col in output_columns if col in df.columns]
    export_df = df[available_columns].copy()
    
    # Get unique days in the data
    unique_days = export_df['Days'].unique() if 'Days' in export_df.columns else []
    print(f"\nUnique days in data: {list(unique_days)}")
    
    # Sort by timeslot and room
    sort_cols = [col for col in ['Time', 'Room'] if col in export_df.columns]
    if sort_cols:
        export_df = export_df.sort_values(sort_cols)
    
    export_df.to_csv(output_path, index=False)
    print(f"\n[OK] Schedule exported to: {output_path}")
    print(f"  Columns: {', '.join(available_columns)}")
    print(f"  Rows: {len(export_df)}")

def export_to_excel(schedule_data, output_path):
    """Export schedule to Excel with formatting and cleaned column names"""
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
        
        # By room
        if 'Room' in export_df.columns:
            room_view = export_df.sort_values(['Room', 'Days', 'Time'])
            room_view.to_excel(writer, sheet_name='By Room', index=False)
        
        # Statistics
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
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        # Format all sheets
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column('A:K', 18)
    
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
# MAIN
# ======================================================================
def main():
    # Default checkpoint path
    DEFAULT_CHECKPOINT = "C:/ray_logs/PPO_Day_Patterns/PPO_day_pattern_env_4a1a5_00000_0_2025-10-10_20-54-42/checkpoint_000049"
    
    parser = argparse.ArgumentParser(description="Export schedule from trained SAHA model")
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                       help=f'Path to checkpoint (default: {DEFAULT_CHECKPOINT})')
    parser.add_argument('--output', type=str, default='schedule_output',
                       help='Output directory name')
    parser.add_argument('--format', type=str, choices=['csv', 'excel', 'both'], default='both',
                       help='Export format')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum generation steps')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic policy')
    
    args = parser.parse_args()
    
    # Use checkpoint path
    checkpoint_path = args.checkpoint
    
    # Verify checkpoint exists
    if checkpoint_path and not os.path.exists(checkpoint_path):
        print(f"WARNING: Specified checkpoint not found: {checkpoint_path}")
        print("Searching for alternative checkpoint...")
        checkpoint_path = find_latest_checkpoint()
        
        if checkpoint_path is None:
            print("ERROR: No checkpoint found. Train a model first.")
            sys.exit(1)
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Generate schedule
    schedule_data, stats = generate_schedule(
        checkpoint_path,
        max_steps=args.max_steps,
        deterministic=args.deterministic
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
        csv_path = output_dir / 'schedule.csv'
        export_to_csv(schedule_data, csv_path)
    
    if args.format in ['excel', 'both']:
        excel_path = output_dir / 'schedule.xlsx'
        export_to_excel(schedule_data, excel_path)
    
    # Export teacher schedules
    faculty_dir = output_dir / 'faculty'
    export_teacher_schedules(schedule_data, faculty_dir)
    
    # Save statistics
    stats_path = output_dir / 'statistics.txt'
    with open(stats_path, 'w') as f:
        f.write("SCHEDULE GENERATION STATISTICS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Completion Rate: {stats['completion_rate']:.1f}%\n")
        f.write(f"Subjects Placed: {stats['placed']}/{stats['total']}\n")
        f.write(f"Generation Steps: {stats['steps']}\n")
        f.write(f"Conflicts: {stats['conflicts']}\n\n")
        f.write("Constraint Violations:\n")
        for key, value in stats['fail_stats'].items():
            if value > 0:
                f.write(f"  {key}: {value}\n")
    
    print(f"\n{'='*80}")
    print(f"ALL EXPORTS COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"\nFiles generated:")
    print(f"  - schedule.csv / schedule.xlsx (cleaned, dept prefixes removed)")
    print(f"  - faculty/*.csv (individual faculty schedules)")
    print(f"  - statistics.txt")
    print(f"\nColumn names match cleaner script format:")
    print(f"  Subject, Faculty, Section, Days, Time, Room, Campus")
    print(f"  Meetings_Per_Week, Duration_Min, Subject_Area")

if __name__ == "__main__":
    main()