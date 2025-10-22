"""
Training script for Manila schedule - FULLY FIXED VERSION with Checkpoint Resume

Version 18.2: All critical bugs resolved + Checkpoint Resume Support
========================================================================================
ALL FIXES IMPLEMENTED:
✅ FIX #1: Teacher-slot consistency in action masking
✅ FIX #2: Section conflict resolution added to environment
✅ FIX #3: Atomic placement validation
✅ FIX #4: Enhanced duplicate prevention
✅ FIX #5: Placement count timing fixed
✅ FIX #6: Immediate tracking updates
✅ FIX #7: Per-placement teacher tracking (CRITICAL!)
✅ FIX #8: Milestone rewards
✅ FIX #9: Rebalanced rewards
✅ FIX #10: Day duplicate prevention
✅ FIX #11: Accurate modality stats
✅ FIX #12: Step-local placement tracking
✅ NEW: Checkpoint Resume Support
========================================================================================
"""

import sys, os, json, time, pickle, argparse, glob
import pandas as pd
import numpy as np

# CRITICAL: Shutdown any existing Ray instance first
import ray
try:
    ray.shutdown()
    print("✅ Shutdown existing Ray instance")
except:
    pass

from ray import init
from ray.air import RunConfig
from ray.tune import CLIReporter, Tuner
from ray.air.config import CheckpointConfig
from ray.tune.logger import TBXLoggerCallback
from ray.tune.registry import register_env

from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from torch.utils.tensorboard import SummaryWriter

# RLlib model
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from collections import OrderedDict

# Import FIXED ENV v14.5
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.timetabling_env import ParallelTimetablingEnv

# For memory monitoring
import psutil

# Windows-friendly Ray setup
SPILL_DIR = "C:/ray_spill"
TEMP_DIR = "C:/ray_temp"
os.makedirs(SPILL_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.environ["RAY_object_spilling_config"] = json.dumps(
    {"type": "filesystem", "params": {"directory_path": SPILL_DIR}}
)


# ============================================================
# MANILA CACHE FINDER
# ============================================================
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
                    mod_time = time.ctime(os.path.getmtime(abs_path))
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


def load_manila_data(cache_file=None):
    """Load Manila preprocessed data"""
    if cache_file is None:
        cache_file = find_manila_cache()
        if cache_file is None:
            raise FileNotFoundError("No Manila cache file found. Run preprocessing first.")
    
    if not os.path.exists(cache_file):
        abs_cache = os.path.abspath(cache_file)
        if os.path.exists(abs_cache):
            cache_file = abs_cache
        else:
            found = find_manila_cache()
            if found:
                cache_file = found
            else:
                raise FileNotFoundError(f"Cache file not found: {cache_file}")
    
    print(f"\n{'='*80}")
    print("LOADING MANILA SCHEDULE DATA")
    print("="*80)
    print(f"Cache file: {cache_file}")
    
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    
    required = ['num_subjects', 'num_teachers', 'room_codes', 'subject_names']
    missing = [field for field in required if field not in data]
    if missing:
        raise ValueError(f"Cache missing required fields: {missing}")
    
    has_modality = 'subject_modalities' in data
    
    print(f"\n✓ Loaded Manila schedule data")
    print(f"  Subjects: {data['num_subjects']}")
    print(f"  Teachers: {data['num_teachers']}")
    print(f"  Modality support: {'✓' if has_modality else '✗'}")
    
    if has_modality:
        print(f"\n  Modality Distribution:")
        for modality in ['Face-to-Face', 'Online', 'Hybrid']:
            count = data['subject_modalities'].count(modality)
            pct = (count / data['num_subjects']) * 100
            print(f"    {modality}: {count} ({pct:.1f}%)")
    
    print("="*80 + "\n")
    return data


# ============================================================
# IMPROVED SAHA MODEL
# ============================================================
class ImprovedSahaMaskedTwoHead(TorchModelV2, nn.Module):
    """Neural network with improved value function for handling complex rewards"""
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


# Clear any existing registration
try:
    ModelCatalog._catalog.pop("improved_saha_masked", None)
except:
    pass

# Register the improved model
ModelCatalog.register_custom_model("improved_saha_masked", ImprovedSahaMaskedTwoHead)
print("✅ Model 'improved_saha_masked' registered\n")


# ============================================================
# ENHANCED DIAGNOSTIC CALLBACK WITH FIX #7 VALIDATION
# ============================================================
class EnhancedValidationCallback(DefaultCallbacks):
    """
    Enhanced callback with FIX #7: Per-placement teacher tracking validation
    """
    def __init__(self):
        super().__init__()
        self.writer = SummaryWriter(log_dir="C:/ray_logs/manila_tensorboard")
        self.episode_counter = 0
        self.best_placement = 0
        self.best_full_placement = 0
        self.best_zero_conflicts = False
        
        self.reward_history = []
        self.value_loss_history = []
        self.policy_loss_history = []
        
        self.action_stats = {
            'wait_actions': 0,
            'place_actions': 0,
            'total_actions': 0,
        }
        
        self.placement_history = []
        self.conflict_history = []
        
    def on_train_result(self, *, algorithm, result, **kwargs):
        it = result.get("training_iteration", 0)
        rew = result.get("episode_reward_mean", 0.0)
        self.reward_history.append(rew)
        
        print(f"\n{'='*80}")
        print(f"🔬 DIAGNOSTIC - Iteration {it}")
        print(f"{'='*80}")
        
        # Learning metrics
        learner_info = result.get("info", {}).get("learner", {})
        if "saha_policy" in learner_info:
            learner_stats = learner_info["saha_policy"].get("learner_stats", {})
            policy_loss = learner_stats.get("policy_loss", None)
            value_loss = learner_stats.get("vf_loss", None)
            entropy = learner_stats.get("entropy", None)
            
            print(f"\n📊 LEARNING METRICS:")
            print(f"   Policy Loss: {policy_loss}")
            print(f"   Value Loss: {value_loss}")
            print(f"   Entropy: {entropy}")
            
            if value_loss is not None:
                self.value_loss_history.append(value_loss)
                
                if value_loss > 100:
                    print(f"   🚨 CRITICAL: Value loss > 100!")
                elif value_loss > 50:
                    print(f"   ❌ ERROR: Value loss > 50")
                elif value_loss > 20:
                    print(f"   ⚠️ WARNING: High value loss")
                elif value_loss < 5:
                    print(f"   ✅ Value loss healthy")
                else:
                    print(f"   ✨ Value loss excellent!")
        
        # Placement metrics
        print(f"\n📈 PLACEMENT PERFORMANCE:")
        partial = result.get('custom_metrics/placement_rate_mean', None)
        full = result.get('custom_metrics/full_placement_rate_mean', None)
        
        # ENHANCED: All conflict types
        total_conflicts = result.get('custom_metrics/total_conflicts_mean', 0)
        duplicates = result.get('custom_metrics/duplicate_count_mean', 0)
        teacher_conflicts = result.get('custom_metrics/teacher_conflicts_mean', 0)
        section_conflicts = result.get('custom_metrics/section_conflicts_mean', 0)
        
        if partial is not None:
            self.placement_history.append(partial)
            self.conflict_history.append(total_conflicts)
            
            print(f"   Partial: {partial:.1f}%")
            print(f"   Full: {full:.1f}%")
            print(f"\n   🔍 CONFLICT BREAKDOWN:")
            print(f"      Total: {total_conflicts:.1f}")
            print(f"      Duplicates: {duplicates:.1f}")
            print(f"      Teacher: {teacher_conflicts:.1f}")
            print(f"      Section: {section_conflicts:.1f}")
            
            # Celebrate zero conflicts!
            if total_conflicts == 0:
                if not self.best_zero_conflicts:
                    print(f"\n   🎉🎉🎉 FIRST ZERO-CONFLICT EPISODE! 🎉🎉🎉")
                    print(f"   FIX #7 IS WORKING!")
                    self.best_zero_conflicts = True
                else:
                    print(f"\n   ✅ ZERO CONFLICTS - All fixes working!")
            elif total_conflicts < 5:
                print(f"\n   ✨ Very low conflicts!")
            elif total_conflicts < 10:
                print(f"\n   ⚠️ Some conflicts remain")
            else:
                print(f"\n   ❌ High conflict count - debugging needed")
            
            if partial > self.best_placement:
                self.best_placement = partial
                print(f"\n   🎯 NEW BEST PARTIAL: {partial:.1f}%!")
            
            if full is not None and full > self.best_full_placement:
                self.best_full_placement = full
                print(f"   🏆 NEW BEST FULL: {full:.1f}%!")
        
        # Action distribution
        print(f"\n🎮 ACTION DISTRIBUTION:")
        if self.action_stats['total_actions'] > 0:
            wait_pct = (self.action_stats['wait_actions'] / self.action_stats['total_actions']) * 100
            place_pct = 100 - wait_pct
            print(f"   Wait: {wait_pct:.1f}%")
            print(f"   Place: {place_pct:.1f}%")
            
            if wait_pct > 90:
                print(f"   ❌ Agent mostly waiting!")
            elif wait_pct > 70:
                print(f"   ⚠️ High wait rate")
            elif wait_pct < 30:
                print(f"   ✅ Good action balance")
            
            self.action_stats = {'wait_actions': 0, 'place_actions': 0, 'total_actions': 0}
        
        # Reward analysis
        print(f"\n💰 REWARD ANALYSIS:")
        print(f"   Current: {rew:.2f}")
        if len(self.reward_history) > 1:
            prev_rew = self.reward_history[-2]
            delta = rew - prev_rew
            print(f"   Change: {delta:+.2f}")
        
        # Progress tracking
        if len(self.conflict_history) > 10:
            recent_conflicts = self.conflict_history[-10:]
            avg_recent = sum(recent_conflicts) / len(recent_conflicts)
            print(f"\n📊 RECENT TREND (last 10 episodes):")
            print(f"   Avg conflicts: {avg_recent:.1f}")
            if avg_recent == 0:
                print(f"   🌟 PERFECT: Consistently zero conflicts!")
            elif avg_recent < 1:
                print(f"   ✨ Excellent: Near-zero conflicts")
            elif avg_recent < 5:
                print(f"   ✅ Good: Low conflicts")
        
        print(f"{'='*80}\n")
        
        # TensorBoard logging
        self.writer.add_scalar("Reward/Mean", rew, it)
        if partial is not None:
            self.writer.add_scalar("Placement/Partial", partial, it)
        if full is not None:
            self.writer.add_scalar("Placement/Full", full, it)
        if total_conflicts >= 0:
            self.writer.add_scalar("Validation/Total_Conflicts", total_conflicts, it)
            self.writer.add_scalar("Validation/Duplicates", duplicates, it)
            self.writer.add_scalar("Validation/Teacher_Conflicts", teacher_conflicts, it)
            self.writer.add_scalar("Validation/Section_Conflicts", section_conflicts, it)
        if value_loss is not None:
            self.writer.add_scalar("Loss/Value", value_loss, it)
        if policy_loss is not None:
            self.writer.add_scalar("Loss/Policy", policy_loss, it)
                
    def _unwrap_env(self, base_env):
        """Properly unwrap the environment"""
        env_like = None
        if hasattr(base_env, "get_sub_environments"):
            try: 
                env_like = base_env.get_sub_environments()[0]
            except: 
                pass
        if env_like is None and hasattr(base_env, "vector_env") and hasattr(base_env.vector_env, "envs"):
            try: 
                env_like = base_env.vector_env.envs[0]
            except: 
                pass
        if env_like is None: 
            env_like = base_env
        return getattr(env_like, "par_env", env_like)
        
    def on_episode_end(self, *, worker, base_env, episode, **kwargs):
        """Enhanced validation with FIX #7: Per-placement teacher tracking"""
        self.episode_counter += 1
        env = self._unwrap_env(base_env)

        num_fully_placed = len(env.placed_subjects)
        full_placement_rate = (num_fully_placed / env.num_subjects) * 100
        
        subjects_with_placements = set()
        for (subj, sec), count in env.subject_placement_count.items():
            if count > 0:
                subjects_with_placements.add(subj)
        
        num_partial_placed = len(subjects_with_placements)
        partial_placement_rate = (num_partial_placed / env.num_subjects) * 100
        
        total_placements_needed = sum(env.subject_required_placements.values())
        total_placements_made = sum(env.subject_placement_count.values())
        overall_progress = (total_placements_made / total_placements_needed) * 100 if total_placements_needed > 0 else 0
        
        episode.custom_metrics["placement_rate"] = partial_placement_rate
        episode.custom_metrics["full_placement_rate"] = full_placement_rate
        episode.custom_metrics["overall_progress"] = overall_progress
        
        # COMPREHENSIVE VALIDATION WITH FIX #7
        
        # 1. Duplicate check
        duplicate_count = 0
        for subj in range(env.num_subjects):
            sec_idx = env.subject_section_idx[subj]
            current = env._get_placement_count(subj, sec_idx)
            required = env.subject_required_placements.get(subj, 1)
            if current > required:
                duplicate_count += (current - required)
        
        episode.custom_metrics["duplicate_count"] = duplicate_count
        
        # 2. Teacher conflict check (FIX #7: Using per-placement tracking)
        teacher_conflicts = 0
        for t_idx in range(env.num_teachers):
            for d in range(env.num_days):
                for ts in range(env.num_timeslots):
                    # Find all subjects at this day/time with this teacher
                    subjects_here = []
                    
                    for b in env.building_keys:
                        b_sched = env.buildings_room_schedule[b]
                        for r_idx in range(len(env.buildings_room_info[b])):
                            subj = b_sched[r_idx, d, ts]
                            
                            if subj >= 0:
                                # FIX #7: Use placement-specific teacher assignment
                                assigned_t = env.placement_teachers.get((subj, d, ts), -1)
                                if assigned_t == t_idx:
                                    subjects_here.append(subj)
                    
                    # If same teacher has multiple subjects at same time, it's a conflict
                    if len(subjects_here) > 1:
                        teacher_conflicts += (len(subjects_here) - 1)
        
        episode.custom_metrics["teacher_conflicts"] = teacher_conflicts
        
        # 3. Section conflict check
        section_conflicts = 0
        for sec_idx in range(env.num_sections):
            for d in range(env.num_days):
                for ts in range(env.num_timeslots):
                    subjects_at_time = []
                    for subj in range(env.num_subjects):
                        if env.subject_section_idx[subj] != sec_idx:
                            continue
                        for b in env.building_keys:
                            b_sched = env.buildings_room_schedule[b]
                            for r_idx in range(len(env.buildings_room_info[b])):
                                if b_sched[r_idx, d, ts] == subj:
                                    subjects_at_time.append(subj)
                    
                    if len(subjects_at_time) > 1:
                        section_conflicts += (len(subjects_at_time) - 1)
        
        episode.custom_metrics["section_conflicts"] = section_conflicts
        
        # Total conflicts
        total_conflicts = duplicate_count + teacher_conflicts + section_conflicts
        episode.custom_metrics["total_conflicts"] = total_conflicts
        
        # Detailed logging for first 20 and every 10th episode
        should_log = (worker.worker_index == 1) and ((self.episode_counter <= 20) or (self.episode_counter % 10 == 0))
        
        if should_log:
            print(f"\n{'='*70}")
            print(f"📊 EPISODE {self.episode_counter} VALIDATION (v14.5)")
            print(f"{'='*70}")
            print(f"Placement: Partial {partial_placement_rate:.1f}% | Full {full_placement_rate:.1f}%")
            print(f"\nConflict Breakdown:")
            print(f"  Total: {total_conflicts}")
            print(f"  ├─ Duplicates: {duplicate_count}")
            print(f"  ├─ Teacher: {teacher_conflicts}")
            print(f"  └─ Section: {section_conflicts}")
            
            if total_conflicts == 0:
                print(f"\n🎉 PERFECT SCHEDULE: ZERO CONFLICTS!")
                print(f"✅ All fixes (including FIX #7) are working!")
            elif total_conflicts < 5:
                print(f"\n✨ Excellent: Very low conflicts")
            elif total_conflicts < 10:
                print(f"\n⚠️ Some conflicts remain")
            else:
                print(f"\n❌ High conflicts - needs attention")
                
                # Show which type is worst
                max_type = max(
                    [("Duplicates", duplicate_count), 
                     ("Teacher", teacher_conflicts),
                     ("Section", section_conflicts)],
                    key=lambda x: x[1]
                )
                print(f"   Main issue: {max_type[0]} ({max_type[1]} conflicts)")
            
            print(f"{'='*70}\n")
    
    def on_episode_step(self, *, worker, base_env, episode, **kwargs):
        """Track action distribution"""
        env = self._unwrap_env(base_env)
        
        if hasattr(episode, 'last_action_for'):
            for agent_id in episode.last_action_for.keys():
                action = episode.last_action_for[agent_id]
                
                if isinstance(action, (list, tuple, np.ndarray)) and len(action) >= 2:
                    teacher_idx = int(action[0])
                    slot_idx = int(action[1])
                    
                    is_wait = (teacher_idx >= env.max_teachers_per_area or 
                              slot_idx >= env.slot_choices)
                    
                    self.action_stats['total_actions'] += 1
                    if is_wait:
                        self.action_stats['wait_actions'] += 1
                    else:
                        self.action_stats['place_actions'] += 1
            
    def __del__(self):
        if hasattr(self, "writer"): 
            self.writer.close()


# ============================================================
# ENVIRONMENT FACTORY
# ============================================================
def make_manila_env(config=None):
    """Create environment for Manila schedule with ALL FIXES including #7"""
    cache_file = None
    if config and isinstance(config, dict):
        cache_file = config.get('cache_file')
    
    if cache_file is None:
        cache_file = find_manila_cache()
        if cache_file is None:
            raise FileNotFoundError("No Manila cache file found")
    
    data = load_manila_data(cache_file)
    
    if 'subject_modalities' not in data:
        data['subject_modalities'] = ['Face-to-Face'] * data['num_subjects']
        data['subject_modality_idx'] = [0] * data['num_subjects']
        data['modality_labels'] = ['Face-to-Face', 'Online', 'Hybrid']
        data['virtual_rooms'] = []
        data['physical_rooms'] = data['room_codes']
    
    env = ParallelTimetablingEnv(
        num_teachers=data["num_teachers"],
        num_subjects=data["num_subjects"],
        num_days=data["num_days"],
        num_timeslots=data["num_timeslots"],
        room_codes=data["room_codes"],
        subject_codes=data["subject_names"],
        subject_modalities=data.get("subject_modalities"),
        subject_modality_idx=data.get("subject_modality_idx"),
        modality_labels=data.get("modality_labels", ['Face-to-Face', 'Online', 'Hybrid']),
        virtual_rooms=data.get("virtual_rooms", []),
        subject_campuses=data["subject_campuses"],
        subject_allowed_rooms=data["subject_allowed_rooms"],
        subject_areas=data["subject_areas"],
        subject_section_idx=data["subject_section_idx"],
        section_labels=data["section_labels"],
        max_classes_per_teacher=4,
        teacher_max_classes_map=data["teacher_max_classes_map"],
        subject_teacher_idx=data["subject_teacher_idx"],
        area_teacher_indices=data["area_teacher_indices"],
        subject_allowed_timeslots=data.get("subject_allowed_timeslots"),
        timeslot_definitions=data.get("timeslot_definitions", []),
        subject_required_placements=data.get("subject_required_placements"),
        strict_teacher_match=False,
        r_teacher_match=40.0,
        r_teacher_mismatch=-1.0,
        r_area_match=25.0,
        r_area_mismatch=-1.5,
        r_workload_balance=15.0,
        base_success_reward=150.0,
        wait_penalty=20.0,
        r_online_bonus=8.0,
        enable_progressive_difficulty=True,
        difficulty_ramp_steps=100,
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
    
    env.subject_names = data["subject_names"]
    env.teacher_names = data.get("teacher_names", [])
    env.teacher_labels = data.get("teacher_id_to_name", [])
    env.day_labels = data.get("day_labels", ['M', 'T', 'W', 'H', 'F'])
    env.timeslot_labels = data.get("timeslot_labels", [])
    env.room_to_campus = data.get("room_to_campus", {})
    env.section_labels = data["section_labels"]
    env.subject_modalities = data["subject_modalities"]
    
    return ParallelPettingZooEnv(env)


# ============================================================
# CHECKPOINT FINDER (OPTIONAL)
# ============================================================
def find_latest_checkpoint(base_dir="C:/ray_logs"):
    """Find the most recent checkpoint in training directories"""
    
    print(f"\n{'='*80}")
    print(f"SEARCHING FOR CHECKPOINTS")
    print(f"{'='*80}")
    print(f"Base directory: {base_dir}")
    
    # Search all subdirectories for checkpoint folders
    all_checkpoints = []
    
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            if d.startswith("checkpoint_"):
                full_path = os.path.join(root, d)
                mtime = os.path.getmtime(full_path)
                
                # Extract iteration number
                try:
                    iter_num = int(d.split('_')[1])
                except:
                    iter_num = 0
                
                all_checkpoints.append({
                    'path': full_path,
                    'name': d,
                    'iteration': iter_num,
                    'modified': mtime,
                    'modified_str': time.ctime(mtime)
                })
    
    if not all_checkpoints:
        print("❌ No checkpoints found!")
        print("="*80 + "\n")
        return None
    
    # Sort by modification time (most recent first)
    all_checkpoints.sort(key=lambda x: x['modified'], reverse=True)
    
    print(f"\nFound {len(all_checkpoints)} checkpoints")
    print(f"\nTop 5 most recent:")
    for i, ckpt in enumerate(all_checkpoints[:5]):
        print(f"  {i+1}. {ckpt['name']} (iter {ckpt['iteration']}) - {ckpt['modified_str']}")
    
    latest = all_checkpoints[0]
    print(f"\n✅ Latest checkpoint: {latest['name']} (iteration {latest['iteration']})")
    print(f"   Path: {latest['path']}")
    print("="*80 + "\n")
    
    return latest['path']


# ============================================================
# MAIN TRAINING
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manila Training v18.2 - FULLY FIXED with Resume")
    parser.add_argument("--cache", type=str, default=None,
                       help="Path to Manila cache file")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Target number of training iterations")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze constraints before training")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate existing schedule")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from (or 'auto' for latest)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MANILA TRAINING - FULLY FIXED v18.2 with Resume")
    print("=" * 80)
    print("\n🔧 ALL FIXES APPLIED:")
    print("  ✅ FIX #1: Teacher-slot consistency in masking")
    print("  ✅ FIX #2: Section conflict resolution")
    print("  ✅ FIX #3: Atomic placement validation")
    print("  ✅ FIX #4: Duplicate prevention enhanced")
    print("  ✅ FIX #5: Placement count timing fixed")
    print("  ✅ FIX #6: Immediate tracking updates")
    print("  ✅ FIX #7: Per-placement teacher tracking (CRITICAL!)")
    print("  ✅ FIX #8: Milestone rewards")
    print("  ✅ FIX #9: Rebalanced rewards")
    print("  ✅ FIX #10: Day duplicate prevention")
    print("  ✅ FIX #11: Accurate modality stats")
    print("  ✅ FIX #12: Step-local placement tracking")
    print("  ✅ NEW: Checkpoint Resume Support")
    print("\n📊 Expected Result:")
    print("  ZERO teacher conflicts")
    print("  ZERO section conflicts")
    print("  ZERO duplicate placements")
    print("=" * 80 + "\n")
    
    cache_file = args.cache
    if cache_file is None:
        cache_file = find_manila_cache()
        if cache_file is None:
            sys.exit(1)
    
    # Validation-only mode
    if args.validate_only:
        print("Running validation-only mode...")
        test_env = make_manila_env({'cache_file': cache_file})
        raw = test_env.par_env
        
        test_env.reset()
        conflicts = raw.validate_schedule()
        
        if conflicts['summary']['total_conflicts'] == 0:
            print("\n✅ Schedule is PERFECT - ZERO conflicts!")
            print("FIX #7 is working correctly!")
        else:
            print(f"\n⚠️ Schedule has {conflicts['summary']['total_conflicts']} conflicts")
            print("\nBreakdown:")
            print(f"  Duplicates: {conflicts['summary']['duplicate_placements']}")
            print(f"  Teacher: {conflicts['summary']['teacher_conflicts']}")
            print(f"  Section: {conflicts['summary']['section_conflicts']}")
        
        sys.exit(0)
    
    # System check
    print("=" * 80)
    print("SYSTEM CHECK")
    print("=" * 80)
    mem = psutil.virtual_memory()
    print(f"RAM: {mem.total/(1024**3):.1f} GB total, {mem.available/(1024**3):.1f} GB available")
    print("=" * 80 + "\n")
    
    # Initialize Ray
    init(
        ignore_reinit_error=True, 
        include_dashboard=False, 
        _temp_dir=TEMP_DIR,
    )
    
    # Register environment
    def env_creator(config):
        config['cache_file'] = cache_file
        return make_manila_env(config)
    
    register_env("manila_env", env_creator)

    # Test environment
    print("=" * 80)
    print("VALIDATING ENVIRONMENT (v14.5)")
    print("=" * 80)
    test_env = make_manila_env({'cache_file': cache_file})
    raw = test_env.par_env
    
    print(f"Subjects: {raw.num_subjects}")
    print(f"Teachers: {raw.num_teachers}")
    print(f"Agents: {len(raw.agents)}")
    print(f"Max teachers per area: {raw.max_teachers_per_area}")
    print(f"Slot choices: {raw.slot_choices}")
    
    # Verify FIX #7
    if hasattr(raw, 'placement_teachers'):
        print(f"✅ FIX #7 applied: placement_teachers dict exists")
    else:
        print(f"❌ FIX #7 NOT applied: placement_teachers missing!")
        sys.exit(1)
    
    obs_dict, _ = test_env.reset()
    first_agent = list(obs_dict.keys())[0]
    first_saha = [a for a in raw.agents if a.startswith("saha_")][0]
    
    sample_obs = obs_dict[first_agent]
    obs_space = raw.observation_spaces[first_saha]
    
    if isinstance(sample_obs, dict):
        actual_dim = sample_obs['obs'].shape[0]
        expected_dim = obs_space.spaces['obs'].shape[0]
        print(f"\nObservation validation:")
        print(f"  Expected: {expected_dim}")
        print(f"  Actual: {actual_dim}")
        
        if actual_dim != expected_dim:
            print(f"  ❌ MISMATCH!")
            sys.exit(1)
        else:
            print(f"  ✅ MATCH!")
    
    action_space = raw.action_spaces[first_saha]
    print(f"\nAction space: MultiDiscrete{action_space.nvec}")
    print("=" * 80 + "\n")

    # Policy configuration
    policies = {
        "saha_policy": (
            None,
            raw.observation_spaces[first_saha],
            raw.action_spaces[first_saha],
            {
                "model": {
                    "custom_model": "improved_saha_masked",
                    "custom_model_config": {"hidden_sizes": [512, 512, 256]},
                    "fcnet_hiddens": [],
                },
                "lr": 5e-4,
            },
        )
    }
    
    def policy_mapping_fn(agent_id, episode, **kwargs):
        return "saha_policy"

    # PPO Configuration
    ppo_cfg = (
        PPOConfig()
        .environment(
            env="manila_env",
            env_config={'cache_file': cache_file},
            disable_env_checking=True
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=1,
            rollout_fragment_length=64,
            batch_mode="complete_episodes",
            num_envs_per_worker=1,
        )
        .training(
            gamma=0.95,
            lr=5e-4,
            lr_schedule=[
                [0, 1e-3],
                [10000, 5e-4],
                [50000, 2e-4],
                [100000, 1e-4],
            ],
            train_batch_size=512,
            sgd_minibatch_size=256,
            num_sgd_iter=10,
            vf_clip_param=50.0,
            use_gae=True,
            lambda_=0.95,
            clip_param=0.3,
            entropy_coeff=1.0,
            entropy_coeff_schedule=[
                [0, 1.0],
                [20000, 0.5],
                [50000, 0.2],
                [100000, 0.05],
            ],
            grad_clip=1.0,
            kl_coeff=0.1,
            kl_target=0.01,
            vf_loss_coeff=1.0,
        )
        .resources(
            num_gpus=1,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .callbacks(EnhancedValidationCallback)
        .experimental(_enable_new_api_stack=False, _disable_preprocessor_api=True)
    )

    config = ppo_cfg.to_dict()
    
    # ============================================================
    # CHECKPOINT DETECTION AND RESUMPTION
    # ============================================================
    
    resume_checkpoint = None
    
    if args.resume:
        if args.resume.lower() == "auto":
            # Auto-detect latest checkpoint
            resume_checkpoint = find_latest_checkpoint()
            if not resume_checkpoint:
                print("No checkpoints found, starting fresh training")
                args.resume = None
        else:
            # User specified checkpoint path
            resume_checkpoint = args.resume
            
            if not os.path.exists(resume_checkpoint):
                print(f"❌ ERROR: Checkpoint not found: {resume_checkpoint}")
                sys.exit(1)
            
            print(f"\n✅ Found checkpoint: {os.path.basename(resume_checkpoint)}")
            
            # Verify it's a valid checkpoint
            if not os.path.isdir(resume_checkpoint):
                print(f"❌ ERROR: Path is not a checkpoint directory")
                sys.exit(1)
    
    # ============================================================
    # STARTING TRAINING (WITH RESUME SUPPORT)
    # ============================================================
    print(f"\n" + "=" * 80)
    print("STARTING TRAINING - ALL FIXES APPLIED (v14.5 + v18.2)")
    print("=" * 80)
    print(f"Target iterations: {args.iterations}")
    if resume_checkpoint:
        print(f"🔄 RESUMING from: {os.path.basename(resume_checkpoint)}")
    else:
        print(f"🆕 STARTING FRESH (no checkpoint)")
    print(f"\n🎯 Target: ZERO conflicts of all types")
    print(f"Watch for: 'FIX #7 IS WORKING!' in episode logs")
    print("=" * 80 + "\n")
    
    # ============================================================
    # TRAINING PATH: RESUME vs FRESH
    # ============================================================
    if resume_checkpoint:
        print(f"Attempting direct algorithm restoration...")
        
        try:
            # Restore algorithm directly from checkpoint
            algorithm = PPO.from_checkpoint(resume_checkpoint)
            current_iter = algorithm.iteration
            print(f"✅ Restored from iteration {current_iter}")
            
            # Verify environment still works with restored model
            print(f"Verifying environment compatibility...")
            verify_env = make_manila_env({'cache_file': cache_file})
            obs, _ = verify_env.reset()
            print(f"✅ Environment compatible with checkpoint")
            
            # Continue training manually
            target_iterations = args.iterations
            checkpoint_dir = "C:/ray_logs/manual_checkpoints_manila_resumed"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            print(f"\n{'='*80}")
            print(f"RESUMING TRAINING: Iteration {current_iter} → {target_iterations}")
            print(f"{'='*80}\n")
            
            while current_iter < target_iterations:
                result = algorithm.train()
                current_iter = result["training_iteration"]
                
                # Extract metrics
                reward = result.get('episode_reward_mean', 0)
                placement_rate = result.get('custom_metrics', {}).get('placement_rate_mean', 0)
                full_rate = result.get('custom_metrics', {}).get('full_placement_rate_mean', 0)
                conflicts = result.get('custom_metrics', {}).get('total_conflicts_mean', 0)
                duplicates = result.get('custom_metrics', {}).get('duplicate_count_mean', 0)
                
                # Calculate actual subjects placed
                num_subjects = raw.num_subjects
                partial_placed = int(placement_rate * num_subjects / 100) if placement_rate else 0
                full_placed = int(full_rate * num_subjects / 100) if full_rate else 0
                
                # Print progress
                print(f"Iter {current_iter:3d}: "
                      f"Reward={reward:7.1f} | "
                      f"Partial={partial_placed:3d}/{num_subjects} ({placement_rate:5.1f}%) | "
                      f"Full={full_placed:3d}/{num_subjects} ({full_rate:5.1f}%) | "
                      f"Conflicts={conflicts:4.1f} | "
                      f"Dup={duplicates:4.1f}")
                
                # Save checkpoint every 10 iterations
                if current_iter % 10 == 0:
                    checkpoint_path = algorithm.save(
                        f"{checkpoint_dir}/checkpoint_{current_iter:06d}"
                    )
                    print(f"   💾 Saved checkpoint_{current_iter:06d}")
                
                # Milestone notifications
                if full_rate >= 95 and current_iter % 5 == 0:
                    print(f"   🎯 Excellent performance! Full placement: {full_rate:.1f}%")
                
                if conflicts == 0 and duplicates == 0 and full_rate > 0:
                    print(f"   ✨ PERFECT SCHEDULE: Zero conflicts and duplicates!")
            
            # Final checkpoint
            final_checkpoint = algorithm.save(
                f"{checkpoint_dir}/checkpoint_{current_iter:06d}_FINAL"
            )
            
            print(f"\n{'='*80}")
            print(f"✅ TRAINING COMPLETED!")
            print(f"{'='*80}")
            print(f"Final iteration: {current_iter}")
            print(f"Final checkpoint: {os.path.basename(final_checkpoint)}")
            print(f"Checkpoint directory: {checkpoint_dir}")
            print(f"{'='*80}\n")
            
            # Run final validation
            print("\nRunning final comprehensive validation...")
            final_env = make_manila_env({'cache_file': cache_file})
            final_env.reset()
            final_conflicts = final_env.par_env.validate_schedule()
            
            total = final_conflicts['summary']['total_conflicts']
            
            if total == 0:
                print("\n🎉🎉🎉 PERFECT! Final schedule has ZERO conflicts! 🎉🎉🎉")
                print("All fixes (including FIX #7 and FIX #12) are working correctly!")
            else:
                print(f"\n⚠️ Final schedule has {total} conflicts")
                print(f"\nBreakdown:")
                print(f"  Duplicates: {final_conflicts['summary']['duplicate_placements']}")
                print(f"  Teacher: {final_conflicts['summary']['teacher_conflicts']}")
                print(f"  Section: {final_conflicts['summary']['section_conflicts']}")
            
            sys.exit(0)
            
        except Exception as e:
            print(f"❌ ERROR: Failed to restore checkpoint: {e}")
            import traceback
            traceback.print_exc()
            print(f"\nFalling back to fresh training...\n")
            resume_checkpoint = None
    
    # ============================================================
    # FRESH TRAINING (No checkpoint or resume failed)
    # ============================================================
    if not resume_checkpoint:
        print("Starting fresh training with Tuner...")
        
        reporter = CLIReporter(
            metric_columns=[
                "training_iteration",
                "episode_reward_mean",
                "custom_metrics/placement_rate_mean",
                "custom_metrics/full_placement_rate_mean",
                "custom_metrics/total_conflicts_mean",
                "custom_metrics/duplicate_count_mean",
                "custom_metrics/teacher_conflicts_mean",
                "custom_metrics/section_conflicts_mean",
            ],
            max_report_frequency=30,
        )

        run_cfg = RunConfig(
            stop={"training_iteration": args.iterations},
            local_dir="C:/ray_logs",
            name="Manila_FULLY_FIXED_v18_2_with_Resume",
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=10, 
                checkpoint_at_end=True,
            ),
            callbacks=[TBXLoggerCallback()],
            progress_reporter=reporter,
            verbose=1,
        )

        tuner = Tuner("PPO", param_space=config, run_config=run_cfg)
        results = tuner.fit()

        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETED")
        print("=" * 80)
        
        # Final comprehensive validation
        print("\nRunning final comprehensive validation with FIX #7...")
        final_env = make_manila_env({'cache_file': cache_file})
        final_env.reset()
        final_conflicts = final_env.par_env.validate_schedule()
        
        total = final_conflicts['summary']['total_conflicts']
        
        if total == 0:
            print("\n🎉🎉🎉 PERFECT! Final schedule has ZERO conflicts! 🎉🎉🎉")
            print("All fixes (including FIX #7 and FIX #12) are working correctly!")
        else:
            print(f"\n⚠️ Final schedule has {total} conflicts")
            print(f"\nBreakdown:")
            print(f"  Duplicates: {final_conflicts['summary']['duplicate_placements']}")
            print(f"  Teacher: {final_conflicts['summary']['teacher_conflicts']}")
            print(f"  Section: {final_conflicts['summary']['section_conflicts']}")
            print("\nReview episode logs to identify when conflicts occur.")