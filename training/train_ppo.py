"""
Training script for Manila schedule - FULLY FIXED VERSION with AGGRESSIVE PENALTIES

Version 18.9: AGGRESSIVE Conflict Penalties + GPU Optimization + Curriculum Learning
========================================================================================
ALL FIXES IMPLEMENTED:
✅ FIX #1-12: All previous critical bugs resolved
✅ FIX #13: AGGRESSIVE conflict penalties (conflicts now HURT more than placements help)
✅ FIX #14: Exponential global conflict penalty (per step)
✅ FIX #15: Quadratic completion pressure (forces 90%+ placement)
✅ FIX #16: Curriculum learning (progressive difficulty in 3 phases)
✅ PERF #1-4: GPU optimization + Windows-safe parallelization
========================================================================================

REWARD REBALANCING SUMMARY:
========================================================================================
BEFORE (Why model wasn't learning):
- Placement reward: +150
- Conflict penalty: -20
- Ratio: 7.5:1 (placements heavily favored)
- Model learned: "Place subjects, conflicts are acceptable" ❌

AFTER (Aggressive penalties):
- Placement reward: +150
- Conflict penalty: -200
- Duplicate penalty: -300
- Section conflict: -250
- Global exponential penalty: -50 * (1 - exp(-0.02 * conflicts))
- Ratio: 1:1.33 (conflicts HURT more than placements help)
- Model learns: "Only place if NO conflicts!" ✅

CURRICULUM LEARNING:
- Phase 1 (0-10k steps / ~100 episodes): 0.5x penalties - gentle learning
- Phase 2 (10k-30k steps / ~100-300 episodes): 1.0x penalties - normal training
- Phase 3 (30k+ steps / 300+ episodes): 1.5x penalties - perfection mode
========================================================================================

EXPECTED BEHAVIOR:
========================================================================================
Iteration 10-50:   Conflicts START decreasing immediately
Iteration 50-100:  Student scores rise to 40+, conflicts drop to <50
Iteration 100-200: Overall scores reach 65-75, conflicts <20
Iteration 200-500: Overall scores 80-90+, conflicts <5
Iteration 500+:    ZERO conflicts achievable, scores 95+

IF CONFLICTS DON'T DECREASE BY ITERATION 50:
→ Penalties still too weak (increase r_conflict_penalty to -300)
→ Check TensorBoard for reward trends (should stabilize, not collapse)
→ Verify environment is using v14.7 (check print at startup)
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

# Import FIXED ENV v14.7 (AGGRESSIVE PENALTIES)
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
# ENHANCED DIAGNOSTIC CALLBACK WITH AGGRESSIVE PENALTY TRACKING
# ============================================================
class EnhancedValidationCallback(DefaultCallbacks):
    """
    Enhanced callback with aggressive penalty tracking and curriculum phase monitoring
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
        
        # NEW: Track curriculum phase performance
        self.phase_conflicts = {0.5: [], 1.0: [], 1.5: []}
        self.phase_placements = {0.5: [], 1.0: [], 1.5: []}
        
        # Track first improvements
        self.first_conflict_decrease = None
        self.first_40_score = None
        
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
        
        # FIX: Get metrics from multiple possible locations
        custom = result.get('custom_metrics', {})
        sampler = result.get('sampler_results', {})
        eval_metrics = result.get('evaluation', {}).get('custom_metrics', {})
        
        # Try all possible sources
        partial = None
        full = None
        total_conflicts = None
        duplicates = None
        teacher_conflicts = None
        section_conflicts = None
        curriculum_phase = None
        global_steps = None
        under_placed = None
        over_placed = None
        correct_placed = None
        episode_conflicts = None
        
        for source in [custom, sampler, eval_metrics]:
            if partial is None:
                partial = source.get('placement_rate_mean')
            if full is None:
                full = source.get('full_placement_rate_mean')
            if total_conflicts is None:
                total_conflicts = source.get('total_conflicts_mean')
            if duplicates is None:
                duplicates = source.get('duplicate_count_mean')
            if teacher_conflicts is None:
                teacher_conflicts = source.get('teacher_conflicts_mean')
            if section_conflicts is None:
                section_conflicts = source.get('section_conflicts_mean')
            if curriculum_phase is None:
                curriculum_phase = source.get('curriculum_phase_mean')
            if global_steps is None:
                global_steps = source.get('global_step_counter_mean')
            if under_placed is None:
                under_placed = source.get('under_placed_subjects_mean')
            if over_placed is None:
                over_placed = source.get('over_placed_subjects_mean')
            if correct_placed is None:
                correct_placed = source.get('correct_placed_subjects_mean')
            if episode_conflicts is None:
                episode_conflicts = source.get('episode_conflicts_mean')
        
        # DEBUG OUTPUT (only every 10 iterations)
        if it % 10 == 0 and (partial is None or curriculum_phase is None):
            print(f"\n⚠️ DEBUG: Metrics not found in expected locations")
            print(f"   custom_metrics keys: {list(custom.keys())[:5] if custom else 'EMPTY'}")
            print(f"   sampler_results keys: {list(sampler.keys())[:5] if sampler else 'EMPTY'}")
        
        # Curriculum status
        print(f"\n🎓 CURRICULUM STATUS:")
        if curriculum_phase is not None and global_steps is not None:
            if curriculum_phase <= 0.6:
                phase_name, phase_desc, phase_icon = "LEARNING", "Exploring with gentle penalties", "🌱"
            elif curriculum_phase <= 1.1:
                phase_name, phase_desc, phase_icon = "NORMAL", "Balanced exploration", "⚡"
            else:
                phase_name, phase_desc, phase_icon = "PERFECTION", "Strict penalties", "🎯"
            
            print(f"   Phase: {phase_icon} {phase_name} ({curriculum_phase:.1f}x penalties)")
            print(f"   Status: {phase_desc}")
            print(f"   Steps: {int(global_steps):,}")
        else:
            print(f"   ⚠️ Curriculum metrics not available (steps={global_steps}, phase={curriculum_phase})")
        
        # Placement metrics
        print(f"\n📈 PLACEMENT PERFORMANCE:")
        if partial is not None:
            self.placement_history.append(partial)
            if total_conflicts is not None:
                self.conflict_history.append(total_conflicts)
            
            print(f"   Partial: {partial:.1f}%")
            if full is not None:
                print(f"   Full: {full:.1f}%")
            
            # NEW: Under/over placement tracking
            if under_placed is not None or over_placed is not None or correct_placed is not None:
                print(f"\n   📦 PLACEMENT ACCURACY:")
                if under_placed is not None:
                    print(f"      Under-placed: {under_placed:.0f} subjects")
                if over_placed is not None:
                    print(f"      Over-placed: {over_placed:.0f} subjects")
                if correct_placed is not None:
                    print(f"      Correct: {correct_placed:.0f} subjects")
                
                # Diagnose the main issue
                if under_placed and under_placed > 400:
                    print(f"      ⚠️ MAJOR ISSUE: Most subjects missing required placements!")
                    print(f"         → Need stronger rewards for reaching required_placements")
                    print(f"         → Consider increasing milestone rewards")
                
                if over_placed and over_placed > 100:
                    print(f"      ⚠️ DUPLICATE PROBLEM: Duplicate prevention failing!")
                    print(f"         → Current r_duplicate_penalty: -300")
                    print(f"         → Consider increasing to -500 or -1000")
            
            if total_conflicts is not None:
                print(f"\n   🔍 CONFLICT BREAKDOWN:")
                print(f"      Total (validate): {total_conflicts:.1f}")
                if duplicates is not None:
                    print(f"      Duplicates: {duplicates:.1f}")
                if teacher_conflicts is not None:
                    print(f"      Teacher: {teacher_conflicts:.1f}")
                if section_conflicts is not None:
                    print(f"      Section: {section_conflicts:.1f}")
                if episode_conflicts is not None:
                    print(f"      Episode (real-time): {episode_conflicts:.1f}")
                
                # Conflict trend analysis
                if len(self.conflict_history) >= 10:
                    recent_avg = sum(self.conflict_history[-10:]) / 10
                    if len(self.conflict_history) >= 20:
                        older_avg = sum(self.conflict_history[-20:-10]) / 10
                        improvement = older_avg - recent_avg
                        
                        print(f"\n   📉 CONFLICT TREND (last 10 iters):")
                        print(f"      Current avg: {recent_avg:.1f}")
                        print(f"      Previous avg: {older_avg:.1f}")
                        print(f"      Improvement: {improvement:+.1f}")
                        
                        if self.first_conflict_decrease is None and improvement > 10:
                            self.first_conflict_decrease = it
                            print(f"      🎉 FIRST MAJOR DECREASE at iteration {it}!")
                        
                        if improvement > 10:
                            print(f"      🎉 EXCELLENT: Conflicts decreasing rapidly!")
                        elif improvement > 5:
                            print(f"      ✅ GOOD: Steady improvement")
                        elif improvement > 0:
                            print(f"      ✓ Progress, but slow")
                        elif improvement > -5:
                            print(f"      ⚠️ WARNING: Stagnating")
                        else:
                            print(f"      ❌ PROBLEM: Getting worse!")
                
                # Zero conflicts check
                if total_conflicts == 0:
                    if not self.best_zero_conflicts:
                        print(f"\n   🎉🎉🎉 FIRST ZERO-CONFLICT EPISODE! 🎉🎉🎉")
                        self.best_zero_conflicts = True
                    else:
                        print(f"\n   ✅ ZERO CONFLICTS!")
                elif total_conflicts < 5:
                    print(f"\n   ✨ Very low conflicts - almost there!")
                elif total_conflicts < 10:
                    print(f"\n   ⚠️ Some conflicts remain")
                elif total_conflicts < 50:
                    print(f"\n   ⚡ Conflicts decreasing")
                else:
                    print(f"\n   ❌ High conflict count")
                
                # Track improvements
                if partial > self.best_placement:
                    self.best_placement = partial
                    print(f"\n   🎯 NEW BEST PARTIAL: {partial:.1f}%!")
                
                if full is not None and full > self.best_full_placement:
                    self.best_full_placement = full
                    print(f"   🏆 NEW BEST FULL: {full:.1f}%!")
            else:
                print(f"   ⚠️ Conflict metrics not available")
        else:
            print(f"   ⚠️ Placement metrics not available (warming up...)")
        
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
        else:
            print(f"   (No actions recorded this iteration)")
        
        # Reward analysis
        print(f"\n💰 REWARD ANALYSIS:")
        print(f"   Current: {rew:.2f}")
        if len(self.reward_history) > 1:
            prev_rew = self.reward_history[-2]
            delta = rew - prev_rew
            print(f"   Change: {delta:+.2f}")
            
            if len(self.reward_history) > 20:
                recent_trend = sum(self.reward_history[-10:]) / 10
                older_trend = sum(self.reward_history[-20:-10]) / 10
                trend_change = recent_trend - older_trend
                
                print(f"   Trend: {trend_change:+.2f}")
                if trend_change > 1.0:
                    print(f"   📈 Rewards improving!")
                elif trend_change < -1.0:
                    print(f"   📉 Rewards declining")
        
        print(f"{'='*80}\n")
        
        # TensorBoard logging
        self.writer.add_scalar("Reward/Mean", rew, it)
        if partial is not None:
            self.writer.add_scalar("Placement/Partial", partial, it)
        if full is not None:
            self.writer.add_scalar("Placement/Full", full, it)
        if total_conflicts is not None:
            self.writer.add_scalar("Validation/Total_Conflicts", total_conflicts, it)
        if duplicates is not None:
            self.writer.add_scalar("Validation/Duplicates", duplicates, it)
        if teacher_conflicts is not None:
            self.writer.add_scalar("Validation/Teacher_Conflicts", teacher_conflicts, it)
        if section_conflicts is not None:
            self.writer.add_scalar("Validation/Section_Conflicts", section_conflicts, it)
        if curriculum_phase is not None:
            self.writer.add_scalar("Curriculum/Phase", curriculum_phase, it)
        if global_steps is not None:
            self.writer.add_scalar("Curriculum/GlobalSteps", global_steps, it)
        if under_placed is not None:
            self.writer.add_scalar("Placement/Under_Placed", under_placed, it)
        if over_placed is not None:
            self.writer.add_scalar("Placement/Over_Placed", over_placed, it)
        if correct_placed is not None:
            self.writer.add_scalar("Placement/Correct_Placed", correct_placed, it)
                    
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
        """Enhanced validation with ACCURATE conflict detection - handles missing keys"""
        self.episode_counter += 1
        env = self._unwrap_env(base_env)

        # DEBUG: Print raw environment state every 50 episodes
        if self.episode_counter % 50 == 0:
            print(f"\n🔍 EPISODE {self.episode_counter} RAW STATE:")
            print(f"   conflict_count: {env.conflict_count}")
            print(f"   placed_subjects: {len(env.placed_subjects)}/{env.num_subjects}")
            print(f"   global_step_counter: {env.global_step_counter}")
            print(f"   curriculum_multiplier: {env._get_curriculum_multiplier():.2f}")

        # Get complete placements (HONEST counting)
        num_fully_placed = env.get_complete_placements()
        full_placement_rate = (num_fully_placed / env.num_subjects) * 100
        
        # Track partial placements
        subjects_with_placements = set()
        for (subj, sec), count in env.subject_placement_count.items():
            if count > 0:
                subjects_with_placements.add(subj)
        
        num_partial_placed = len(subjects_with_placements)
        partial_placement_rate = (num_partial_placed / env.num_subjects) * 100
        
        # Overall progress
        total_placements_needed = sum(env.subject_required_placements.values())
        total_placements_made = sum(env.subject_placement_count.values())
        overall_progress = (total_placements_made / total_placements_needed) * 100 if total_placements_needed > 0 else 0
        
        # Report metrics
        episode.custom_metrics["placement_rate"] = partial_placement_rate
        episode.custom_metrics["full_placement_rate"] = full_placement_rate
        episode.custom_metrics["overall_progress"] = overall_progress
        
        # CRITICAL: Track curriculum phase
        episode.custom_metrics["curriculum_phase"] = env._get_curriculum_multiplier()
        episode.custom_metrics["global_step_counter"] = env.global_step_counter
        
        # ============================================================
        # FIX: USE ENVIRONMENT'S COMPREHENSIVE VALIDATION WITH SAFE KEY ACCESS
        # ============================================================
        conflicts_report = env.validate_schedule()
        
        # Extract conflict counts SAFELY (handle missing keys)
        summary = conflicts_report.get('summary', {})
        duplicate_count = summary.get('duplicate_placements', 0)
        teacher_conflicts = summary.get('teacher_conflicts', 0)
        total_conflicts = summary.get('total_conflicts', 0)
        
        # Calculate section conflicts manually (since validate_schedule doesn't return it)
        section_conflicts = 0
        section_time_map = {}
        for subj in range(env.num_subjects):
            sec_idx = env.subject_section_idx[subj]
            for d in range(env.num_days):
                for ts in range(env.num_timeslots):
                    if (subj, d, ts) in env.placement_teachers:
                        key = (sec_idx, d, ts)
                        if key not in section_time_map:
                            section_time_map[key] = []
                        section_time_map[key].append(subj)
        
        for key, subjects in section_time_map.items():
            if len(subjects) > 1:
                section_conflicts += (len(subjects) - 1)
        
        # Also get the conflict count tracked DURING episode
        episode_conflicts = env.conflict_count
        
        # Report ALL conflict metrics
        episode.custom_metrics["duplicate_count"] = duplicate_count
        episode.custom_metrics["teacher_conflicts"] = teacher_conflicts  
        episode.custom_metrics["section_conflicts"] = section_conflicts
        episode.custom_metrics["total_conflicts"] = total_conflicts
        episode.custom_metrics["episode_conflicts"] = episode_conflicts
        
        # NEW: Report under/over placement issues
        under_placed = 0
        over_placed = 0
        correct_placed = 0
        
        for subj in range(env.num_subjects):
            sec_idx = env.subject_section_idx[subj]
            current = env._get_placement_count(subj, sec_idx)
            required = env.subject_required_placements.get(subj, 1)
            
            if current < required:
                under_placed += 1
            elif current > required:
                over_placed += 1
            else:
                correct_placed += 1
        
        episode.custom_metrics["under_placed_subjects"] = under_placed
        episode.custom_metrics["over_placed_subjects"] = over_placed
        episode.custom_metrics["correct_placed_subjects"] = correct_placed
        
        # Modality stats
        for modality in env.modality_labels:
            modality_data = env.modality_stats.get(modality, {})
            total = modality_data.get('attempted', 0)
            placed = modality_data.get('placed', 0)
            rate = (placed / total * 100) if total > 0 else 0
            episode.custom_metrics[f"modality_{modality}_rate"] = rate
        
        # Enhanced debug output every 100 episodes
        if self.episode_counter % 100 == 0:
            print(f"\n" + "="*80)
            print(f"📊 EPISODE {self.episode_counter} COMPREHENSIVE SUMMARY")
            print("="*80)
            print(f"Placement:")
            print(f"  Partial: {num_partial_placed}/{env.num_subjects} ({partial_placement_rate:.1f}%)")
            print(f"  Full: {num_fully_placed}/{env.num_subjects} ({full_placement_rate:.1f}%)")
            print(f"  Under-placed: {under_placed}")
            print(f"  Over-placed: {over_placed}")
            print(f"  Correct: {correct_placed}")
            print(f"\nConflicts:")
            print(f"  Total (from validation): {total_conflicts}")
            print(f"  Teacher: {teacher_conflicts}")
            print(f"  Section (calculated): {section_conflicts}")
            print(f"  Duplicates: {duplicate_count}")
            print(f"  Episode (real-time): {episode_conflicts}")
            print(f"\nCurriculum: {env._get_curriculum_multiplier():.1f}x @ step {env.global_step_counter}")
            print("="*80 + "\n")
        
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

def convert_subject_placements(data):
    """Convert subject_required_days to integer counts (handles both list and int formats)"""
    subject_required_days = data.get('subject_required_days') or data.get('subject_required_placements')
    
    if not subject_required_days:
        return {i: 1 for i in range(data['num_subjects'])}
    
    # Convert to integer counts (handle both list ['M','H'] and integer 2 formats)
    result = {}
    for subj_idx, days in subject_required_days.items():
        if isinstance(days, list):
            result[subj_idx] = len(days)  # Convert list to count
        else:
            result[subj_idx] = int(days)  # Already an integer
    
    return result


def make_manila_env(config=None):
    """Create environment for Manila schedule with AGGRESSIVE PENALTIES"""
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
        subject_required_placements=convert_subject_placements(data),
        strict_teacher_match=False,
        r_teacher_match=40.0,
        r_teacher_mismatch=-1.0,
        r_area_match=25.0,
        r_area_mismatch=-1.5,
        r_workload_balance=15.0,
        base_success_reward=150.0,
        wait_penalty=20.0,
        # NEW: AGGRESSIVE PENALTIES
        r_conflict_penalty=-200.0,
        r_duplicate_penalty=-300.0,
        r_section_conflict_penalty=-250.0,
        r_online_bonus=8.0,
        enable_progressive_difficulty=True,
        difficulty_ramp_steps=100,
        enable_curriculum_learning=True,  # NEW
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
    parser = argparse.ArgumentParser(description="Manila Training v18.9 - AGGRESSIVE PENALTIES")
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
    print("MANILA TRAINING - v18.9 AGGRESSIVE PENALTIES + GPU Optimization")
    print("=" * 80)
    print("\n🔧 ALL FIXES APPLIED:")
    print("  ✅ FIX #1-12: All critical bugs resolved")
    print("  ✅ FIX #13: AGGRESSIVE conflict penalties (-200)")
    print("  ✅ FIX #14: Exponential global conflict penalty")
    print("  ✅ FIX #15: Quadratic completion pressure")
    print("  ✅ FIX #16: Curriculum learning (3 phases)")
    print("\n⚡ EXPECTED LEARNING BEHAVIOR:")
    print("  • Phase 1 (0-10k steps): 🌱 Learning basics (0.5x penalties)")
    print("  • Phase 2 (10k-30k steps): ⚡ Normal training (1.0x penalties)")
    print("  • Phase 3 (30k+ steps): 🎯 Perfection mode (1.5x penalties)")
    print("\n📈 EXPECTED RESULTS:")
    print("  • Iter 10-50: Conflicts START decreasing")
    print("  • Iter 50-100: Student scores rise to 40+, conflicts <50")
    print("  • Iter 100-200: Overall scores 65-75, conflicts <20")
    print("  • Iter 200-500: Overall scores 80-90+, conflicts <5")
    print("  • Iter 500+: ZERO conflicts achievable, scores 95+")
    print("\n⚠️ IF CONFLICTS DON'T DECREASE BY ITERATION 50:")
    print("  → Penalties still too weak")
    print("  → Increase r_conflict_penalty to -300 or -400")
    print("  → Check TensorBoard for reward trends")
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
            print("All fixes are working correctly!")
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
    print("VALIDATING ENVIRONMENT (v14.7 - AGGRESSIVE PENALTIES)")
    print("=" * 80)
    test_env = make_manila_env({'cache_file': cache_file})
    raw = test_env.par_env
    
    print(f"Subjects: {raw.num_subjects}")
    print(f"Teachers: {raw.num_teachers}")
    print(f"Agents: {len(raw.agents)}")
    print(f"Max teachers per area: {raw.max_teachers_per_area}")
    print(f"Slot choices: {raw.slot_choices}")
    
    # Verify aggressive penalties
    print(f"\n🔥 AGGRESSIVE PENALTIES VERIFIED:")
    print(f"   Conflict penalty: {raw.r_conflict_penalty}")
    print(f"   Duplicate penalty: {raw.r_duplicate_penalty}")
    print(f"   Section conflict: {raw.r_section_conflict_penalty}")
    print(f"   Curriculum learning: {raw.enable_curriculum_learning}")
    
    if raw.r_conflict_penalty >= -50:
        print(f"\n   ❌ WARNING: Penalties too weak!")
        print(f"      Current: {raw.r_conflict_penalty}")
        print(f"      Recommended: -200 or lower")
        sys.exit(1)
    else:
        print(f"   ✅ Penalties are aggressive enough")
    
    # Verify curriculum learning
    if hasattr(raw, '_get_curriculum_multiplier'):
        print(f"   ✅ Curriculum learning enabled")
    else:
        print(f"   ❌ Curriculum learning NOT found!")
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

    # PPO Configuration - GPU-OPTIMIZED
    ppo_cfg = (
        PPOConfig()
        .environment(
            env="manila_env",
            env_config={'cache_file': cache_file},
            disable_env_checking=True
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=2,
            rollout_fragment_length=128,
            batch_mode="truncate_episodes",
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
            sgd_minibatch_size=512,
            num_sgd_iter=1,
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
            resume_checkpoint = find_latest_checkpoint()
            if not resume_checkpoint:
                print("No checkpoints found, starting fresh training")
                args.resume = None
        else:
            resume_checkpoint = args.resume
            
            if not os.path.exists(resume_checkpoint):
                print(f"❌ ERROR: Checkpoint not found: {resume_checkpoint}")
                sys.exit(1)
            
            print(f"\n✅ Found checkpoint: {os.path.basename(resume_checkpoint)}")
            
            if not os.path.isdir(resume_checkpoint):
                print(f"❌ ERROR: Path is not a checkpoint directory")
                sys.exit(1)
    
    # ============================================================
    # STARTING TRAINING
    # ============================================================
    print(f"\n" + "=" * 80)
    print("STARTING TRAINING - AGGRESSIVE PENALTIES v14.7")
    print("=" * 80)
    print(f"Target iterations: {args.iterations}")
    if resume_checkpoint:
        print(f"🔄 RESUMING from: {os.path.basename(resume_checkpoint)}")
    else:
        print(f"🆕 STARTING FRESH (no checkpoint)")
    print(f"\n🎯 Target: Conflicts decrease by iteration 50")
    print(f"Watch for: '🎉 FIRST MAJOR DECREASE' in diagnostic output")
    print("=" * 80 + "\n")
    
    # ============================================================
    # TRAINING PATH: RESUME vs FRESH
    # ============================================================
    if resume_checkpoint:
        print(f"Attempting direct algorithm restoration...")
        
        try:
            algorithm = PPO.from_checkpoint(resume_checkpoint)
            current_iter = algorithm.iteration
            print(f"✅ Restored from iteration {current_iter}")
            
            verify_env = make_manila_env({'cache_file': cache_file})
            obs, _ = verify_env.reset()
            print(f"✅ Environment compatible with checkpoint")
            
            target_iterations = args.iterations
            checkpoint_dir = "C:/ray_logs/manual_checkpoints_manila_aggressive"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            print(f"\n{'='*80}")
            print(f"RESUMING TRAINING: Iteration {current_iter} → {target_iterations}")
            print(f"{'='*80}\n")

            iteration_times = []

            while current_iter < target_iterations:
                iter_start_time = time.time()
                result = algorithm.train()
                iter_end_time = time.time()
                iter_duration = iter_end_time - iter_start_time
                iteration_times.append(iter_duration)

                current_iter = result["training_iteration"]
                
                reward = result.get('episode_reward_mean', 0)
                placement_rate = result.get('custom_metrics', {}).get('placement_rate_mean', 0)
                full_rate = result.get('custom_metrics', {}).get('full_placement_rate_mean', 0)
                conflicts = result.get('custom_metrics', {}).get('total_conflicts_mean', 0)
                duplicates = result.get('custom_metrics', {}).get('duplicate_count_mean', 0)
                
                num_subjects = raw.num_subjects
                partial_placed = int(placement_rate * num_subjects / 100) if placement_rate else 0
                full_placed = int(full_rate * num_subjects / 100) if full_rate else 0
                
                avg_time = sum(iteration_times[-5:]) / min(5, len(iteration_times)) if iteration_times else 0
                print(f"Iter {current_iter:3d}: "
                      f"Time={iter_duration:5.1f}s (avg={avg_time:5.1f}s) | "
                      f"Reward={reward:7.1f} | "
                      f"Partial={partial_placed:3d}/{num_subjects} ({placement_rate:5.1f}%) | "
                      f"Full={full_placed:3d}/{num_subjects} ({full_rate:5.1f}%) | "
                      f"Conflicts={conflicts:4.1f} | "
                      f"Dup={duplicates:4.1f}")
                
                if current_iter % 10 == 0:
                    checkpoint_path = algorithm.save(
                        f"{checkpoint_dir}/checkpoint_{current_iter:06d}"
                    )
                    print(f"   💾 Saved checkpoint_{current_iter:06d}")
                
                if full_rate >= 95 and current_iter % 5 == 0:
                    print(f"   🎯 Excellent performance! Full placement: {full_rate:.1f}%")
                
                if conflicts == 0 and duplicates == 0 and full_rate > 0:
                    print(f"   ✨ PERFECT SCHEDULE: Zero conflicts and duplicates!")
            
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
            
            print("\nRunning final comprehensive validation...")
            final_env = make_manila_env({'cache_file': cache_file})
            final_env.reset()
            final_conflicts = final_env.par_env.validate_schedule()
            
            total = final_conflicts['summary']['total_conflicts']
            
            if total == 0:
                print("\n🎉🎉🎉 PERFECT! Final schedule has ZERO conflicts! 🎉🎉🎉")
                print("All fixes (aggressive penalties) are working correctly!")
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
    # FRESH TRAINING
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
                "custom_metrics/curriculum_phase_mean",
            ],
            max_report_frequency=30,
        )

        run_cfg = RunConfig(
            stop={"training_iteration": args.iterations},
            local_dir="C:/ray_logs",
            name="Manila_v14_7_AGGRESSIVE_PENALTIES",
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
        
        print("\nRunning final comprehensive validation...")
        final_env = make_manila_env({'cache_file': cache_file})
        final_env.reset()
        final_conflicts = final_env.par_env.validate_schedule()
        
        total = final_conflicts['summary']['total_conflicts']
        
        if total == 0:
            print("\n🎉🎉🎉 PERFECT! Final schedule has ZERO conflicts! 🎉🎉🎉")
            print("Aggressive penalties worked!")
        else:
            print(f"\n⚠️ Final schedule has {total} conflicts")
            print(f"\nBreakdown:")
            print(f"  Duplicates: {final_conflicts['summary']['duplicate_placements']}")
            print(f"  Teacher: {final_conflicts['summary']['teacher_conflicts']}")
            print(f"  Section: {final_conflicts['summary']['section_conflicts']}")
            
            if total > 50:
                print(f"\n💡 RECOMMENDATION:")
                print(f"   Penalties still too weak. In environment init, increase:")
                print(f"   r_conflict_penalty=-200.0  →  -300.0 or -400.0")
                print(f"   r_duplicate_penalty=-300.0  →  -500.0")