"""
Analyze which subjects cannot be placed
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ray import init, shutdown
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from envs.timetabling_env import ParallelTimetablingEnv
import pickle

def load_cached_data(cache_file='cached_environment_data_MANILA_MODALITY.pkl'):
    with open(cache_file, 'rb') as f:
        return pickle.load(f)

def make_env(cache_file=None):
    """Create environment - simplified version"""
    data = load_cached_data(cache_file)
    
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
        enable_repair_pass=False,
    )
    
    return ParallelPettingZooEnv(env)

def analyze_unplaced_subjects(checkpoint_path):
    """Run episode and analyze what didn't get placed"""
    
    print("\n" + "="*80)
    print("ANALYZING UNPLACED SUBJECTS")
    print("="*80)
    
    init(ignore_reinit_error=True, include_dashboard=False)
    
    from ray.rllib.models import ModelCatalog
    from train_ppo import ImprovedSahaMaskedTwoHead  # Import your model
    
    ModelCatalog.register_custom_model("improved_saha_masked", ImprovedSahaMaskedTwoHead)
    register_env("manila_env", lambda cfg: make_env())
    
    # Load checkpoint
    algo = PPO.from_checkpoint(checkpoint_path)
    
    # Create environment
    env = make_env()
    raw_env = env.par_env
    
    # Run one episode
    obs_dict, _ = env.reset()
    agents = raw_env.agents
    done = {agent: False for agent in agents}
    step = 0
    
    while step < 400 and not all(done.values()):
        actions = {}
        for agent in agents:
            if not done.get(agent, False):
                action = algo.compute_single_action(
                    obs_dict[agent],
                    policy_id="saha_policy",
                    explore=False  # Deterministic
                )
                actions[agent] = action
        
        obs_dict, rewards, dones, truncs, infos = env.step(actions)
        
        for agent in agents:
            done[agent] = dones.get(agent, False) or truncs.get(agent, False)
        
        step += 1
    
    # Analyze unplaced subjects
    print(f"\n📊 EPISODE RESULTS:")
    print(f"Steps: {step}")
    print(f"Placed: {len(raw_env.placed_subjects)}/{raw_env.num_subjects} ({len(raw_env.placed_subjects)/raw_env.num_subjects*100:.1f}%)")
    
    unplaced = []
    for s in range(raw_env.num_subjects):
        if s not in raw_env.placed_subjects:
            modality = raw_env.subject_modalities[s]
            area = raw_env.subject_areas[s]
            allowed_rooms = raw_env.subject_allowed_rooms[s]
            allowed_timeslots = raw_env.subject_allowed_timeslots[s]
            required = raw_env.subject_required_placements.get(s, 1)
            sec_idx = raw_env.subject_section_idx[s]
            
            # Check constraints
            issues = []
            
            if not allowed_rooms:
                issues.append("No specific room constraints")
            else:
                # Check modality-room match
                if modality == 'Online':
                    has_virtual = any(raw_env._is_virtual_room(r) for r in allowed_rooms)
                    if not has_virtual:
                        issues.append("Online but no virtual rooms")
                elif modality == 'Face-to-Face':
                    has_physical = any(not raw_env._is_virtual_room(r) for r in allowed_rooms)
                    if not has_physical:
                        issues.append("F2F but no physical rooms")
            
            if len(allowed_timeslots) < required:
                issues.append(f"Only {len(allowed_timeslots)} timeslots but needs {required}")
            
            # Check if section was heavily scheduled
            section_usage = 0
            skey = f"section_{sec_idx}"
            if skey in raw_env.section_schedules:
                section_usage = raw_env.section_schedules[skey].sum()
            
            if section_usage > raw_env.num_days * raw_env.num_timeslots * 0.8:
                issues.append(f"Section heavily scheduled ({section_usage} slots)")
            
            unplaced.append({
                'idx': s,
                'name': raw_env.subject_codes[s],
                'modality': modality,
                'area': area,
                'rooms': len(allowed_rooms),
                'timeslots': len(allowed_timeslots),
                'required': required,
                'section': raw_env.section_labels[sec_idx],
                'section_usage': section_usage,
                'issues': issues
            })
    
    print(f"\n{'='*80}")
    print(f"UNPLACED SUBJECTS BREAKDOWN")
    print(f"{'='*80}")
    print(f"Total unplaced: {len(unplaced)}")
    
    # Group by modality
    by_modality = {}
    for subj in unplaced:
        mod = subj['modality']
        if mod not in by_modality:
            by_modality[mod] = []
        by_modality[mod].append(subj)
    
    print(f"\nBy Modality:")
    for mod, subjects in by_modality.items():
        print(f"  {mod}: {len(subjects)}")
    
    # Group by area
    by_area = {}
    for subj in unplaced:
        area = subj['area']
        if area not in by_area:
            by_area[area] = []
        by_area[area].append(subj)
    
    print(f"\nBy Area:")
    for area, subjects in sorted(by_area.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"  {area}: {len(subjects)}")
    
    # Show subjects with issues
    subjects_with_issues = [s for s in unplaced if s['issues']]
    if subjects_with_issues:
        print(f"\n{'='*80}")
        print(f"SUBJECTS WITH CONSTRAINT ISSUES ({len(subjects_with_issues)})")
        print(f"{'='*80}")
        for i, subj in enumerate(subjects_with_issues[:20]):
            print(f"\n{i+1}. {subj['name']}")
            print(f"   Modality: {subj['modality']}, Area: {subj['area']}")
            print(f"   Allowed rooms: {subj['rooms']}, timeslots: {subj['timeslots']}")
            print(f"   Required placements: {subj['required']}")
            print(f"   Section: {subj['section']} (usage: {subj['section_usage']} slots)")
            print(f"   Issues:")
            for issue in subj['issues']:
                print(f"     • {issue}")
    
    # Show all unplaced (grouped)
    print(f"\n{'='*80}")
    print(f"ALL UNPLACED SUBJECTS")
    print(f"{'='*80}")
    for i, subj in enumerate(unplaced):
        print(f"{i+1:3d}. {subj['name'][:60]:<60} | {subj['modality'][:10]:<10} | {subj['area'][:15]:<15}")
    
    shutdown()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    args = parser.parse_args()
    
    analyze_unplaced_subjects(args.checkpoint)