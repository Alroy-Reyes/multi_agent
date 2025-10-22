"""
Smart random schedule generator - NO MODEL NEEDED
Tests if environment can actually place subjects successfully

This uses random valid actions instead of a trained model
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.timetabling_env import ParallelTimetablingEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

def load_manila_data():
    """Load Manila cache"""
    cache_file = "cached_environment_data_MANILA_MODALITY.pkl"
    
    if not os.path.exists(cache_file):
        print("ERROR: Cache file not found")
        return None
    
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✓ Loaded cache: {data['num_subjects']} subjects")
    return data

def create_environment(data):
    """Create environment with proper configuration"""
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
    )
    
    # Add metadata
    env.subject_names = data["subject_names"]
    env.teacher_labels = data.get("teacher_id_to_name", [])
    env.day_labels = data.get("day_labels", ['M', 'T', 'W', 'H', 'F'])
    env.timeslot_labels = data.get("timeslot_labels", [])
    env.room_to_campus = data.get("room_to_campus", {})
    env.section_labels = data["section_labels"]
    env.subject_modalities = data["subject_modalities"]
    
    return ParallelPettingZooEnv(env)

def smart_random_action(obs, agent_id=None):
    """
    Choose a random VALID action based on action masks
    Much smarter than pure random!
    """
    if not isinstance(obs, dict):
        # No masks, just random
        return np.array([0, 0], dtype=np.int64)
    
    teacher_mask = obs.get('teacher_mask')
    slot_mask = obs.get('slot_mask')
    
    if teacher_mask is None or slot_mask is None:
        return np.array([0, 0], dtype=np.int64)
    
    # Find valid actions
    valid_teachers = np.where(teacher_mask > 0.5)[0]
    valid_slots = np.where(slot_mask > 0.5)[0]
    
    # Exclude "wait" actions (last indices)
    if len(valid_teachers) > 1:
        valid_teachers = valid_teachers[:-1]  # Remove wait action
    if len(valid_slots) > 1:
        valid_slots = valid_slots[:-1]  # Remove wait action
    
    # If we have valid actions, choose randomly
    if len(valid_teachers) > 0 and len(valid_slots) > 0:
        teacher_idx = np.random.choice(valid_teachers)
        slot_idx = np.random.choice(valid_slots)
        return np.array([teacher_idx, slot_idx], dtype=np.int64)
    
    # Otherwise wait
    return np.array([len(teacher_mask)-1, len(slot_mask)-1], dtype=np.int64)

def generate_schedule_random(max_steps=500):
    """Generate schedule using smart random policy"""
    
    print("\n" + "="*80)
    print("SMART RANDOM SCHEDULE GENERATION")
    print("(No trained model needed - tests environment)")
    print("="*80)
    
    # Load data and create environment
    data = load_manila_data()
    if data is None:
        return None, None
    
    env = create_environment(data)
    raw_env = env.par_env
    
    print(f"\nEnvironment Info:")
    print(f"  Subjects: {raw_env.num_subjects}")
    print(f"  Teachers: {raw_env.num_teachers}")
    print(f"  Max steps: {max_steps}")
    
    # Reset
    obs_dict, _ = env.reset()
    agents = list(obs_dict.keys())
    
    done = {agent: False for agent in agents}
    step = 0
    
    print("\nGenerating with smart random actions...")
    
    while step < max_steps and not all(done.values()):
        actions = {}
        
        # Smart random actions for each agent
        for agent in agents:
            if not done.get(agent, False):
                actions[agent] = smart_random_action(obs_dict[agent], agent)
        
        # Step
        obs_dict, rewards, dones, truncs, infos = env.step(actions)
        
        # Update done status
        for agent in agents:
            done[agent] = dones.get(agent, False) or truncs.get(agent, False)
        
        step += 1
        
        # Progress
        if step % 50 == 0:
            placed = len(raw_env.placed_subjects)
            print(f"  Step {step}: {placed}/{raw_env.num_subjects} subjects placed ({placed/raw_env.num_subjects*100:.1f}%)")
    
    # Final stats
    placed = len(raw_env.placed_subjects)
    completion_rate = (placed / raw_env.num_subjects) * 100
    
    print(f"\n{'='*80}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Steps: {step}")
    print(f"Subjects placed: {placed}/{raw_env.num_subjects} ({completion_rate:.1f}%)")
    print(f"Conflicts: {raw_env.conflict_count}")
    
    # Fail stats
    print(f"\nTop constraint violations:")
    sorted_fails = sorted(raw_env.fail_stats.items(), key=lambda x: x[1], reverse=True)
    for key, value in sorted_fails[:10]:
        if value > 0:
            print(f"  {key}: {value}")
    
    return env, {
        'placed': placed,
        'total': raw_env.num_subjects,
        'completion_rate': completion_rate,
        'steps': step,
        'conflicts': raw_env.conflict_count,
        'fail_stats': dict(raw_env.fail_stats),
    }

def extract_schedule(env):
    """Extract schedule from environment"""
    raw_env = env.par_env
    schedule_data = []
    
    for building in raw_env.building_keys:
        schedule = raw_env.buildings_room_schedule[building]
        num_rooms = schedule.shape[0]
        
        for room_idx in range(num_rooms):
            room_code = raw_env.buildings_room_info[building][room_idx]
            
            for day_idx in range(raw_env.num_days):
                for ts_idx in range(raw_env.num_timeslots):
                    subject_idx = schedule[room_idx, day_idx, ts_idx]
                    
                    if subject_idx >= 0:
                        subject_name = raw_env.subject_names[subject_idx]
                        teacher_idx = raw_env.subject_assignments[subject_idx]
                        
                        # Get teacher name
                        if hasattr(raw_env, 'teacher_labels') and teacher_idx < len(raw_env.teacher_labels):
                            teacher_name = raw_env.teacher_labels[teacher_idx]
                        else:
                            teacher_name = f"Teacher_{teacher_idx}"
                        
                        day_label = raw_env.day_labels[day_idx] if day_idx < len(raw_env.day_labels) else f"Day_{day_idx}"
                        timeslot_label = raw_env.timeslot_labels[ts_idx] if ts_idx < len(raw_env.timeslot_labels) else f"Slot_{ts_idx}"
                        
                        section_idx = raw_env.subject_section_idx[subject_idx]
                        section_label = raw_env.section_labels[section_idx]
                        
                        modality = raw_env.subject_modalities[subject_idx]
                        
                        schedule_data.append({
                            'Subject': subject_name,
                            'Faculty': teacher_name,
                            'Section': section_label,
                            'Days': day_label,
                            'Time': timeslot_label,
                            'Room': room_code,
                            'Modality': modality,
                        })
    
    return schedule_data

def main():
    """Main execution"""
    
    # Generate schedule
    env, stats = generate_schedule_random(max_steps=500)
    
    if env is None:
        print("ERROR: Failed to create environment")
        return
    
    # Extract schedule
    schedule_data = extract_schedule(env)
    
    if not schedule_data:
        print("\nWARNING: No schedule data generated!")
        print("\nThis means:")
        print("  • Environment constraints are too restrictive, OR")
        print("  • Smart random policy isn't smart enough")
        return
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("smart_random_output") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export to CSV
    df = pd.DataFrame(schedule_data)
    csv_path = output_dir / "schedule_smart_random.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"EXPORT COMPLETE")
    print(f"{'='*80}")
    print(f"Output: {csv_path}")
    print(f"Entries: {len(df)}")
    print(f"\nCompletion rate: {stats['completion_rate']:.1f}%")
    
    if stats['completion_rate'] > 50:
        print("\n✅ Good completion! Environment is working.")
        print("   → Your trained model checkpoint is just too early/broken.")
        print("   → Solution: Use a later checkpoint (50+) or retrain.")
    elif stats['completion_rate'] > 20:
        print("\n⚠️ Moderate completion. Environment works but has issues.")
        print("   → Some constraints might be too restrictive.")
        print("   → A trained model should do MUCH better than this.")
    else:
        print("\n❌ Low completion. Environment or constraints have problems.")
        print("   → Check constraint definitions in cache file.")
        print("   → Some subjects might be impossible to place.")
    
    print(f"\n💡 Comparison:")
    print(f"   Smart random: {stats['completion_rate']:.1f}%")
    print(f"   Your model:   0.0%")
    print(f"   → Model is WORSE than random! Needs retraining.")

if __name__ == "__main__":
    main()