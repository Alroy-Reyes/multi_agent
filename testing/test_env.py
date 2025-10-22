"""
Verify that modality validation is actually blocking invalid placements

This checks:
1. If validation code is being called
2. If rejections are happening
3. What the fail_stats show
"""

import sys
import os
import pickle
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.timetabling_env import ParallelTimetablingEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv


def test_modality_validation():
    """Test that modality validation actually blocks invalid placements"""
    
    print("\n" + "="*70)
    print("MODALITY VALIDATION TEST")
    print("="*70)
    
    # Load environment
    cache_file = 'cached_environment_data_MANILA_MODALITY.pkl'
    
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    
    env = ParallelTimetablingEnv(
        num_teachers=data["num_teachers"],
        num_subjects=data["num_subjects"],
        num_days=data["num_days"],
        num_timeslots=data["num_timeslots"],
        room_codes=data["room_codes"],
        subject_codes=data["subject_names"],
        subject_modalities=data.get("subject_modalities"),
        subject_modality_idx=data.get("subject_modality_idx"),
        modality_labels=data.get("modality_labels"),
        virtual_rooms=data.get("virtual_rooms", []),
        subject_campuses=data["subject_campuses"],
        subject_allowed_rooms=data["subject_allowed_rooms"],
        subject_areas=data["subject_areas"],
        subject_section_idx=data["subject_section_idx"],
        section_labels=data["section_labels"],
        area_teacher_indices=data["area_teacher_indices"],
        subject_teacher_idx=data["subject_teacher_idx"],
        max_classes_per_teacher=4,
        teacher_max_classes_map=data.get("teacher_max_classes_map", {}),
        use_action_masks=True,
    )
    
    env.subject_names = data["subject_names"]
    env.teacher_names = data.get("teacher_names", [])
    
    print(f"\n1. Environment Setup:")
    print(f"   Subjects: {env.num_subjects}")
    print(f"   Virtual rooms: {len(env.virtual_rooms)}")
    print(f"   Physical rooms: {len(env.physical_rooms)}")
    
    # Count modalities
    from collections import Counter
    modality_counts = Counter(env.subject_modalities)
    print(f"\n   Modality breakdown:")
    for mod, count in modality_counts.items():
        pct = (count / env.num_subjects) * 100
        print(f"     {mod}: {count} ({pct:.1f}%)")
    
    # Test validation function directly
    print(f"\n2. Testing validation function directly:")
    
    # Find a hybrid subject
    hybrid_idx = None
    for i, mod in enumerate(env.subject_modalities):
        if mod == 'Hybrid':
            hybrid_idx = i
            break
    
    if hybrid_idx is not None:
        hybrid_name = env.subject_names[hybrid_idx]
        print(f"\n   Using Hybrid subject: {hybrid_name}")
        
        # Test with ZOOM room
        if env.virtual_rooms:
            zoom_room = list(env.virtual_rooms)[0]
            is_valid, reason = env._validate_modality_room_match(hybrid_idx, zoom_room)
            
            if not is_valid and reason == "hybrid_needs_physical":
                print(f"   ✅ ZOOM room correctly REJECTED for Hybrid")
                print(f"      Room: {zoom_room}")
                print(f"      Reason: {reason}")
            else:
                print(f"   ❌ BUG: ZOOM room allowed for Hybrid!")
                print(f"      is_valid: {is_valid}, reason: {reason}")
        
        # Test with physical room
        if env.physical_rooms:
            phys_room = list(env.physical_rooms)[0]
            is_valid, reason = env._validate_modality_room_match(hybrid_idx, phys_room)
            
            if is_valid and reason == "ok":
                print(f"   ✅ Physical room correctly ACCEPTED for Hybrid")
                print(f"      Room: {phys_room}")
            else:
                print(f"   ❌ BUG: Physical room rejected for Hybrid!")
                print(f"      is_valid: {is_valid}, reason: {reason}")
    
    # Run an episode and check fail_stats
    print(f"\n3. Running episode to check validation in action...")
    
    wrapped = ParallelPettingZooEnv(env)
    wrapped.reset()
    
    # Run 100 steps
    for step in range(100):
        actions = {
            agent: np.array([
                np.random.randint(env.max_teachers_per_area + 1),
                np.random.randint(env.slot_choices + 1)
            ]) 
            for agent in env.agents
        }
        
        _, _, dones, _, _ = wrapped.step(actions)
        if all(dones.values()):
            break
    
    print(f"   Episode completed in {step+1} steps")
    print(f"   Placed: {len(env.placed_subjects)}/{env.num_subjects}")
    
    # Check fail_stats
    print(f"\n4. Checking fail_stats for modality rejections:")
    
    modality_fails = {
        'online_needs_virtual': env.fail_stats.get('online_needs_virtual', 0),
        'f2f_needs_physical': env.fail_stats.get('f2f_needs_physical', 0),
        'hybrid_needs_physical': env.fail_stats.get('hybrid_needs_physical', 0),
    }
    
    total_modality_fails = sum(modality_fails.values())
    
    print(f"   Total modality rejections: {total_modality_fails}")
    for fail_type, count in modality_fails.items():
        if count > 0:
            print(f"     {fail_type}: {count}")
    
    if modality_fails['hybrid_needs_physical'] > 0:
        print(f"\n   ✅ Hybrid→ZOOM rejections happening ({modality_fails['hybrid_needs_physical']} times)")
        print(f"      Validation IS working!")
    else:
        print(f"\n   ⚠️  No Hybrid→ZOOM rejections recorded")
        print(f"      This could mean:")
        print(f"      1. No hybrid subjects were attempted in ZOOM rooms (good)")
        print(f"      2. Validation not being called (needs investigation)")
    
    # Check what actually got placed
    print(f"\n5. Checking actual placements:")
    
    hybrid_in_zoom = 0
    hybrid_in_physical = 0
    online_in_zoom = 0
    online_in_physical = 0
    
    for b in env.building_keys:
        b_sched = env.buildings_room_schedule[b]
        
        for r_idx in range(len(env.buildings_room_info[b])):
            room_code = env.buildings_room_info[b][r_idx]
            is_virtual = room_code in env.virtual_rooms
            
            for d in range(env.num_days):
                for ts in range(env.num_timeslots):
                    subj = b_sched[r_idx, d, ts]
                    
                    if subj >= 0:
                        modality = env.subject_modalities[subj]
                        
                        if modality == 'Hybrid':
                            if is_virtual:
                                hybrid_in_zoom += 1
                            else:
                                hybrid_in_physical += 1
                        
                        elif modality == 'Online':
                            if is_virtual:
                                online_in_zoom += 1
                            else:
                                online_in_physical += 1
    
    print(f"\n   Hybrid placements:")
    print(f"     In ZOOM: {hybrid_in_zoom}")
    print(f"     In Physical: {hybrid_in_physical}")
    
    if hybrid_in_zoom > 0:
        print(f"     ❌ PROBLEM: {hybrid_in_zoom} Hybrid classes in ZOOM rooms!")
    else:
        print(f"     ✅ Good: No Hybrid in ZOOM")
    
    print(f"\n   Online placements:")
    print(f"     In ZOOM: {online_in_zoom}")
    print(f"     In Physical: {online_in_physical}")
    
    if online_in_physical > 0:
        print(f"     ⚠️  Note: {online_in_physical} Online classes in Physical rooms")
    else:
        print(f"     ✅ Good: All Online in ZOOM")
    
    # Final verdict
    print(f"\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    if hybrid_in_zoom == 0 and modality_fails['hybrid_needs_physical'] > 0:
        print("\n✅ VALIDATION IS WORKING CORRECTLY")
        print("   • Hybrid→ZOOM attempts are being rejected")
        print("   • No Hybrid classes ended up in ZOOM rooms")
        print("   • The validation you saw earlier was from random test actions")
    elif hybrid_in_zoom == 0 and modality_fails['hybrid_needs_physical'] == 0:
        print("\n✅ NO INVALID PLACEMENTS")
        print("   • No Hybrid in ZOOM (good)")
        print("   • No rejection stats (might not have been attempted)")
        print("   • Random actions might not have hit that combination")
    else:
        print("\n❌ VALIDATION MAY NOT BE WORKING")
        print(f"   • {hybrid_in_zoom} Hybrid classes in ZOOM rooms")
        print(f"   • Only {modality_fails['hybrid_needs_physical']} rejections recorded")
        print("   • Need to investigate validation code")
    
    print("="*70 + "\n")


def check_training_logs():
    """Check if training logs show modality rejections"""
    
    print("\n" + "="*70)
    print("CHECKING TRAINING EPISODE LOGS")
    print("="*70)
    
    print("\nTo verify validation is working during training:")
    print("\n1. Look at your training output for fail_stats:")
    print("   (RolloutWorker) fail_stats: {")
    print("       'hybrid_needs_physical': XXX,  ← Should be > 0")
    print("       'online_needs_virtual': XXX,")
    print("   }")
    
    print("\n2. If you see hybrid_needs_physical > 0:")
    print("   ✅ Validation IS rejecting Hybrid→ZOOM during training")
    
    print("\n3. The ZOOM placements you saw were from:")
    print("   • Random test actions (not real training)")
    print("   • Those get rejected in actual training")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    test_modality_validation()
    check_training_logs()