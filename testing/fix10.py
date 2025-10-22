"""
Quick test to verify FIX #10 is working
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.timetabling_env import ParallelTimetablingEnv
import pickle

def test_day_duplicate_prevention():
    """Test that same-day duplicates are prevented"""
    
    print("\n" + "="*80)
    print("TESTING FIX #10: Day Duplicate Prevention")
    print("="*80)
    
    # Load cache
    cache_file = "cached_environment_data_MANILA_MODALITY.pkl"
    
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    
    # Create minimal env
    env = ParallelTimetablingEnv(
        num_teachers=data["num_teachers"],
        num_subjects=data["num_subjects"],
        num_days=data["num_days"],
        num_timeslots=data["num_timeslots"],
        room_codes=data["room_codes"],
        subject_codes=data["subject_names"],
        subject_campuses=data["subject_campuses"],
        subject_allowed_rooms=data["subject_allowed_rooms"],
        subject_areas=data["subject_areas"],
        subject_section_idx=data["subject_section_idx"],
        section_labels=data["section_labels"],
        max_classes_per_teacher=4,
        teacher_max_classes_map=data["teacher_max_classes_map"],
        subject_teacher_idx=data["subject_teacher_idx"],
        area_teacher_indices=data["area_teacher_indices"],
        max_timesteps=50,
    )
    
    env.reset()
    
    # Test scenario: Try to place same subject on same day twice
    test_subject = 0
    test_day = 0
    
    print(f"\nTest Subject: {env.subject_codes[test_subject]}")
    print(f"Test Day: Day {test_day}")
    
    # First placement on day 0
    env.subject_day_usage[test_subject] = {test_day}
    
    print(f"\nAfter first placement:")
    print(f"  subject_day_usage[{test_subject}] = {env.subject_day_usage[test_subject]}")
    
    # Try to place again on same day
    can_place, reason = env._can_place_on_day(test_subject, test_day, 0)
    
    print(f"\nAttempt to place on same day again:")
    print(f"  can_place: {can_place}")
    print(f"  reason: {reason}")
    
    if not can_place and reason == "day_duplicate":
        print("\n✅ FIX #10 IS WORKING!")
        print("   Same-day duplicate was correctly prevented")
        return True
    else:
        print("\n❌ FIX #10 IS NOT WORKING!")
        print("   Same-day duplicate was NOT prevented")
        return False

if __name__ == "__main__":
    success = test_day_duplicate_prevention()
    sys.exit(0 if success else 1)