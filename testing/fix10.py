# quick_check.py
import pickle
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.timetabling_env import ParallelTimetablingEnv

with open('cached_environment_data_MANILA_MODALITY.pkl', 'rb') as f:
    data = pickle.load(f)

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
)

print("="*60)
print("QUICK FIX #12 CHECK")
print("="*60)

# Check 1: Does the attribute exist?
if hasattr(env, 'subject_required_placements'):
    print("✅ subject_required_placements exists")
    
    # Sample a few
    sample = list(env.subject_required_placements.items())[:5]
    print(f"\nSample values:")
    for subj_idx, req in sample:
        print(f"  Subject {subj_idx}: requires {req} placements")
else:
    print("❌ subject_required_placements MISSING")

# Check 2: Does method exist?
if hasattr(env, '_get_placement_count'):
    print("\n✅ _get_placement_count() exists")
else:
    print("\n❌ _get_placement_count() MISSING")

# Check 3: Test the logic
print("\n" + "-"*60)
print("TESTING MAX PLACEMENT LOGIC")
print("-"*60)

env.subject_required_placements[0] = 2  # Set to 2 placements
env.subject_placement_count[(0, 0)] = 2  # Already at 2

can_place, reason = env._can_place_on_day(0, 2, 0)

print(f"Subject 0: Required=2, Current=2")
print(f"Attempt 3rd placement: can_place={can_place}, reason='{reason}'")

if not can_place and reason == "max_placements_reached":
    print("\n✅ FIX #12 LOGIC WORKS!")
else:
    print(f"\n❌ FIX #12 LOGIC BROKEN!")
    print(f"   Expected: can_place=False, reason='max_placements_reached'")
    print(f"   Got: can_place={can_place}, reason='{reason}'")

print("="*60)