# check_timeslots.py
import pickle

print("="*60)
print("TIMESLOT CONSTRAINT ANALYSIS")
print("="*60)

with open('cached_environment_data_MANILA_MODALITY.pkl', 'rb') as f:
    data = pickle.load(f)

if 'subject_allowed_timeslots' in data:
    timeslot_counts = [len(slots) for slots in data['subject_allowed_timeslots']]
    print(f"\nTimeslots per subject:")
    print(f"  Min: {min(timeslot_counts)}")
    print(f"  Max: {max(timeslot_counts)}")
    print(f"  Average: {sum(timeslot_counts)/len(timeslot_counts):.1f}")
    print(f"\nSubjects with only 1 timeslot: {sum(1 for c in timeslot_counts if c == 1)}/{len(timeslot_counts)}")
    print(f"Subjects with 2-5 timeslots: {sum(1 for c in timeslot_counts if 2 <= c <= 5)}/{len(timeslot_counts)}")
    print(f"Subjects with 6+ timeslots: {sum(1 for c in timeslot_counts if c >= 6)}/{len(timeslot_counts)}")
    
    # Show some examples
    print(f"\nFirst 10 subjects timeslot counts:")
    for i in range(min(10, len(timeslot_counts))):
        print(f"  Subject {i}: {timeslot_counts[i]} allowed timeslots")
    
    # Show total timeslots available
    print(f"\nTotal timeslots in environment: {data['num_timeslots']}")
    
    # Percentage analysis
    restricted = sum(1 for c in timeslot_counts if c < data['num_timeslots'] * 0.5)
    print(f"\nHighly restricted subjects (< 50% of timeslots): {restricted}/{len(timeslot_counts)} ({restricted/len(timeslot_counts)*100:.1f}%)")
else:
    print("\n❌ No 'subject_allowed_timeslots' found in cache!")
    print("This means subjects can use ANY timeslot (good for learning)")

print("="*60)