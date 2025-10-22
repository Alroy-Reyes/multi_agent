import pickle

# Load cache
with open('cached_environment_data_MANILA_MODALITY.pkl', 'rb') as f:
    data = pickle.load(f)

# Check first subject
slots = data['subject_allowed_timeslots'][0]
print(f"First subject timeslots: {slots}")
print(f"Count: {len(slots)}")
print(f"Effective slots (× 7 days): {len(slots) * 7}")

# Expected after expansion:
# Count: 12-13
# Effective: 84-91