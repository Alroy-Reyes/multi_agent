import pickle

with open('cached_environment_data_MANILA_MODALITY.pkl', 'rb') as f:
    data = pickle.load(f)

# Check subject 889
idx = 889
print(f"Subject {idx}:")
print(f"  Name: {data['subject_names'][idx]}")
print(f"  Modality: {data['subject_modalities'][idx]}")
print(f"  Is Async: {data['subject_is_async'][idx]}")  # ← THE KEY
print(f"  Campuses: {data['subject_campuses'][idx]}")
print(f"  Allowed rooms: {data['subject_allowed_rooms'][idx][:5]}")


# I bet you'll see:
# ```
# Is Async: True  ← THIS is the problem!
# Campuses: ['X']  ← Gets async building instead of virtual