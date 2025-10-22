import pandas as pd

# Load both cleaned schedules
ai_schedule = pd.read_csv('cleaned_generated_schedule_standard_format.csv')
standard_schedule = pd.read_csv('standardized_schedule.csv').dropna(subset=['Subject'])

print("=== Dataset Overview ===")
print(f"AI Generated: {len(ai_schedule)} classes")
print(f"Standard: {len(standard_schedule)} classes")

# Create unique identifiers for each class (Subject + Section)
ai_schedule['class_id'] = ai_schedule['Subject'].str.strip() + " | " + ai_schedule['Section'].str.strip()
standard_schedule['class_id'] = standard_schedule['Subject'].str.strip() + " | " + standard_schedule['Section'].str.strip()

ai_classes = set(ai_schedule['class_id'])
std_classes = set(standard_schedule['class_id'])

# Calculate matches
matched = ai_classes & std_classes
missing_from_ai = std_classes - ai_classes
extra_in_ai = ai_classes - std_classes

print(f"\n=== Class-Level Comparison ===")
print(f"Matched classes: {len(matched)} ({len(matched)/len(std_classes)*100:.1f}%)")
print(f"Missing from AI: {len(missing_from_ai)}")
print(f"Extra in AI (not in standard): {len(extra_in_ai)}")

# Show missing classes
if len(missing_from_ai) > 0:
    print(f"\n=== Missing Classes (first 30) ===")
    for i, class_id in enumerate(sorted(missing_from_ai)[:30], 1):
        subject, section = class_id.split(" | ")
        print(f"{i:2d}. {subject} - {section}")

# Analyze by subject
print(f"\n=== Subject Analysis ===")
ai_subjects = set(ai_schedule['Subject'].str.strip())
std_subjects = set(standard_schedule['Subject'].str.strip())

subject_overlap = ai_subjects & std_subjects
print(f"Common subjects: {len(subject_overlap)}")
print(f"Subjects only in standard: {len(std_subjects - ai_subjects)}")

if len(std_subjects - ai_subjects) > 0:
    print(f"\nSubjects in standard but not scheduled by AI:")
    for subj in sorted(std_subjects - ai_subjects)[:15]:
        count = len(standard_schedule[standard_schedule['Subject'] == subj])
        print(f"  - {subj} ({count} sections)")

# Save detailed comparison
missing_df = standard_schedule[standard_schedule['class_id'].isin(missing_from_ai)]
missing_df.to_csv('missing_from_ai_schedule.csv', index=False)
print(f"\nDetailed missing classes saved to 'missing_from_ai_schedule.csv'")