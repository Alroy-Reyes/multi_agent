import pandas as pd

# Load your schedule
df = pd.read_csv('schedule_summary.csv')

# Count occurrences of each (Subject, Section) pair
duplicates = df.groupby(['Subject', 'Section']).size().reset_index(name='Count')

# Flag potential issues
duplicates['Status'] = duplicates.apply(lambda row:
    'OK (90-min 2x)' if row['Count'] == 2 else
    'OK (single meeting)' if row['Count'] == 1 else
    'VIOLATION (3+ times)', axis=1
)

# Show all entries with 3+ occurrences (true violations)
true_violations = duplicates[duplicates['Count'] >= 3]

print("=== DUPLICATE ANALYSIS ===")
print(f"Total unique (Subject, Section) pairs: {len(duplicates)}")
print(f"\nBreakdown:")
print(f"  - 1 occurrence: {len(duplicates[duplicates['Count'] == 1])}")
print(f"  - 2 occurrences: {len(duplicates[duplicates['Count'] == 2])} (90-min classes)")
print(f"  - 3+ occurrences: {len(true_violations)} ⚠️ VIOLATIONS")

if len(true_violations) > 0:
    print("\n🚨 TRUE VIOLATIONS FOUND:")
    print(true_violations.to_string(index=False))
else:
    print("\n✅ No violations found - all 2x entries are valid 90-min classes")