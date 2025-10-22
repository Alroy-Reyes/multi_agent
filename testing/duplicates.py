"""
Detailed Duplicate Checker
Checks for duplicates in both cache and during training
"""

import pickle
import pandas as pd
from collections import defaultdict

def check_cache_duplicates():
    """Check cache file for duplicates"""
    
    print("\n" + "="*80)
    print("CACHE DUPLICATE CHECKER")
    print("="*80)
    
    with open('cached_environment_data_MANILA_MODALITY.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print(f"\nLoaded cache:")
    print(f"  Total subjects: {data['num_subjects']}")
    print(f"  Unique base subjects: {data.get('num_unique_base_subjects', 'N/A')}")
    
    # Check for day suffixes
    subjects_with_days = defaultdict(list)
    day_patterns = ['_M', '_T', '_W', '_H', '_F', '_S', '_MH', '_TF', '_TTH']
    
    for name in data['subject_names']:
        # Check if ends with day pattern
        has_day = any(name.endswith(pattern) for pattern in day_patterns)
        if has_day:
            # Remove day suffix
            for pattern in day_patterns:
                if name.endswith(pattern):
                    base = name[:-len(pattern)]
                    subjects_with_days[base].append(name)
                    break
    
    if subjects_with_days:
        print(f"\n⚠️ FOUND {len(subjects_with_days)} SUBJECTS WITH DAY SUFFIXES")
        print(f"   Total instances: {sum(len(v) for v in subjects_with_days.values())}")
        print(f"\n   First 10 examples:")
        for i, (base, instances) in enumerate(list(subjects_with_days.items())[:10]):
            print(f"   {i+1}. {base}")
            for inst in instances:
                print(f"      - {inst}")
        print(f"\n   ❌ RERUN PREPROCESSING TO FIX!")
    else:
        print(f"\n✅ NO DAY SUFFIXES FOUND - Cache is clean!")
    
    # Check modality counts
    print(f"\n{'='*80}")
    print("MODALITY DISTRIBUTION IN CACHE")
    print("="*80)
    
    for modality in ['Face-to-Face', 'Online', 'Hybrid']:
        count = data['subject_modalities'].count(modality)
        pct = (count / data['num_subjects']) * 100
        print(f"  {modality}: {count} subjects ({pct:.1f}%)")
    
    # Check subject_required_days if present
    if 'subject_required_days' in data:
        print(f"\n{'='*80}")
        print("SUBJECT REQUIRED DAYS")
        print("="*80)
        
        days_counts = defaultdict(int)
        for days in data['subject_required_days'].values():
            days_counts[len(days)] += 1
        
        print(f"  Subjects by number of required days:")
        for num_days, count in sorted(days_counts.items()):
            print(f"    {num_days} day(s): {count} subjects")
        
        total_placements = sum(len(days) for days in data['subject_required_days'].values())
        print(f"\n  Total placements needed: {total_placements}")
        print(f"  Average days per subject: {total_placements / data['num_subjects']:.2f}")
    
    return len(subjects_with_days) == 0

def check_training_duplicates(env):
    """Check environment during training for duplicates"""
    
    print("\n" + "="*80)
    print("TRAINING DUPLICATE CHECKER")
    print("="*80)
    
    conflicts = env.validate_schedule()
    
    duplicate_count = conflicts['summary']['duplicate_placements']
    
    if duplicate_count > 0:
        print(f"\n⚠️ FOUND {duplicate_count} DUPLICATE PLACEMENTS")
        print(f"\n   First 10 duplicates:")
        for i, dup in enumerate(conflicts['duplicate_subjects'][:10]):
            print(f"\n   {i+1}. {dup['subject']}")
            print(f"      Required: {dup['required']} | Actual: {dup['actual']}")
            print(f"      Placements:")
            for p in dup['placements']:
                print(f"        - Day {p['day']}, TS {p['timeslot']}, Room {p['room_code']}")
        return False
    else:
        print(f"\n✅ NO DUPLICATES IN TRAINING")
        return True

if __name__ == "__main__":
    # Check cache
    cache_ok = check_cache_duplicates()
    
    if not cache_ok:
        print(f"\n{'='*80}")
        print("RECOMMENDATION")
        print("="*80)
        print("Rerun preprocessing to remove day suffixes:")
        print("  python preprocess_manila_csv.py")