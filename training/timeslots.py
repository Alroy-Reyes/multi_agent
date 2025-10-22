# create_timeslots_manila.py - COMPLETE FIXED VERSION
"""
Extract timeslots from Manila_Summary_CLEANED.csv and update cache
FIXED: Correctly determines subject durations from CSV data
"""
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Set
from collections import defaultdict, Counter
import os
import re

def extract_manila_timeslots():
    """
    Extract timeslots from Manila CSV and update cache
    """
    
    print("=" * 80)
    print("EXTRACTING MANILA TIMESLOTS (FIXED VERSION)")
    print("=" * 80)
    
    # Load Manila CSV
    csv_path = 'Manila_Summary_CLEANED.csv'
    print(f"\nReading {csv_path}...")
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        return None
        
    df = pd.read_csv(csv_path)
    print(f"Total entries: {len(df)}")
    
    # Calculate duration from time string
    def calc_duration(time_str):
        if not time_str or pd.isna(time_str):
            return 0
        try:
            time_str = str(time_str).strip().upper()
            
            # Skip ASYNC
            if time_str in ['ASYNC', 'ASYNCHRONOUS', '', 'NAN']:
                return 0
            
            # Parse HH:MM-HH:MM or HHMM-HHMM
            patterns = [
                r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})",
                r"(\d{1,2})(\d{2})\s*-\s*(\d{1,2})(\d{2})",
            ]
            
            for pattern in patterns:
                match = re.search(pattern, time_str)
                if match:
                    groups = match.groups()
                    if len(groups) == 4:
                        h1, m1, h2, m2 = groups
                        start_mins = int(h1) * 60 + int(m1)
                        end_mins = int(h2) * 60 + int(m2)
                        return end_mins - start_mins
            
            return 0
        except:
            return 0
    
    # Normalize time strings
    def normalize_time(time_str):
        if not time_str or pd.isna(time_str):
            return None
        
        time_str = str(time_str).strip().upper()
        
        if time_str in ['ASYNC', 'ASYNCHRONOUS', '', 'NAN']:
            return None
        
        patterns = [
            r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})",
            r"(\d{1,2})(\d{2})\s*-\s*(\d{1,2})(\d{2})",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, time_str)
            if match:
                groups = match.groups()
                if len(groups) == 4:
                    h1, m1, h2, m2 = groups
                    start = f"{h1.zfill(2)}{m1.zfill(2)}"
                    end = f"{h2.zfill(2)}{m2.zfill(2)}"
                    return f"{start}-{end}"
        
        return None
    
    # Collect timeslot information AND subject durations
    timeslot_info = {}
    subject_timeslot_usage = defaultdict(set)
    subject_duration_map = defaultdict(list)  # Track all durations per subject
    
    # Process regular Time column
    for _, row in df.iterrows():
        time = normalize_time(row.get('Time'))
        subject = str(row.get('Subject', '')).strip()
        modality = str(row.get('Modality', '')).strip()
        
        if time and subject:
            duration = calc_duration(row.get('Time'))
            
            if duration > 0:
                if time not in timeslot_info:
                    timeslot_info[time] = {
                        'duration': duration,
                        'count': 0,
                        'subjects': set(),
                        'modalities': set()
                    }
                
                timeslot_info[time]['count'] += 1
                timeslot_info[time]['subjects'].add(subject)
                timeslot_info[time]['modalities'].add(modality)
                subject_timeslot_usage[subject].add((time, duration, modality))
                
                # Track duration for this subject
                subject_duration_map[subject.upper()].append(duration)
    
    # Process Async Time column
    async_timeslots = set()
    if 'Async Time' in df.columns:
        for _, row in df.iterrows():
            async_time = normalize_time(row.get('Async Time'))
            if async_time:
                async_timeslots.add(async_time)
    
    # Sort timeslots
    sorted_timeslots = sorted(timeslot_info.items(), key=lambda x: x[0])
    
    # Categorize by duration
    timeslots_30 = [(t, info) for t, info in sorted_timeslots if info['duration'] == 30]
    timeslots_60 = [(t, info) for t, info in sorted_timeslots if info['duration'] == 60]
    timeslots_90 = [(t, info) for t, info in sorted_timeslots if info['duration'] == 90]
    timeslots_120 = [(t, info) for t, info in sorted_timeslots if info['duration'] == 120]
    timeslots_other = [(t, info) for t, info in sorted_timeslots 
                       if info['duration'] not in [30, 60, 90, 120]]
    
    print(f"\n{'='*60}")
    print("DISCOVERED TIMESLOT STRUCTURE")
    print("="*60)
    
    if timeslots_30:
        print(f"\n30-MINUTE SLOTS ({len(timeslots_30)}):")
        for time, info in timeslots_30[:5]:  # Show first 5
            print(f"  {time} - {info['count']} classes")
    
    if timeslots_60:
        print(f"\n60-MINUTE SLOTS ({len(timeslots_60)}):")
        for time, info in timeslots_60[:5]:
            print(f"  {time} - {info['count']} classes")
    
    if timeslots_90:
        print(f"\n90-MINUTE SLOTS ({len(timeslots_90)}):")
        for time, info in timeslots_90[:5]:
            print(f"  {time} - {info['count']} classes")
    
    if timeslots_120:
        print(f"\n120-MINUTE SLOTS ({len(timeslots_120)}):")
        for time, info in timeslots_120[:5]:
            print(f"  {time} - {info['count']} classes")
    
    if timeslots_other:
        print(f"\nOTHER DURATION SLOTS ({len(timeslots_other)}):")
        for time, info in timeslots_other:
            print(f"  {time} - {info['duration']}min, {info['count']} classes")
    
    if async_timeslots:
        print(f"\nASYNC TIMESLOTS: {len(async_timeslots)}")
    
    # Load existing cache
    cache_path = 'cached_environment_data_MANILA_MODALITY.pkl'
    
    if not os.path.exists(cache_path):
        print(f"\n⚠ Cache file not found: {cache_path}")
        print("Run preprocess_manila_csv.py first!")
        return None
    
    print(f"\n{'='*60}")
    print(f"Loading {cache_path}...")
    
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    
    subject_names = data['subject_names']
    num_subjects = len(subject_names)
    
    # Create timeslot definitions
    timeslot_definitions = []
    timeslot_idx = 0
    
    # Add all real timeslots in order: 30min, 60min, 90min, 120min, other
    for time_list in [timeslots_30, timeslots_60, timeslots_90, timeslots_120, timeslots_other]:
        for time, info in time_list:
            timeslot_definitions.append({
                'start': time[:4],
                'end': time[5:],
                'time': time,
                'duration': info['duration'],
                'type': 'standard' if info['duration'] == 90 else 
                       ('short' if info['duration'] == 30 else
                        ('hour' if info['duration'] == 60 else
                         ('extended' if info['duration'] == 120 else 'other'))),
                'index': timeslot_idx
            })
            timeslot_idx += 1
    
    # Add ASYNC timeslot
    timeslot_definitions.append({
        'start': 'ASYNC',
        'end': 'ASYNC',
        'time': 'ASYNC',
        'duration': 0,
        'type': 'async',
        'index': timeslot_idx
    })
    
    total_timeslots = len(timeslot_definitions)
    
    # Assign subject-timeslot mappings
    subject_allowed_timeslots = []
    subject_timeslot_types = []
    
    for subj_name in subject_names:
        # Parse subject name: Dept_Subject_Section_Modality
        parts = subj_name.split('_')
        if len(parts) >= 2:
            subj_clean = parts[1].strip().upper()
        else:
            subj_clean = subj_name.upper()
        
        allowed_indices = []
        primary_type = 'standard'
        
        # Check if async
        is_async_instance = False
        subj_idx = subject_names.index(subj_name)
        if 'subject_is_async' in data and subj_idx < len(data['subject_is_async']):
            is_async_instance = data['subject_is_async'][subj_idx]
        
        if is_async_instance:
            # Async instances only use ASYNC timeslot
            async_idx = timeslot_idx  # Last added
            allowed_indices = [async_idx]
            primary_type = 'async'
        else:
            # Find matching timeslots from CSV data
            for csv_subject, timeslot_set in subject_timeslot_usage.items():
                csv_subj_upper = csv_subject.upper()
                
                if csv_subj_upper in subj_clean or subj_clean in csv_subj_upper:
                    for time_str, duration, modality in timeslot_set:
                        for slot_def in timeslot_definitions:
                            if slot_def['time'] == time_str:
                                allowed_indices.append(slot_def['index'])
                                if duration == 120:
                                    primary_type = 'extended'
                                elif duration == 60:
                                    primary_type = 'hour'
                                elif duration == 30:
                                    primary_type = 'short'
                                break
                    break
            
            # Fallback: allow timeslots based on type
            if not allowed_indices:
                for slot_def in timeslot_definitions:
                    if slot_def['type'] == 'standard':
                        allowed_indices.append(slot_def['index'])
                primary_type = 'standard'
        
        # Remove duplicates
        allowed_indices = sorted(list(set(allowed_indices)))
        
        if not allowed_indices:
            print(f"⚠ Warning: No timeslots for {subj_name[:40]}..., using standard")
            allowed_indices = [i for i, td in enumerate(timeslot_definitions) 
                             if td['type'] == 'standard']
        
        subject_allowed_timeslots.append(allowed_indices)
        subject_timeslot_types.append(primary_type)
    
    # Statistics
    type_counts = defaultdict(int)
    for stype in subject_timeslot_types:
        type_counts[stype] += 1
    
    print(f"\n{'='*60}")
    print("SUBJECT TIMESLOT CATEGORIZATION")
    print("="*60)
    
    for slot_type in ['standard', 'short', 'hour', 'extended', 'async', 'other']:
        count = type_counts.get(slot_type, 0)
        if count > 0:
            pct = (count / num_subjects) * 100
            print(f"\n{slot_type.upper()} type ({count} subjects, {pct:.1f}%):")
            examples = [subject_names[i] for i, t in enumerate(subject_timeslot_types) 
                       if t == slot_type][:3]
            for ex in examples:
                num_slots = len(subject_allowed_timeslots[subject_names.index(ex)])
                print(f"  - {ex[:60]}... ({num_slots} timeslots)")
    
    # Update cache with timeslot data
    data['num_timeslots'] = total_timeslots
    data['timeslot_definitions'] = timeslot_definitions
    data['timeslot_labels'] = [td['time'] for td in timeslot_definitions]
    data['subject_allowed_timeslots'] = subject_allowed_timeslots
    data['subject_timeslot_types'] = subject_timeslot_types
    
    # ============================================================
    # FIXED: Calculate required placements from ACTUAL CSV durations
    # ============================================================
    print(f"\n{'='*60}")
    print("DETERMINING ACTUAL SUBJECT DURATIONS (FIXED)")
    print("="*60)
    
    subject_required_placements = {}
    subject_actual_durations = {}
    
    for s in range(num_subjects):
        subj_name = subject_names[s]
        
        # Parse subject name to get original subject
        parts = subj_name.split('_')
        if len(parts) >= 2:
            subj_clean = parts[1].strip().upper()
        else:
            subj_clean = subj_name.upper()
        
        # Check if async instance
        is_async_instance = False
        if 'subject_is_async' in data and s < len(data['subject_is_async']):
            is_async_instance = data['subject_is_async'][s]
        
        if is_async_instance:
            # Async instances need 1 placement
            actual_duration = 0
            subject_required_placements[s] = 1
        else:
            # Find actual duration from CSV data
            actual_duration = 90  # Default
            
            # Try to find in duration map
            for csv_subject, durations in subject_duration_map.items():
                if csv_subject in subj_clean or subj_clean in csv_subject:
                    # Use most common duration for this subject
                    if durations:
                        duration_counter = Counter(durations)
                        actual_duration = duration_counter.most_common(1)[0][0]
                    break
            
            # Calculate placements: 90-min needs 2, others need 1
            subject_required_placements[s] = 2 if actual_duration == 90 else 1
        
        subject_actual_durations[s] = actual_duration
    
    # Store in data
    data['subject_required_placements'] = subject_required_placements
    data['subject_actual_durations'] = subject_actual_durations
    
    # Show statistics
    duration_counts = Counter(subject_actual_durations.values())
    
    print(f"\nActual duration distribution from CSV:")
    for dur in sorted(duration_counts.keys()):
        count = duration_counts[dur]
        pct = (count / num_subjects) * 100
        print(f"  {dur:3d} min: {count:4d} subjects ({pct:5.1f}%)")
    
    needs_2 = sum(1 for v in subject_required_placements.values() if v == 2)
    needs_1 = sum(1 for v in subject_required_placements.values() if v == 1)
    
    print(f"\nPlacement requirements:")
    print(f"  2 placements (90-min classes): {needs_2}")
    print(f"  1 placement (other durations):  {needs_1}")
    
    # Verify against what we found in CSV
    csv_90_min = sum(1 for durations in subject_duration_map.values() 
                     if durations and Counter(durations).most_common(1)[0][0] == 90)
    csv_other = len(subject_duration_map) - csv_90_min
    
    print(f"\nVerification against raw CSV:")
    print(f"  90-min in CSV: {csv_90_min}")
    print(f"  Other in CSV:  {csv_other}")
    print(f"  Match: {'✓' if abs(needs_2 - csv_90_min) < 50 else '✗ CHECK DATA'}")
    
    # Save updated cache
    output_file = 'cached_environment_data_MANILA_MODALITY.pkl'
    
    print(f"\n{'='*60}")
    print("SAVING UPDATED CACHE")
    print("="*60)
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"✓ Saved to: {output_file}")
    
    # Verify the save
    print(f"\nVerifying saved data...")
    with open(output_file, 'rb') as f:
        verify_data = pickle.load(f)
    
    if 'subject_required_placements' in verify_data:
        print(f"✓ subject_required_placements found in saved cache")
        verify_needs_2 = sum(1 for v in verify_data['subject_required_placements'].values() if v == 2)
        verify_needs_1 = sum(1 for v in verify_data['subject_required_placements'].values() if v == 1)
        print(f"  - 2 placements: {verify_needs_2}")
        print(f"  - 1 placement: {verify_needs_1}")
    else:
        print(f"✗ WARNING: subject_required_placements NOT in saved cache!")
    
    print(f"\n{'='*60}")
    print("COMPLETE!")
    print("="*60)
    print(f"\nTimeslot structure:")
    print(f"  - Total timeslots: {total_timeslots}")
    print(f"  - 30 min: {len(timeslots_30)}")
    print(f"  - 60 min: {len(timeslots_60)}")
    print(f"  - 90 min: {len(timeslots_90)}")
    print(f"  - 120 min: {len(timeslots_120)}")
    if timeslots_other:
        print(f"  - Other: {len(timeslots_other)}")
    print(f"  - Async: 1")
    print("="*60)
    
    return data


if __name__ == "__main__":
    # Check if preprocessed cache exists
    if not os.path.exists('cached_environment_data_MANILA_MODALITY.pkl'):
        print("❌ Please run preprocess_manila_csv.py first!")
        import sys
        sys.exit(1)
    
    data = extract_manila_timeslots()
    
    if data:
        print("\n✅ Timeslot extraction completed!")
        print("\nNext steps:")
        print("1. Verify the placement requirements match your CSV")
        print("2. Re-run training:")
        print("   python train_manila_schedule.py --iterations 200")
        print("\nThe environment will now show:")
        print("  ✓ Loaded subject_required_placements from cache")
    else:
        print("\n❌ Timeslot extraction failed.")