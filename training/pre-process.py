# preprocess_manila_csv.py - FIXED: Remove day from unique identifier
"""
Preprocess Manila_Summary_CLEANED.csv with proper deduplication
Removes day from subject identifier to prevent treating same subject as different subjects
"""
import pickle
import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from area_subject_resolver import assign_area


def preprocess_manila_schedule(
    csv_path="Manila_Summary_CLEANED.csv", 
    output_filename="cached_environment_data_MANILA_MODALITY.pkl"
):
    """
    Main preprocessing function for Manila schedule
    """
    
    print("=" * 80)
    print("PREPROCESSING MANILA SCHEDULE - FIXED DEDUPLICATION")
    print("=" * 80)
    print(f"Processing: {csv_path}\n")
    
    # === LOAD CSV ===
    try:
        df_raw = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df_raw)} rows")
        print(f"✓ Columns: {list(df_raw.columns)}\n")
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        return None
    
    # === DATA CLEANING ===
    print("=" * 40)
    print("DATA CLEANING")
    print("=" * 40)
    
    # Remove completely empty rows
    empty_mask = df_raw.isna().all(axis=1) | (df_raw.astype(str).eq('').all(axis=1))
    empty_count = empty_mask.sum()
    df_clean = df_raw[~empty_mask].copy()
    
    print(f"• Removed {empty_count} empty rows")
    print(f"• Working with {len(df_clean)} data rows\n")
    
    # Clean text columns
    text_columns = ['Subject', 'Faculty', 'Day', 'Time', 'Room', 'Department', 
                    'Section', 'Modality']
    
    for col in text_columns:
        if col in df_clean.columns:
            df_clean[col] = (df_clean[col]
                            .fillna('')
                            .astype(str)
                            .str.replace(r'\s+', ' ', regex=True)
                            .str.strip())
    
    # === DEDUPLICATE CSV ROWS ===
    print("=" * 40)
    print("CSV DEDUPLICATION")
    print("=" * 40)
    
    # Create composite key for each row
    df_clean['csv_key'] = (
        df_clean['Subject'].str.upper() + "|||" +
        df_clean['Department'].str.upper() + "|||" +
        df_clean['Section'].str.upper() + "|||" +
        df_clean['Faculty'].str.upper() + "|||" +
        df_clean.get('Modality', '').str.upper() + "|||" +
        df_clean.get('Day', '').str.upper() + "|||" +
        df_clean.get('Time', '').str.upper()
    )
    
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset='csv_key', keep='first')
    after_dedup = len(df_clean)
    duplicates_removed = before_dedup - after_dedup
    
    print(f"• Rows before deduplication: {before_dedup}")
    print(f"• Rows after deduplication: {after_dedup}")
    print(f"• Duplicate rows removed: {duplicates_removed}")
    
    df_clean = df_clean.drop(columns=['csv_key'])
    
    # === STANDARDIZE MODALITY ===
    print(f"\n{'='*40}")
    print("MODALITY STANDARDIZATION")
    print("=" * 40)
    
    if 'Modality' not in df_clean.columns:
        print("⚠ No Modality column - adding default Face-to-Face")
        df_clean['Modality'] = 'Face-to-Face'
    
    modality_mapping = {
        'F2F': 'Face-to-Face', 'FTF': 'Face-to-Face', 'FACE TO FACE': 'Face-to-Face',
        'FACE-TO-FACE': 'Face-to-Face', 'IN-PERSON': 'Face-to-Face', 'PHYSICAL': 'Face-to-Face',
        'ONSITE': 'Face-to-Face', 'TYPE C - IN-PERSON': 'Face-to-Face', 'TYPE C': 'Face-to-Face',
        'C': 'Face-to-Face',
        'ONLINE': 'Online', 'VIRTUAL': 'Online', 'REMOTE': 'Online', 'ASYNCHRONOUS': 'Online',
        'ASYNC': 'Online', 'PURE ONLINE': 'Online', 'TYPE A - PURE ONLINE': 'Online',
        'TYPE A': 'Online', 'A': 'Online',
        'HYBRID': 'Hybrid', 'BLENDED': 'Hybrid', 'MIXED': 'Hybrid', 'HY-FLEX': 'Hybrid',
        'HYFLEX': 'Hybrid', 'TYPE B - HYBRID': 'Hybrid', 'TYPE B': 'Hybrid', 'B': 'Hybrid',
    }
    
    df_clean['Modality'] = (df_clean['Modality'].str.strip().str.upper().replace(modality_mapping))
    
    valid_modalities = {'Face-to-Face', 'Online', 'Hybrid'}
    unknown_mask = ~df_clean['Modality'].isin(valid_modalities)
    
    if unknown_mask.any():
        for idx in df_clean[unknown_mask].index:
            room = str(df_clean.loc[idx, 'Room']).upper()
            if 'ZOOM' in room or 'ONLINE' in room or not room or room == 'NAN':
                df_clean.loc[idx, 'Modality'] = 'Online'
            else:
                df_clean.loc[idx, 'Modality'] = 'Face-to-Face'
    
    modality_counts = df_clean['Modality'].value_counts()
    print(f"\nModality Distribution (in CSV):")
    for modality, count in modality_counts.items():
        pct = (count / len(df_clean)) * 100
        print(f"  • {modality}: {count} ({pct:.1f}%)")
    
    # === VALIDATE ESSENTIAL FIELDS ===
    print(f"\n{'='*40}")
    print("FIELD VALIDATION")
    print("=" * 40)
    
    essential_fields = ['Subject', 'Faculty', 'Department', 'Section']
    
    for field in essential_fields:
        if field not in df_clean.columns:
            df_clean[field] = 'UNKNOWN' if field != 'Faculty' else 'TBD'
            continue
            
        field_valid = (df_clean[field] != '') & df_clean[field].notna()
        invalid_count = (~field_valid).sum()
        
        if invalid_count > 0:
            if field == 'Faculty':
                df_clean.loc[~field_valid, field] = 'TBD'
            elif field == 'Department':
                df_clean.loc[~field_valid, field] = 'UNKNOWN'
            elif field == 'Section':
                df_clean.loc[~field_valid, field] = 'GEN'
            elif field == 'Subject':
                df_clean.loc[~field_valid, field] = 'SUBJECT'
    
    for col in ['Day', 'Time']:
        if col not in df_clean.columns:
            df_clean[col] = 'M' if col == 'Day' else '0800-0930'
    
    for idx, row in df_clean.iterrows():
        if not str(row.get('Day', '')).strip():
            df_clean.loc[idx, 'Day'] = 'M'
    
    print(f"✓ Validated {len(df_clean)} rows")
    
    # === PROCESS DAYS ===
    print(f"\n{'='*40}")
    print("DAY PATTERN PROCESSING")
    print("=" * 40)
    
    DAY_TOKENS_ORDER = ["MH", "TF", "TTH", "T", "H", "F", "M", "W", "S"]
    DAY_REGEX = r"TTH|MH|TF|M|T|W|H|F|S"
    
    def tokenize_days_exact(cell: str) -> list:
        return re.findall(DAY_REGEX, (str(cell) or "").upper())
    
    all_days = set()
    days_series = df_clean['Day'].apply(tokenize_days_exact)
    for days in days_series:
        all_days.update(days)
    
    day_labels = [d for d in DAY_TOKENS_ORDER if d in all_days]
    if not day_labels:
        day_labels = ['M', 'T', 'W', 'H', 'F']
    num_days = len(day_labels)
    
    print(f"• Found {num_days} day patterns: {day_labels}")
    
    # === EXPAND ROWS FOR EACH DAY ===
    print(f"\n{'='*40}")
    print("ROW EXPANSION (TRACKING DAYS)")
    print("=" * 40)
    
    expanded_data = []
    
    for idx, row in df_clean.iterrows():
        sync_days = tokenize_days_exact(row.get("Day", ""))
        modality = row["Modality"]
        
        if sync_days:
            for day in sync_days:
                expanded_data.append({
                    'subject': row["Subject"],
                    'faculty': row["Faculty"],
                    'department': row["Department"],
                    'section': row["Section"],
                    'room': row.get("Room", ""),
                    'day': day,
                    'time': row.get("Time", ""),
                    'modality': modality,
                    'original_row': idx
                })
        else:
            expanded_data.append({
                'subject': row["Subject"],
                'faculty': row["Faculty"],
                'department': row["Department"],
                'section': row["Section"],
                'room': row.get("Room", ""),
                'day': 'M',
                'time': row.get("Time", "0800-0930"),
                'modality': modality,
                'original_row': idx
            })
    
    expanded_df = pd.DataFrame(expanded_data)
    print(f"• Rows after day expansion: {len(expanded_df)}")
    
    # === DEDUPLICATE EXPANDED DATA ===
    print(f"\n{'='*40}")
    print("EXPANDED DATA DEDUPLICATION")
    print("=" * 40)
    
    expanded_df['unique_key'] = (
        expanded_df['department'].str.upper() + "|||" +
        expanded_df['subject'].str.upper() + "|||" +
        expanded_df['section'].str.upper() + "|||" +
        expanded_df['modality'].str.upper() + "|||" +
        expanded_df['day'].str.upper()
    )
    
    before_dedup = len(expanded_df)
    expanded_df = expanded_df.drop_duplicates(subset='unique_key', keep='first')
    after_dedup = len(expanded_df)
    duplicates_removed = before_dedup - after_dedup
    
    print(f"• Instances before deduplication: {before_dedup}")
    print(f"• Instances after deduplication: {after_dedup}")
    print(f"• Duplicate instances removed: {duplicates_removed}")
    
    expanded_df = expanded_df.drop(columns=['unique_key'])
    
    # === GROUP BY BASE SUBJECT (CRITICAL FIX) ===
    print(f"\n{'='*40}")
    print("GROUPING BY BASE SUBJECT (NO DAY SUFFIX)")
    print("=" * 40)
    
    # Create base subject identifier WITHOUT day
    expanded_df['base_subject_id'] = (
        expanded_df['department'] + "_" +
        expanded_df['subject'] + "_" +
        expanded_df['section'] + "_" +
        expanded_df['modality']
    )
    
    # Group by base subject to collect all days
    subject_groups = expanded_df.groupby('base_subject_id').agg({
        'subject': 'first',
        'faculty': 'first',
        'department': 'first',
        'section': 'first',
        'room': 'first',
        'time': 'first',
        'modality': 'first',
        'day': lambda x: list(x),  # Collect all days as list
        'original_row': 'first'
    }).reset_index()
    
    print(f"• Before grouping: {len(expanded_df)} day-instances")
    print(f"• After grouping: {len(subject_groups)} unique base subjects")
    print(f"• Instances removed: {len(expanded_df) - len(subject_groups)}")
    
    # Store days per subject
    subject_required_days = {}
    for idx, row in subject_groups.iterrows():
        subject_required_days[idx] = row['day']
    
    # === CREATE UNIQUE IDENTIFIERS (NO DAY SUFFIX) ===
    
    subject_names = subject_groups['base_subject_id'].tolist()
    subject_modalities = subject_groups['modality'].tolist()
    subject_departments = subject_groups['department'].tolist()
    section_names = subject_groups['section'].tolist()
    room_assignments = subject_groups['room'].tolist()
    
    # Create teacher identifiers
    teacher_names = [
        f"{row['faculty']}_{row['department']}"
        for _, row in subject_groups.iterrows()
    ]
    
    num_subjects = len(subject_names)
    
    print(f"\n✓ Created {num_subjects} unique base subjects")
    print(f"  Average days per subject: {sum(len(days) for days in subject_required_days.values()) / num_subjects:.2f}")
    
    # === CALCULATE UNIQUE BASE SUBJECTS ===
    unique_base_subjects = len(subject_groups)
    total_day_instances = sum(len(days) for days in subject_required_days.values())
    avg_days_per_subject = total_day_instances / unique_base_subjects if unique_base_subjects > 0 else 1
    
    print(f"\n{'='*40}")
    print("BASE SUBJECT ANALYSIS")
    print("=" * 40)
    print(f"• Unique base subjects: {unique_base_subjects}")
    print(f"• Total day-instances needed: {total_day_instances}")
    print(f"• Average days per subject: {avg_days_per_subject:.2f}")
    
    # === PROCESS TEACHERS (DLSU POLICY) ===
    print(f"\n{'='*40}")
    print("TEACHER PROCESSING (DLSU POLICY)")
    print("=" * 40)
    
    unique_teachers = pd.unique(pd.Series(teacher_names))
    teacher_id_to_name = list(unique_teachers)
    teacher_name_to_idx = {name: i for i, name in enumerate(teacher_id_to_name)}
    num_teachers = len(teacher_id_to_name)
    subject_teacher_idx = [teacher_name_to_idx[t] for t in teacher_names]
    
    print(f"• Unique teachers: {num_teachers}")
    
    MAX_SUBJECTS_PER_TEACHER = 4
    
    print(f"\n📚 DLSU Teaching Load Policy:")
    print(f"  • Maximum: 12 units per teacher")
    print(f"  • Units per subject: 3")
    print(f"  • Maximum subjects: {MAX_SUBJECTS_PER_TEACHER}")
    print(f"\n🔢 Capacity Calculation:")
    print(f"  • {MAX_SUBJECTS_PER_TEACHER} subjects × {avg_days_per_subject:.2f} days/subject")
    print(f"  • = {MAX_SUBJECTS_PER_TEACHER * avg_days_per_subject:.1f} day-instances per teacher")
    
    teacher_modality_counts = defaultdict(lambda: {'Online': 0, 'Face-to-Face': 0, 'Hybrid': 0})
    
    for i, t_idx in enumerate(subject_teacher_idx):
        modality = subject_modalities[i]
        # Count total day instances for this teacher
        days_count = len(subject_required_days.get(i, [1]))
        teacher_modality_counts[t_idx][modality] += days_count
    
    teacher_max_classes_map = {}
    
    for i in range(num_teachers):
        counts = teacher_modality_counts[i]
        total = sum(counts.values())
        
        base_capacity = MAX_SUBJECTS_PER_TEACHER * avg_days_per_subject
        
        if total == 0:
            cap = base_capacity
        else:
            online_ratio = counts['Online'] / total
            
            if online_ratio > 0.7:
                cap = base_capacity * 1.1
            elif online_ratio > 0.4:
                cap = base_capacity * 1.05
            else:
                cap = base_capacity
        
        teacher_max_classes_map[f"teacher_{i}"] = int(cap)
    
    capacities = list(teacher_max_classes_map.values())
    print(f"\n• Instance capacity per teacher: {min(capacities)}-{max(capacities)} day-instances")
    print(f"  (Equivalent to ~{min(capacities)/avg_days_per_subject:.1f}-{max(capacities)/avg_days_per_subject:.1f} subjects)")
    
    # === PROCESS ROOMS ===
    print(f"\n{'='*40}")
    print("ROOM PROCESSING")
    print("=" * 40)
    
    VIRTUAL_ROOMS = [f"ZOOM{i:03d}" for i in range(1, 51)]
    
    def process_room_by_modality(row):
        modality = row['modality']
        room = str(row['room']).strip().upper()
        
        if modality == 'Online':
            if not room or room in ['NAN', '', 'TBD', 'NONE']:
                return 'ZOOM001'
            elif any(keyword in room for keyword in ['ZOOM', 'ONLINE', 'VIRTUAL', 'TEAMS', 'MEET']):
                return room
            else:
                return 'ZOOM001'
        elif modality == 'Hybrid':
            if room and room not in ['NAN', '', 'TBD', 'ZOOM', 'ONLINE']:
                return room
            else:
                return 'A101'
        else:
            if room and room not in ['NAN', '', 'TBD', 'ZOOM', 'ONLINE']:
                return room
            else:
                return 'A101'
    
    subject_groups['processed_room'] = subject_groups.apply(process_room_by_modality, axis=1)
    
    all_rooms = subject_groups['processed_room'].unique()
    
    physical_rooms = [r for r in all_rooms if not any(kw in r for kw in ['ZOOM', 'ONLINE', 'VIRTUAL'])]
    virtual_rooms_in_data = [r for r in all_rooms if 'ZOOM' in r or 'ONLINE' in r or 'VIRTUAL' in r]
    
    room_codes = (
        sorted(physical_rooms) + 
        sorted(virtual_rooms_in_data) +
        [r for r in VIRTUAL_ROOMS if r not in virtual_rooms_in_data][:20]
    )
    
    virtual_room_set = set([r for r in room_codes if any(kw in r for kw in ['ZOOM', 'ONLINE', 'VIRTUAL'])])
    physical_room_set = set([r for r in room_codes if r not in virtual_room_set])
    
    print(f"• Physical rooms: {len(physical_room_set)}")
    print(f"• Virtual rooms: {len(virtual_room_set)}")
    print(f"• Total rooms: {len(room_codes)}")
    
    # === MODALITY ENCODING ===
    modality_to_idx = {'Face-to-Face': 0, 'Online': 1, 'Hybrid': 2}
    subject_modality_idx = [modality_to_idx[m] for m in subject_modalities]
    
    # === SUBJECT-ROOM CONSTRAINTS ===
    print(f"\n{'='*40}")
    print("ROOM CONSTRAINTS")
    print("=" * 40)
    
    subject_room_map = defaultdict(list)
    
    for idx, row in subject_groups.iterrows():
        subj = subject_names[idx]
        room = row['processed_room']
        modality = row['modality']
        
        if modality == 'Online':
            for vr in virtual_room_set:
                if vr not in subject_room_map[subj]:
                    subject_room_map[subj].append(vr)
        elif modality == 'Hybrid':
            if room in physical_room_set and room not in subject_room_map[subj]:
                subject_room_map[subj].append(room)
            for vr in list(virtual_room_set)[:5]:
                if vr not in subject_room_map[subj]:
                    subject_room_map[subj].append(vr)
        else:
            if room in physical_room_set and room not in subject_room_map[subj]:
                subject_room_map[subj].append(room)
    
    for subj in subject_names:
        if not subject_room_map[subj]:
            idx = subject_names.index(subj)
            modality = subject_modalities[idx]
            if modality == 'Online':
                subject_room_map[subj] = list(virtual_room_set)[:10]
            else:
                subject_room_map[subj] = list(physical_room_set)[:5]
    
    subject_allowed_rooms = [subject_room_map.get(subj, []) for subj in subject_names]
    
    print(f"• Room constraints created for {len(subject_room_map)} subjects")
    
    # === CAMPUS/BUILDING MAPPINGS ===
    print(f"\n{'='*40}")
    print("CAMPUS/BUILDING MAPPINGS")
    print("=" * 40)
    
    def extract_building(room_code):
        room_str = str(room_code).strip().upper()
        if 'ZOOM' in room_str or 'ONLINE' in room_str or 'VIRTUAL' in room_str:
            return 'Z'
        if room_str and len(room_str) > 0 and room_str[0].isalpha():
            return room_str[0]
        return 'U'
    
    room_to_campus = {room: extract_building(room) for room in room_codes}
    buildings = sorted(list(set(room_to_campus.values())))
    
    print(f"• Buildings/campuses: {buildings}")
    
    virtual_buildings = set()
    for vroom in virtual_room_set:
        vbuild = extract_building(vroom)
        virtual_buildings.add(vbuild)
    
    print(f"• Virtual buildings (ZOOM): {virtual_buildings}")
    
    # === SUBJECT-CAMPUS MAPPINGS ===
    print(f"\n{'='*40}")
    print("SUBJECT-CAMPUS ASSIGNMENTS")
    print("=" * 40)
    
    subject_campuses = []

    for i, subj in enumerate(subject_names):
        modality = subject_modalities[i]
        allowed_rooms = subject_allowed_rooms[i]
    
        if modality == 'Online':
            online_buildings = list(virtual_buildings)
            subject_campuses.append(online_buildings)
        elif modality == 'Hybrid':
            hybrid_buildings = []
            for room in allowed_rooms:
                building = extract_building(room)
                if building not in virtual_buildings and building not in ['U']:
                    if building not in hybrid_buildings:
                        hybrid_buildings.append(building)
            for vb in virtual_buildings:
                if vb not in hybrid_buildings:
                    hybrid_buildings.append(vb)
            if not hybrid_buildings:
                hybrid_buildings = [b for b in buildings if b not in ['U']]
            subject_campuses.append(hybrid_buildings)
        else:
            if allowed_rooms:
                f2f_buildings = []
                for room in allowed_rooms:
                    building = extract_building(room)
                    if building not in virtual_buildings and building not in ['U']:
                        if building not in f2f_buildings:
                            f2f_buildings.append(building)
                if f2f_buildings:
                    subject_campuses.append(f2f_buildings)
                else:
                    subject_campuses.append([b for b in buildings 
                                            if b not in virtual_buildings and b not in ['U']])
            else:
                subject_campuses.append([b for b in buildings 
                                        if b not in virtual_buildings and b not in ['U']])
    
    online_count = 0
    online_with_virtual = 0
    for i, modality in enumerate(subject_modalities):
        if modality == 'Online':
            online_count += 1
            has_virtual = any(b in virtual_buildings for b in subject_campuses[i])
            if has_virtual:
                online_with_virtual += 1
    
    print(f"✓ Online subjects: {online_count}")
    print(f"✓ Online with virtual building access: {online_with_virtual}")
    
    # === PROCESS AREAS ===
    print(f"\n{'='*40}")
    print("SUBJECT AREA PROCESSING")
    print("=" * 40)
    
    base_subjects = [name.split("_")[1] if "_" in name else name for name in subject_names]
    subject_areas = [assign_area(subj) for subj in base_subjects]
    area_labels = sorted(list(set(subject_areas)))
    
    print(f"• Subject areas: {len(area_labels)}")
    for area in area_labels[:10]:
        count = subject_areas.count(area)
        print(f"    {area}: {count} subjects")
    
    # === PROCESS SECTIONS ===
    
    unique_sections = pd.unique(pd.Series([
        f"{dept}_{sect}" for dept, sect in zip(subject_departments, section_names)
    ]))
    section_labels = sorted(list(unique_sections))
    section_name_to_idx = {name: i for i, name in enumerate(section_labels)}
    subject_section_idx = [
        section_name_to_idx[f"{dept}_{sect}"]
        for dept, sect in zip(subject_departments, section_names)
    ]
    
    print(f"• Sections: {len(section_labels)}")
    
    # === AREA-TEACHER MAPPINGS ===
    print(f"\n{'='*40}")
    print("AREA-TEACHER MAPPINGS")
    print("=" * 40)
    
    teacher_to_areas = defaultdict(set)
    for subj_idx, teacher_idx in enumerate(subject_teacher_idx):
        area = subject_areas[subj_idx]
        teacher_to_areas[teacher_idx].add(area)
    
    area_to_teachers = defaultdict(list)
    for teacher_idx, areas in teacher_to_areas.items():
        if areas:
            area_counts = defaultdict(int)
            for subj_idx, t_idx in enumerate(subject_teacher_idx):
                if t_idx == teacher_idx:
                    area_counts[subject_areas[subj_idx]] += 1
            primary_area = max(area_counts.keys(), key=lambda a: area_counts[a])
            area_to_teachers[primary_area].append(teacher_idx)
    
    unassigned_teachers = [i for i in range(num_teachers) if i not in teacher_to_areas]
    for i, teacher_idx in enumerate(unassigned_teachers):
        area = area_labels[i % len(area_labels)]
        area_to_teachers[area].append(teacher_idx)
    
    for area in area_labels:
        if not area_to_teachers[area] and num_teachers > 0:
            area_to_teachers[area].append(0)
    
    area_teacher_indices = {area: sorted(area_to_teachers[area]) for area in area_labels}
    max_teachers_per_area = max(len(teachers) for teachers in area_teacher_indices.values()) if area_teacher_indices else 1
    
    for area in area_labels[:5]:
        print(f"  {area}: {len(area_teacher_indices[area])} teachers")
    
    # === TIMESLOT PROCESSING ===
    print(f"\n{'='*40}")
    print("TIMESLOT PROCESSING")
    print("=" * 40)
    
    def normalize_time(time_str):
        if not time_str or str(time_str).strip() == "" or str(time_str).lower() == 'nan':
            return "0800-0930"
        
        time_str = str(time_str).strip()
        
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
        
        return "0800-0930"
    
    time_data = subject_groups['time'].apply(normalize_time)
    timeslot_labels = sorted(list(pd.unique(time_data)))
    num_timeslots = len(timeslot_labels)
    
    print(f"• Timeslots found: {num_timeslots}")
    
    # === CREATE FINAL CACHED DATA ===
    print(f"\n{'='*80}")
    print("CREATING CACHED DATA")
    print("=" * 80)
    
    cached_data = {
        # Metadata
        'is_unified': True,
        'is_cleaned': True,
        'has_modality': True,
        'is_deduplicated': True,
        'dlsu_compliant': True,
        'source_file': csv_path,
        
        # Core dimensions
        'num_teachers': num_teachers,
        'num_subjects': num_subjects,
        'num_days': num_days,
        'num_timeslots': num_timeslots,
        'max_teachers_per_area': max_teachers_per_area,
        'num_unique_base_subjects': unique_base_subjects,
        'avg_days_per_subject': avg_days_per_subject,
        
        # NEW: Store required days per subject
        'subject_required_days': subject_required_days,
        
        # Main data arrays
        'room_codes': room_codes,
        'subject_names': subject_names,
        'subject_departments': subject_departments,
        'subject_modalities': subject_modalities,
        'subject_modality_idx': subject_modality_idx,
        'subject_campuses': subject_campuses,
        'subject_areas': subject_areas,
        'subject_allowed_rooms': subject_allowed_rooms,
        'subject_section_idx': subject_section_idx,
        'section_labels': section_labels,
        'teacher_max_classes_map': teacher_max_classes_map,
        'subject_teacher_idx': subject_teacher_idx,
        'area_teacher_indices': area_teacher_indices,
        
        # Labels and mappings
        'teacher_id_to_name': teacher_id_to_name,
        'teacher_names': teacher_names,
        'day_labels': day_labels,
        'timeslot_labels': timeslot_labels,
        'room_to_campus': room_to_campus,
        'area_labels': area_labels,
        
        # Modality-specific data
        'modality_labels': ['Face-to-Face', 'Online', 'Hybrid'],
        'physical_rooms': sorted(list(physical_room_set)),
        'virtual_rooms': sorted(list(virtual_room_set)),
        'virtual_buildings': sorted(list(virtual_buildings)),
    }
    
    # Save to file
    with open(output_filename, 'wb') as f:
        pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # === FINAL SUMMARY ===
    print(f"\n{'='*80}")
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    
    print(f"\nData Summary:")
    print(f"  Original CSV rows: {len(df_raw)}")
    print(f"  After cleaning: {len(df_clean)}")
    print(f"  Unique base subjects: {unique_base_subjects}")
    print(f"  Total day-instances: {total_day_instances}")
    print(f"  Average days per subject: {avg_days_per_subject:.2f}")
    
    print(f"\nModality Breakdown:")
    for modality in ['Face-to-Face', 'Online', 'Hybrid']:
        count = subject_modalities.count(modality)
        pct = (count / num_subjects) * 100
        print(f"  {modality}: {count} ({pct:.1f}%)")
    
    print(f"\nDLSU Compliance:")
    print(f"  Maximum subjects per teacher: {MAX_SUBJECTS_PER_TEACHER}")
    print(f"  Average instance capacity: {sum(capacities)/len(capacities):.1f} day-instances")
    print(f"  Equivalent to: ~{sum(capacities)/len(capacities)/avg_days_per_subject:.1f} subjects")
    
    print(f"\nResources:")
    print(f"  Teachers: {num_teachers}")
    print(f"  Rooms: {len(room_codes)}")
    print(f"  Buildings: {len(buildings)}")
    print(f"  Days: {num_days}")
    print(f"  Timeslots: {num_timeslots}")
    print(f"  Subject Areas: {len(area_labels)}")
    print(f"  Sections: {len(section_labels)}")
    
    print(f"\n✓ Cached to: {output_filename}")
    print(f"{'='*80}\n")
    
    return cached_data


if __name__ == "__main__":
    result = preprocess_manila_schedule()
    
    if result:
        print("✅ Manila schedule preprocessing completed successfully!")
        print(f"\nKey Stats:")
        print(f"  • {result['num_unique_base_subjects']} unique base subjects")
        print(f"  • {result['num_subjects']} total subjects (after grouping)")
        print(f"  • ~{result['avg_days_per_subject']:.2f} days per subject")
        print(f"  • NO day suffixes in subject names")
        print("\nNext steps:")
        print("1. Run: python create_timeslots_manila.py")
        print("2. Train: python train_manila_schedule.py --iterations 300")
    else:
        print("❌ Preprocessing failed. Check your CSV file.")