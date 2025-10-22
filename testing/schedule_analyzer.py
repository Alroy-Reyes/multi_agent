"""
Comprehensive Schedule Validator - Find REAL duplicates and conflicts

This validates the ACTUAL exported schedule, not just the environment's internal state
"""

import pandas as pd
import sys
from collections import defaultdict


def validate_exported_schedule(csv_file='exported_schedule.csv'):
    """
    Validate exported schedule and find ALL duplicates and conflicts
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE SCHEDULE VALIDATION")
    print("="*80)
    
    # Load the CSV
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"❌ File not found: {csv_file}")
        print(f"\nPlease export your schedule first:")
        print(f"  python export_schedule.py --checkpoint <path>")
        return
    
    print(f"\n📊 Schedule Overview:")
    print(f"  Total entries: {len(df)}")
    print(f"  Columns: {', '.join(df.columns)}")
    
    # Required columns
    required = ['Subject', 'Section', 'Faculty', 'Days', 'Time', 'Room']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"\n❌ Missing required columns: {missing}")
        return
    
    # ==================================================================
    # 1. FIND SECTION DUPLICATES (Same subject+section at same time)
    # ==================================================================
    print(f"\n{'='*80}")
    print("1. CHECKING SECTION DUPLICATES")
    print("="*80)
    print("(Same subject + section scheduled multiple times)")
    
    section_counts = defaultdict(int)
    section_details = defaultdict(list)
    
    for idx, row in df.iterrows():
        key = (row['Subject'], row['Section'])
        section_counts[key] += 1
        section_details[key].append({
            'Days': row['Days'],
            'Time': row['Time'],
            'Room': row['Room'],
            'Faculty': row['Faculty']
        })
    
    # Find duplicates (subjects scheduled more than expected)
    duplicates = []
    for (subject, section), count in section_counts.items():
        if count > 2:  # Most subjects should have at most 2 meetings
            duplicates.append((subject, section, count))
    
    if duplicates:
        print(f"\n❌ Found {len(duplicates)} subjects with excessive placements:")
        for subject, section, count in sorted(duplicates, key=lambda x: -x[2])[:20]:
            print(f"\n  {subject} - {section}: {count} placements")
            for detail in section_details[(subject, section)]:
                print(f"    • {detail['Days']} {detail['Time']} | {detail['Room']} | {detail['Faculty']}")
    else:
        print("\n✅ No excessive duplicates found")
    
    # ==================================================================
    # 2. CHECK EXACT DUPLICATES (Same subject+section+time)
    # ==================================================================
    print(f"\n{'='*80}")
    print("2. CHECKING EXACT DUPLICATES")
    print("="*80)
    print("(Same subject + section + time slot)")
    
    exact_duplicates = []
    time_slot_map = defaultdict(list)
    
    for idx, row in df.iterrows():
        key = (row['Subject'], row['Section'], row['Days'], row['Time'])
        time_slot_map[key].append({
            'Room': row['Room'],
            'Faculty': row['Faculty'],
            'index': idx
        })
    
    for key, entries in time_slot_map.items():
        if len(entries) > 1:
            exact_duplicates.append((key, entries))
    
    if exact_duplicates:
        print(f"\n❌ Found {len(exact_duplicates)} EXACT duplicates:")
        for (subject, section, days, time), entries in exact_duplicates[:10]:
            print(f"\n  {subject} - {section} @ {days} {time}:")
            for entry in entries:
                print(f"    • Room: {entry['Room']} | Faculty: {entry['Faculty']}")
    else:
        print("\n✅ No exact duplicates found")
    
    # ==================================================================
    # 3. TEACHER CONFLICTS (Same teacher at same time)
    # ==================================================================
    print(f"\n{'='*80}")
    print("3. CHECKING TEACHER CONFLICTS")
    print("="*80)
    
    teacher_schedule = defaultdict(list)
    
    for idx, row in df.iterrows():
        key = (row['Faculty'], row['Days'], row['Time'])
        teacher_schedule[key].append({
            'Subject': row['Subject'],
            'Section': row['Section'],
            'Room': row['Room']
        })
    
    teacher_conflicts = []
    for key, classes in teacher_schedule.items():
        if len(classes) > 1:
            teacher_conflicts.append((key, classes))
    
    if teacher_conflicts:
        print(f"\n❌ Found {len(teacher_conflicts)} teacher conflicts:")
        for (teacher, days, time), classes in teacher_conflicts[:10]:
            print(f"\n  {teacher} @ {days} {time}:")
            for cls in classes:
                print(f"    • {cls['Subject']} - {cls['Section']} | {cls['Room']}")
    else:
        print("\n✅ No teacher conflicts found")
    
    # ==================================================================
    # 4. SECTION CONFLICTS (Same section at same time)
    # ==================================================================
    print(f"\n{'='*80}")
    print("4. CHECKING SECTION CONFLICTS")
    print("="*80)
    
    section_schedule = defaultdict(list)
    
    for idx, row in df.iterrows():
        key = (row['Section'], row['Days'], row['Time'])
        section_schedule[key].append({
            'Subject': row['Subject'],
            'Faculty': row['Faculty'],
            'Room': row['Room']
        })
    
    section_conflicts = []
    for key, classes in section_schedule.items():
        if len(classes) > 1:
            section_conflicts.append((key, classes))
    
    if section_conflicts:
        print(f"\n❌ Found {len(section_conflicts)} section conflicts:")
        for (section, days, time), classes in section_conflicts[:10]:
            print(f"\n  {section} @ {days} {time}:")
            for cls in classes:
                print(f"    • {cls['Subject']} | {cls['Faculty']} | {cls['Room']}")
    else:
        print("\n✅ No section conflicts found")
    
    # ==================================================================
    # 5. ROOM CONFLICTS (Same room at same time)
    # ==================================================================
    print(f"\n{'='*80}")
    print("5. CHECKING ROOM CONFLICTS")
    print("="*80)
    
    room_schedule = defaultdict(list)
    
    for idx, row in df.iterrows():
        key = (row['Room'], row['Days'], row['Time'])
        room_schedule[key].append({
            'Subject': row['Subject'],
            'Section': row['Section'],
            'Faculty': row['Faculty']
        })
    
    room_conflicts = []
    for key, classes in room_schedule.items():
        if len(classes) > 1:
            room_conflicts.append((key, classes))
    
    if room_conflicts:
        print(f"\n❌ Found {len(room_conflicts)} room conflicts:")
        for (room, days, time), classes in room_conflicts[:10]:
            print(f"\n  {room} @ {days} {time}:")
            for cls in classes:
                print(f"    • {cls['Subject']} - {cls['Section']} | {cls['Faculty']}")
    else:
        print("\n✅ No room conflicts found")
    
    # ==================================================================
    # SUMMARY
    # ==================================================================
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    
    total_conflicts = (
        len(duplicates) +
        len(exact_duplicates) +
        len(teacher_conflicts) +
        len(section_conflicts) +
        len(room_conflicts)
    )
    
    print(f"\nTotal issues found: {total_conflicts}")
    print(f"  • Excessive duplicates: {len(duplicates)}")
    print(f"  • Exact duplicates: {len(exact_duplicates)}")
    print(f"  • Teacher conflicts: {len(teacher_conflicts)}")
    print(f"  • Section conflicts: {len(section_conflicts)}")
    print(f"  • Room conflicts: {len(room_conflicts)}")
    
    if total_conflicts == 0:
        print(f"\n🎉 PERFECT! Schedule has ZERO conflicts!")
    else:
        print(f"\n❌ Schedule has {total_conflicts} issues that need fixing")
    
    # Placement stats
    unique_subjects = df.groupby(['Subject', 'Section']).size()
    print(f"\n📊 Placement Statistics:")
    print(f"  Unique subject-sections: {len(unique_subjects)}")
    print(f"  Total placements: {len(df)}")
    print(f"  Average placements per subject: {len(df) / len(unique_subjects):.2f}")
    print(f"  Subjects with 1 placement: {sum(unique_subjects == 1)}")
    print(f"  Subjects with 2 placements: {sum(unique_subjects == 2)}")
    print(f"  Subjects with 3+ placements: {sum(unique_subjects >= 3)}")
    
    print("="*80 + "\n")
    
    return {
        'total_conflicts': total_conflicts,
        'duplicates': duplicates,
        'exact_duplicates': exact_duplicates,
        'teacher_conflicts': teacher_conflicts,
        'section_conflicts': section_conflicts,
        'room_conflicts': room_conflicts,
    }


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'exported_schedule.csv'
    
    print(f"\n📁 Validating: {csv_file}")
    validate_exported_schedule(csv_file)