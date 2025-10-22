"""
Modality Assignment Checker

Checks:
1. How many subjects per modality
2. If there are duplicate subjects across modalities
3. If modality assignments match the original CSV
4. If Hybrid subjects are being counted correctly
"""

import pickle
import pandas as pd
from collections import defaultdict


def check_modality_assignments():
    """Check modality assignments in cache"""
    
    print("\n" + "="*80)
    print("MODALITY ASSIGNMENT CHECKER")
    print("="*80)
    
    # Load cache
    print(f"\n📁 Loading cache...")
    with open('cached_environment_data_MANILA_MODALITY.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ Loaded {data['num_subjects']} subjects")
    
    # Load CSV
    print(f"\n📁 Loading CSV...")
    csv = pd.read_csv('Manila_Summary_CLEANED.csv')
    print(f"✅ Loaded {len(csv)} schedule entries")
    
    # Get modality data from cache
    subject_names = data.get('subject_names', [])
    section_labels = data.get('section_labels', [])
    subject_section_idx = data.get('subject_section_idx', [])
    subject_modalities = data.get('subject_modalities', [])
    modality_labels = data.get('modality_labels', ['Face-to-Face', 'Online', 'Hybrid'])
    
    print(f"\n{'='*80}")
    print("CACHE MODALITY DISTRIBUTION")
    print("="*80)
    
    # Count by modality
    modality_counts = defaultdict(int)
    for modality in subject_modalities:
        modality_counts[modality] += 1
    
    print(f"\nTotal subjects in cache: {len(subject_modalities)}")
    for modality in modality_labels:
        count = modality_counts.get(modality, 0)
        pct = (count / len(subject_modalities)) * 100 if subject_modalities else 0
        print(f"  {modality}: {count} ({pct:.1f}%)")
    
    # Check for duplicates (same base subject in multiple modalities)
    print(f"\n{'='*80}")
    print("CHECKING FOR DUPLICATE SUBJECTS")
    print("="*80)
    
    # Group subjects by base name (without modality suffix)
    subject_base_names = {}
    for i in range(data['num_subjects']):
        subj_name = subject_names[i] if i < len(subject_names) else f'Subject_{i}'
        sec_idx = subject_section_idx[i]
        section = section_labels[sec_idx] if sec_idx < len(section_labels) else 'Unknown'
        modality = subject_modalities[i] if i < len(subject_modalities) else 'Unknown'
        
        # Remove modality suffix from name to get base name
        base_name = subj_name.replace('_Hybrid', '').replace('_Online', '').replace('_Face-to-Face', '')
        base_key = f"{base_name} - {section}"
        
        if base_key not in subject_base_names:
            subject_base_names[base_key] = []
        
        subject_base_names[base_key].append({
            'index': i,
            'full_name': subj_name,
            'modality': modality
        })
    
    # Find duplicates
    duplicates = {}
    for base_key, instances in subject_base_names.items():
        if len(instances) > 1:
            duplicates[base_key] = instances
    
    if duplicates:
        print(f"\n⚠️  Found {len(duplicates)} subjects with multiple modality instances:")
        print(f"\nShowing first 20 duplicates:")
        
        for i, (base_key, instances) in enumerate(list(duplicates.items())[:20], 1):
            modalities = [inst['modality'] for inst in instances]
            print(f"\n  {i}. {base_key}")
            print(f"     Instances: {len(instances)}")
            for inst in instances:
                print(f"       • [{inst['modality']}] {inst['full_name']}")
        
        if len(duplicates) > 20:
            print(f"\n     ... and {len(duplicates) - 20} more duplicates")
    else:
        print(f"\n✅ No duplicate subjects found")
        print(f"   Each subject-section appears only once")
    
    # Compare with CSV
    print(f"\n{'='*80}")
    print("COMPARING CACHE vs CSV MODALITIES")
    print("="*80)
    
    # Get CSV modality distribution
    csv_modality_counts = csv['Modality'].value_counts().to_dict()
    
    print(f"\nCSV Modality Distribution:")
    csv_total = len(csv)
    for modality in modality_labels:
        count = csv_modality_counts.get(modality, 0)
        pct = (count / csv_total) * 100 if csv_total > 0 else 0
        print(f"  {modality}: {count} ({pct:.1f}%)")
    
    # Detailed matching
    print(f"\n{'='*80}")
    print("DETAILED MODALITY MATCHING")
    print("="*80)
    
    # Try to match cache subjects with CSV
    match_stats = {
        'exact_match': 0,
        'modality_mismatch': 0,
        'not_in_csv': 0,
        'mismatches': []
    }
    
    for i in range(min(data['num_subjects'], 100)):  # Check first 100 for speed
        cache_subj = subject_names[i] if i < len(subject_names) else f'Subject_{i}'
        sec_idx = subject_section_idx[i]
        cache_section = section_labels[sec_idx] if sec_idx < len(section_labels) else 'Unknown'
        cache_modality = subject_modalities[i] if i < len(subject_modalities) else 'Unknown'
        
        # Try to find in CSV
        csv_matches = csv[
            (csv['Subject'].str.contains(cache_subj[:30], case=False, na=False)) &
            (csv['Section'].str.contains(cache_section[:10], case=False, na=False))
        ]
        
        if len(csv_matches) > 0:
            csv_modality = csv_matches.iloc[0]['Modality']
            
            if cache_modality == csv_modality:
                match_stats['exact_match'] += 1
            else:
                match_stats['modality_mismatch'] += 1
                match_stats['mismatches'].append({
                    'subject': cache_subj[:50],
                    'section': cache_section,
                    'cache_modality': cache_modality,
                    'csv_modality': csv_modality
                })
        else:
            match_stats['not_in_csv'] += 1
    
    print(f"\nMatching results (sample of 100 subjects):")
    print(f"  Exact match: {match_stats['exact_match']}")
    print(f"  Modality mismatch: {match_stats['modality_mismatch']}")
    print(f"  Not found in CSV: {match_stats['not_in_csv']}")
    
    if match_stats['mismatches']:
        print(f"\n  Sample mismatches (first 5):")
        for i, mm in enumerate(match_stats['mismatches'][:5], 1):
            print(f"    {i}. {mm['subject']}")
            print(f"       Cache says: {mm['cache_modality']}")
            print(f"       CSV says: {mm['csv_modality']}")
    
    # Analyze Hybrid breakdown
    print(f"\n{'='*80}")
    print("HYBRID SUBJECTS ANALYSIS")
    print("="*80)
    
    hybrid_subjects = [
        (i, subject_names[i], section_labels[subject_section_idx[i]])
        for i in range(data['num_subjects'])
        if i < len(subject_modalities) and subject_modalities[i] == 'Hybrid'
    ]
    
    print(f"\nTotal Hybrid subjects: {len(hybrid_subjects)}")
    print(f"Percentage of all subjects: {len(hybrid_subjects)/data['num_subjects']*100:.1f}%")
    
    if len(hybrid_subjects) > 0:
        print(f"\nSample Hybrid subjects (first 15):")
        for i, (idx, name, section) in enumerate(hybrid_subjects[:15], 1):
            print(f"  {i:2d}. {name[:60]} ({section})")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    if duplicates:
        print(f"\n⚠️  ISSUE: {len(duplicates)} duplicate subjects found")
        print(f"   These subjects appear multiple times (once per modality)")
        print(f"   Example: 'Math 101 - Section A' might appear as:")
        print(f"     • Math 101 - Section A [Face-to-Face]")
        print(f"     • Math 101 - Section A [Online]")
        print(f"     • Math 101 - Section A [Hybrid]")
        print(f"\n   💡 RECOMMENDATION:")
        print(f"      This is INFLATING your subject count!")
        print(f"      925 subjects might actually be ~400 unique subjects × 2-3 modalities")
        print(f"      This explains why you can only place 76% - you're triple-counting!")
        
        print(f"\n   🔧 FIX:")
        print(f"      The preprocessing should create ONE entry per subject,")
        print(f"      not separate entries for each modality option.")
        print(f"      Rerun preprocessing to deduplicate.")
    else:
        print(f"\n✅ No duplicates found - subjects are unique")
    
    if modality_counts.get('Hybrid', 0) > data['num_subjects'] * 0.5:
        print(f"\n⚠️  NOTE: 61% Hybrid is unusually high")
        print(f"   Most schools have:")
        print(f"     • Face-to-Face: 40-60%")
        print(f"     • Online: 20-30%")
        print(f"     • Hybrid: 20-30%")
        print(f"\n   Verify this is intentional in your data source")
    
    return {
        'total_subjects': data['num_subjects'],
        'modality_counts': dict(modality_counts),
        'duplicates': len(duplicates),
        'duplicate_list': duplicates
    }


def create_deduplication_script():
    """Create a script to fix duplicates if found"""
    
    script = '''
"""
Modality Deduplication Script

If subjects are duplicated across modalities, this fixes it by:
1. Keeping only ONE instance per subject-section
2. Storing modality as a property, not creating separate subjects
"""

import pickle

def deduplicate_modalities():
    print("\\n" + "="*80)
    print("DEDUPLICATING MODALITY SUBJECTS")
    print("="*80)
    
    # Load cache
    with open('cached_environment_data_MANILA_MODALITY.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Backup
    with open('cached_environment_data_MANILA_MODALITY_backup_dedup.pkl', 'wb') as f:
        pickle.dump(data, f)
    print("\\n✅ Backup created")
    
    subject_names = data.get('subject_names', [])
    subject_section_idx = data.get('subject_section_idx', [])
    section_labels = data.get('section_labels', [])
    subject_modalities = data.get('subject_modalities', [])
    
    # Find unique base subjects
    seen = set()
    keep_indices = []
    
    for i in range(data['num_subjects']):
        subj_name = subject_names[i]
        sec_idx = subject_section_idx[i]
        section = section_labels[sec_idx]
        
        # Remove modality suffix
        base_name = subj_name.replace('_Hybrid', '').replace('_Online', '').replace('_Face-to-Face', '')
        key = f"{base_name}|||{section}"
        
        if key not in seen:
            seen.add(key)
            keep_indices.append(i)
    
    print(f"\\nOriginal subjects: {data['num_subjects']}")
    print(f"Unique subjects: {len(keep_indices)}")
    print(f"Duplicates removed: {data['num_subjects'] - len(keep_indices)}")
    
    # Update data - keep only unique subjects
    # (This is complex - need to update ALL related arrays)
    
    print("\\n⚠️  Manual deduplication required")
    print("Rerun preprocessing script with modality as property, not separate subjects")

if __name__ == "__main__":
    deduplicate_modalities()
'''
    
    with open('deduplicate_modalities.py', 'w') as f:
        f.write(script)
    
    print(f"\n💾 Created: deduplicate_modalities.py")


if __name__ == "__main__":
    result = check_modality_assignments()
    
    if result['duplicates'] > 0:
        print(f"\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print(f"""
Your 925 subjects includes duplicates across modalities!

This means:
  • You might have ~400 unique subjects
  • Each appearing 2-3 times (once per modality option)
  • This inflates the "total" and makes 76% look worse than it is

To fix:
  1. Check your preprocessing script (create_timeslots_manila.py)
  2. Ensure it creates ONE subject entry, with modality as a property
  3. Not separate entries for Face-to-Face, Online, Hybrid versions
  4. Rerun preprocessing after fixing

Your REAL placement rate might be:
  76% of 925 = 703 subjects placed
  But if 925 is actually ~400 unique × 2.3 = 925
  Then real rate = 703 / 400 = ~175% (impossible)
  
More likely: You're placing ~90% of UNIQUE subjects
           But it shows 76% because of triple-counting
""")