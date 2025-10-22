"""
SAME-DAY DUPLICATE DETECTOR
Finds subjects scheduled multiple times on the same day for same section
"""
import pandas as pd
import sys

def expand_day_codes(day_code):
    """
    Expand day codes like 'TF' into individual days
    
    Common patterns:
    M = Monday
    T = Tuesday  
    W = Wednesday
    TH/H = Thursday
    F = Friday
    TF = Thursday + Friday
    MW = Monday + Wednesday
    etc.
    """
    day_code = str(day_code).upper().strip()
    
    # Single day codes
    day_map = {
        'M': ['Monday'],
        'T': ['Tuesday'],
        'W': ['Wednesday'],
        'H': ['Thursday'],
        'TH': ['Thursday'],
        'F': ['Friday'],
        'S': ['Saturday'],
    }
    
    # Check if it's a single day
    if day_code in day_map:
        return day_map[day_code]
    
    # Handle multi-day codes
    days = []
    
    # Common combinations
    if 'TF' in day_code:
        days.extend(['Thursday', 'Friday'])
        day_code = day_code.replace('TF', '')
    
    if 'MW' in day_code:
        days.extend(['Monday', 'Wednesday'])
        day_code = day_code.replace('MW', '')
    
    if 'MH' in day_code or 'MTH' in day_code:
        days.extend(['Monday', 'Thursday'])
        day_code = day_code.replace('MH', '').replace('MTH', '')
    
    # Parse remaining single characters
    for char in day_code:
        if char in day_map:
            days.extend(day_map[char])
    
    return list(set(days))  # Remove duplicates

def check_same_day_duplicates(csv_path):
    """Check for same subject/section appearing multiple times on same day"""
    
    print("\n" + "="*80)
    print("SAME-DAY DUPLICATE DETECTOR")
    print("="*80)
    
    df = pd.read_csv(csv_path)
    
    print(f"\nTotal entries: {len(df)}")
    
    # Expand each entry to individual days
    expanded_rows = []
    
    for idx, row in df.iterrows():
        days = expand_day_codes(row['Days'])
        
        for day in days:
            expanded_rows.append({
                'Subject': row['Subject'],
                'Section': row['Section'],
                'Day': day,
                'Time': row['Time'],
                'Faculty': row['Faculty'],
                'Room': row['Room'],
                'Original_Days': row['Days']
            })
    
    expanded_df = pd.DataFrame(expanded_rows)
    
    print(f"Expanded to {len(expanded_df)} individual day entries")
    
    # Now check for duplicates on same day
    print("\n" + "="*80)
    print("CHECKING FOR SAME-DAY DUPLICATES")
    print("="*80)
    
    # Group by Subject, Section, Day
    day_counts = expanded_df.groupby(['Subject', 'Section', 'Day']).size().reset_index(name='Count')
    
    # Find duplicates (appearing more than once on same day)
    same_day_dups = day_counts[day_counts['Count'] > 1]
    
    if len(same_day_dups) > 0:
        print(f"\n❌ FOUND {len(same_day_dups)} SAME-DAY DUPLICATES!")
        print("\nSubjects scheduled multiple times on the SAME DAY:\n")
        
        for idx, dup in same_day_dups.iterrows():
            subj = dup['Subject']
            sec = dup['Section']
            day = dup['Day']
            count = dup['Count']
            
            print(f"{subj} ({sec}) - {day}")
            print(f"  Scheduled {count} times on {day}:")
            
            # Get all entries for this subject/section/day
            entries = expanded_df[
                (expanded_df['Subject'] == subj) & 
                (expanded_df['Section'] == sec) & 
                (expanded_df['Day'] == day)
            ]
            
            for _, entry in entries.iterrows():
                print(f"    • {entry['Time']} - {entry['Faculty']} - {entry['Room']} (from '{entry['Original_Days']}')")
            
            print()
        
        print(f"TOTAL SAME-DAY CONFLICTS: {len(same_day_dups)}")
        
    else:
        print("\n✅ NO SAME-DAY DUPLICATES FOUND")
        print("All subjects appear at most once per day for each section")
    
    # Show day code interpretations
    print("\n" + "="*80)
    print("DAY CODE INTERPRETATIONS USED")
    print("="*80)
    
    unique_day_codes = df['Days'].unique()
    
    for code in sorted(unique_day_codes):
        expanded = expand_day_codes(code)
        print(f"  {code:5s} → {', '.join(expanded)}")
    
    print("\n" + "="*80)
    
    # SPECIFIC CHECK: Your example
    print("\nCHECKING YOUR HOMEROOM EXAMPLE:")
    print("="*80)
    
    homeroom_abm11a = df[(df['Subject'] == 'HOMEROOM') & (df['Section'] == 'ABM 11-A')]
    
    if len(homeroom_abm11a) > 0:
        print(f"\nHOMEROOM (ABM 11-A): {len(homeroom_abm11a)} entries found\n")
        
        for idx, row in homeroom_abm11a.iterrows():
            days_expanded = expand_day_codes(row['Days'])
            print(f"Entry {idx}:")
            print(f"  Days code: {row['Days']}")
            print(f"  Expands to: {', '.join(days_expanded)}")
            print(f"  Time: {row['Time']}")
            print(f"  Faculty: {row['Faculty']}")
            print()
        
        # Check for Friday overlap
        friday_entries = expanded_df[
            (expanded_df['Subject'] == 'HOMEROOM') & 
            (expanded_df['Section'] == 'ABM 11-A') & 
            (expanded_df['Day'] == 'Friday')
        ]
        
        print(f"Total Friday HOMEROOM entries: {len(friday_entries)}")
        
        if len(friday_entries) > 1:
            print("❌ CONFLICT: Multiple HOMEROOM sessions on Friday!")
            for _, entry in friday_entries.iterrows():
                print(f"  • {entry['Time']} (from day code '{entry['Original_Days']}')")
        else:
            print("✅ No Friday conflict")
    else:
        print("\nHOMEROOM (ABM 11-A) not found in schedule")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_same_day_duplicates.py <csv_file>")
        sys.exit(1)
    
    check_same_day_duplicates(sys.argv[1])