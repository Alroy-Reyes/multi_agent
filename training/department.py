#!/usr/bin/env python3
# check_missing_departments.py - Find rows with missing department data

import pandas as pd
import numpy as np

def check_missing_departments(csv_file="standardized_schedule.csv"):
    """
    Analyze the CSV file to find rows with missing department information.
    Distinguishes between completely empty rows and data rows missing departments.
    """
    print(f"Analyzing: {csv_file}")
    print("=" * 50)
    
    # Load the CSV
    try:
        df = pd.read_csv(csv_file)
        print(f"✓ Loaded CSV successfully")
        print(f"  Total rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        return
    
    # Identify completely empty rows (all fields are empty/null)
    empty_row_mask = df.isna().all(axis=1) | (df.astype(str).eq('').all(axis=1))
    empty_rows = empty_row_mask.sum()
    
    # Identify rows with ANY data (not completely empty)
    data_rows_mask = ~empty_row_mask
    data_rows = data_rows_mask.sum()
    
    print(f"\nRow Classification:")
    print("-" * 20)
    print(f"Total rows: {len(df)}")
    print(f"Completely empty rows: {empty_rows}")
    print(f"Rows with data: {data_rows}")
    
    # Now analyze departments only in data rows
    data_df = df[data_rows_mask].copy()
    
    print(f"\nDepartment Analysis (Data Rows Only):")
    print("-" * 40)
    
    if len(data_df) == 0:
        print("No data rows found!")
        return
    
    # Check various types of missing values in data rows
    # 1. Completely null/NaN values
    null_mask = data_df['Department'].isna()
    null_count = null_mask.sum()
    
    # 2. Empty strings
    empty_mask = data_df['Department'].astype(str).eq('')
    empty_count = empty_mask.sum()
    
    # 3. Whitespace-only strings
    whitespace_mask = data_df['Department'].astype(str).str.strip().eq('')
    whitespace_count = whitespace_mask.sum()
    
    # 4. "nan" strings
    nan_string_mask = data_df['Department'].astype(str).str.lower().eq('nan')
    nan_string_count = nan_string_mask.sum()
    
    # Combined missing mask for data rows
    missing_dept_mask = null_mask | empty_mask | whitespace_mask | nan_string_mask
    missing_dept_count = missing_dept_mask.sum()
    
    print(f"In data rows only:")
    print(f"  Null/NaN departments: {null_count}")
    print(f"  Empty string departments: {empty_count}")
    print(f"  Whitespace-only departments: {whitespace_count}")
    print(f"  'nan' string departments: {nan_string_count}")
    print(f"  Total missing departments: {missing_dept_count}")
    print(f"  Valid departments: {len(data_df) - missing_dept_count}")
    print(f"  Missing percentage: {(missing_dept_count/len(data_df))*100:.1f}%" if len(data_df) > 0 else "  Missing percentage: 0%")
    
    # Show department distribution (only from data rows)
    print(f"\nDepartment Distribution (Data Rows Only):")
    print("-" * 40)
    dept_counts = data_df['Department'].value_counts(dropna=False)
    for dept, count in dept_counts.head(15).items():
        print(f"  {str(dept):<20}: {count:3d} rows")
    
    if len(dept_counts) > 15:
        print(f"  ... and {len(dept_counts) - 15} more departments")
    
    # Show problematic DATA rows (not empty rows)
    if missing_dept_count > 0:
        print(f"\nData Rows with Missing Departments ({missing_dept_count} rows):")
        print("-" * 65)
        
        missing_data_rows = data_df[missing_dept_mask].copy()
        
        # Show sample of problematic rows
        display_rows = missing_data_rows.head(10)
        
        print(f"{'Row':<4} | {'Subject':<20} | {'Faculty':<20} | {'Day':<5} | {'Time':<10} | {'Room':<8}")
        print("-" * 80)
        
        for idx, row in display_rows.iterrows():
            subject = str(row['Subject'])[:20]
            faculty = str(row['Faculty'])[:20] 
            day = str(row['Day'])[:5]
            time = str(row['Time'])[:10]
            room = str(row['Room'])[:8]
            print(f"{idx:<4} | {subject:<20} | {faculty:<20} | {day:<5} | {time:<10} | {room:<8}")
        
        if len(missing_data_rows) > 10:
            print(f"... and {len(missing_data_rows) - 10} more data rows with missing departments")
        
        # Save problematic rows to file
        output_file = "data_rows_missing_departments.csv"
        missing_data_rows.to_csv(output_file, index=True)
        print(f"\n✓ Saved problematic data rows to: {output_file}")
        
        # Show full details of first few problematic rows
        print(f"\nDetailed View of First 3 Problematic Rows:")
        print("-" * 50)
        for i, (idx, row) in enumerate(missing_data_rows.head(3).iterrows()):
            print(f"\nRow {idx}:")
            for col in data_df.columns:
                value = str(row[col])
                print(f"  {col:<12}: '{value}'")
            if i >= 2:
                break
        
    else:
        print(f"\n✓ All data rows have departments! No missing department information in actual scheduling data.")
    
    # Summary recommendations
    print(f"\nSummary & Recommendations:")
    print("-" * 30)
    
    if empty_rows > 0:
        print(f"• Remove {empty_rows} completely empty rows")
    
    if missing_dept_count > 0:
        print(f"• Fix {missing_dept_count} data rows missing departments")
        print(f"• Options: assign 'UNKNOWN', infer from faculty/room, manual review")
    else:
        print(f"• ✓ No department fixes needed")
    
    print(f"• Usable data: {len(data_df) - missing_dept_count} rows ready for training")
    
    print(f"\nClean Dataset Info:")
    print("-" * 20)
    clean_data = data_df[~missing_dept_mask] if missing_dept_count > 0 else data_df
    print(f"Clean rows: {len(clean_data)}")
    print(f"Unique subjects: {clean_data['Subject'].nunique()}")
    print(f"Unique faculty: {clean_data['Faculty'].nunique()}")
    print(f"Unique departments: {clean_data['Department'].nunique()}")
    
    return missing_dept_mask if missing_dept_count > 0 else None

if __name__ == "__main__":
    check_missing_departments()