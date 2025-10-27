"""
SCHEDULE VALIDITY CHECKER V2 - TEACHER & STUDENT POV WITH SEVERITY WEIGHTING

Validates schedule from users' perspectives with nuanced severity-based scoring.

Scoring Methodology:
- Starts at 100 points (perfect)
- Deducts points based on issue severity
- Different weights for critical vs minor issues
- Based on educational research and best practices

UPDATED: Now handles both "Day" and "Days" column names automatically
"""
import pandas as pd
import sys
from collections import defaultdict
from datetime import datetime
import json

class SchedulePOVValidatorV2:
    """Schedule validation with severity-weighted scoring"""
    
    # SEVERITY WEIGHTS - Based on educational research
    TEACHER_WEIGHTS = {
        'conflict': 20,              # Impossible - must fix
        'campus_change_tight': 10,   # <30 min - very difficult
        'campus_change_rushed': 15,  # <15 min - nearly impossible
        'back_to_back': 5,           # No break - tiring
        'very_tight': 8,             # <10 min break - difficult
        'day_10plus_hours': 15,      # Exhausting
        'day_8plus_hours': 8,        # Heavy
        'day_6plus_hours': 3,        # Moderate
        'week_40plus_hours': 25,     # Unsustainable
        'week_35plus_hours': 15,     # Very heavy
        'week_30plus_hours': 10,     # Heavy
        'week_25plus_hours': 5,      # Full load
        'fragmented_5hours': 12,     # Very inefficient
        'fragmented_3hours': 6,      # Inefficient
        'very_early': 5,             # Before 7 AM
        'very_late': 5,              # After 8 PM
    }
    
    STUDENT_WEIGHTS = {
        'conflict': 30,              # Impossible - CRITICAL
        'campus_change_impossible': 20,  # <15 min - can't do
        'campus_change_tight': 12,   # <20 min - very difficult
        'no_break': 10,              # Back-to-back - exhausting
        'very_tight': 12,            # <10 min - rushed
        'day_9plus_hours': 20,       # Too exhausting
        'day_7plus_hours': 10,       # Heavy
        'day_6plus_hours': 5,        # Moderate
        'classes_9plus': 15,         # Too many transitions
        'classes_8plus': 8,          # Many transitions
        'classes_7plus': 4,          # Moderate
        'very_early': 8,             # Before 7 AM - hard to concentrate
        'late_evening': 6,           # After 7 PM - fatigue
        'very_late': 10,             # After 8 PM - unsafe
        'long_day': 8,               # 10+ hour span on campus
    }
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.teacher_details = {}
        self.student_details = {}
        
        print(f"\n📋 Detecting column names...")
        print(f"Available columns: {', '.join(self.df.columns)}")
        
        # AUTO-DETECT: Day column
        day_variations = ['Day', 'Days', 'day', 'days']
        self.day_column = self._detect_column(day_variations, 'Day/Days')
        
        # AUTO-DETECT: Teacher column
        teacher_variations = ['Teacher', 'Teachers', 'Instructor', 'Instructors', 
                             'Faculty', 'faculty', 'teacher', 'teachers']
        self.teacher_column = self._detect_column(teacher_variations, 'Teacher')
        
        # AUTO-DETECT: Section column
        section_variations = ['Section', 'Sections', 'Class', 'section', 'sections']
        self.section_column = self._detect_column(section_variations, 'Section')
        
        # AUTO-DETECT: Subject column
        subject_variations = ['Subject', 'Subjects', 'Course', 'subject', 'subjects']
        self.subject_column = self._detect_column(subject_variations, 'Subject')
        
        # AUTO-DETECT: Room column
        room_variations = ['Room', 'Rooms', 'room', 'rooms']
        self.room_column = self._detect_column(room_variations, 'Room')
        
        # AUTO-DETECT: Campus column (OPTIONAL)
        campus_variations = ['Campus', 'campus', 'Location', 'location', 'Building', 'building']
        try:
            self.campus_column = self._detect_column(campus_variations, 'Campus')
            self.has_campus = True
        except ValueError:
            self.campus_column = None
            self.has_campus = False
            print(f"⚠️  Note: No Campus/Location column found - campus change analysis will be skipped")
        
        # AUTO-DETECT: Time column
        time_variations = ['Time', 'time', 'Schedule', 'schedule']
        self.time_column = self._detect_column(time_variations, 'Time')
        
        print(f"\n✓ Column mapping:")
        print(f"  Day:     '{self.day_column}'")
        print(f"  Teacher: '{self.teacher_column}'")
        print(f"  Section: '{self.section_column}'")
        print(f"  Subject: '{self.subject_column}'")
        print(f"  Room:    '{self.room_column}'")
        if self.has_campus:
            print(f"  Campus:  '{self.campus_column}'")
        else:
            print(f"  Campus:  Not found (will use 'Unknown')")
        print(f"  Time:    '{self.time_column}'")
        
        # Validate time formats and report issues
        self._validate_time_formats()
    
    def _detect_column(self, variations, column_type):
        """Detect which variation of a column name exists in the DataFrame"""
        for var in variations:
            if var in self.df.columns:
                return var
        
        # If not found, raise error with available columns
        raise ValueError(
            f"Could not find {column_type} column. Tried: {variations}\n"
            f"Available columns: {list(self.df.columns)}"
        )
    
    def _validate_time_formats(self):
        """Check for and report time format issues"""
        invalid_count = 0
        
        for idx, time_val in enumerate(self.df[self.time_column]):
            start, end, duration = self.parse_time(time_val)
            if start is None:
                invalid_count += 1
        
        if invalid_count > 0:
            print(f"\n⚠️  WARNING: {invalid_count} entries have invalid time formats and will be skipped.")
            print(f"   These entries won't be included in schedule analysis.")
            print()
        
    def expand_day_codes(self, day_code):
        """Expand day codes into individual days"""
        day_code = str(day_code).upper().strip()
        
        day_map = {
            'M': 'Monday',
            'T': 'Tuesday', 
            'W': 'Wednesday',
            'H': 'Thursday',
            'TH': 'Thursday',
            'F': 'Friday',
            'S': 'Saturday',
        }
        
        if day_code in day_map:
            return [day_map[day_code]]
        
        days = []
        
        if 'TF' in day_code:
            days.extend(['Tuesday', 'Friday'])
            day_code = day_code.replace('TF', '')
        if 'MW' in day_code:
            days.extend(['Monday', 'Wednesday'])
            day_code = day_code.replace('MW', '')
        if 'MH' in day_code:
            days.extend(['Monday', 'Thursday'])
            day_code = day_code.replace('MH', '')
        
        for char in day_code:
            if char in day_map and day_map[char] not in days:
                days.append(day_map[char])
        
        return days if days else [day_code]
    
    def parse_time(self, time_str):
        """Parse time string to minutes"""
        try:
            time_str = str(time_str).strip()
            
            if '-' in time_str:
                start, end = time_str.split('-')
            else:
                return None, None, None
            
            start = start.replace(':', '')
            end = end.replace(':', '')
            
            if len(start) == 4:
                start_hour = int(start[:2])
                start_min = int(start[2:])
            else:
                return None, None, None
                
            if len(end) == 4:
                end_hour = int(end[:2])
                end_min = int(end[2:])
            else:
                return None, None, None
            
            start_minutes = start_hour * 60 + start_min
            end_minutes = end_hour * 60 + end_min
            duration = end_minutes - start_minutes
            
            return start_minutes, end_minutes, duration
        except:
            return None, None, None
    
    def format_time(self, minutes):
        """Convert minutes to time string"""
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"
    
    def times_overlap(self, start1, end1, start2, end2):
        """Check if two time ranges overlap"""
        if None in [start1, end1, start2, end2]:
            return False
        return start1 < end2 and start2 < end1
    
    # ========================================================================
    # TEACHER VALIDATION WITH SEVERITY SCORING
    # ========================================================================
    
    def calculate_teacher_score(self, report):
        """
        Calculate teacher score based on issue severity
        
        FIXED: Properly handles DataFrame checking
        """
        # FIX: Check for None/empty report properly
        if report is None:
            return 100, {}
        
        if 'schedule' not in report:
            return 100, {}
        
        schedule = report.get('schedule')
        if schedule is None or (isinstance(schedule, pd.DataFrame) and schedule.empty):
            return 100, {}
        
        score = 100
        deductions = {}
        
        # CRITICAL: Double-booked (teaching 2 classes at once)
        if report.get('conflicts', 0) > 0:
            penalty = report['conflicts'] * self.TEACHER_WEIGHTS['conflict']
            deductions['conflicts'] = penalty
            score -= penalty
        
        # MAJOR: Campus changes
        for day in schedule['Day'].unique():
            day_schedule = schedule[schedule['Day'] == day].sort_values('Start')
            
            for i in range(len(day_schedule) - 1):
                class1 = day_schedule.iloc[i]
                class2 = day_schedule.iloc[i + 1]
                
                if class1['Campus'] != class2['Campus'] and class1['Campus'] != 'Unknown':
                    gap_minutes = class2['Start'] - class1['End']
                    
                    if gap_minutes < 15:
                        key = 'campus_change_rushed'
                        deductions[key] = deductions.get(key, 0) + self.TEACHER_WEIGHTS[key]
                        score -= self.TEACHER_WEIGHTS[key]
                    elif gap_minutes < 30:
                        key = 'campus_change_tight'
                        deductions[key] = deductions.get(key, 0) + self.TEACHER_WEIGHTS[key]
                        score -= self.TEACHER_WEIGHTS[key]
        
        # MODERATE: Break times
        for day in schedule['Day'].unique():
            day_schedule = schedule[schedule['Day'] == day].sort_values('Start')
            
            for i in range(len(day_schedule) - 1):
                gap_minutes = day_schedule.iloc[i + 1]['Start'] - day_schedule.iloc[i]['End']
                
                if gap_minutes == 0:
                    key = 'back_to_back'
                    deductions[key] = deductions.get(key, 0) + self.TEACHER_WEIGHTS[key]
                    score -= self.TEACHER_WEIGHTS[key]
                elif gap_minutes < 10:
                    key = 'very_tight'
                    deductions[key] = deductions.get(key, 0) + self.TEACHER_WEIGHTS[key]
                    score -= self.TEACHER_WEIGHTS[key]
        
        # MAJOR: Daily hours
        for day in schedule['Day'].unique():
            day_schedule = schedule[schedule['Day'] == day].sort_values('Start')
            
            total_minutes = day_schedule['Duration'].sum()
            hours = total_minutes / 60
            
            if hours >= 10:
                key = 'day_10plus_hours'
                deductions[key] = deductions.get(key, 0) + self.TEACHER_WEIGHTS[key]
                score -= self.TEACHER_WEIGHTS[key]
            elif hours >= 8:
                key = 'day_8plus_hours'
                deductions[key] = deductions.get(key, 0) + self.TEACHER_WEIGHTS[key]
                score -= self.TEACHER_WEIGHTS[key]
            elif hours >= 6:
                key = 'day_6plus_hours'
                deductions[key] = deductions.get(key, 0) + self.TEACHER_WEIGHTS[key]
                score -= self.TEACHER_WEIGHTS[key]
        
        # CRITICAL: Weekly hours
        total_minutes = schedule['Duration'].sum()
        weekly_hours = total_minutes / 60
        
        if weekly_hours >= 40:
            key = 'week_40plus_hours'
            deductions[key] = deductions.get(key, 0) + self.TEACHER_WEIGHTS[key]
            score -= self.TEACHER_WEIGHTS[key]
        elif weekly_hours >= 35:
            key = 'week_35plus_hours'
            deductions[key] = deductions.get(key, 0) + self.TEACHER_WEIGHTS[key]
            score -= self.TEACHER_WEIGHTS[key]
        elif weekly_hours >= 30:
            key = 'week_30plus_hours'
            deductions[key] = deductions.get(key, 0) + self.TEACHER_WEIGHTS[key]
            score -= self.TEACHER_WEIGHTS[key]
        elif weekly_hours >= 25:
            key = 'week_25plus_hours'
            deductions[key] = deductions.get(key, 0) + self.TEACHER_WEIGHTS[key]
            score -= self.TEACHER_WEIGHTS[key]
        
        # MODERATE: Fragmentation (many small gaps)
        for day in schedule['Day'].unique():
            day_schedule = schedule[schedule['Day'] == day].sort_values('Start')
            
            if len(day_schedule) > 1:
                gaps = []
                for i in range(len(day_schedule) - 1):
                    gap = day_schedule.iloc[i + 1]['Start'] - day_schedule.iloc[i]['End']
                    gaps.append(gap)
                
                long_gaps = sum(1 for g in gaps if g >= 180)  # 3+ hour gaps
                
                if long_gaps >= 2:
                    key = 'fragmented_5hours'
                    deductions[key] = deductions.get(key, 0) + self.TEACHER_WEIGHTS[key]
                    score -= self.TEACHER_WEIGHTS[key]
                elif long_gaps >= 1:
                    key = 'fragmented_3hours'
                    deductions[key] = deductions.get(key, 0) + self.TEACHER_WEIGHTS[key]
                    score -= self.TEACHER_WEIGHTS[key]
        
        # MINOR: Time of day issues
        for _, row in schedule.iterrows():
            if row['Start'] < 7 * 60:  # Before 7 AM
                key = 'very_early'
                deductions[key] = deductions.get(key, 0) + self.TEACHER_WEIGHTS[key]
                score -= self.TEACHER_WEIGHTS[key]
            
            if row['End'] > 20 * 60:  # After 8 PM
                key = 'very_late'
                deductions[key] = deductions.get(key, 0) + self.TEACHER_WEIGHTS[key]
                score -= self.TEACHER_WEIGHTS[key]
        
        return max(score, 0), deductions
    
    def validate_teacher_schedule(self, teacher):
        """Validate a specific teacher's schedule"""
        teacher_classes = self.df[self.df[self.teacher_column] == teacher].copy()
        
        if teacher_classes.empty:
            return None
        
        expanded_schedule = []
        for _, row in teacher_classes.iterrows():
            start, end, duration = self.parse_time(row.get(self.time_column, ''))
            
            if start is None:
                continue
            
            day_col_value = row.get(self.day_column, '')
            days = self.expand_day_codes(day_col_value)
            
            for day in days:
                expanded_schedule.append({
                    'Day': day,
                    'Subject': row.get(self.subject_column, 'Unknown'),
                    'Section': row.get(self.section_column, 'Unknown'),
                    'Room': row.get(self.room_column, 'Unknown'),
                    'Campus': row.get(self.campus_column, 'Unknown') if self.has_campus else 'Unknown',
                    'Start': start,
                    'End': end,
                    'Duration': duration,
                    'Time_Str': row.get(self.time_column, '')
                })
        
        schedule_df = pd.DataFrame(expanded_schedule)
        
        if schedule_df.empty:
            return None
        
        conflicts = 0
        conflict_details = []
        
        for day in schedule_df['Day'].unique():
            day_schedule = schedule_df[schedule_df['Day'] == day].sort_values('Start')
            
            for i in range(len(day_schedule)):
                for j in range(i + 1, len(day_schedule)):
                    class1 = day_schedule.iloc[i]
                    class2 = day_schedule.iloc[j]
                    
                    if self.times_overlap(class1['Start'], class1['End'], 
                                        class2['Start'], class2['End']):
                        conflicts += 1
                        conflict_details.append({
                            'day': day,
                            'class1': f"{class1['Subject']} ({class1['Section']})",
                            'class2': f"{class2['Subject']} ({class2['Section']})",
                            'time1': class1['Time_Str'],
                            'time2': class2['Time_Str']
                        })
        
        weekly_hours = schedule_df['Duration'].sum() / 60
        total_classes = len(schedule_df)
        unique_subjects = schedule_df['Subject'].nunique()
        
        report = {
            'teacher': teacher,
            'schedule': schedule_df,
            'conflicts': conflicts,
            'conflict_details': conflict_details,
            'weekly_hours': weekly_hours,
            'total_classes': total_classes,
            'unique_subjects': unique_subjects,
        }
        
        score, deductions = self.calculate_teacher_score(report)
        report['score'] = score
        report['deductions'] = deductions
        
        return report
    
    # ========================================================================
    # STUDENT/SECTION VALIDATION WITH SEVERITY SCORING
    # ========================================================================
    
    def calculate_section_score(self, report):
        """Calculate section score based on issue severity"""
        if report is None:
            return 100, {}
        
        if 'schedule' not in report:
            return 100, {}
        
        schedule = report.get('schedule')
        if schedule is None or (isinstance(schedule, pd.DataFrame) and schedule.empty):
            return 100, {}
        
        score = 100
        deductions = {}
        
        # CRITICAL: Conflicts (attending 2 classes at once)
        if report.get('conflicts', 0) > 0:
            penalty = report['conflicts'] * self.STUDENT_WEIGHTS['conflict']
            deductions['conflicts'] = penalty
            score -= penalty
        
        # MAJOR: Campus changes
        for day in schedule['Day'].unique():
            day_schedule = schedule[schedule['Day'] == day].sort_values('Start')
            
            for i in range(len(day_schedule) - 1):
                class1 = day_schedule.iloc[i]
                class2 = day_schedule.iloc[i + 1]
                
                if class1['Campus'] != class2['Campus'] and class1['Campus'] != 'Unknown':
                    gap_minutes = class2['Start'] - class1['End']
                    
                    if gap_minutes < 15:
                        key = 'campus_change_impossible'
                        deductions[key] = deductions.get(key, 0) + self.STUDENT_WEIGHTS[key]
                        score -= self.STUDENT_WEIGHTS[key]
                    elif gap_minutes < 20:
                        key = 'campus_change_tight'
                        deductions[key] = deductions.get(key, 0) + self.STUDENT_WEIGHTS[key]
                        score -= self.STUDENT_WEIGHTS[key]
        
        # MAJOR: Break times
        for day in schedule['Day'].unique():
            day_schedule = schedule[schedule['Day'] == day].sort_values('Start')
            
            for i in range(len(day_schedule) - 1):
                gap_minutes = day_schedule.iloc[i + 1]['Start'] - day_schedule.iloc[i]['End']
                
                if gap_minutes == 0:
                    key = 'no_break'
                    deductions[key] = deductions.get(key, 0) + self.STUDENT_WEIGHTS[key]
                    score -= self.STUDENT_WEIGHTS[key]
                elif gap_minutes < 10:
                    key = 'very_tight'
                    deductions[key] = deductions.get(key, 0) + self.STUDENT_WEIGHTS[key]
                    score -= self.STUDENT_WEIGHTS[key]
        
        # MAJOR: Daily hours
        for day in schedule['Day'].unique():
            day_schedule = schedule[schedule['Day'] == day].sort_values('Start')
            
            total_minutes = day_schedule['Duration'].sum()
            hours = total_minutes / 60
            
            if hours >= 9:
                key = 'day_9plus_hours'
                deductions[key] = deductions.get(key, 0) + self.STUDENT_WEIGHTS[key]
                score -= self.STUDENT_WEIGHTS[key]
            elif hours >= 7:
                key = 'day_7plus_hours'
                deductions[key] = deductions.get(key, 0) + self.STUDENT_WEIGHTS[key]
                score -= self.STUDENT_WEIGHTS[key]
            elif hours >= 6:
                key = 'day_6plus_hours'
                deductions[key] = deductions.get(key, 0) + self.STUDENT_WEIGHTS[key]
                score -= self.STUDENT_WEIGHTS[key]
        
        # MODERATE: Class count per day
        for day in schedule['Day'].unique():
            day_schedule = schedule[schedule['Day'] == day]
            class_count = len(day_schedule)
            
            if class_count >= 9:
                key = 'classes_9plus'
                deductions[key] = deductions.get(key, 0) + self.STUDENT_WEIGHTS[key]
                score -= self.STUDENT_WEIGHTS[key]
            elif class_count >= 8:
                key = 'classes_8plus'
                deductions[key] = deductions.get(key, 0) + self.STUDENT_WEIGHTS[key]
                score -= self.STUDENT_WEIGHTS[key]
            elif class_count >= 7:
                key = 'classes_7plus'
                deductions[key] = deductions.get(key, 0) + self.STUDENT_WEIGHTS[key]
                score -= self.STUDENT_WEIGHTS[key]
        
        # MODERATE: Time of day issues
        for _, row in schedule.iterrows():
            if row['Start'] < 7 * 60:  # Before 7 AM
                key = 'very_early'
                deductions[key] = deductions.get(key, 0) + self.STUDENT_WEIGHTS[key]
                score -= self.STUDENT_WEIGHTS[key]
            
            if row['End'] > 20 * 60:  # After 8 PM
                key = 'very_late'
                deductions[key] = deductions.get(key, 0) + self.STUDENT_WEIGHTS[key]
                score -= self.STUDENT_WEIGHTS[key]
            elif row['End'] > 19 * 60:  # After 7 PM
                key = 'late_evening'
                deductions[key] = deductions.get(key, 0) + self.STUDENT_WEIGHTS[key]
                score -= self.STUDENT_WEIGHTS[key]
        
        # MODERATE: Long day (time span on campus)
        for day in schedule['Day'].unique():
            day_schedule = schedule[schedule['Day'] == day].sort_values('Start')
            
            if len(day_schedule) > 0:
                first_class_start = day_schedule.iloc[0]['Start']
                last_class_end = day_schedule.iloc[-1]['End']
                day_span_hours = (last_class_end - first_class_start) / 60
                
                if day_span_hours >= 10:
                    key = 'long_day'
                    deductions[key] = deductions.get(key, 0) + self.STUDENT_WEIGHTS[key]
                    score -= self.STUDENT_WEIGHTS[key]
        
        return max(score, 0), deductions
    
    def validate_section_schedule(self, section):
        """Validate a specific section's schedule"""
        section_classes = self.df[self.df[self.section_column] == section].copy()
        
        if section_classes.empty:
            return None
        
        expanded_schedule = []
        for _, row in section_classes.iterrows():
            start, end, duration = self.parse_time(row.get(self.time_column, ''))
            
            if start is None:
                continue
            
            day_col_value = row.get(self.day_column, '')
            days = self.expand_day_codes(day_col_value)
            
            for day in days:
                expanded_schedule.append({
                    'Day': day,
                    'Subject': row.get(self.subject_column, 'Unknown'),
                    'Teacher': row.get(self.teacher_column, 'Unknown'),
                    'Room': row.get(self.room_column, 'Unknown'),
                    'Campus': row.get(self.campus_column, 'Unknown') if self.has_campus else 'Unknown',
                    'Start': start,
                    'End': end,
                    'Duration': duration,
                    'Time_Str': row.get(self.time_column, '')
                })
        
        schedule_df = pd.DataFrame(expanded_schedule)
        
        if schedule_df.empty:
            return None
        
        conflicts = 0
        conflict_details = []
        
        for day in schedule_df['Day'].unique():
            day_schedule = schedule_df[schedule_df['Day'] == day].sort_values('Start')
            
            for i in range(len(day_schedule)):
                for j in range(i + 1, len(day_schedule)):
                    class1 = day_schedule.iloc[i]
                    class2 = day_schedule.iloc[j]
                    
                    if self.times_overlap(class1['Start'], class1['End'], 
                                        class2['Start'], class2['End']):
                        conflicts += 1
                        conflict_details.append({
                            'day': day,
                            'subject1': class1['Subject'],
                            'subject2': class2['Subject'],
                            'time1': class1['Time_Str'],
                            'time2': class2['Time_Str']
                        })
        
        total_subjects = len(schedule_df)
        unique_teachers = schedule_df['Teacher'].nunique()
        
        report = {
            'section': section,
            'schedule': schedule_df,
            'conflicts': conflicts,
            'conflict_details': conflict_details,
            'total_subjects': total_subjects,
            'unique_teachers': unique_teachers,
        }
        
        score, deductions = self.calculate_section_score(report)
        report['score'] = score
        report['deductions'] = deductions
        
        return report
    
    def analyze_all_teachers(self):
        """Analyze all teachers with severity weighting"""
        print("\n" + "="*80)
        print("TEACHER PERSPECTIVE ANALYSIS (Severity-Weighted)")
        print("="*80)
        
        teachers = self.df[self.teacher_column].unique()
        teacher_reports = {}
        total_score = 0
        score_distribution = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
        
        for teacher in teachers:
            report = self.validate_teacher_schedule(teacher)
            if report:
                teacher_reports[teacher] = report
                total_score += report['score']
                
                # Grade distribution
                if report['score'] >= 90:
                    score_distribution['A'] += 1
                elif report['score'] >= 80:
                    score_distribution['B'] += 1
                elif report['score'] >= 70:
                    score_distribution['C'] += 1
                elif report['score'] >= 60:
                    score_distribution['D'] += 1
                else:
                    score_distribution['F'] += 1
        
        avg_score = total_score / len(teachers) if len(teachers) > 0 else 0
        
        print(f"\nTotal Teachers: {len(teachers)}")
        print(f"Average Score: {avg_score:.1f}/100")
        print(f"\nGrade Distribution:")
        print(f"  A (90-100): {score_distribution['A']} teachers")
        print(f"  B (80-89):  {score_distribution['B']} teachers")
        print(f"  C (70-79):  {score_distribution['C']} teachers")
        print(f"  D (60-69):  {score_distribution['D']} teachers")
        print(f"  F (0-59):   {score_distribution['F']} teachers")
        
        # Show worst schedules
        worst_teachers = sorted(teacher_reports.items(), key=lambda x: x[1]['score'])[:5]
        
        if worst_teachers:
            print(f"\n{'='*80}")
            print("TEACHERS WITH LOWEST SCORES")
            print("="*80)
            
            for teacher, report in worst_teachers:
                print(f"\n{teacher}: {report['score']:.1f}/100")
                print(f"  Weekly Hours: {report['weekly_hours']:.1f}")
                print(f"  Total Classes: {report['total_classes']}")
                
                if report.get('deductions'):
                    print(f"  Deductions:")
                    for issue_type, points in sorted(report['deductions'].items(), key=lambda x: x[1], reverse=True)[:3]:
                        print(f"    • {issue_type}: -{points} points")
        
        self.teacher_details = {
            'overall_score': avg_score,
            'distribution': score_distribution,
            'total_count': len(teachers),
            'reports': teacher_reports,
        }
        
        print(f"\n{'='*80}")
        print(f"TEACHER OVERALL SCORE: {avg_score:.1f}/100")
        print("="*80)
        
        return teacher_reports

    def analyze_all_sections(self):
        """Analyze all sections with severity weighting"""
        print("\n" + "="*80)
        print("STUDENT/SECTION PERSPECTIVE ANALYSIS (Severity-Weighted)")
        print("="*80)
        
        sections = self.df[self.section_column].unique()
        section_reports = {}
        total_score = 0
        score_distribution = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
        
        for section in sections:
            report = self.validate_section_schedule(section)
            if report:
                section_reports[section] = report
                total_score += report['score']
                
                # Grade distribution
                if report['score'] >= 90:
                    score_distribution['A'] += 1
                elif report['score'] >= 80:
                    score_distribution['B'] += 1
                elif report['score'] >= 70:
                    score_distribution['C'] += 1
                elif report['score'] >= 60:
                    score_distribution['D'] += 1
                else:
                    score_distribution['F'] += 1
        
        # FIX: Check length instead of truthiness
        avg_score = total_score / len(sections) if len(sections) > 0 else 0
        
        print(f"\nTotal Sections: {len(sections)}")
        print(f"Average Score: {avg_score:.1f}/100")
        print(f"\nGrade Distribution:")
        print(f"  A (90-100): {score_distribution['A']} sections")
        print(f"  B (80-89):  {score_distribution['B']} sections")
        print(f"  C (70-79):  {score_distribution['C']} sections")
        print(f"  D (60-69):  {score_distribution['D']} sections")
        print(f"  F (0-59):   {score_distribution['F']} sections")
        
        # Show worst schedules
        worst_sections = sorted(section_reports.items(), key=lambda x: x[1]['score'])[:5]
        
        if worst_sections:
            print(f"\n{'='*80}")
            print("SECTIONS WITH LOWEST SCORES")
            print("="*80)
            
            for section, report in worst_sections:
                print(f"\n{section}: {report['score']:.1f}/100")
                print(f"  Subjects: {report['total_subjects']}")
                
                if report.get('deductions'):
                    print(f"  Deductions:")
                    for issue_type, points in sorted(report['deductions'].items(), key=lambda x: x[1], reverse=True)[:3]:
                        print(f"    • {issue_type}: -{points} points")
        
        self.student_details = {
            'overall_score': avg_score,
            'distribution': score_distribution,
            'total_count': len(sections),
            'reports': section_reports,
        }
        
        print(f"\n{'='*80}")
        print(f"STUDENT/SECTION OVERALL SCORE: {avg_score:.1f}/100")
        print("="*80)
        
        return section_reports
    
    def generate_comprehensive_report(self):
        """Generate complete report with severity scoring"""
        print("\n" + "="*80)
        print("SCHEDULE VALIDITY CHECK V2 - SEVERITY-WEIGHTED POV ANALYSIS")
        print("="*80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Schedule File: {self.csv_path}")
        print(f"Total Entries: {len(self.df)}")
        print("="*80)
        
        # Run analyses
        teacher_reports = self.analyze_all_teachers()
        section_reports = self.analyze_all_sections()
        
        # Calculate overall score (weighted: students 60%, teachers 40%)
        teacher_score = self.teacher_details['overall_score']
        student_score = self.student_details['overall_score']
        overall_score = (teacher_score * 0.4) + (student_score * 0.6)
        
        print("\n" + "="*80)
        print("OVERALL POV VALIDITY SCORES (SEVERITY-WEIGHTED)")
        print("="*80)
        print(f"Teacher Perspective:  {teacher_score:.1f}/100")
        print(f"Student Perspective:  {student_score:.1f}/100")
        print(f"Overall POV Score:    {overall_score:.1f}/100")
        print("="*80)
        
        # Grade and recommendations
        if overall_score >= 95:
            grade = "A+ (Excellent)"
            status = "✅ SCHEDULE IS EXCELLENT"
            recommendation = "Schedule is ready for deployment with minor to no issues."
        elif overall_score >= 90:
            grade = "A (Excellent)"
            status = "✅ SCHEDULE IS VERY GOOD"
            recommendation = "Schedule is high quality with only minor issues."
        elif overall_score >= 85:
            grade = "B+ (Very Good)"
            status = "✅ SCHEDULE IS GOOD"
            recommendation = "Schedule is acceptable with some room for improvement."
        elif overall_score >= 80:
            grade = "B (Good)"
            status = "⚠️ SCHEDULE IS ACCEPTABLE"
            recommendation = "Schedule works but has several issues to address."
        elif overall_score >= 70:
            grade = "C (Acceptable)"
            status = "⚠️ SCHEDULE HAS ISSUES"
            recommendation = "Schedule needs improvement. Continue training to iteration 200+."
        elif overall_score >= 60:
            grade = "D (Poor)"
            status = "❌ SCHEDULE HAS MAJOR ISSUES"
            recommendation = "Schedule has significant problems. Fix environment and retrain."
        else:
            grade = "F (Fail)"
            status = "❌ SCHEDULE IS UNUSABLE"
            recommendation = "Schedule is not viable. Critical fixes needed in environment code."
        
        print(f"\nGrade: {grade}")
        print(f"Status: {status}")
        print(f"\n{recommendation}")
        
        # Detailed recommendations
        print("\n" + "="*80)
        print("DETAILED RECOMMENDATIONS")
        print("="*80)
        
        if student_score < 70:
            print("\n🚨 CRITICAL STUDENT ISSUES:")
            worst_section = min(section_reports.items(), key=lambda x: x[1]['score'])
            print(f"   Worst section: {worst_section[0]} ({worst_section[1]['score']:.1f}/100)")
            if worst_section[1]['conflicts'] > 0:
                print(f"   • FIX SAME-DAY DUPLICATES (FIX #10 not working)")
                print(f"   • {worst_section[1]['conflicts']} scheduling conflicts found")
        
        if teacher_score < 70:
            print("\n⚠️ TEACHER WORKLOAD ISSUES:")
            worst_teacher = min(teacher_reports.items(), key=lambda x: x[1]['score'])
            print(f"   Worst teacher: {worst_teacher[0]} ({worst_teacher[1]['score']:.1f}/100)")
            if worst_teacher[1]['weekly_hours'] > 35:
                print(f"   • Heavy workload: {worst_teacher[1]['weekly_hours']:.1f} hours/week")
        
        if overall_score >= 85:
            print("\n✅ SCHEDULE QUALITY IS GOOD")
            print("   Consider these minor improvements:")
            print("   • Review lowest-scoring teachers/sections")
            print("   • Optimize break times where possible")
            print("   • Balance workload distribution")
        
        print("\n" + "="*80)
        
        return {
            'overall_score': overall_score,
            'teacher_score': teacher_score,
            'student_score': student_score,
            'grade': grade,
            'status': status,
            'recommendation': recommendation,
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: python schedule_pov_validator_v2.py <csv_file>")
        print("\nExample:")
        print('  python schedule_pov_validator_v2.py "manila_schedule.csv"')
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    validator = SchedulePOVValidatorV2(csv_path)
    summary = validator.generate_comprehensive_report()
    
    # Exit code based on score
    if summary['overall_score'] >= 80:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()