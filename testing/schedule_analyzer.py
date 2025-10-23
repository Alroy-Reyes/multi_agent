"""
SCHEDULE VALIDITY CHECKER V2 - TEACHER & STUDENT POV WITH SEVERITY WEIGHTING

Validates schedule from users' perspectives with nuanced severity-based scoring.

Scoring Methodology:
- Starts at 100 points (perfect)
- Deducts points based on issue severity
- Different weights for critical vs minor issues
- Based on educational research and best practices
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
                    gap = class2['Start'] - class1['End']
                    
                    if gap < 15:  # Nearly impossible
                        deductions['campus_rushed'] = deductions.get('campus_rushed', 0) + self.TEACHER_WEIGHTS['campus_change_rushed']
                        score -= self.TEACHER_WEIGHTS['campus_change_rushed']
                    elif gap < 30:  # Very difficult
                        deductions['campus_tight'] = deductions.get('campus_tight', 0) + self.TEACHER_WEIGHTS['campus_change_tight']
                        score -= self.TEACHER_WEIGHTS['campus_change_tight']
        
        # MODERATE: Break times
        for day in schedule['Day'].unique():
            day_schedule = schedule[schedule['Day'] == day].sort_values('Start')
            
            for i in range(len(day_schedule) - 1):
                class1 = day_schedule.iloc[i]
                class2 = day_schedule.iloc[i + 1]
                gap = class2['Start'] - class1['End']
                
                if gap == 0:  # Back-to-back
                    deductions['back_to_back'] = deductions.get('back_to_back', 0) + self.TEACHER_WEIGHTS['back_to_back']
                    score -= self.TEACHER_WEIGHTS['back_to_back']
                elif gap < 10:  # Very tight
                    deductions['very_tight'] = deductions.get('very_tight', 0) + self.TEACHER_WEIGHTS['very_tight']
                    score -= self.TEACHER_WEIGHTS['very_tight']
        
        # DAILY WORKLOAD
        daily_hours = report.get('daily_hours', {})
        for day, hours in daily_hours.items():
            if hours >= 10:
                deductions['day_10plus'] = deductions.get('day_10plus', 0) + self.TEACHER_WEIGHTS['day_10plus_hours']
                score -= self.TEACHER_WEIGHTS['day_10plus_hours']
            elif hours >= 8:
                deductions['day_8plus'] = deductions.get('day_8plus', 0) + self.TEACHER_WEIGHTS['day_8plus_hours']
                score -= self.TEACHER_WEIGHTS['day_8plus_hours']
            elif hours >= 6:
                deductions['day_6plus'] = deductions.get('day_6plus', 0) + self.TEACHER_WEIGHTS['day_6plus_hours']
                score -= self.TEACHER_WEIGHTS['day_6plus_hours']
        
        # WEEKLY WORKLOAD
        weekly_hours = report.get('weekly_hours', 0)
        if weekly_hours >= 40:
            deductions['week_40plus'] = self.TEACHER_WEIGHTS['week_40plus_hours']
            score -= self.TEACHER_WEIGHTS['week_40plus_hours']
        elif weekly_hours >= 35:
            deductions['week_35plus'] = self.TEACHER_WEIGHTS['week_35plus_hours']
            score -= self.TEACHER_WEIGHTS['week_35plus_hours']
        elif weekly_hours >= 30:
            deductions['week_30plus'] = self.TEACHER_WEIGHTS['week_30plus_hours']
            score -= self.TEACHER_WEIGHTS['week_30plus_hours']
        elif weekly_hours >= 25:
            deductions['week_25plus'] = self.TEACHER_WEIGHTS['week_25plus_hours']
            score -= self.TEACHER_WEIGHTS['week_25plus_hours']
        
        # SCHEDULE FRAGMENTATION (wasted time)
        wasted_minutes = report.get('wasted_time_minutes', 0)
        if wasted_minutes >= 300:  # 5+ hours
            deductions['fragmented_5h'] = self.TEACHER_WEIGHTS['fragmented_5hours']
            score -= self.TEACHER_WEIGHTS['fragmented_5hours']
        elif wasted_minutes >= 180:  # 3+ hours
            deductions['fragmented_3h'] = self.TEACHER_WEIGHTS['fragmented_3hours']
            score -= self.TEACHER_WEIGHTS['fragmented_3hours']
        
        # EARLY/LATE CLASSES
        for day in schedule['Day'].unique():
            day_schedule = schedule[schedule['Day'] == day].sort_values('Start')
            
            if len(day_schedule) > 0:
                first_start = day_schedule.iloc[0]['Start']
                last_end = day_schedule.iloc[-1]['End']
                
                if first_start < 7 * 60:  # Before 7 AM
                    deductions['very_early'] = deductions.get('very_early', 0) + self.TEACHER_WEIGHTS['very_early']
                    score -= self.TEACHER_WEIGHTS['very_early']
                
                if last_end > 20 * 60:  # After 8 PM
                    deductions['very_late'] = deductions.get('very_late', 0) + self.TEACHER_WEIGHTS['very_late']
                    score -= self.TEACHER_WEIGHTS['very_late']
        
        return max(0, score), deductions
    
    def validate_teacher_schedule(self, teacher_name):
        """Validate schedule from teacher's perspective"""
        
        teacher_classes = self.df[self.df['Faculty'] == teacher_name].copy()
        
        if len(teacher_classes) == 0:
            return None
        
        # Expand to individual days
        expanded = []
        for idx, row in teacher_classes.iterrows():
            days = self.expand_day_codes(row['Days'])
            for day in days:
                start, end, duration = self.parse_time(row['Time'])
                expanded.append({
                    'Day': day,
                    'Start': start,
                    'End': end,
                    'Duration': duration,
                    'Time_Str': row['Time'],
                    'Subject': row['Subject'],
                    'Section': row['Section'],
                    'Room': row['Room'],
                    'Campus': row.get('Campus', 'Unknown'),
                })
        
        schedule_df = pd.DataFrame(expanded)
        
        issues = []
        
        # Check conflicts
        conflicts = []
        for day in schedule_df['Day'].unique():
            day_schedule = schedule_df[schedule_df['Day'] == day].sort_values('Start')
            
            for i in range(len(day_schedule)):
                for j in range(i + 1, len(day_schedule)):
                    class1 = day_schedule.iloc[i]
                    class2 = day_schedule.iloc[j]
                    
                    if self.times_overlap(class1['Start'], class1['End'], 
                                         class2['Start'], class2['End']):
                        conflicts.append({
                            'day': day,
                            'class1': f"{class1['Subject']} ({class1['Section']}) at {class1['Time_Str']}",
                            'class2': f"{class2['Subject']} ({class2['Section']}) at {class2['Time_Str']}",
                        })
                        issues.append(f"❌ CONFLICT [{-self.TEACHER_WEIGHTS['conflict']} pts]: {day} - Double-booked")
        
        # Daily hours
        daily_hours = {}
        for day in schedule_df['Day'].unique():
            day_schedule = schedule_df[schedule_df['Day'] == day]
            total_duration = day_schedule['Duration'].sum()
            total_hours = total_duration / 60
            daily_hours[day] = total_hours
        
        # Weekly hours
        total_classes = len(teacher_classes)
        total_weekly_hours = sum(daily_hours.values())
        
        # Calculate wasted time
        wasted_time = 0
        for day in schedule_df['Day'].unique():
            day_schedule = schedule_df[schedule_df['Day'] == day].sort_values('Start')
            
            if len(day_schedule) > 1:
                first_class = day_schedule.iloc[0]['Start']
                last_class = day_schedule.iloc[-1]['End']
                total_teaching = day_schedule['Duration'].sum()
                total_span = last_class - first_class
                gap_time = total_span - total_teaching
                wasted_time += gap_time
        
        report = {
            'teacher': teacher_name,
            'total_classes': total_classes,
            'weekly_hours': total_weekly_hours,
            'daily_hours': daily_hours,
            'conflicts': len(conflicts),
            'wasted_time_minutes': wasted_time,
            'issues': issues,
            'schedule': schedule_df,
        }
        
        # Calculate score
        score, deductions = self.calculate_teacher_score(report)
        report['score'] = score
        report['deductions'] = deductions
        
        return report
    
    # ========================================================================
    # STUDENT VALIDATION WITH SEVERITY SCORING
    # ========================================================================
    
    def calculate_section_score(self, report):
        """
        Calculate section/student score based on issue severity
        
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
        
        # CRITICAL: Students in 2 places at once
        if report.get('conflicts', 0) > 0:
            penalty = report['conflicts'] * self.STUDENT_WEIGHTS['conflict']
            deductions['conflicts'] = penalty
            score -= penalty
        
        # MAJOR: Campus changes (harder for students - no car)
        for day in schedule['Day'].unique():
            day_schedule = schedule[schedule['Day'] == day].sort_values('Start')
            
            for i in range(len(day_schedule) - 1):
                class1 = day_schedule.iloc[i]
                class2 = day_schedule.iloc[i + 1]
                
                if class1['Campus'] != class2['Campus'] and class1['Campus'] != 'Unknown':
                    gap = class2['Start'] - class1['End']
                    
                    if gap < 15:  # Impossible for students
                        deductions['campus_impossible'] = deductions.get('campus_impossible', 0) + self.STUDENT_WEIGHTS['campus_change_impossible']
                        score -= self.STUDENT_WEIGHTS['campus_change_impossible']
                    elif gap < 20:  # Very difficult
                        deductions['campus_tight'] = deductions.get('campus_tight', 0) + self.STUDENT_WEIGHTS['campus_change_tight']
                        score -= self.STUDENT_WEIGHTS['campus_change_tight']
        
        # MODERATE: Break times (students need time to eat, rest, socialize)
        for day in schedule['Day'].unique():
            day_schedule = schedule[schedule['Day'] == day].sort_values('Start')
            
            for i in range(len(day_schedule) - 1):
                class1 = day_schedule.iloc[i]
                class2 = day_schedule.iloc[i + 1]
                gap = class2['Start'] - class1['End']
                
                if gap == 0:  # No break
                    deductions['no_break'] = deductions.get('no_break', 0) + self.STUDENT_WEIGHTS['no_break']
                    score -= self.STUDENT_WEIGHTS['no_break']
                elif gap < 10:  # Very tight
                    deductions['very_tight'] = deductions.get('very_tight', 0) + self.STUDENT_WEIGHTS['very_tight']
                    score -= self.STUDENT_WEIGHTS['very_tight']
        
        # DAILY CLASS HOURS (cognitive load)
        daily_hours = report.get('daily_hours', {})
        for day, hours in daily_hours.items():
            if hours >= 9:
                deductions['day_9plus'] = deductions.get('day_9plus', 0) + self.STUDENT_WEIGHTS['day_9plus_hours']
                score -= self.STUDENT_WEIGHTS['day_9plus_hours']
            elif hours >= 7:
                deductions['day_7plus'] = deductions.get('day_7plus', 0) + self.STUDENT_WEIGHTS['day_7plus_hours']
                score -= self.STUDENT_WEIGHTS['day_7plus_hours']
            elif hours >= 6:
                deductions['day_6plus'] = deductions.get('day_6plus', 0) + self.STUDENT_WEIGHTS['day_6plus_hours']
                score -= self.STUDENT_WEIGHTS['day_6plus_hours']
        
        # NUMBER OF CLASSES (transitions are exhausting)
        daily_classes = report.get('daily_classes', {})
        for day, num_classes in daily_classes.items():
            if num_classes >= 9:
                deductions['classes_9plus'] = deductions.get('classes_9plus', 0) + self.STUDENT_WEIGHTS['classes_9plus']
                score -= self.STUDENT_WEIGHTS['classes_9plus']
            elif num_classes >= 8:
                deductions['classes_8plus'] = deductions.get('classes_8plus', 0) + self.STUDENT_WEIGHTS['classes_8plus']
                score -= self.STUDENT_WEIGHTS['classes_8plus']
            elif num_classes >= 7:
                deductions['classes_7plus'] = deductions.get('classes_7plus', 0) + self.STUDENT_WEIGHTS['classes_7plus']
                score -= self.STUDENT_WEIGHTS['classes_7plus']
        
        # EARLY/LATE CLASSES (affects learning)
        for day in schedule['Day'].unique():
            day_schedule = schedule[schedule['Day'] == day].sort_values('Start')
            
            if len(day_schedule) > 0:
                first_start = day_schedule.iloc[0]['Start']
                last_end = day_schedule.iloc[-1]['End']
                
                if first_start < 7 * 60:  # Before 7 AM - hard to concentrate
                    deductions['very_early'] = deductions.get('very_early', 0) + self.STUDENT_WEIGHTS['very_early']
                    score -= self.STUDENT_WEIGHTS['very_early']
                
                if last_end > 20 * 60:  # After 8 PM - safety concern
                    deductions['very_late'] = deductions.get('very_late', 0) + self.STUDENT_WEIGHTS['very_late']
                    score -= self.STUDENT_WEIGHTS['very_late']
                elif last_end > 19 * 60:  # After 7 PM - fatigue
                    deductions['late_evening'] = deductions.get('late_evening', 0) + self.STUDENT_WEIGHTS['late_evening']
                    score -= self.STUDENT_WEIGHTS['late_evening']
                
                # Long day on campus
                span = last_end - first_start
                if span > 10 * 60:  # 10+ hours
                    deductions['long_day'] = deductions.get('long_day', 0) + self.STUDENT_WEIGHTS['long_day']
                    score -= self.STUDENT_WEIGHTS['long_day']
        
        return max(0, score), deductions
    
    def validate_section_schedule(self, section_name):
        """Validate schedule from section's perspective"""
        
        section_classes = self.df[self.df['Section'] == section_name].copy()
        
        if len(section_classes) == 0:
            return None
        
        # Expand to individual days
        expanded = []
        for idx, row in section_classes.iterrows():
            days = self.expand_day_codes(row['Days'])
            for day in days:
                start, end, duration = self.parse_time(row['Time'])
                expanded.append({
                    'Day': day,
                    'Start': start,
                    'End': end,
                    'Duration': duration,
                    'Time_Str': row['Time'],
                    'Subject': row['Subject'],
                    'Faculty': row['Faculty'],
                    'Room': row['Room'],
                    'Campus': row.get('Campus', 'Unknown'),
                })
        
        schedule_df = pd.DataFrame(expanded)
        
        issues = []
        
        # Check conflicts
        conflicts = []
        for day in schedule_df['Day'].unique():
            day_schedule = schedule_df[schedule_df['Day'] == day].sort_values('Start')
            
            for i in range(len(day_schedule)):
                for j in range(i + 1, len(day_schedule)):
                    class1 = day_schedule.iloc[i]
                    class2 = day_schedule.iloc[j]
                    
                    if self.times_overlap(class1['Start'], class1['End'], 
                                         class2['Start'], class2['End']):
                        conflicts.append({
                            'day': day,
                            'class1': f"{class1['Subject']} at {class1['Time_Str']}",
                            'class2': f"{class2['Subject']} at {class2['Time_Str']}",
                        })
                        issues.append(f"❌ CONFLICT [{-self.STUDENT_WEIGHTS['conflict']} pts]: {day} - Students in 2 places at once")
        
        # Daily stats
        daily_hours = {}
        daily_classes = {}
        for day in schedule_df['Day'].unique():
            day_schedule = schedule_df[schedule_df['Day'] == day]
            total_duration = day_schedule['Duration'].sum()
            total_hours = total_duration / 60
            num_classes = len(day_schedule)
            daily_hours[day] = total_hours
            daily_classes[day] = num_classes
        
        report = {
            'section': section_name,
            'total_subjects': len(section_classes),
            'daily_hours': daily_hours,
            'daily_classes': daily_classes,
            'conflicts': len(conflicts),
            'issues': issues,
            'schedule': schedule_df,
        }
        
        # Calculate score
        score, deductions = self.calculate_section_score(report)
        report['score'] = score
        report['deductions'] = deductions
        
        return report
    
    # ========================================================================
    # ANALYSIS FUNCTIONS
    # ========================================================================
    
    def analyze_all_teachers(self):
        """Analyze all teachers with severity weighting"""
        print("\n" + "="*80)
        print("TEACHER PERSPECTIVE ANALYSIS (Severity-Weighted)")
        print("="*80)
        
        teachers = self.df['Faculty'].unique()
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
        
        # FIX: Check length instead of truthiness
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
                print(f"  Classes: {report['total_classes']} | Weekly Hours: {report['weekly_hours']:.1f}")
                
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
        
        sections = self.df['Section'].unique()
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
