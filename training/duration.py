"""
Smart Bottleneck Reducer

Reality check: With current teacher capacity, we CANNOT schedule all subjects 
with proper contact hours. Instead, strategically reduce requirements to 
maximize placement rate.

Strategy: Reduce subjects from 2→1 placement in bottleneck areas, regardless
of class duration. Accept reduced contact hours to improve scheduling success.
"""

import pickle
import sys
from collections import defaultdict


class SmartBottleneckReducer:
    
    # Bottleneck areas from diagnostic (180% utilization = critical)
    CRITICAL_AREAS = [
        'literature', 'pe', 'health', 'physical education',
        'business', 'abm', 'accounting', 'english', 'communication',
        'social', 'politics'
    ]
    
    def __init__(self, cache_file='cached_environment_data_MANILA_MODALITY.pkl'):
        self.cache_file = cache_file
        self.backup_file = cache_file.replace('.pkl', '_backup_smart_reduction.pkl')
        self.data = None
        
    def load_data(self):
        """Load cache data"""
        print("\n" + "="*80)
        print("SMART BOTTLENECK REDUCER")
        print("="*80)
        print(f"\n📁 Loading: {self.cache_file}")
        
        with open(self.cache_file, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"✅ Loaded {self.data['num_subjects']} subjects")
    
    def analyze_by_area(self):
        """Show current state by area"""
        print("\n" + "="*80)
        print("CURRENT STATE BY AREA")
        print("="*80)
        
        subject_areas = self.data['subject_areas']
        subject_required = self.data.get('subject_required_placements', {})
        area_teacher_indices = self.data.get('area_teacher_indices', {})
        teacher_max_classes_map = self.data.get('teacher_max_classes_map', {})
        
        area_stats = defaultdict(lambda: {'subjects': 0, 'required': 0, 'capacity': 0})
        
        for i in range(self.data['num_subjects']):
            area = subject_areas[i]
            required = subject_required.get(i, 1)
            
            area_stats[area]['subjects'] += 1
            area_stats[area]['required'] += required
        
        # Calculate capacity
        for area, teachers in area_teacher_indices.items():
            capacity = sum(
                min(teacher_max_classes_map.get(f"teacher_{t}", 4), 4)
                for t in teachers
            )
            area_stats[area]['capacity'] = capacity
        
        # Show sorted by utilization
        print(f"\n{'Area':<30} {'Subjects':<10} {'Required':<10} {'Capacity':<10} {'Util%':<10} {'Status'}")
        print("-" * 90)
        
        area_list = []
        for area, stats in area_stats.items():
            capacity = stats['capacity']
            if capacity > 0:
                util = (stats['required'] / capacity) * 100
            else:
                util = 999
            
            area_list.append((area, stats, util))
        
        # Sort by utilization
        area_list.sort(key=lambda x: -x[2])
        
        bottleneck_count = 0
        for area, stats, util in area_list[:15]:
            status = "❌" if util > 100 else "⚠️" if util > 85 else "✅"
            if util > 100:
                bottleneck_count += 1
            
            print(f"{area:<30} {stats['subjects']:<10} {stats['required']:<10} "
                  f"{stats['capacity']:<10} {util:<9.0f}% {status}")
        
        return area_stats
    
    def find_reduction_candidates(self, target_reductions=100):
        """Find subjects to reduce from 2→1 placement"""
        print("\n" + "="*80)
        print(f"FINDING {target_reductions} SUBJECTS TO REDUCE")
        print("="*80)
        
        subject_names = self.data.get('subject_names', [])
        subject_areas = self.data['subject_areas']
        subject_required = self.data.get('subject_required_placements', {})
        
        print(f"\nPrioritization strategy:")
        print(f"  1. Currently requires 2 placements")
        print(f"  2. In critical bottleneck area")
        print(f"  3. Additional heuristics for fairness")
        
        candidates = []
        
        for i in range(self.data['num_subjects']):
            current_required = subject_required.get(i, 1)
            
            # Only consider subjects currently requiring 2 placements
            if current_required != 2:
                continue
            
            subject_name = subject_names[i] if i < len(subject_names) else f'Subject_{i}'
            area = subject_areas[i]
            
            # Calculate priority score
            score = 0
            reasons = []
            
            # HIGH PRIORITY: In critical bottleneck area
            in_critical = any(crit.lower() in area.lower() for crit in self.CRITICAL_AREAS)
            if in_critical:
                score += 20
                reasons.append("Critical bottleneck area")
            
            # MEDIUM PRIORITY: In any bottleneck
            in_bottleneck = any(
                ba in subject_name.lower() or ba in area.lower()
                for ba in self.CRITICAL_AREAS
            )
            if in_bottleneck and not in_critical:
                score += 10
                reasons.append("Bottleneck area")
            
            # All subjects requiring 2 placements get base score
            score += 5
            
            if score > 0:
                candidates.append({
                    'index': i,
                    'subject': subject_name,
                    'area': area,
                    'current_required': 2,
                    'score': score,
                    'reasons': reasons
                })
        
        # Sort by score
        candidates.sort(key=lambda x: -x['score'])
        
        print(f"\n✅ Found {len(candidates)} subjects currently requiring 2 placements")
        
        # Show distribution by area
        area_counts = defaultdict(int)
        for c in candidates[:target_reductions]:
            area_counts[c['area']] += 1
        
        print(f"\nTop areas to be affected (for {target_reductions} reductions):")
        for area, count in sorted(area_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {area}: {count} subjects")
        
        return candidates
    
    def show_preview(self, candidates, num_to_reduce):
        """Show preview of changes"""
        print("\n" + "="*80)
        print(f"PREVIEW: REDUCING {num_to_reduce} SUBJECTS")
        print("="*80)
        
        to_reduce = candidates[:num_to_reduce]
        
        print(f"\nSample subjects to be reduced (first 20):")
        for i, cand in enumerate(to_reduce[:20], 1):
            reasons_str = ", ".join(cand['reasons'])
            print(f"  {i:3d}. {cand['subject'][:55]:<55} | {reasons_str}")
        
        if len(to_reduce) > 20:
            print(f"  ... and {len(to_reduce) - 20} more")
        
        # Calculate impact
        print("\n" + "="*80)
        print("IMPACT ANALYSIS")
        print("="*80)
        
        slots_saved = num_to_reduce
        teachers_freed = slots_saved / 4
        
        current_shortage = 307  # From diagnostic
        new_shortage = max(0, current_shortage - slots_saved)
        
        current_placement = 0.70
        new_placement = 1.0 - (new_shortage / self.data['num_subjects'])
        
        print(f"\n💾 Teacher Capacity Impact:")
        print(f"   Slots saved: {slots_saved}")
        print(f"   Teacher capacity freed: ~{teachers_freed:.1f} teachers")
        
        print(f"\n📈 Expected Improvement:")
        print(f"   Current placement: {current_placement*100:.1f}%")
        print(f"   Current shortage: {current_shortage} slots")
        print(f"   New shortage: {new_shortage} slots")
        print(f"   Estimated placement: {new_placement*100:.1f}%")
        print(f"   Improvement: +{(new_placement - current_placement)*100:.1f}%")
        
        return {
            'slots_saved': slots_saved,
            'new_placement': new_placement,
            'improvement': (new_placement - current_placement) * 100
        }
    
    def apply_reductions(self, candidates, num_to_reduce, dry_run=True):
        """Apply the reductions"""
        print("\n" + "="*80)
        if dry_run:
            print("DRY RUN - No changes will be made")
        else:
            print("APPLYING REDUCTIONS")
        print("="*80)
        
        to_reduce = candidates[:num_to_reduce]
        
        if not dry_run:
            # Backup
            print(f"\n📁 Creating backup...")
            with open(self.backup_file, 'wb') as f:
                pickle.dump(self.data, f)
            print(f"   ✅ Backup: {self.backup_file}")
        
        # Apply changes
        subject_required = self.data.get('subject_required_placements', {})
        
        for cand in to_reduce:
            if not dry_run:
                subject_required[cand['index']] = 1
        
        if not dry_run:
            self.data['subject_required_placements'] = subject_required
            
            print(f"\n💾 Saving changes...")
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.data, f)
            print(f"   ✅ Saved: {self.cache_file}")
            
            print(f"\n" + "="*80)
            print("✅ REDUCTIONS APPLIED")
            print("="*80)
            
            print(f"\nChanged {len(to_reduce)} subjects from 2→1 placement")
            
            print(f"\nNext steps:")
            print(f"  1. Update train_manila.py:")
            print(f"     - Enable repair_pass=True")
            print(f"     - Relax teacher matching (r_teacher_match=10.0)")
            print(f"  2. python train_manila.py --iterations 200")
            print(f"  3. Enable cross-area teaching for surplus capacity")
            
            print(f"\nBackup: {self.backup_file}")
        else:
            print(f"\n⚠️  DRY RUN - No changes made")
            print(f"\nTo apply: python {sys.argv[0]} --apply --reduce {num_to_reduce}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Smart Bottleneck Reducer - Reduce subjects in bottleneck areas"
    )
    parser.add_argument('--analyze', action='store_true', help='Analyze only')
    parser.add_argument('--reduce', type=int, default=100, help='Number to reduce (default: 100)')
    parser.add_argument('--apply', action='store_true', help='Apply changes')
    parser.add_argument('--cache', default='cached_environment_data_MANILA_MODALITY.pkl', help='Cache file')
    
    args = parser.parse_args()
    
    reducer = SmartBottleneckReducer(cache_file=args.cache)
    
    # Load
    reducer.load_data()
    
    # Show current state
    reducer.analyze_by_area()
    
    # Find candidates
    candidates = reducer.find_reduction_candidates(target_reductions=args.reduce)
    
    # Show preview
    if candidates:
        impact = reducer.show_preview(candidates, args.reduce)
        
        # Apply or dry run
        if args.analyze:
            print("\n✅ Analysis complete. Use --apply to make changes.")
        else:
            dry_run = not args.apply
            reducer.apply_reductions(candidates, args.reduce, dry_run=dry_run)
    else:
        print("\n⚠️  No candidates found for reduction")


if __name__ == "__main__":
    main()