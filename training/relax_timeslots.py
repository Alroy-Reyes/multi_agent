"""
FIXED Timeslot Expansion for Per-Day Encoding

Fixes the issue where expansion only reaches 16/224 instead of target 89/224 slots

Key fix: Ensure ALL available timeslots are considered, not just adjacent ones
"""

import pickle
import sys


def expand_perday_timeslots(target_percentage=0.4):
    """
    Expand timeslots for per-day encoding
    
    Args:
        target_percentage: Target coverage (0.4 = 40%)
    """
    
    print("\n" + "="*70)
    print("EXPANDING TIMESLOTS (PER-DAY ENCODING) - FIXED VERSION")
    print("="*70)
    
    cache_file = 'cached_environment_data_MANILA_MODALITY.pkl'
    backup_file = 'cached_environment_data_MANILA_MODALITY_backup.pkl'
    
    # Load data
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    
    num_days = data['num_days']
    num_timeslots = data['num_timeslots']
    total_slots = num_days * num_timeslots
    
    print(f"\nSchedule structure:")
    print(f"  Days: {num_days}")
    print(f"  Timeslots per day: {num_timeslots}")
    print(f"  Total slots: {total_slots}")
    
    # Calculate target timeslots needed per day
    target_total_slots = int(total_slots * target_percentage)
    target_timeslots_per_day = int(target_total_slots / num_days)
    
    # Add 1 if rounding down loses too much
    if (target_timeslots_per_day * num_days) < (target_total_slots - num_days/2):
        target_timeslots_per_day += 1
    
    actual_effective_slots = target_timeslots_per_day * num_days
    actual_coverage = actual_effective_slots / total_slots * 100
    
    print(f"\nTarget coverage: {target_percentage*100:.0f}%")
    print(f"  Total slots needed: {target_total_slots}")
    print(f"  Timeslots per day needed: {target_timeslots_per_day}")
    print(f"  Actual effective slots: {actual_effective_slots}")
    print(f"  Actual coverage: {actual_coverage:.1f}%")
    
    # Get current state
    allowed = data['subject_allowed_timeslots']
    original_counts = [len(slots) for slots in allowed]
    
    print(f"\nCurrent state:")
    print(f"  Average timeslots per subject: {sum(original_counts)/len(original_counts):.1f}")
    print(f"  Min timeslots: {min(original_counts)}")
    print(f"  Max timeslots: {max(original_counts)}")
    print(f"  Average effective slots: {sum(original_counts)/len(original_counts) * num_days:.1f}")
    print(f"  Average coverage: {sum(original_counts)/len(original_counts) * num_days / total_slots * 100:.1f}%")
    
    # Backup
    print(f"\n📁 Creating backup...")
    with open(backup_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"   ✅ Backup saved: {backup_file}")
    
    # Expand each subject
    print(f"\n🔧 Expanding timeslots to {target_timeslots_per_day} per subject...")
    
    new_allowed_timeslots = []
    expanded_count = 0
    
    for i, current_slots in enumerate(allowed):
        current_count = len(current_slots)
        
        if current_count < target_timeslots_per_day:
            # Need to expand
            current_set = set(current_slots)
            
            # FIXED: Build a comprehensive ranked list of candidates
            # Priority 1: Immediately adjacent slots
            priority1 = []
            for slot in current_slots:
                if slot - 1 >= 0 and slot - 1 not in current_set:
                    priority1.append(slot - 1)
                if slot + 1 < num_timeslots and slot + 1 not in current_set:
                    priority1.append(slot + 1)
            
            # Priority 2: Slots within ±2 distance
            priority2 = []
            for slot in current_slots:
                for offset in [-2, 2]:
                    candidate = slot + offset
                    if (0 <= candidate < num_timeslots and 
                        candidate not in current_set and 
                        candidate not in priority1):
                        priority2.append(candidate)
            
            # Priority 3: Slots within ±3 distance
            priority3 = []
            for slot in current_slots:
                for offset in [-3, 3]:
                    candidate = slot + offset
                    if (0 <= candidate < num_timeslots and 
                        candidate not in current_set and 
                        candidate not in priority1 and 
                        candidate not in priority2):
                        priority3.append(candidate)
            
            # Priority 4: Slots within ±5 distance
            priority4 = []
            for slot in current_slots:
                for offset in range(-5, 6):
                    if offset in [-4, -3, -2, -1, 0, 1, 2, 3]:
                        continue  # Already covered
                    candidate = slot + offset
                    if (0 <= candidate < num_timeslots and 
                        candidate not in current_set and 
                        candidate not in priority1 and 
                        candidate not in priority2 and
                        candidate not in priority3):
                        priority4.append(candidate)
            
            # Priority 5: All remaining slots
            priority5 = []
            all_used = (set(priority1) | set(priority2) | set(priority3) | 
                       set(priority4) | current_set)
            for slot in range(num_timeslots):
                if slot not in all_used:
                    priority5.append(slot)
            
            # Combine all candidates in priority order (remove duplicates)
            candidates = []
            for p_list in [priority1, priority2, priority3, priority4, priority5]:
                for slot in p_list:
                    if slot not in candidates:
                        candidates.append(slot)
            
            # Start with current slots and add candidates until target reached
            new_slots = list(current_slots)
            slots_needed = target_timeslots_per_day - len(new_slots)
            
            # FIXED: Ensure we actually add the needed slots
            for candidate in candidates:
                if len(new_slots) >= target_timeslots_per_day:
                    break
                if candidate not in new_slots:
                    new_slots.append(candidate)
            
            # CRITICAL: Verify we reached the target
            if len(new_slots) < target_timeslots_per_day:
                print(f"   ⚠️  WARNING: Subject {i} only reached {len(new_slots)}/{target_timeslots_per_day} timeslots")
                print(f"      Current: {current_slots}")
                print(f"      Available candidates: {len(candidates)}")
                print(f"      num_timeslots: {num_timeslots}")
            
            new_slots.sort()
            new_allowed_timeslots.append(new_slots)
            expanded_count += 1
            
            # Show first 5 expansions
            if i < 5:
                effective_before = current_count * num_days
                effective_after = len(new_slots) * num_days
                print(f"\n   Subject {i} ({data['subject_names'][i][:30]}):")
                print(f"     Before: {current_slots}")
                print(f"             ({current_count} timeslots × {num_days} days = {effective_before} effective slots)")
                print(f"     After:  {new_slots}")
                print(f"             ({len(new_slots)} timeslots × {num_days} days = {effective_after} effective slots)")
                print(f"     Coverage: {effective_before}/{total_slots} ({effective_before/total_slots*100:.1f}%) → {effective_after}/{total_slots} ({effective_after/total_slots*100:.1f}%)")
        else:
            # Already has enough or more timeslots
            new_allowed_timeslots.append(current_slots)
            if i < 5 and current_count > target_timeslots_per_day:
                print(f"\n   Subject {i}: Already has {current_count} timeslots (keeping as-is)")
    
    # Update data
    data['subject_allowed_timeslots'] = new_allowed_timeslots
    
    new_counts = [len(slots) for slots in new_allowed_timeslots]
    new_effective = [c * num_days for c in new_counts]
    
    print(f"\n✅ Expansion complete!")
    print(f"   Subjects expanded: {expanded_count}/{len(allowed)}")
    print(f"\n   Timeslots per subject:")
    print(f"     Before: avg={sum(original_counts)/len(original_counts):.1f}, min={min(original_counts)}, max={max(original_counts)}")
    print(f"     After:  avg={sum(new_counts)/len(new_counts):.1f}, min={min(new_counts)}, max={max(new_counts)}")
    print(f"\n   Effective slots per subject (timeslots × {num_days} days):")
    print(f"     Before: avg={sum(original_counts)*num_days/len(original_counts):.1f}")
    print(f"     After:  avg={sum(new_counts)*num_days/len(new_counts):.1f}")
    print(f"\n   Coverage:")
    print(f"     Before: {sum(original_counts)*num_days/len(original_counts)/total_slots*100:.1f}%")
    print(f"     After:  {sum(new_counts)*num_days/len(new_counts)/total_slots*100:.1f}%")
    
    # Verify target was reached
    avg_effective_slots = sum(new_counts) * num_days / len(new_counts)
    if avg_effective_slots < target_total_slots * 0.95:
        print(f"\n   ⚠️  WARNING: Average effective slots ({avg_effective_slots:.1f}) is below target ({target_total_slots})")
        print(f"       Expected ~{target_timeslots_per_day} timeslots per subject")
        print(f"       Got ~{sum(new_counts)/len(new_counts):.1f} timeslots per subject")
    
    # Recalculate constraint severity
    hard = sum(1 for c in new_effective if c < total_slots * 0.3)
    medium = sum(1 for c in new_effective if total_slots * 0.3 <= c < total_slots * 0.6)
    easy = sum(1 for c in new_effective if c >= total_slots * 0.6)
    
    print(f"\n📊 New constraint severity:")
    print(f"   HARD (< {int(total_slots*0.3)} effective slots):   {hard} subjects ({hard/len(new_counts)*100:.1f}%)")
    print(f"   MEDIUM ({int(total_slots*0.3)}-{int(total_slots*0.6)} effective slots): {medium} subjects ({medium/len(new_counts)*100:.1f}%)")
    print(f"   EASY (≥ {int(total_slots*0.6)} effective slots):    {easy} subjects ({easy/len(new_counts)*100:.1f}%)")
    
    # Save
    print(f"\n💾 Saving expanded constraints...")
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"   ✅ Saved: {cache_file}")
    
    print("\n" + "="*70)
    print("✅ EXPANSION COMPLETE")
    print("="*70)
    print(f"\nResults:")
    print(f"  • Average timeslots: {sum(original_counts)/len(original_counts):.1f} → {sum(new_counts)/len(new_counts):.1f}")
    print(f"  • Average effective slots: {sum(original_counts)*num_days/len(original_counts):.1f} → {sum(new_counts)*num_days/len(new_counts):.1f}")
    print(f"  • Average coverage: {sum(original_counts)*num_days/len(original_counts)/total_slots*100:.1f}% → {sum(new_counts)*num_days/len(new_counts)/total_slots*100:.1f}%")
    print(f"  • Constraint severity: {hard} HARD, {medium} MEDIUM, {easy} EASY")
    
    print(f"\nBackup: {backup_file}")
    print(f"\nNext steps:")
    print(f"  1. Verify expansion: python -c \"import pickle; d=pickle.load(open('{cache_file}','rb')); print(f'Avg timeslots: {{sum(len(s) for s in d[\\\"subject_allowed_timeslots\\\"])/len(d[\\\"subject_allowed_timeslots\\\"]):.1f}}')\"")
    print(f"  2. Start training: python train_manila.py --iterations 50")
    print(f"  3. Expected placement: 82-88%")
    print(f"  4. To restore: copy {backup_file} → {cache_file}")
    print("="*70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--apply":
        # Get target percentage
        target = 0.4  # Default: 40% coverage
        
        if len(sys.argv) > 2:
            try:
                target = float(sys.argv[2])
                if not 0.1 <= target <= 1.0:
                    print("⚠️  Target must be between 0.1 and 1.0")
                    sys.exit(1)
            except ValueError:
                print("⚠️  Invalid target percentage")
                sys.exit(1)
        
        print(f"\n🎯 Applying expansion with {target*100:.0f}% target...")
        expand_perday_timeslots(target)
        
    else:
        # Show info
        print("\n" + "="*70)
        print("TIMESLOT EXPANSION TOOL (PER-DAY ENCODING) - FIXED")
        print("="*70)
        
        print("\nThis script fixes the expansion to actually reach target coverage")
        print("\nYour data uses per-day timeslot encoding:")
        print("  • Current: ~5 timeslots per subject")
        print("  • Effective: ~35 slots (5 timeslots × 7 days)")
        print("  • Coverage: ~15.6% of total 224 slots")
        
        print("\nExpansion options:")
        print("\n  1. MODERATE - 40% coverage (Recommended)")
        print("     Command: python expand_perday_timeslots.py --apply 0.4")
        print("     Result: 13 timeslots × 7 days = 91 effective slots")
        print("     Coverage: 40.6% (91/224 slots)")
        print("     Expected placement: 82-88%")
        
        print("\n  2. AGGRESSIVE - 50% coverage")
        print("     Command: python expand_perday_timeslots.py --apply 0.5")
        print("     Result: 16 timeslots × 7 days = 112 effective slots")
        print("     Coverage: 50.0% (112/224 slots)")
        print("     Expected placement: 88-93%")
        
        print("\n  3. VERY AGGRESSIVE - 60% coverage")
        print("     Command: python expand_perday_timeslots.py --apply 0.6")
        print("     Result: 19 timeslots × 7 days = 133 effective slots")
        print("     Coverage: 59.4% (133/224 slots)")
        print("     Expected placement: 90-95%")
        
        print("\n  4. CONSERVATIVE - 30% coverage")
        print("     Command: python expand_perday_timeslots.py --apply 0.3")
        print("     Result: 10 timeslots × 7 days = 70 effective slots")
        print("     Coverage: 31.3% (70/224 slots)")
        print("     Expected placement: 75-80%")
        
        print("\nWhat the FIX changes:")
        print("  ✅ More comprehensive candidate building (5 priority levels)")
        print("  ✅ Verification that target is actually reached")
        print("  ✅ Better diagnostic output showing effective slots")
        print("  ✅ Warnings if expansion fails to reach target")
        
        print("\nWhat happens:")
        print("  ✅ Creates backup of current cache")
        print("  ✅ Expands timeslots while prioritizing adjacent slots")
        print("  ✅ Keeps existing slots (e.g., [17,18,19,20,21])")
        print("  ✅ Adds nearby slots first, then expands outward")
        print("  ✅ Saves updated cache with verification")
        
        print("\n💡 Recommendation: Start with --apply 0.4")
        print("   This gives 89-91 effective slots per subject (40% coverage)")
        print("="*70 + "\n")