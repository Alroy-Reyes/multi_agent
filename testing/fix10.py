"""
Comprehensive verification that FIX #10 is fully implemented
Checks all 5 required components
"""
import sys
import os
import inspect
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def verify_fix10():
    """Run comprehensive FIX #10 verification"""
    
    print("\n" + "="*80)
    print("FIX #10 COMPREHENSIVE VERIFICATION")
    print("="*80)
    
    passed = []
    failed = []
    warnings = []
    
    # ========================================================================
    # STEP 1: Load environment
    # ========================================================================
    print("\n[1/7] Loading environment...")
    
    try:
        cache_file = "cached_environment_data_MANILA_MODALITY.pkl"
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        
        from envs.timetabling_env import ParallelTimetablingEnv
        
        env = ParallelTimetablingEnv(
            num_teachers=data["num_teachers"],
            num_subjects=data["num_subjects"],
            num_days=data["num_days"],
            num_timeslots=data["num_timeslots"],
            room_codes=data["room_codes"],
            subject_codes=data["subject_names"],
            subject_campuses=data["subject_campuses"],
            subject_allowed_rooms=data["subject_allowed_rooms"],
            subject_areas=data["subject_areas"],
            subject_section_idx=data["subject_section_idx"],
            section_labels=data["section_labels"],
            max_classes_per_teacher=4,
            teacher_max_classes_map=data["teacher_max_classes_map"],
            subject_teacher_idx=data["subject_teacher_idx"],
            area_teacher_indices=data["area_teacher_indices"],
        )
        
        print("✅ Environment loaded successfully")
        passed.append("Environment loading")
        
    except Exception as e:
        print(f"❌ Failed to load environment: {e}")
        failed.append("Environment loading")
        return False
    
    # ========================================================================
    # STEP 2: Check if subject_day_usage exists in __init__
    # ========================================================================
    print("\n[2/7] Checking subject_day_usage attribute...")
    
    if hasattr(env, 'subject_day_usage'):
        if isinstance(env.subject_day_usage, dict):
            print("✅ subject_day_usage exists and is a dict")
            passed.append("subject_day_usage attribute")
        else:
            print(f"❌ subject_day_usage exists but wrong type: {type(env.subject_day_usage)}")
            failed.append("subject_day_usage type")
    else:
        print("❌ subject_day_usage attribute NOT FOUND")
        print("\n   ADD THIS to __init__:")
        print("   self.subject_day_usage = {}  # After self.placement_teachers = {}")
        failed.append("subject_day_usage attribute")
    
    # ========================================================================
    # STEP 3: Check if _can_place_on_day method exists
    # ========================================================================
    print("\n[3/7] Checking _can_place_on_day() method...")
    
    if hasattr(env, '_can_place_on_day'):
        # Check method signature
        sig = inspect.signature(env._can_place_on_day)
        params = list(sig.parameters.keys())
        
        if 'subject_idx' in params and 'day_idx' in params:
            print("✅ _can_place_on_day() method exists with correct signature")
            passed.append("_can_place_on_day method")
            
            # Check if it returns tuple
            try:
                test_result = env._can_place_on_day(0, 0, 0)
                if isinstance(test_result, tuple) and len(test_result) == 2:
                    print("✅ Method returns (bool, str) tuple")
                    passed.append("_can_place_on_day return type")
                else:
                    print(f"⚠️ Method returns {type(test_result)}, expected tuple")
                    warnings.append("_can_place_on_day return type")
            except Exception as e:
                print(f"⚠️ Could not test method: {e}")
                warnings.append("_can_place_on_day testing")
        else:
            print(f"❌ _can_place_on_day() has wrong parameters: {params}")
            failed.append("_can_place_on_day signature")
    else:
        print("❌ _can_place_on_day() method NOT FOUND")
        print("\n   ADD THIS METHOD (around line 400):")
        print("""
    def _can_place_on_day(self, subject_idx, day_idx, timeslot_idx):
        used_days = self.subject_day_usage.get(subject_idx, set())
        if not used_days:
            return True, "first_placement"
        if day_idx in used_days:
            return False, "day_duplicate"
        # ... rest of checks
        """)
        failed.append("_can_place_on_day method")
    
    # ========================================================================
    # STEP 4: Test the logic actually works
    # ========================================================================
    print("\n[4/7] Testing day duplicate prevention logic...")
    
    if hasattr(env, '_can_place_on_day') and hasattr(env, 'subject_day_usage'):
        test_subject = 0
        test_day = 0
        
        # Test 1: First placement should be allowed
        can_place, reason = env._can_place_on_day(test_subject, test_day, 0)
        if can_place and reason == "first_placement":
            print("✅ First placement allowed correctly")
            passed.append("First placement logic")
        else:
            print(f"❌ First placement failed: can_place={can_place}, reason={reason}")
            failed.append("First placement logic")
        
        # Test 2: Same day duplicate should be blocked
        env.subject_day_usage[test_subject] = {test_day}
        can_place, reason = env._can_place_on_day(test_subject, test_day, 0)
        
        if can_place == False and reason == "day_duplicate":
            print("✅ Same-day duplicate correctly blocked")
            passed.append("Day duplicate prevention")
        else:
            print(f"❌ Same-day duplicate NOT blocked: can_place={can_place}, reason={reason}")
            print("\n   CHECK _can_place_on_day() implementation:")
            print("   if day_idx in used_days:")
            print("       return False, 'day_duplicate'")
            failed.append("Day duplicate prevention")
        
        # Test 3: Different day should be allowed
        different_day = 2
        can_place, reason = env._can_place_on_day(test_subject, different_day, 0)
        
        if can_place == True:
            print("✅ Different day correctly allowed")
            passed.append("Different day logic")
        else:
            print(f"⚠️ Different day blocked: reason={reason}")
            if reason == "max_placements_reached":
                print("   (This is expected if subject already at max placements)")
                warnings.append("Different day blocked by FIX #12")
            else:
                failed.append("Different day logic")
    else:
        print("⚠️ Skipping logic tests (missing prerequisites)")
        failed.append("Logic testing")
    
    # ========================================================================
    # STEP 5: Check if reset() clears subject_day_usage
    # ========================================================================
    print("\n[5/7] Checking reset() method...")
    
    if hasattr(env, 'subject_day_usage'):
        # Pollute the dict
        env.subject_day_usage[0] = {0, 1, 2}
        env.subject_day_usage[1] = {3, 4}
        
        # Reset
        env.reset()
        
        if env.subject_day_usage == {}:
            print("✅ reset() correctly clears subject_day_usage")
            passed.append("reset() clears tracking")
        else:
            print(f"❌ reset() does NOT clear subject_day_usage: {env.subject_day_usage}")
            print("\n   ADD THIS to reset() method:")
            print("   self.subject_day_usage = {}")
            failed.append("reset() clears tracking")
    else:
        print("⚠️ Cannot test reset (attribute missing)")
        failed.append("reset() testing")
    
    # ========================================================================
    # STEP 6: Check step() method calls _can_place_on_day
    # ========================================================================
    print("\n[6/7] Checking if step() calls _can_place_on_day()...")
    
    if hasattr(env, 'step'):
        step_source = inspect.getsource(env.step)
        
        if '_can_place_on_day' in step_source:
            print("✅ step() method calls _can_place_on_day()")
            passed.append("step() calls check")
            
            # Check if it's before placement
            if step_source.index('_can_place_on_day') < step_source.index('buildings_room_schedule'):
                print("✅ Check is called BEFORE placement")
                passed.append("Check ordering")
            else:
                print("⚠️ Check might be after placement (verify manually)")
                warnings.append("Check ordering")
        else:
            print("❌ step() does NOT call _can_place_on_day()")
            print("\n   ADD THIS to step() BEFORE placement:")
            print("""
    can_place, reason = self._can_place_on_day(subj, d, ts)
    if not can_place:
        rewards[agent] -= 10.0
        self.fail_stats[reason] += 1
        continue
            """)
            failed.append("step() calls check")
    else:
        print("⚠️ Cannot analyze step() method")
        failed.append("step() analysis")
    
    # ========================================================================
    # STEP 7: Check if step() updates subject_day_usage after placement
    # ========================================================================
    print("\n[7/7] Checking if step() updates subject_day_usage...")
    
    if hasattr(env, 'step'):
        step_source = inspect.getsource(env.step)
        
        if 'subject_day_usage' in step_source and '.add(' in step_source:
            print("✅ step() updates subject_day_usage after placement")
            passed.append("step() updates tracking")
        else:
            print("❌ step() does NOT update subject_day_usage")
            print("\n   ADD THIS to step() AFTER placement:")
            print("""
    if subj not in self.subject_day_usage:
        self.subject_day_usage[subj] = set()
    self.subject_day_usage[subj].add(d)
            """)
            failed.append("step() updates tracking")
    else:
        print("⚠️ Cannot analyze step() method")
        failed.append("step() update analysis")
    
    # ========================================================================
    # FINAL REPORT
    # ========================================================================
    print("\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80)
    
    print(f"\n✅ PASSED: {len(passed)}")
    for item in passed:
        print(f"   • {item}")
    
    if warnings:
        print(f"\n⚠️ WARNINGS: {len(warnings)}")
        for item in warnings:
            print(f"   • {item}")
    
    if failed:
        print(f"\n❌ FAILED: {len(failed)}")
        for item in failed:
            print(f"   • {item}")
    
    print("\n" + "="*80)
    
    if not failed:
        print("✅ FIX #10 IS COMPLETE AND WORKING!")
        print("\nYou can now:")
        print("  1. Delete old checkpoints: rm -rf C:/ray_logs/Manila_*")
        print("  2. Retrain from scratch: python training_manila.py --iterations 200")
        print("  3. Expected: Same-day duplicates drop from 121 to 0-5")
        print("="*80)
        return True
    else:
        print("❌ FIX #10 IS INCOMPLETE")
        print(f"\nYou need to fix {len(failed)} issue(s) before retraining.")
        print("\nSee error messages above for specific fixes needed.")
        print("="*80)
        return False


if __name__ == "__main__":
    success = verify_fix10()
    sys.exit(0 if success else 1)