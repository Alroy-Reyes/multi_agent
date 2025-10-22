"""
Plateau Breaker & Diagnostic Tool

When training gets stuck at X% and stops improving, this diagnoses why
and provides solutions to break through the plateau.
"""

import os
import glob
import json


def analyze_training_logs():
    """Analyze Ray logs to see what's happening"""
    
    print("\n" + "="*80)
    print("TRAINING PLATEAU DIAGNOSTIC")
    print("="*80)
    
    # Check for Ray results
    ray_results_dir = "C:/ray_logs"
    
    if not os.path.exists(ray_results_dir):
        print(f"\n❌ Ray logs not found at {ray_results_dir}")
        return
    
    # Find the most recent experiment
    experiments = glob.glob(os.path.join(ray_results_dir, "Manila*"))
    if not experiments:
        print(f"\n❌ No Manila experiments found")
        return
    
    latest_exp = max(experiments, key=os.path.getmtime)
    print(f"\n📁 Analyzing: {latest_exp}")
    
    # Look for result.json or progress.csv
    result_files = glob.glob(os.path.join(latest_exp, "**", "result.json"), recursive=True)
    
    if not result_files:
        print(f"\n⚠️  No result files found")
        return
    
    print(f"\n✅ Found {len(result_files)} result files")
    
    # Analyze the metrics
    print(f"\n" + "="*80)
    print("CONVERGENCE ANALYSIS")
    print("="*80)
    
    # Read latest results
    latest_results = []
    for f in sorted(result_files)[-10:]:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                latest_results.append(data)
        except:
            continue
    
    if not latest_results:
        print("\n❌ Could not read result files")
        return
    
    # Check key metrics
    metrics_to_check = [
        'episode_reward_mean',
        'custom_metrics/placement_rate_mean',
        'info/learner/saha_policy/learner_stats/entropy',
        'info/learner/saha_policy/learner_stats/policy_loss',
        'info/learner/saha_policy/learner_stats/vf_loss',
    ]
    
    print(f"\n📊 Last 5 iterations:")
    print(f"{'Iter':<8} {'Placement%':<12} {'Reward':<10} {'Entropy':<10} {'Status'}")
    print("-" * 60)
    
    for result in latest_results[-5:]:
        iteration = result.get('training_iteration', '?')
        placement = result.get('custom_metrics/placement_rate_mean', 0)
        reward = result.get('episode_reward_mean', 0)
        
        # Check entropy
        entropy = None
        try:
            entropy = result.get('info', {}).get('learner', {}).get('saha_policy', {}).get('learner_stats', {}).get('entropy')
        except:
            pass
        
        status = "✅" if placement > 75 else "⚠️"
        
        print(f"{iteration:<8} {placement:<11.1f}% {reward:<9.2f} {str(entropy):<10} {status}")


def diagnose_plateau():
    """Diagnose why training is stuck"""
    
    print("\n" + "="*80)
    print("PLATEAU DIAGNOSIS")
    print("="*80)
    
    print("""
When training plateaus at 76% and stops improving, it's usually one of these:

1. ❌ ENTROPY COLLAPSED (Most Common)
   • Agent stopped exploring, always does same thing
   • Entropy should be > 0.2, if it's < 0.05 → problem
   • Solution: Increase entropy coefficient

2. ❌ LEARNING RATE TOO LOW
   • Model updates too small to escape local optimum
   • Solution: Increase learning rate or reset schedule

3. ❌ VALUE FUNCTION CONVERGED INCORRECTLY
   • Model learned wrong value estimates
   • Value loss should be decreasing; if stuck high → problem
   • Solution: Increase vf_loss_coeff or clip_param

4. ❌ POLICY SATURATED
   • Found a "good enough" strategy and won't deviate
   • Solution: Add noise, increase KL target

5. ⚠️  HARD CAPACITY LIMIT
   • Actually impossible to do better with current constraints
   • Check if 76% is theoretical maximum
   • Solution: Reduce requirements or hire teachers

6. ⚠️  REWARD SHAPING ISSUE
   • Getting good reward for suboptimal behavior
   • Solution: Adjust reward weights
""")


def provide_solutions():
    """Provide concrete solutions"""
    
    print("\n" + "="*80)
    print("SOLUTIONS TO BREAK PLATEAU")
    print("="*80)
    
    print("""
SOLUTION 1: INCREASE EXPLORATION (Most Effective)
==================================================

In train_manila.py, update PPO config:

.training(
    # ... existing params ...
    
    # CRITICAL: Boost entropy to force exploration
    entropy_coeff=2.5,              # Increased from 1.5
    entropy_coeff_schedule=[
        [0, 2.5],                   # Start MUCH higher
        [50000, 1.5],               # Stay high longer
        [150000, 0.8],              # Reduce slower
        [300000, 0.3],              # Keep some exploration
    ],
    
    # Increase KL target to allow bigger policy changes
    kl_target=0.03,                 # Increased from 0.01
    
    # ... rest ...
)

Expected: Forces agent to try new strategies


SOLUTION 2: RESET LEARNING RATE
================================

.training(
    # ... existing params ...
    
    # Reset to higher learning rate
    lr=1e-3,                        # Increased from 5e-4
    lr_schedule=[
        [0, 1e-3],
        [30000, 5e-4],
        [100000, 2e-4],
    ],
    
    # ... rest ...
)

Expected: Allows bigger updates to escape plateau


SOLUTION 3: INCREASE VALUE FUNCTION FLEXIBILITY
===============================================

.training(
    # ... existing params ...
    
    # Allow value function to adjust more
    vf_clip_param=100.0,            # Increased from 50.0
    vf_loss_coeff=2.0,              # Increased from 1.0
    
    # Allow bigger policy changes
    clip_param=0.4,                 # Increased from 0.3
    
    # ... rest ...
)

Expected: Value function can adapt better


SOLUTION 4: CURRICULUM RESTART
===============================

.training(
    # ... existing params ...
    
    # Add noise to actions
    exploration_config={
        "type": "Random",
        "random_timesteps": 10000,  # Random actions for first 10k steps
    },
    
    # ... rest ...
)

Expected: Breaks out of learned patterns


SOLUTION 5: CHECK IF IT'S A HARD LIMIT
======================================

Run capacity analysis:

    python plateau_analyzer.py

This will tell you if 76% is actually the theoretical maximum
with current teacher capacity.

If capacity shortage > 150 slots:
    → 76% might be the limit
    → Need to reduce requirements or hire teachers

If capacity shortage < 100 slots:
    → Should be able to reach 85-90%
    → Problem is RL training, not capacity


SOLUTION 6: COMBINED APPROACH (RECOMMENDED)
===========================================

Apply ALL of the above:

1. Increase entropy_coeff to 2.5
2. Reset learning rate to 1e-3
3. Increase clip_param to 0.4
4. Increase vf_clip_param to 100.0
5. Train for 300+ iterations (longer than before)

Expected improvement: 76% → 85-90%


SOLUTION 7: NUCLEAR OPTION - FULL RESET
=======================================

If nothing works:

1. Delete Ray logs: rmdir /s C:\\ray_logs
2. Reduce requirements by 150 subjects:
   python smart_bottleneck_reducer.py --apply --reduce 150
3. Start fresh training with boosted exploration:
   python train_manila.py --iterations 300

Expected: 76% → 92-95%
""")


def create_config_template():
    """Create a ready-to-use config"""
    
    config = '''
# =================================================================
# PLATEAU-BREAKING CONFIGURATION
# =================================================================
# Copy this into train_manila.py to break through 76% plateau

ppo_cfg = (
    PPOConfig()
    .environment(
        env="manila_env",
        env_config={'cache_file': cache_file},
        disable_env_checking=True
    )
    .framework("torch")
    .rollouts(
        num_rollout_workers=1,
        rollout_fragment_length=64,
        batch_mode="complete_episodes",
        num_envs_per_worker=1,
    )
    .training(
        gamma=0.95,
        
        # SOLUTION 2: Higher learning rate
        lr=1e-3,                                    # INCREASED
        lr_schedule=[
            [0, 1e-3],
            [30000, 5e-4],
            [100000, 2e-4],
            [200000, 1e-4],
        ],
        
        train_batch_size=512,
        sgd_minibatch_size=256,
        num_sgd_iter=10,
        
        # SOLUTION 3: More flexible value function
        vf_clip_param=100.0,                        # INCREASED
        vf_loss_coeff=2.0,                          # INCREASED
        
        use_gae=True,
        lambda_=0.95,
        
        # SOLUTION 3: Bigger policy changes allowed
        clip_param=0.4,                             # INCREASED
        
        # SOLUTION 1: MUCH higher exploration
        entropy_coeff=2.5,                          # CRITICAL!
        entropy_coeff_schedule=[
            [0, 2.5],                               # Start VERY high
            [50000, 1.5],                           # Stay high
            [150000, 0.8],                          # Reduce slowly
            [300000, 0.3],                          # Keep exploring
        ],
        
        grad_clip=1.0,
        
        # SOLUTION 1: Allow bigger policy updates
        kl_coeff=0.1,
        kl_target=0.03,                             # INCREASED
    )
    .resources(
        num_gpus=1,
    )
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
    )
    .callbacks(EnhancedValidationCallback)
    .experimental(_enable_new_api_stack=False, _disable_preprocessor_api=True)
)

# Train LONGER
# python train_manila.py --iterations 300
'''
    
    with open('plateau_breaking_config.txt', 'w') as f:
        f.write(config)
    
    print("\n" + "="*80)
    print("CONFIGURATION FILE CREATED")
    print("="*80)
    print("\n✅ Saved to: plateau_breaking_config.txt")
    print("\nCopy the contents into train_manila.py")
    print("Then run: python train_manila.py --iterations 300")


if __name__ == "__main__":
    # Run diagnostics
    analyze_training_logs()
    diagnose_plateau()
    provide_solutions()
    
    # Create config
    create_config_template()
    
    print("\n" + "="*80)
    print("QUICK START")
    print("="*80)
    print("""
Fastest way to break through 76% plateau:

1. Update train_manila.py with plateau_breaking_config.txt
2. Run: python train_manila.py --iterations 300
3. Watch entropy in logs - should stay > 0.5 for first 50 iterations
4. Placement should start improving after ~30 iterations

If STILL stuck after 100 iterations:
    → Run: python plateau_analyzer.py
    → Check if 76% is actually the theoretical maximum
    → May need to reduce requirements further
""")