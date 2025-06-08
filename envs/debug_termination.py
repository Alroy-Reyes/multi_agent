# envs/debug_termination.py

import numpy as np
from timetabling_env import TimetablingEnv

# ----------------------------------------------------------------------
# This script will:
#   1) Instantiate your TimetablingEnv (v2).
#   2) Step through it with random actions until either:
#       a) all agents have terminated (env.agents becomes empty), or
#       b) we hit a maximum‐step cutoff (to catch infinite loops).
#   3) Print out how many steps it took, or warn if it never terminated.
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # 1) Create the environment exactly as you do in training.
    env = TimetablingEnv(
        num_sahas=2,
        num_cmas=2,
        num_teachers=5,
        num_subjects=6,
        num_timeslots=5,
        buildings_room_info={
            0: ["lecture", "lab"],
            1: ["lecture", "lecture"],
        },
        max_classes_per_teacher=3,
    )

    # 2) Reset it. PettingZoo’s AEC wrapper will call `reset()` internally.
    obs = env.reset()  # we don't actually need obs for this test

    MAX_STEPS = 1000
    for step in range(1, MAX_STEPS + 1):
        # If there are no more active agents, the environment has terminated properly.
        if not env.agents:
            print(f"\n✅ Environment terminated cleanly after {step-1} steps.")
            break

        # Otherwise, grab the currently‐selected agent and sample a random action.
        current_agent = env.agent_selection
        action_space = env.action_space(current_agent)
        random_action = action_space.sample()

        # Step the env with that random action.
        env.step(random_action)

    else:
        # If we never broke out, we've exceeded MAX_STEPS without termination.
        print(f"\n❌ Environment did NOT terminate within {MAX_STEPS} steps.")
