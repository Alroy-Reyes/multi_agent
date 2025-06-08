# debug_env.py

from timetabling_env import TimetablingEnv
import numpy as np

if __name__ == "__main__":
    # 1) Construct exactly the same TimetablingEnv you pass into RLlib:
    env = TimetablingEnv(
        num_sahas=2,
        num_cmas=2,
        num_teachers=5,
        num_subjects=6,
        num_timeslots=5,
        buildings_room_info={0: ["lecture", "lab"], 1: ["lecture", "lecture"]},
        max_classes_per_teacher=3,
    )

    # 2) Reset the AECEnv.  (PettingZoo AEC API: reset() does not return obs/infos here;
    #    instead we immediately call observe() on agent_selection.)
    env.reset()

    # 3) Manually walk through the “AEC cycle” until no agents remain.
    step = 0
    while env.agents:  # as long as there is at least one active agent
        agent = env.agent_selection

        # Sample a random valid action from that agent's action space
        act_space = env.action_space(agent)
        action = act_space.sample()

        # Step once
        env.step(action)

        # Optionally record when each agent terminates
        if env.terminations.get(agent, False):
            print(f"   → Agent '{agent}' just terminated at step {step}.")

        step += 1
        # (For safety, break out if it goes too long):
        if step > 500:
            print("Environment did not terminate within 500 steps; there is probably a bug.")
            break

    if not env.agents:
        print(f"\n✅  Environment terminated after {step} total steps.")
        print("Final subject_assignments array:", env.subject_assignments)
        print("Final room schedules per building:")
        for bldg_id, schedule in env.buildings_room_schedule.items():
            print(f"  Building {bldg_id} schedule:\n{schedule}")
    env.close()
