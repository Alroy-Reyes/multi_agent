#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from timetabling_env import TimetablingEnv  # adjust import path if needed


def load_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    # Subjects, timeslots, faculty, rooms
    subjects    = df["Subject"].unique().tolist()
    timeslots   = df["Time"].unique().tolist()
    faculty     = df["Faculty"].unique().tolist()
    room_codes  = df["Room"].unique().tolist()  # flat list of all room IDs

    return {
        "subjects":      subjects,
        "timeslots":     timeslots,
        "faculty":       faculty,
        "num_subjects":   len(subjects),
        "num_timeslots":  len(timeslots),
        "num_teachers":   len(faculty),
        "room_codes":     room_codes,
    }


def main():
    # 0) CSV path
    here     = os.path.dirname(__file__)
    csv_path = os.path.join(here, "../Math_Subjects_Schedule.csv")

    # 1) Load data from CSV
    cfg       = load_from_csv(csv_path)
    subjects  = cfg["subjects"]
    timeslots = cfg["timeslots"]
    faculty   = cfg["faculty"]
    print(f"Detected rooms: {cfg['room_codes']}")

    # 2) Seed for reproducibility
    np.random.seed(42)

    # 3) Instantiate and reset the env
    env = TimetablingEnv(
        num_sahas=4,
        num_teachers=cfg["num_teachers"],
        num_subjects=cfg["num_subjects"],
        num_timeslots=cfg["num_timeslots"],
        room_codes=cfg["room_codes"],
        max_classes_per_teacher=3,  # adjust as needed
    )
    env.reset()

    # 4) Track negotiation stats
    cma_stats = {"total": 0, "resolved": 0, "failed": 0}

    step, max_steps = 0, 200
    while env.agents and step < max_steps:
        agent = env.agent_selection
        obs   = env.observe(agent)

        # CMA picks the slot with highest “free” score, SAHA random
        if agent.startswith("cma"):
            action = int(np.argmax(obs))
        else:
            action = env.action_spaces[agent].sample()

        # step environment
        env.step(action)
        reward = env.rewards[agent]

        # record conflicts
        if agent.startswith("cma") and abs(reward) == 2:
            cma_stats["total"]   += 1
            if reward > 0:
                cma_stats["resolved"] += 1
            else:
                cma_stats["failed"]   += 1

        next_agent = env.agent_selection if env.agents else "None"
        print(f"Step {step:03d} | {agent:5s} → act={action:2d}  r={reward:+.0f} | Next: {next_agent}")
        step += 1

    # 5) Summary
    print(f"\nFinished after {step} steps")
    print(f"CMA conflicts: {cma_stats['total']}  resolved: {cma_stats['resolved']}  failed: {cma_stats['failed']}")

    # 6) Final room schedules, with subject, teacher, and timeslot
    print("\nFinal room schedules:")
    for bldg, sched in env.buildings_room_schedule.items():
        rooms = env.buildings_room_info[bldg]
        print(f"Building {bldg} (rooms {rooms}):")
        # Print header row of timeslot labels
        header = "       " + "  ".join([f"[{t}]" for t in timeslots])
        print(header)
        for room_idx, room_code in enumerate(rooms):
            row_entries = []
            for t_idx, t_label in enumerate(timeslots):
                subj_idx = sched[room_idx, t_idx]
                if subj_idx >= 0:
                    subj_name   = subjects[subj_idx]
                    teacher_id  = env.subject_assignments[subj_idx]
                    teacher_name= faculty[teacher_id]
                    row_entries.append(f"{subj_name}->{teacher_name}")
                else:
                    row_entries.append("--")
            print(f"{room_code:6s}: " + " | ".join(row_entries))
        print()

if __name__ == "__main__":
    main()
