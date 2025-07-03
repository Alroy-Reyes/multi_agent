# training/train_ppo.py

import sys
import os
import pandas as pd
import numpy as np
from ray import init
from ray.air import RunConfig
from ray.tune import CLIReporter, Tuner
from ray.tune.logger import TBXLoggerCallback
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# allow importing your custom environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.timetabling_env import TimetablingEnv

# ──────────────────────────────
# 1) LOAD & PARSE YOUR CSV
# ──────────────────────────────
csv_path = os.path.join(os.path.dirname(__file__), "../Math_Subjects_Schedule.csv")
df = pd.read_csv(csv_path)

# build full per‐row lists
subject_names   = df["Subject"].tolist()
teacher_names   = df["Faculty"].tolist()
timeslot_labels = df["Time"].tolist()
room_codes      = df["Room"].tolist()

num_subjects   = len(subject_names)
num_teachers   = len(set(teacher_names))   # only for sizing the env
num_timeslots  = len(set(timeslot_labels)) # only for sizing the env
print(f"Detected rooms: {room_codes}")

# helper to build schedule DataFrame
def build_schedule_dataframe(env):
    rows = []
    # for each building b
    for b in env.building_keys:
        rooms = env.buildings_room_info[b]
        sched = env.buildings_room_schedule[b]
        for room_idx, room_code in enumerate(rooms):
            for ts in range(env.num_timeslots):
                subj_idx = sched[room_idx, ts]
                if subj_idx < 0:
                    # empty slot
                    rows.append({
                        "Building": b,
                        "Room":     room_code,
                        "Timeslot": env.timeslot_labels[ts],
                        "Subject":  None,
                        "Teacher":  None,
                    })
                else:
                    rows.append({
                        "Building": b,
                        "Room":     room_code,
                        "Timeslot": env.timeslot_labels[ts],
                        "Subject":  env.subject_names[subj_idx],
                        "Teacher":  env.teacher_names[subj_idx],
                    })
    return pd.DataFrame(rows)

# ──────────────────────────────
# 2) CALLBACKS
# ──────────────────────────────
class ScheduleCallback(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result, **kwargs):
        it  = result["training_iteration"]
        ts  = result["timesteps_total"]
        rew = result["episode_reward_mean"]
        print(f"[DEBUG] train() finished iteration={it:<3}  timesteps={ts:<6}  reward={rew:.2f}")
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)

    def on_episode_end(self, *, worker, base_env, episode, **kwargs):
        # extract the raw pettingzoo env
        sub_env = base_env.get_sub_environments()[0].env
        df_sched = build_schedule_dataframe(sub_env)

        # print the current timetable
        print("\n=== Current Timetable at Episode End ===")
        print(df_sched.to_string(index=False))
        print("========================================\n")

        super().on_episode_end(worker=worker, base_env=base_env, episode=episode, **kwargs)

# ──────────────────────────────
# 3) ENV FACTORY
# ──────────────────────────────
def make_env():
    # pass in our per‐row lists
    base_env = TimetablingEnv(
        num_sahas=4,
        num_teachers=num_teachers,
        num_subjects=num_subjects,
        num_timeslots=num_timeslots,
        room_codes=room_codes,
        max_classes_per_teacher=5,
    )
    # attach the lists for the callback to use
    base_env.subject_names   = subject_names
    base_env.teacher_names   = teacher_names
    base_env.timeslot_labels = timeslot_labels

    base_env.metadata["is_parallelizable"] = False
    return PettingZooEnv(base_env)

# ──────────────────────────────
# 4) MAIN
# ──────────────────────────────
if __name__ == "__main__":
    # Start Ray
    init(ignore_reinit_error=True, include_dashboard=False)

    # Register the env
    register_env("timetabling_env_v5", lambda cfg: make_env())

    # Dummy to grab spaces and building keys
    dummy = make_env()
    dummy.reset()
    raw = dummy.env

    # ──────────────────────────────
    # 5) MULTI-AGENT POLICIES
    # ──────────────────────────────
    # SAHA policy
    obs_s = raw.observation_spaces["saha_0"]
    act_s = raw.action_spaces["saha_0"]
    policies = {
        "saha_policy": (
            None,
            obs_s,
            act_s,
            {"model": {"fcnet_hiddens": [128, 128], "fcnet_activation": "relu"}},
        ),
    }
    # CMA policy per building key
    for b in raw.building_keys:
        obs_c = raw.observation_spaces[f"cma_{b}"]
        act_c = raw.action_spaces[f"cma_{b}"]
        policies[f"cma_{b}_policy"] = (
            None,
            obs_c,
            act_c,
            {"model": {"fcnet_hiddens": [128, 128], "fcnet_activation": "relu"}},
        )

    def policy_mapping_fn(agent_id, episode, **kwargs):
        if agent_id.startswith("saha"):
            return "saha_policy"
        return f"{agent_id}_policy"

    # ──────────────────────────────
    # 6) PPO CONFIG
    # ──────────────────────────────
    ppo_cfg = (
        PPOConfig()
        .environment(env="timetabling_env_v5", disable_env_checking=True)
        .framework("torch")
        .rollouts(
            num_rollout_workers=0,
            rollout_fragment_length=7,
            batch_mode="complete_episodes",
        )
        .training(
            gamma=0.95,
            lr=2e-4,
            train_batch_size=256,
            sgd_minibatch_size=64,
            entropy_coeff=0.01,
        )
        .resources(num_gpus=0)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(policies.keys()),
            count_steps_by="env_steps",
        )
        .callbacks(ScheduleCallback)
    )

    config = ppo_cfg.to_dict()

    # ──────────────────────────────
    # 7) LAUNCH TUNER
    # ──────────────────────────────
    reporter = CLIReporter(
        parameter_columns=["env", "lr", "train_batch_size"],
        metric_columns=[
            "episode_reward_mean",
            "timesteps_total",
            "episode_len_mean",
            "custom_metrics/valid_schedule_mean",
            "custom_metrics/negotiation_success_rate_mean",
        ],
        max_report_frequency=1,
    )

    run_cfg = RunConfig(
        stop={"training_iteration": 100},
        local_dir=os.path.expanduser("~/ray_results"),
        name="PPO_Timetabling_v5",
        callbacks=[TBXLoggerCallback()],
        progress_reporter=reporter,
        verbose=1,
        log_to_file="ppo_train_v5.log",
    )

    tuner = Tuner(
        "PPO",
        param_space=config,
        run_config=run_cfg,
    )

    tuner.fit()
