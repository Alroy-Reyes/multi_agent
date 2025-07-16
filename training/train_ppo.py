# train_timetabling_v5_with_days_and_compact_summary.py

import sys
import os
import re
import pandas as pd
import numpy as np
from ray import init
from ray.air import RunConfig
from ray.tune import CLIReporter, Tuner
from ray.air.config import CheckpointConfig
from ray.tune.logger import TBXLoggerCallback
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from torch.utils.tensorboard import SummaryWriter

# allow importing your custom environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.timetabling_env import TimetablingEnv

# ─── 1) LOAD & PARSE YOUR CSV ─────────────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(__file__), "../standardized_schedule-TERM1-2022-2023.csv")
df = pd.read_csv(csv_path)

orig_subjects = df["Subject"].tolist()
orig_teachers = df["Faculty"].tolist()

# ─── 2) EXTRACT EXACT DAY CODES ────────────────────────────────────────────────
day_lists = (
    df["Day"]
      .dropna().astype(str)
      .str.replace(r"\n", "", regex=True)
      .str.findall(r"MH|TF|[MTHWF]")
      .tolist()
)
flat_days  = {d for sub in day_lists for d in sub}
order      = ["M","T","W","H","F","MH","TF"]
day_labels = [d for d in order if d in flat_days]
num_days   = len(day_labels)

# ─── 3) EXPAND EACH COURSE FOR EACH DAY CODE ───────────────────────────────────
df_rooms           = df[df["Room"].notna()]
room_to_campus     = df_rooms.set_index("Room")["Campus"].to_dict()
building_to_campus = {r[0]: c for r, c in room_to_campus.items()}
teacher_to_campus  = (
    df[["Faculty","Campus"]]
      .drop_duplicates("Faculty")
      .set_index("Faculty")["Campus"]
      .to_dict()
)

expanded_subjects = []
expanded_teachers = []
expanded_campuses = []

for subj, teach, days in zip(orig_subjects, orig_teachers, day_lists):
    allowed = [b for b, c in building_to_campus.items() if c == teacher_to_campus[teach]]
    for d in days:
        expanded_subjects.append(subj)
        expanded_teachers.append(teach)
        expanded_campuses.append(allowed)

subject_names    = expanded_subjects
teacher_names    = expanded_teachers
subject_campuses = expanded_campuses
num_subjects     = len(subject_names)
num_teachers     = len(set(teacher_names))

# ─── 4) TIMESLOT LABELS ────────────────────────────────────────────────────────
timeslot_labels = (
    df["Time"]
      .dropna().astype(str)
      .str.split(r"\n").explode()
      .str.strip().loc[lambda s: s != ""]
      .drop_duplicates()
      .tolist()
)
num_timeslots = len(timeslot_labels)
room_codes    = sorted(df_rooms["Room"].unique().tolist())

# ─── 5) BUILD SCHEDULE DATAFRAME ──────────────────────────────────────────────
def build_schedule_dataframe(env):
    rows = []
    for b in env.building_keys:
        rooms = env.buildings_room_info[b]
        sched = env.buildings_room_schedule[b]  # (n_rooms, num_days, num_timeslots)
        for i, room in enumerate(rooms):
            campus = env.room_to_campus.get(room, "UNKNOWN")
            for day_idx, day in enumerate(env.day_labels):
                for ts in range(env.num_timeslots):
                    sub = sched[i, day_idx, ts]
                    rows.append({
                        "Campus":   campus,
                        "Room":     room,
                        "Day":      day,
                        "Timeslot": env.timeslot_labels[ts],
                        "Subject":  None if sub < 0 else env.subject_names[sub],
                        "Teacher":  None if sub < 0 else env.teacher_names[sub],
                    })
    return pd.DataFrame(rows)

# ─── 6) CUSTOM CALLBACK WITH COMPACT SUMMARY ───────────────────────────────────
class ScheduleCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.writer = SummaryWriter(log_dir="C:/ray_logs/custom_tensorboard")

    def on_train_result(self, *, algorithm, result, **kwargs):
        it, ts, rew = (
            result["training_iteration"],
            result["timesteps_total"],
            result["episode_reward_mean"],
        )
        print(f"[DEBUG] iter={it:<3} timesteps={ts:<6} reward={rew:.2f}")
        self.writer.add_scalar("Reward/EpisodeRewardMean", rew, it)
        self.writer.add_scalar("Timesteps/Total", ts, it)
        for k, v in result.get("custom_metrics", {}).items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"Custom/{k}", v, it)

    def on_episode_end(self, *, worker, base_env, episode, **kwargs):
        sub_env = base_env.get_sub_environments()[0].env
        df_sched = build_schedule_dataframe(sub_env).fillna("")

        # print full timetable
        print("\n=== Full Timetable ===")
        print(df_sched.to_string(index=False))
        print("======================\n")

        # compact one-line summary per Subject/Teacher
        assigned = df_sched[df_sched["Subject"].notna()]
        summary = (
            assigned
            .groupby(["Subject", "Teacher"])
            .agg(
                Rooms     = ("Room"    , lambda x: ", ".join(sorted(set(x)))      ),
                Days      = ("Day"     , lambda x: ", ".join(sorted(set(x)))      ),
                Timeslots = ("Timeslot", lambda x: ", ".join(sorted(set(x)))      ),
            )
            .reset_index()
        )

        print("=== Assignment Summary ===")
        for _, row in summary.iterrows():
            print(
                f"{row.Subject} ({row.Teacher}) -  "
                f"Rooms: {row.Rooms} | "
                f"Days: {row.Days} | "
                f"Timeslots: {row.Timeslots}"
            )
            # blank line after each summary entry
            print()
        print("==========================\n")

    def __del__(self):
        if hasattr(self, "writer"):
            self.writer.close()

# ─── 7) ENV FACTORY ──────────────────────────────────────────────────────────
def make_env():
    env = TimetablingEnv(
        num_sahas=4,
        num_teachers=num_teachers,
        num_subjects=num_subjects,
        num_days=num_days,
        num_timeslots=num_timeslots,
        room_codes=room_codes,
        subject_campuses=subject_campuses,
        max_classes_per_teacher=5,
    )
    env.subject_names     = subject_names
    env.teacher_names     = teacher_names
    env.day_labels        = day_labels
    env.timeslot_labels   = timeslot_labels
    env.room_to_campus    = room_to_campus
    env.metadata["is_parallelizable"] = False
    return PettingZooEnv(env)

# ─── 8) MAIN TRAINING LOOP ───────────────────────────────────────────────────
if __name__ == "__main__":
    init(ignore_reinit_error=True, include_dashboard=False)
    register_env("timetabling_env_v5", lambda cfg: make_env())

    dummy = make_env()
    dummy.reset()
    raw = dummy.env

    # define multi-agent policies
    policies = {}
    obs_s, act_s = raw.observation_spaces["saha_0"], raw.action_spaces["saha_0"]
    policies["saha_policy"] = (
        None, obs_s, act_s,
        {"model": {"fcnet_hiddens": [128, 128], "fcnet_activation": "relu"}}
    )
    for b in raw.building_keys:
        key   = f"cma_{b}_policy"
        obs_c = raw.observation_spaces[f"cma_{b}"]
        act_c = raw.action_spaces[f"cma_{b}"]
        policies[key] = (
            None, obs_c, act_c,
            {"model": {"fcnet_hiddens": [128, 128], "fcnet_activation": "relu"}}
        )

    def policy_mapping_fn(agent_id, episode, **kwargs):
        return "saha_policy" if agent_id.startswith("saha") else f"{agent_id}_policy"

    ppo_cfg = (
        PPOConfig()
        .environment(env="timetabling_env_v5", disable_env_checking=True)
        .framework("torch")
        .rollouts(num_rollout_workers=0, rollout_fragment_length=7, batch_mode="complete_episodes")
        .training(gamma=0.95, lr=2e-4, train_batch_size=256, sgd_minibatch_size=64, entropy_coeff=0.01)
        .resources(num_gpus=0)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn,
                     policies_to_train=list(policies.keys()), count_steps_by="env_steps")
        .callbacks(ScheduleCallback)
    )

    config   = ppo_cfg.to_dict()
    reporter = CLIReporter(
        parameter_columns=["env","lr","train_batch_size"],
        metric_columns=[
            "episode_reward_mean","timesteps_total","episode_len_mean",
            "custom_metrics/valid_schedule_mean",
            "custom_metrics/negotiation_success_rate_mean",
            "custom_metrics/error_rate_mean",
        ],
        max_report_frequency=1,
    )
    run_cfg = RunConfig(
        stop={"training_iteration": 500},
        local_dir="C:/ray_logs",
        name="PPO_Timetabling_v5",
        checkpoint_config=CheckpointConfig(checkpoint_frequency=0, num_to_keep=1),
        callbacks=[TBXLoggerCallback()],
        progress_reporter=reporter,
        verbose=1,
        log_to_file="ppo_train_v5.log",
    )
    Tuner("PPO", param_space=config, run_config=run_cfg).fit()
