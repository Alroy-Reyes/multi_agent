import sys
import os
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

# allow importing your custom environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.timetabling_env import TimetablingEnv

# ─────────────────────────────────────────
# 1) LOAD & PARSE YOUR CSV
# ─────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(__file__), "../standardized_schedule.csv")
df       = pd.read_csv(csv_path)

subject_names = df["Subject"].tolist()
teacher_names = df["Faculty"].tolist()

# ←— CLEAN TIMESLOTS: split on newline, explode, strip, drop dupes
timeslot_labels = (
    df["Time"]
     .dropna()                    # 1) drop real NaNs
     .astype(str)
     .str.split(r"\n")            # 2) split multi‐line
     .explode()                   # 3) one per row
     .str.strip()                 # 4) trim whitespace
     .loc[lambda s: s != ""]     # 5) throw away any "" entries
     .drop_duplicates()           # 6) unique, in order
     .tolist()
)
num_timeslots = len(timeslot_labels)

# filter out any rows with missing Room before using them
df_rooms   = df[df["Room"].notna()].copy()
room_codes = sorted(df_rooms["Room"].unique().tolist())

num_subjects  = len(subject_names)
num_teachers  = len(set(teacher_names))

print(f"Number of subjects detected:        {num_subjects}")
print(f"Number of unique teachers detected: {num_teachers}")
print(f"Number of unique timeslots:         {num_timeslots}")
print(f"Number of unique rooms (non-null):  {len(room_codes)}")

# ─────────────────────────────────────────
# 1b) DETECT CAMPUS INFO
# ─────────────────────────────────────────
room_to_campus = df_rooms.set_index("Room")["Campus"].to_dict()
teacher_to_campus = (
    df[["Faculty", "Campus"]]
      .drop_duplicates(subset=["Faculty"])
      .set_index("Faculty")["Campus"]
      .to_dict()
)

campuses = sorted(set(df["Campus"]))
print(f"Detected campuses: {campuses}")
print(f"Number of unique campuses detected: {len(campuses)}")

print("\nTeacher → Campus mapping:")
for teacher, campus in teacher_to_campus.items():
    print(f"  {teacher:<20} → {campus}")

print(f"\nRoom → Campus mapping:\n  {room_to_campus}\n")

# ─────────────────────────────────────────
# COMPUTE building → campus lookup
# ─────────────────────────────────────────
building_to_campus = {room[0]: campus for room, campus in room_to_campus.items()}

# ─────────────────────────────────────────
# COMPUTE subject_campuses
# ─────────────────────────────────────────
subject_campuses = []
for subj_idx, teacher in enumerate(teacher_names):
    campus = teacher_to_campus.get(teacher)
    allowed = [b for b, c in building_to_campus.items() if c == campus]
    if not allowed:
        raise ValueError(f"No building found for campus '{campus}' (teacher {teacher})")
    subject_campuses.append(allowed)

print(f"Subject → allowed buildings (by campus): {subject_campuses}")

# helper to build schedule DataFrame
def build_schedule_dataframe(env):
    rows = []
    for b in env.building_keys:
        rooms = env.buildings_room_info[b]
        sched = env.buildings_room_schedule[b]
        for room_idx, room_code in enumerate(rooms):
            campus = env.room_to_campus.get(room_code, "UNKNOWN")
            for ts in range(env.num_timeslots):
                subj = sched[room_idx, ts]
                rows.append({
                    "Campus":   campus,
                    "Room":     room_code,
                    "Timeslot": env.timeslot_labels[ts],
                    "Subject":  None if subj < 0 else env.subject_names[subj],
                    "Teacher":  None if subj < 0 else env.teacher_names[subj],
                })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────
# 2) CALLBACKS
# ─────────────────────────────────────────
class ScheduleCallback(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result, **kwargs):
        it  = result["training_iteration"]
        ts  = result["timesteps_total"]
        rew = result["episode_reward_mean"]
        print(f"[DEBUG] train() finished iteration={it:<3}  timesteps={ts:<6}  reward={rew:.2f}")
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)

    def on_episode_end(self, *, worker, base_env, episode, **kwargs):
        sub_env  = base_env.get_sub_environments()[0].env
        df_sched = build_schedule_dataframe(sub_env).fillna("")   # drop NaNs
        print("\n=== Current Timetable at Episode End ===")
        print(df_sched.to_string(index=False))
        print("========================================\n")
        super().on_episode_end(worker=worker, base_env=base_env, episode=episode, **kwargs)

# ─────────────────────────────────────────
# 3) ENV FACTORY
# ─────────────────────────────────────────
def make_env():
    base_env = TimetablingEnv(
        num_sahas=4,
        num_teachers=num_teachers,
        num_subjects=num_subjects,
        num_timeslots=num_timeslots,
        room_codes=room_codes,
        subject_campuses=subject_campuses,       # pass in campus constraints
        max_classes_per_teacher=5,
    )
    base_env.subject_names     = subject_names
    base_env.teacher_names     = teacher_names
    base_env.timeslot_labels   = timeslot_labels
    base_env.room_to_campus    = room_to_campus
    base_env.teacher_to_campus = teacher_to_campus
    base_env.metadata["is_parallelizable"] = False
    return PettingZooEnv(base_env)

# ─────────────────────────────────────────
# 4) MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    init(ignore_reinit_error=True, include_dashboard=False)
    register_env("timetabling_env_v5", lambda cfg: make_env())

    # dummy env for policy setup
    dummy = make_env()
    dummy.reset()
    raw   = dummy.env

    # ─────────────────────────────────────────
    # 5) POLICIES & MULTI-AGENT SETUP
    # ─────────────────────────────────────────
    obs_s = raw.observation_spaces["saha_0"]
    act_s = raw.action_spaces   ["saha_0"]
    policies = {
        "saha_policy": (
            None,
            obs_s,
            act_s,
            {"model": {"fcnet_hiddens": [128, 128], "fcnet_activation": "relu"}},
        ),
    }
    for b in raw.building_keys:
        obs_c = raw.observation_spaces[f"cma_{b}"]
        act_c = raw.action_spaces   [f"cma_{b}"]
        policies[f"cma_{b}_policy"] = (
            None,
            obs_c,
            act_c,
            {"model": {"fcnet_hiddens": [128, 128], "fcnet_activation": "relu"}},
        )

    def policy_mapping_fn(agent_id, episode, **kwargs):
        return "saha_policy" if agent_id.startswith("saha") else f"{agent_id}_policy"

    # ─────────────────────────────────────────
    # 6) PPO CONFIG & RUNNER
    # ─────────────────────────────────────────
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
    config   = ppo_cfg.to_dict()
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
        # disable Tune's internal trial-controller checkpointing
        checkpoint_config=CheckpointConfig(checkpoint_frequency=0, num_to_keep=1),
        callbacks=[TBXLoggerCallback()],
        progress_reporter=reporter,
        verbose=1,
        log_to_file="ppo_train_v5.log",
    )

    Tuner("PPO", param_space=config, run_config=run_cfg).fit()
