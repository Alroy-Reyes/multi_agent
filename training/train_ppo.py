# train_ppo.py

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
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv  # ⬅️ use the parallel wrapper

from torch.utils.tensorboard import SummaryWriter

# allow importing your custom environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.timetabling_env import ParallelTimetablingEnv


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
    allowed = [b for b, c in building_to_campus.items() if c == teacher_to_campus.get(teach)]
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
def build_schedule_dataframe(env: ParallelTimetablingEnv) -> pd.DataFrame:
    rows = []
    for b in env.building_keys:
        rooms = env.buildings_room_info[b]
        sched = env.buildings_room_schedule[b]
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


# ─── 6) ENHANCED CALLBACK FOR PARALLEL TRAINING ───────────────────────────────
class ParallelScheduleCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.writer = SummaryWriter(log_dir="C:/ray_logs/parallel_tensorboard")
        self.episode_counter = 0

    def on_train_result(self, *, algorithm, result, **kwargs):
        it = result.get("training_iteration", 0)
        ts = result.get("timesteps_total", 0)
        rew = result.get("episode_reward_mean", 0.0)

        # Log key metrics
        self.writer.add_scalar("Reward/EpisodeRewardMean", rew, it)
        self.writer.add_scalar("Timesteps/Total", ts, it)

        # Log parallel-specific custom metrics (aggregated by RLlib)
        for k, v in result.get("custom_metrics", {}).items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"Custom/{k}", v, it)

        # Log learner stats (if present)
        learner_info = result.get("info", {}).get("learner", {})
        for pid, data in learner_info.items():
            stats = data.get("learner_stats", {})
            for name in ("policy_loss", "vf_loss", "entropy"):
                val = stats.get(name)
                if val is not None:
                    self.writer.add_scalar(f"{pid}/{name}", val, it)

        # Training efficiency metric
        if "time_this_iter_s" in result and result["time_this_iter_s"] > 0:
            throughput = ts / result["time_this_iter_s"]
            self.writer.add_scalar("Performance/Throughput", throughput, it)

    def on_episode_end(self, *, worker, base_env, episode, **kwargs):
        self.episode_counter += 1

        # Get the actual PettingZoo parallel wrapper then unwrap to your env
        wrapped = base_env.get_sub_environments()[0]  # ParallelPettingZooEnv
        env = getattr(wrapped, "par_env", wrapped)    # ParallelTimetablingEnv underneath

        # Build and log a schedule snapshot every 10 episodes
        if self.episode_counter % 10 == 0:
            df_sched = build_schedule_dataframe(env).fillna("")

            # Compact summary and metrics
            assigns = env.subject_assignments
            num_assigned = int((assigns >= 0).sum())
            num_unassigned = int((assigns < 0).sum())

            counts = list(env.teacher_classes.values())
            full_load = sum(1 for c in counts if c >= env.max_classes)
            under_load = sum(1 for c in counts if c < env.max_classes)

            print(f"\n=== Episode {self.episode_counter} Summary ===")
            print(f"Parallel execution completed in {env.timestep} timesteps")
            print(f"Assigned subjects: {num_assigned}/{env.num_subjects}")
            print(f"Conflicts resolved: {env.negotiation_success}/{env.conflict_count}")
            print(f"Faculty utilization: {full_load} full, {under_load} partial")
            print("=" * 40 + "\n")

            # Custom metrics for RLlib aggregation
            episode.custom_metrics["parallel_timesteps"] = env.timestep
            episode.custom_metrics["conflict_rate"] = (
                env.conflict_count / env.timestep if env.timestep > 0 else 0.0
            )
            episode.custom_metrics["assignment_rate"] = (
                num_assigned / env.num_subjects if env.num_subjects > 0 else 0.0
            )

            if counts:
                workload_std = np.std(counts)
                episode.custom_metrics["workload_balance"] = 1.0 / (1.0 + workload_std)

    def __del__(self):
        if hasattr(self, "writer"):
            self.writer.close()


# ─── 7) PARALLEL ENV FACTORY ───────────────────────────────────────────────────
def make_parallel_env():
    env = ParallelTimetablingEnv(
        num_sahas=4,
        num_teachers=num_teachers,
        num_subjects=num_subjects,
        num_days=num_days,
        num_timeslots=num_timeslots,
        room_codes=room_codes,
        subject_campuses=subject_campuses,
        max_classes_per_teacher=4,
        enable_communication=True,  # Enable agent communication
    )
    # Attach labels/lookup tables for callbacks & schedule export
    env.subject_names = subject_names
    env.teacher_names = teacher_names
    env.day_labels = day_labels
    env.timeslot_labels = timeslot_labels
    env.room_to_campus = room_to_campus

    # Return the **parallel** RLlib wrapper
    return ParallelPettingZooEnv(env)


# ─── 8) MAIN TRAINING WITH PARALLEL OPTIMIZATION ───────────────────────────────
if __name__ == "__main__":
    init(ignore_reinit_error=True, include_dashboard=False)
    register_env("parallel_timetabling_env", lambda cfg: make_parallel_env())

    # Smoke test the env
    dummy = make_parallel_env()
    dummy.reset()
    raw = dummy.par_env  # ⬅️ unwrap to underlying ParallelTimetablingEnv

    # Setup policies for all agents
    policies = {}

    # SAHA policies - shared parameters
    obs_s, act_s = raw.observation_spaces["saha_0"], raw.action_spaces["saha_0"]
    policies["saha_policy"] = (
        None, obs_s, act_s,
        {
            "model": {
                "fcnet_hiddens": [256, 256, 128],
                "fcnet_activation": "relu",
                "vf_share_layers": True,
            },
            "lr": 1e-4,
        }
    )

    # CMA policies - one per building
    for b in raw.building_keys:
        key = f"cma_{b}_policy"
        obs_c = raw.observation_spaces[f"cma_{b}"]
        act_c = raw.action_spaces[f"cma_{b}"]
        policies[key] = (
            None, obs_c, act_c,
            {
                "model": {
                    "fcnet_hiddens": [256, 256, 128],
                    "fcnet_activation": "relu",
                    "vf_share_layers": True,
                },
                "lr": 1e-4,
            }
        )

    def policy_mapping_fn(agent_id, episode, **kwargs):
        return "saha_policy" if agent_id.startswith("saha") else f"{agent_id}_policy"

    # Optimized PPO config for parallel execution
    ppo_cfg = (
        PPOConfig()
        .environment(
            env="parallel_timetabling_env",
            disable_env_checking=True,
            env_config={"enable_communication": True}
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=4,
            rollout_fragment_length=50,
            batch_mode="complete_episodes",
            enable_connectors=False,
        )
        .training(
            gamma=0.98,
            lr=5e-5,
            lr_schedule=[
                (0,      5e-5),
                (100000, 2e-5),
                (300000, 5e-6),
            ],
            train_batch_size=4096,
            sgd_minibatch_size=1024,
            num_sgd_iter=15,
            clip_param=0.2,
            vf_clip_param=50000.0,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            grad_clip=10.0,
        )
        .resources(
            num_gpus=1 if os.environ.get("CUDA_VISIBLE_DEVICES") else 0,
            num_cpus_for_local_worker=2,   # CPU for the local (driver) worker
            # Optional: tune these if you want to limit worker CPUs explicitly
            # num_cpus_per_worker=1,
            # num_gpus_per_worker=0,
        )

        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(policies.keys()),
            count_steps_by="env_steps",
        )
        .callbacks(ParallelScheduleCallback)
        .experimental(
            _enable_new_api_stack=False,
            _disable_preprocessor_api=False,
        )
    )

    config = ppo_cfg.to_dict()

    # Enhanced reporting for parallel training
    reporter = CLIReporter(
        parameter_columns=["env", "lr", "train_batch_size"],
        metric_columns=[
            "episode_reward_mean",
            "timesteps_total",
            "episode_len_mean",
            "custom_metrics/parallel_timesteps_mean",
            "custom_metrics/conflict_rate_mean",
            "custom_metrics/assignment_rate_mean",
            "custom_metrics/workload_balance_mean",
            "time_this_iter_s",
        ],
        max_report_frequency=5,
    )

    # Run configuration
    run_cfg = RunConfig(
        stop={
            "training_iteration": 300,
            "episode_reward_mean": 100,
        },
        local_dir="C:/ray_logs",
        name="PPO_Parallel_Timetabling",
        checkpoint_config=CheckpointConfig(
            checkpoint_frequency=10,
            checkpoint_at_end=True,
            num_to_keep=3,
        ),
        callbacks=[TBXLoggerCallback()],
        progress_reporter=reporter,
        verbose=1,
        log_to_file="ppo_parallel_train.log",
    )

    # Start training
    print("Starting parallel multi-agent training...")
    print(f"Agents acting simultaneously: {len(raw.agents)} agents")
    print(f"Communication enabled: True")
    print("-" * 50)

    tuner = Tuner("PPO", param_space=config, run_config=run_cfg)
    results = tuner.fit()

    print("\nTraining completed!")
    print(f"Best reward: {results.get_best_result().metrics['episode_reward_mean']}")
