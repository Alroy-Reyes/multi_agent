# training/train_ppo.py

import sys, os
import numpy as np
from ray import init, tune
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


class TimetableCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        sub_envs = base_env.get_sub_environments()
        petting_env = sub_envs[0]
        raw_env = petting_env.env

        valid = True
        # no duplicate subjects
        for sched in raw_env.buildings_room_schedule.values():
            flat = sched.flatten()
            placed = flat[flat >= 0]
            if len(placed) != len(np.unique(placed)):
                valid = False
                break

        # every subject assigned exactly once
        if not (raw_env.subject_assignments >= 0).all() or \
           np.unique(raw_env.subject_assignments).size != raw_env.num_subjects:
            valid = False

        episode.custom_metrics["valid_schedule"] = int(valid)


def make_env():
    base = TimetablingEnv(
        num_sahas=4,
        num_cmas=2,
        num_teachers=6,
        num_subjects=12,
        num_timeslots=6,
        buildings_room_info={0: ["lecture", "lab"], 1: ["lecture", "lecture", "lab"]},
        max_classes_per_teacher=4,
    )
    return PettingZooEnv(base)


if __name__ == "__main__":
    # 1) Start Ray
    init(ignore_reinit_error=True, include_dashboard=False)

    # 2) Register the environment
    register_env("timetabling_env_v4", lambda cfg: make_env())

    # 3) Dummy env to extract spaces
    wrapped = make_env()
    wrapped.env.reset()
    raw = wrapped.env

    # 4) Define multi-agent policies
    policies = {
        aid: (
            None,
            raw.observation_spaces[aid],
            raw.action_spaces[aid],
            {}
        )
        for aid in raw.possible_agents
    }

    # 5) Build PPO configuration
    ppo_cfg = (
        PPOConfig()
        .environment(env="timetabling_env_v4", disable_env_checking=True)
        .framework("torch")
        .rollouts(
            num_rollout_workers=2,
            rollout_fragment_length=32,
            batch_mode="complete_episodes",
        )
        .training(
            gamma=0.95,
            lr=2e-4,
            train_batch_size=256,
            sgd_minibatch_size=64,
            entropy_coeff=0.01,
        )
        .evaluation(
            evaluation_interval=5,
            evaluation_num_episodes=10,
            evaluation_config={"explore": False},
        )
        .resources(num_gpus=0)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=lambda aid, *args, **kwargs: aid,
        )
        .callbacks(TimetableCallbacks)
    )

    # 5a) Inject custom MLP model hyperparams directly:
    ppo_cfg.model["fcnet_hiddens"]    = [128, 128]
    ppo_cfg.model["fcnet_activation"] = "relu"
    ppo_cfg.model["vf_share_layers"]  = True

    config = ppo_cfg.to_dict()

    # 6) Reporter
    reporter = CLIReporter(
        parameter_columns=["env", "lr", "train_batch_size"],
        metric_columns=[
            "episode_reward_mean",
            "timesteps_total",
            "episode_len_mean",
            "evaluation/episode_reward_mean",
            "custom_metrics/valid_schedule",
        ],
    )

    # 7) Run training via the new Tuner API
    tuner = Tuner(
        "PPO",
        param_space=config,
        run_config=RunConfig(
            stop={"training_iteration": 100},
            local_dir=os.path.expanduser("~/ray_results"),
            name="PPO_Timetabling_v4",
            callbacks=[TBXLoggerCallback()],
            progress_reporter=reporter,
            verbose=1,
            log_to_file="ppo_train_v4.log",
        ),
    )
    tuner.fit()
