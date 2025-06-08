# training/train_ppo.py

import sys
import os
from ray import init, tune
from ray.tune import CLIReporter
from ray.air.config import RunConfig
from ray.tune.logger import TBXLoggerCallback
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.timetabling_env import TimetablingEnv

def make_env():
    base = TimetablingEnv(
        num_sahas=2,
        num_cmas=2,
        num_teachers=5,
        num_subjects=6,
        num_timeslots=5,
        buildings_room_info={0: ["lecture", "lab"], 1: ["lecture", "lecture"]},
        max_classes_per_teacher=3
    )
    return PettingZooEnv(base)

def env_creator(_):
    return make_env()

if __name__ == "__main__":
    # 1) Start Ray
    init(ignore_reinit_error=True, include_dashboard=False)

    # 2) Register the env
    register_env("timetabling_env", lambda config: make_env())

    # 3) Build a dummy to extract obs/action spaces
    dummy = make_env()
    dummy.env.reset()
    raw = dummy.env
    policies = {}
    for agent_id in raw.possible_agents:
        obs_space = dummy.observation_space[agent_id]
        act_space = dummy.action_space[agent_id]
        policies[agent_id] = (None, obs_space, act_space, {})

    # 4) Build PPOConfig
    config = (
        PPOConfig()
        .environment(env="timetabling_env", disable_env_checking=True)
        .framework("torch")
        .rollouts(num_rollout_workers=0, rollout_fragment_length=1, batch_mode="complete_episodes")
        .training(gamma=0.99, lr=1e-3, train_batch_size=6, sgd_minibatch_size=6)
        .resources(num_gpus=0)
        .multi_agent(policies=policies, policy_mapping_fn=lambda aid, *args, **kwargs: aid)
    ).to_dict()

    # 5) Use CLIReporter to show live progress in console
    reporter = CLIReporter(
        parameter_columns=["env", "lr", "train_batch_size"],
        metric_columns=["episode_reward_mean", "policy_reward_mean", "policy_loss", "vf_loss", "timesteps_total"]
    )

    # 6) Run with tune.run rather than Tuner
    tune.run(
        "PPO",
        config=config,
        stop={"training_iteration": 50},   # e.g. run for 10 iterations
        local_dir=os.path.expanduser("~/ray_results"),
        name="PPO_Timetabling",
        callbacks=[TBXLoggerCallback()],
        progress_reporter=reporter,
        verbose=1,
    )
