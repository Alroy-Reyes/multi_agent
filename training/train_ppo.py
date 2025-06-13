# training/train_ppo.py

import sys, os
from ray import init, tune
from ray.tune import CLIReporter
# ← import the official TensorBoard callback
from ray.tune.logger import TBXLoggerCallback
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv

# allow importing your custom environment
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

if __name__ == "__main__":
    # 1) Start Ray
    init(ignore_reinit_error=True, include_dashboard=False)

    # 2) Register your env
    register_env("timetabling_env", lambda cfg: make_env())

    # 3) Build a dummy so we can pull out observation & action spaces
    wrapped = make_env()
    wrapped.env.reset()      # reset the raw AECEnv
    raw = wrapped.env

    policies = {}
    for agent_id in raw.possible_agents:
        # grab the real spaces from the raw environment
        obs_space = raw.observation_space(agent_id)
        act_space = raw.action_space(agent_id)
        policies[agent_id] = (None, obs_space, act_space, {})

    # 4) Build a straightforward PPOConfig (no logger_config/loggers hacks)
    ppo_cfg = (
        PPOConfig()
        .environment(env="timetabling_env", disable_env_checking=True)
        .framework("torch")
        .rollouts(
            num_rollout_workers=0,
            rollout_fragment_length=1,
            batch_mode="complete_episodes",
        )
        .training(
            gamma=0.99,
            lr=1e-3,
            train_batch_size=6,
            sgd_minibatch_size=6,
        )
        .resources(num_gpus=0)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
        )
    )
    config = ppo_cfg.to_dict()

    # 5) CLIReporter for live console metrics
    reporter = CLIReporter(
        parameter_columns=["env", "lr", "train_batch_size"],
        metric_columns=["episode_reward_mean", "timesteps_total"]
    )

    # 6) Run with Tune, using only TBXLoggerCallback to emit TensorBoard events
    tune.run(
        "PPO",
        config=config,
        stop={"training_iteration": 50},
        local_dir=os.path.expanduser("~/ray_results"),
        name="PPO_Timetabling",
        callbacks=[TBXLoggerCallback()],
        progress_reporter=reporter,
        verbose=1,
        log_to_file="output.log",
    )
