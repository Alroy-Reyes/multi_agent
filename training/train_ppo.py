from ray import tune, init
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.timetabling_env import TimetablingEnv

def env_creator(_):
    return PettingZooEnv(TimetablingEnv())

register_env("timetabling_env", env_creator)

if __name__ == "__main__":
    init(ignore_reinit_error=True, include_dashboard=False)

    dummy = TimetablingEnv()
    policies = {
        a: (None, dummy.observation_spaces[a], dummy.action_spaces[a], {})
        for a in dummy.possible_agents
    }

    config = (
        PPOConfig()
        .environment(env="timetabling_env")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
        )
        .framework("torch")
        .rollouts(num_rollout_workers=0)
        .to_dict()
    )

    tune.run(
        "PPO",
        config=config,
        stop={"training_iteration": 10},  # stop after 10 iters
        verbose=2,                        # print each iteration’s result
    )
