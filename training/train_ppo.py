from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
from envs.timetabling_env import TimetablingEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune, init

def env_creator(_):
    return PettingZooEnv(TimetablingEnv())

register_env("timetabling_env", env_creator)

if __name__ == "__main__":
    init(ignore_reinit_error=True)

    env = TimetablingEnv()
    policies = {
        agent: (None, env.observation_spaces[agent], env.action_spaces[agent], {})
        for agent in env.possible_agents
    }

    config = (
        PPOConfig()
        .environment("timetabling_env")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
        )
        .framework("torch")
        .rollouts(num_rollout_workers=0)
    )

    tune.Tuner("PPO", param_space=config.to_dict()).fit()
