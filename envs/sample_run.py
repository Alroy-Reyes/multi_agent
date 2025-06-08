from timetabling_env import TimetablingEnv

# Initialize environment with SAHAs and CMAs, plus room/building info
env = TimetablingEnv(
    num_sahas=2,
    num_cmas=2,
    num_teachers=5,
    num_subjects=6,
    num_timeslots=5,
    buildings_room_info={
        0: ['lecture', 'lab'],
        1: ['lecture', 'lecture']
    },
    max_classes_per_teacher=3
)

obs, info = env.reset()

done = {agent: False for agent in env.agents}
action_pointer = 0

while not all(done.values()) and env.agents:
    agent = env.agent_selection

    if done.get(agent, False):
        env.agent_selection = env._agent_selector.next()
        continue

    action_space = env.action_space(agent)
    action = action_pointer % action_space.n

    print(f"\n▶️ {agent} attempting action: {action}")

    env.step(action)
    env.render()

    if env.terminations.get(agent, False):
        done[agent] = True

    action_pointer += 1

env.close()
