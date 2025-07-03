from timetabling_env import TimetablingEnv
env = TimetablingEnv()
env.reset()
steps = 0
while env.agents:
    a = env.action_spaces[env.agent_selection].sample()
    env.step(a)
    steps += 1
print("Episode terminated in", steps, "agent steps")
