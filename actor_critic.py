from obstacle_tower_env import ObstacleTowerEnv
import numpy as np
from matplotlib import pyplot as plt

# Realtime mode determines whether the environment window will render the scene,
# as well as whether the environment will run at realtime speed. Set this to `True`
# to visual the agent behavior as you would in player mode.

env = ObstacleTowerEnv('./ObstacleTower/obstacletower', retro=False, realtime_mode=True)

# The environment provided has a MultiDiscrete action space, where the 4 dimensions are:

# 0. Movement (No-Op/Forward/Back)
# 1. Camera Rotation (No-Op/Counter-Clockwise/Clockwise)
# 2. Jump (No-Op/Jump)
# 3. Movement (No-Op/Right/Left)

print(env.action_space.nvec)
print(env.observation_space)


#plt.imshow(obs[0])
#plt.show()
#print(env.unwrapped.get_action_meanings())


# tower 0, floor 10 = second room holds key
config = {'tower-seed': 0, 'starting-floor': 10, 'agent-perspective': 0, 'allowed-rooms': 1, 'allowed-modules': 0, 'allowed-floors': 0}
obs = env.reset(config=config)

action = env.action_space.sample()
allowed_action = False
allowed_actions = np.array([np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]), np.array([0, 2, 0, 0]), np.array([1, 0, 1, 0])])

# Took only an action of the allowed actions
while not allowed_action:
    if (allowed_actions == action).all(1).any():
        allowed_action = True
    else:
        action = env.action_space.sample()

action = np.array([1, 0, 0, 0])
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)

action = np.array([0, 1, 0, 0])
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)

action = np.array([1, 0, 0, 0])
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)

action = np.array([0, 2, 0, 0])
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)

action = np.array([1, 0, 0, 0])
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)

action = np.array([0, 2, 0, 0])
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)

action = np.array([1, 0, 0, 0])
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action)
#env.close()

