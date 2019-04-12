import gym
import time
env = gym.make('Breakout-ram-v0')
env.reset()
for _ in range(500):
    env.render()
    env.step(env.action_space.sample())

from gym import envs
print(envs.registry.all())
