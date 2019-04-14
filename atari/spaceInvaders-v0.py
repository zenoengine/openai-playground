import gym
import time
import os
import tensorflow as tf
import numpy as np
import random
from collections import deque

env = gym.make('SpaceInvaders-v0')
obsevation = env.reset()

print("The action Size is ", env.action_space.n)
print("The size of frame : ", env.observation_space)

possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())

normalized_image_shape = (110,84)
def preprocess_frame(frame):
    # make grey, normalize
    normalized_frame = np.mean(frame, -1) / 255.0
    # scale down
    processed_frame = np.resize(normalized_frame, normalized_image_shape)
    return processed_frame

env = gym.make('SpaceInvaders-v0')
env.reset()

for _ in range(500):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample())
    preprocess_frame(observation)

env.close()