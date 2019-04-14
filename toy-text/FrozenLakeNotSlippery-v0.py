import gym
import random
import numpy as np
from gym.envs.registration import register

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

env = gym.make('FrozenLakeNotSlippery-v0')

q_table = np.zeros((env.observation_space.n, env.action_space.n))

def getMaxQ(state):
    max = q_table[state][0]
    for actionQ in q_table[state]:
        if max < actionQ:
            max = actionQ
    return max

def getMaxQIndex(state):
    max = q_table[state][0]
    index = 0
    i = 0
    for actionQ in q_table[state]:
        if max < actionQ:
            max = actionQ
            index = i
        i += 1
    return index

discount_factor = 0.8
learning_rate = 0.7
epsilion = 0.3

for i_episode in range(100):
    observation = env.reset()
    for t in range(100):
        env.render()
        
        current_state = observation
        
        action = getMaxQIndex(current_state)
        
        exploration = random.random()
        if exploration < epsilion:
            action = env.action_space.sample()

        observation, reward, done, info = env.step(action)
        next_state = observation
        
        if done:
            print("Learning Episode: finished after {} timesteps".format(t+1))
            if reward == 0:
                reward -= 10
            else:
                print("Learning Episode: finished and found the goal after {} timesteps".format(t+1))
                reward += 10

        reward -= 1
        
        q_table[current_state][action] = q_table[current_state][action] + learning_rate*(reward + discount_factor*getMaxQ(next_state) - q_table[current_state][action])
        
        if done:
            break;

def test():
    observation = env.reset()
    for t in range(100):
        current_state = observation
        action = getMaxQIndex(current_state)
        observation, reward, done, info = env.step(action)
        env.render()

        if done:
            if reward == 0:
                print("Test episode: Failed after {} timesteps".format(t+1))
            else:
                print("Test episode: Found the goal after {} timesteps".format(t+1))
            break;

test()

env.close()

