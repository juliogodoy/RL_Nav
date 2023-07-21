import pfrl
from pfrl.agents import PPO
import torch
from torch import nn
import numpy
import gym
import random
import math
import time
import gym 
import RVOenv4
import RVOenv2



#env = RVOenv4.RVOEnv()
env = RVOenv4.RVOEnv()#gym.make('CartPole-v0')
#env =gym.make('CartPole-v0')
print('observation space:', env.observation_space)
print('action space:', env.action_space)

obs = env.reset()
print('initial observation:', obs)

action = env.action_space.sample()
obs, r, done, info = env.step(action)
print('next observation:', obs)
print('reward:', r)
print('done:', done)
print('info:', info)
#env.render()
#while True:
#    continue
#breakpoint()

#Parameters
#n_episodes = 500       #Total number of training episodes
#max_episode_len = 200  #How many steps in a rollout/trajectory
#args_gpu = -1          #Use GPU? -1 is false, otherwise device ID
#args_epochs = 50

#env =  RVOenv.RVOEnv()
#obs_space = env.observation_space
#action_space = env.action_space
#print("Observation space:", obs_space)
#print("Action space:", action_space)

#obs = env.reset()
#print('initial observation:', obs)

#action = env.action_space.sample()
#obs, r, done, info = env.step(action)
#print('next observation:', obs)
#print('reward:', r)
#print('done:', done)
#print('info:', info)


class QFunction(torch.nn.Module):

    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.l1 = torch.nn.Linear(obs_size, 50)
        self.l2 = torch.nn.Linear(50, 50)
        self.l3 = torch.nn.Linear(50, n_actions)

    def forward(self, x):
        h = x
        h = torch.nn.functional.relu(self.l1(h))
        h = torch.nn.functional.relu(self.l2(h))
        h = self.l3(h)
        return pfrl.action_value.DiscreteActionValue(h)

obs_size = env.observation_space.low.size
print("Obs size: ", obs_size)

n_actions = env.action_space.n
print("n_actions: ", n_actions)
q_func = QFunction(obs_size, n_actions)
print(q_func)


optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)

# Set the discount factor that discounts future rewards.
gamma = 0.9



#explorer= pfrl.explorers.ExponentialDecayEpsilonGreedy(
#    start_epsilon=0.35, end_epsilon=0.00,decay=0.99, random_action_func=env.action_space.sample)
# Use epsilon-greedy for exploration
explorer = pfrl.explorers.LinearDecayEpsilonGreedy(
    start_epsilon=0.35, end_epsilon=0.00,decay_steps=350000, random_action_func=env.action_space.sample)

# Use epsilon-greedy for exploration
#explorer = pfrl.explorers.ConstantEpsilonGreedy(
#    epsilon=0.3, random_action_func=env.action_space.sample)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)

# Since observations from CartPole-v0 is numpy.float64 while
# As PyTorch only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
phi = lambda x: x.astype(numpy.float32, copy=False)

# Set the device id to use GPU. To use CPU only, set it to -1.
gpu = -1

# Now create an agent that will interact with the environment.
#agent = pfrl.agents.DQN(
#    q_func,
#    optimizer,
#    replay_buffer,
#    gamma,
#    explorer,
#    replay_start_size=500,
#    update_interval=1,
#    target_update_interval=100,
#    phi=phi,
#    gpu=gpu,
#)


agent = pfrl.agents.DoubleDQN(
    q_func,
    optimizer,
    replay_buffer,
    gamma,
    explorer,
    replay_start_size=500,
    update_interval=1,
    target_update_interval=100,
    phi=phi,
    gpu=gpu,
)


n_episodes = 4000
max_episode_len = 200


#import logging
#import sys
#logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

#pfrl.experiments.train_agent(
#    agent,
#    env,
#    steps=2000000,
#    outdir='result_stats', 
#)


GlobalR=0
for i in range(1, n_episodes + 1):
    obs = env.reset()
    R = 0  # return (sum of rewards)
    t = 0  # time step
    #if i>500:
        #explorer.epsilon=0


    while True:
        # Uncomment to watch the behavior in a GUI window
        #if i>500:
            #env.render()
        #env.render()
        if explorer.epsilon<0.01:
            env.render()

        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        #print("Epsilon is: ", explorer.compute_epsilon(t))
        R += reward
        t += 1
        #print("Step: ", t, " reward: ", reward, "action: ", action, " epsilon: ", explorer.epsilon)
        #time.sleep(0.1)

        reset = t == max_episode_len
        agent.observe(obs, reward, done, reset)
        if done or reset:
            #time.sleep(0.8)
            break
    if i % 1 == 0:
        #print('episode:', i, 'R:', R, ' epsilon: ', explorer.epsilon)
        GlobalR=GlobalR+R
        #time.sleep(1)
    #if i>1000:
        #print('episode:', i, 'R:', R, ' epsilon: ', explorer.epsilon)
    if i % 50 == 0:
        print("Episodes ", i-50, "-", i, " Epsilon: ", explorer.epsilon, " AvgReward : ", GlobalR/50)
        GlobalR=0
        #print('statistics:', agent.get_statistics())
print('Finished training.')



# Use epsilon-greedy for exploration
#explorer = pfrl.explorers.ConstantEpsilonGreedy(
#    epsilon=0.0, random_action_func=env.action_space.sample)

print("Starting evaluation with epsilon: ", explorer.epsilon)
n_episodes_eval=50

GlobalR=0
for i in range(1, n_episodes_eval + 1):
    obs = env.reset()
    R = 0  # return (sum of rewards)
    t = 0  # time step
    #if i>500:
        #explorer.epsilon=0

    while True:
        # Uncomment to watch the behavior in a GUI window
        #if i>500:
            #env.render()
        env.render()
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
        #print("Step: ", t, " reward: ", reward, "action: ", action, " epsilon: ", explorer.epsilon)
        #time.sleep(0.1)

        reset = t == max_episode_len
        agent.observe(obs, reward, done, reset)
        if done or reset:
            #time.sleep(0.8)
            break
    if i % 1 == 0:
        print('episode:', i, 'R:', R, ' epsilon: ', explorer.epsilon)
        GlobalR=GlobalR+R
        #time.sleep(1)
    if i>1000:
        print('episode:', i, 'R:', R, ' epsilon: ', explorer.epsilon)
    if i % 50 == 0:
        print("Evaluation episodes ", i-50, "-", i, " Epsilon: ", explorer.epsilon, " AvgReward : ", GlobalR/50)
        GlobalR=0
        #print('statistics:', agent.get_statistics())
print('Finished.')



