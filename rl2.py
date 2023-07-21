import pfrl
from pfrl.agents import PPO
import torch
from torch import nn
import numpy
import random
import math
import RVOenv
import c_cartpole

#Parameters
n_episodes = 500       #Total number of training episodes
max_episode_len = 200  #How many steps in a rollout/trajectory
args_gpu = -1          #Use GPU? -1 is false, otherwise device ID
args_epochs = 50

# import gym; env = gym.make('CartPole-v0') 
# import gym
# env = gym.make('MountainCarContinuous-v0') 
env =  RVOenv.RVOEnv()
#env =  c_cartpole.ContinuousCartPoleEnv()
obs_space = env.observation_space
action_space = env.action_space
print("Observation space:", obs_space)
print("Action space:", action_space)

obs = env.reset()
print('initial observation:', obs)

#rnd_number=random.random()
#if rnd_number<0.5:
#    action=0
#else:
#    action=1

action = env.action_space.sample()
obs, r, done, info = env.step(action)
print('next observation:', obs)
print('reward:', r)
print('done:', done)
print('info:', info)


# Set a random seed used in PFRL
# utils.set_random_seed(seed) #seed = 0

# Normalize observations based on their empirical mean and variance
obs_normalizer = pfrl.nn.EmpiricalNormalization(
    obs_space.low.size, clip_threshold=5
)

#print(obs_space.low)
#print(obs_space.shape)
#print(action_space.low)
#print("---")
obs_size = obs_space.low.size
print("Action space size: ",action_space.n)
action_size =action_space.n

print("Observation space size: ", obs_size)
print("Actions space size: ", action_size)

#Policy Network
policy = torch.nn.Sequential(
    nn.Linear(obs_size, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, action_size),
    pfrl.policies.GaussianHeadWithStateIndependentCovariance( #Gaussian output for stochastic policy
        action_size=action_size,
        var_type="diagonal",
        var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
        var_param_init=0,  # log std = 0 => std = 1
    ),
)

#Value Network
vf = torch.nn.Sequential(
    nn.Linear(obs_size, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1),
)

# Orthogonal initialization is used in the latest openai/baselines PPO
def ortho_init(layer, gain):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)

#Even layers are weights, initialize them with the orthogonal initialization
ortho_init(policy[0], gain=1)
ortho_init(policy[2], gain=1)
ortho_init(policy[4], gain=1e-2) #Google initialization
ortho_init(vf[0], gain=1)
ortho_init(vf[2], gain=1)
ortho_init(vf[4], gain=1e-2)     #Google initialization

# Combine a policy and a value function into a single model
model = pfrl.nn.Branched(policy, vf)

# Use Adam to optimize q_func. eps=1e-2 is for stability.
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)

# Since observations from CartPole-v0 is numpy.float64 while
# As PyTorch only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
phi = lambda x: x.astype(numpy.float32, copy=False)

agent = PPO(
    model,
    optimizer,
    obs_normalizer=obs_normalizer,
    gpu=args_gpu,
    # update_interval=args.update_interval,
    # minibatch_size=args.batch_size,
    epochs=args_epochs,
    clip_eps_vf=None,
    entropy_coef=0,
    standardize_advantages=True,
    gamma=0.995,
    lambd=0.97,
    phi=phi,
)

import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

pfrl.experiments.train_agent_with_evaluation(
    agent,
    env,
    steps=100000,
    eval_n_steps=None,
    eval_n_episodes=1,       
    train_max_episode_len=1000,  
    eval_interval=20,
    outdir='result_stats', 
)

print("done")


# for i in range(1, n_episodes + 1):
#     obs = env.reset()
#     R = 0  # return (sum of rewards)
#     t = 0  # time step
#     while True:
#         action = agent.act(obs)
#         obs, reward, done, _ = env.step(action)
#         R += reward
#         t += 1
#         reset = (t == max_episode_len)
#         agent.observe(obs, reward, done, reset)
#         if done or reset: break
#     if i % 10 == 0:
#         print('episode:', i, 'Return:', R)
#     if i % 50 == 0:
#         print('statistics:', agent.get_statistics())
# print('Finished.')
