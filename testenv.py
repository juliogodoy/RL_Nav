import gym


gym.envs.register('RVOenv-v0', entry_point='rvoenv:RVOenv')
rvoenv = gym.make('RVOenv-v0')