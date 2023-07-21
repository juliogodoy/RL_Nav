#!/usr/bin/env python

import rvo2
import pfrl
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import OpenGL
from collections import namedtuple, deque
from itertools import count
import math
import gym
import OpenGL.GL
import OpenGL.GLUT
import OpenGL.GLU
from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *
print("Imports successful!") # If you see this printed to the console then installation was successful


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is ", device)
M_PI = 3.14159265358979323846
RAND_MAX = 32767

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


def InitGL():
    glShadeModel(GL_SMOOTH);                            # Enable Smooth Shading
    glClearColor(0.5, 0.5, 0.5, 0.5);               # Black Background
    glClearDepth(1.0);                                 # Depth Buffer Setup
    glEnable(GL_DEPTH_TEST);                            # Enables Depth Testing
    glDepthFunc(GL_LEQUAL);                             # The Type Of Depth Testing To Do
    glEnable ( GL_COLOR_MATERIAL );
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);


def reshape(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    q = width/height
    gluOrtho2D(-10, 10, 20, 40)
    #gluOrtho2D(-q*10, q*10, 10, 50 )
    glMatrixMode(GL_MODELVIEW)



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)



def idle():
    glutPostRedisplay()

def renderBitmapString(x,y, font, nombre):    
    glRasterPos2f(x,y)
    for c in range (nombre):
        glutBitmapCharacter(font, *c)

def unit_vector(v):
    return v / np.linalg.norm(v)



"""### Training loop

Finally, the code for training our model.

Here, you can find an ``optimize_model`` function that performs a
single step of the optimization. It first samples a batch, concatenates
all the tensors into a single one, computes $Q(s_t, a_t)$ and
$V(s_{t+1}) = \max_a Q(s_{t+1}, a)$, and combines them into our
loss. By definition we set $V(s) = 0$ if $s$ is a terminal
state. We also use a target network to compute $V(s_{t+1})$ for
added stability. The target network is updated at every step with a 
[soft update](https://arxiv.org/pdf/1509.02971.pdf)_ controlled by 
the hyperparameter ``TAU``, which was previously defined.



"""

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def showScreen():
    glLineWidth(2)
    glutGet(GLUT_ELAPSED_TIME)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # Remove everything from screen (i.e. displays all white)

    for step in range(2000):
        sim.doStep()
        time.sleep(0.01)

        positions = ['(%5.3f, %5.3f)' % sim.getAgentPosition(agent_no)
                 for agent_no in (a0, a1)]
        #print('step=%2i  t=%.3f  %s' % (step, sim.getGlobalTime(), '  '.join(positions)))
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # Remove everything from screen (i.e. displays all white)

        glMatrixMode(GL_MODELVIEW)
        

        for agent_no in range(2):
            glPushMatrix()
            glTranslatef(*sim.getAgentPosition(agent_no),0)
            glColor3f(0.4,0.9,0.0)
            glutSolidSphere(0.4,8,8)
            glDisable( GL_LIGHTING )
            glColor3f(0,0,0)
            random.seed()
            pert_agent =random.random()*RAND_MAX
            #pert_agent_0= random.random()*RAND_MAX
            #pert_agent_1= random.random()*RAND_MAX
            angle = pert_agent*2.0 * M_PI / RAND_MAX
            dist= pert_agent*0.01 * M_PI / RAND_MAX
            #angle_0 = pert_agent_0 * 2.0 * M_PI / RAND_MAX;
            #dist_0 = pert_agent_0 * 0.01 / RAND_MAX;
            #angle_1 = pert_agent_1 * 2.0 * M_PI / RAND_MAX;
            #dist_1 = pert_agent_1 * 0.01 / RAND_MAX;
            if agent_no==0:
                const=1
            else:
                const=-1
            GoalVector = np.array([const*10,28])-np.array(sim.getAgentPosition(agent_no)) 
            unitario = unit_vector(GoalVector)
            sim.setAgentPrefVelocity(agent_no, (unitario[0]+dist*math.cos(angle), unitario[1]+dist*math.sin(angle)))
            #print(" Agent 0 preferred velocity: ", sim.getAgentPrefVelocity(a0))
            #GoalVector_a1 = np.array([-10,28])-np.array(sim.getAgentPosition(a1))
            #unitario_a1 = unit_vector(GoalVector_a1)
            #sim.setAgentPrefVelocity(a1, (unitario_a1[0]+dist_1*math.cos(angle_0), unitario_a1[1]+dist_1*math.sin(angle_1)))
            #print(" Agent 1 preferred velocity: ", sim.getAgentPrefVelocity(a1))         
            glBegin(GL_LINES)
            glVertex3f(0,0,1.0)
            pfrev=  sim.getAgentPrefVelocity(agent_no);
            glVertex3f(pfrev[0],pfrev[1],1.0)
            glEnd()
            glColor3f(0,0,0)
            glPopMatrix()


        
        glMatrixMode(GL_MODELVIEW)

        glPushMatrix();
        glTranslatef(0, 0, 0.0);
        glColor3f(0.0, 0.0, 0.0);
        glBegin(GL_QUADS);
        glVertex2f(8.0,   29.0);
        glVertex2f(-8.0,   29.0);
        glVertex2f(-8.0, 28.42);
        glVertex2f(8.0, 28.42);
        glEnd();
                    
        glBegin(GL_QUADS);
        glVertex2f(8.0,   27.58);
        glVertex2f(-8.0,   27.58);
        glVertex2f(-8.0, 27.0);
        glVertex2f(8.0, 27.0);
        glEnd();

        glBegin(GL_QUADS);
        glVertex2f(8.0,   40);
        glVertex2f(7.5,   40);
        glVertex2f(7.5, 29.0);
        glVertex2f(8.0, 29.0);
        glEnd();

        glBegin(GL_QUADS);
        glVertex2f(8.0,   27);
        glVertex2f(7.5,   27);
        glVertex2f(7.5, 0.0);
        glVertex2f(8.0, 0.0);
        glEnd();

        glBegin(GL_QUADS);
        glVertex2f(-7.5,   40);
        glVertex2f(-8,   40);
        glVertex2f(-8, 29.0);
        glVertex2f(-7.5, 29.0);
        glEnd();

        glBegin(GL_QUADS);
        glVertex2f(-7.5,   27);
        glVertex2f(-8,   27);
        glVertex2f(-8, 0.0);
        glVertex2f(-7.5, 0.0);
        glEnd();
        glPopMatrix();
        glutSwapBuffers()
        glutPostRedisplay()  



    
    glutLeaveMainLoop()     


env = gym.make('CartPole-v0')
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
n_actions = env.action_space.n
q_func = QFunction(obs_size, n_actions)


# Use Adam to optimize q_func. eps=1e-2 is for stability.
optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)

# Set the discount factor that discounts future rewards.
gamma = 0.9

# Use epsilon-greedy for exploration
explorer = pfrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space.sample)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)

# Since observations from CartPole-v0 is numpy.float64 while
# As PyTorch only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
phi = lambda x: x.astype(np.float32, copy=False)

# Set the device id to use GPU. To use CPU only, set it to -1.
gpu = 0

# Now create an agent that will interact with the environment.
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

n_episodes = 300
max_episode_len = 200
for i in range(1, n_episodes + 1):
    obs = env.reset()
    R = 0  # return (sum of rewards)
    t = 0  # time step
    while True:
        # Uncomment to watch the behavior in a GUI window
        # env.render()
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
        reset = t == max_episode_len
        agent.observe(obs, reward, done, reset)
        if done or reset:
            break
    if i % 10 == 0:
        print('episode:', i, 'R:', R)
    if i % 50 == 0:
        print('statistics:', agent.get_statistics())
print('Finished.')








sim = rvo2.PyRVOSimulator(1/50., 15, 10, 5, 1.3, 0.4, 1.5)

# Pass either just the position (the other parameters then use
# the default values passed to the PyRVOSimulator constructor),
# or pass all available parameters.
a0 = sim.addAgent((-10, 28))
a1 = sim.addAgent((10, 28), 15, 10, 5, 1.3, 0.4, 1.5, (0, 0))



# Obstacles are also supported.
#o1 = sim.addObstacle([(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1)])
o1 = sim.addObstacle([(8.0, 29.0), (-8.0, 29.0), (-8.0, 28.42),(8.0, 28.42)])
o2 = sim.addObstacle([(8.0, 27.58), (-8.0, 27.58), (-8.0, 27.0),(8.0, 27.0)])
o3 = sim.addObstacle([(8, 40), (7.5, 40), (7.5, 29), (8,29)])
o4 = sim.addObstacle([(8, 27), (7.5, 27), (7.5, 0), (8,0)])
o5 = sim.addObstacle([(-7.5, 40), (-8, 40), (-8, 29), (-7.5,29)])
o6 = sim.addObstacle([(-7.5, 27), (-8, 27), (-8, 0), (-7.5,0)])

sim.processObstacles()


random.seed()
pert_agent_0= random.random()*RAND_MAX
pert_agent_1= random.random()*RAND_MAX
angle_0 = -0.5+pert_agent_0 * 2.0 * M_PI / RAND_MAX;
dist_0 = pert_agent_0 * 0.01 / RAND_MAX;
angle_1 = -0.5+pert_agent_1 * 2.0 * M_PI / RAND_MAX;
dist_1 = pert_agent_1 * 0.01 / RAND_MAX;
#sim.setAgentPrefVelocity(a0, sim.getAgentPrefVelocity(a0) +  dist * (math.cos(angle), math.sin(angle)))
                  

sim.setAgentPrefVelocity(a0, (1+dist_0*math.cos(angle_0), 0+dist_0*math.sin(angle_0)))
print(" Agent 0 preferred velocity: ", sim.getAgentPrefVelocity(a0))
sim.setAgentPrefVelocity(a1, (-1+dist_1*math.cos(angle_0), 0+dist_1*math.sin(angle_1)))
print(" Agent 1 preferred velocity: ", sim.getAgentPrefVelocity(a1))

print('Simulation has %i agents and %i obstacle vertices in it.' %
      (sim.getNumAgents(), sim.getNumObstacleVertices()))

print('Running simulation')

#### Deep Q learniong init:

# Get number of actions from gym action space
n_actions = 2
# Get the number of state observations
#state, info = env.reset()
state =[0,0,0,0,0,0,0,0] 
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0

#### OpenGL functions and main loop call
glutInit() # Initialize a glut instance which will allow us to customize our window
glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH) # Set the display mode to be colored
glutInitWindowSize(1024, 768)   # Set the width and height of your window
glutInitWindowPosition(0, 0)   # Set the position at which this windows should appear
wind = glutCreateWindow("MRS2023") # Give your window a title
glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS)
InitGL();
glutDisplayFunc(showScreen)  # Tell OpenGL to call the showScreen method continuously
glutReshapeFunc(reshape)
glutIdleFunc(idle)     #showScreen Draw any graphics or shapes in the showScreen function at all times
glutMainLoop()
print("Each agent has ", sim.getAgentRadius(1))

#for step in range(20):
#    sim.doStep()

#    positions = ['(%5.3f, %5.3f)' % sim.getAgentPosition(agent_no)
#                 for agent_no in (a0, a1, a2, a3)]
#    print('step=%2i  t=%.3f  %s' % (step, sim.getGlobalTime(), '  '.join(positions)))

