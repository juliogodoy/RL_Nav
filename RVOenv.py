"""
Classic cart-pole system implemented by Rich Sutton et al.
Continuous version by Ian Danforth, with simplification's form S. J. Guy
"""

import math
import numpy as np
from collections import namedtuple, deque
from itertools import count
import rvo2
import torch
import transformations
import random


M_PI = 3.14159265358979323846
RAND_MAX = 32767
sim = rvo2.PyRVOSimulator(1/5., 15, 10, 5, 1.3, 0.4, 1.5)


def unit_vector(v):
    return v / np.linalg.norm(v)

class RVOEnv():



    def __init__(self):
        ## Main part of the code

        

        # Pass either just the position (the other parameters then use
        # the default values passed to the PyRVOSimulator constructor),
        # or pass all available parameters.
        self.a0 = sim.addAgent((-10, 28))
        self.a1 = sim.addAgent((10, 28), 15, 10, 5, 1.3, 0.4, 1.5, (0, 0))



        # Obstacles are also supported.
        #o1 = sim.addObstacle([(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1)])
        self.o1 = sim.addObstacle([(8.0, 29.0), (-8.0, 29.0), (-8.0, 28.42),(8.0, 28.42)])
        self.o2 = sim.addObstacle([(8.0, 27.58), (-8.0, 27.58), (-8.0, 27.0),(8.0, 27.0)])
        self.o3 = sim.addObstacle([(8, 40), (7.5, 40), (7.5, 29), (8,29)])
        self.o4 = sim.addObstacle([(8, 27), (7.5, 27), (7.5, 0), (8,0)])
        self.o5 = sim.addObstacle([(-7.5, 40), (-8, 40), (-8, 29), (-7.5,29)])
        self.o6 = sim.addObstacle([(-7.5, 27), (-8, 27), (-8, 0), (-7.5,0)])

        sim.processObstacles()

        for agent_no in range(2):
            random.seed()
            pert_agent =random.random()*RAND_MAX
            angle = pert_agent*2.0 * M_PI / RAND_MAX
            dist= pert_agent*0.01 * M_PI / RAND_MAX
            #sim.setAgentPrefVelocity(a0, sim.getAgentPrefVelocity(a0) +  dist * (math.cos(angle), math.sin(angle)))

            if agent_no==0:
                const=1
            else:
                const=-1
            GoalVector = np.array([const*10,28])-np.array(sim.getAgentPosition(agent_no)) 
            unitario = unit_vector(GoalVector)
            sim.setAgentPrefVelocity(agent_no, (unitario[0]+dist*math.cos(angle), unitario[1]+dist*math.sin(angle)))
                  

        self.initialPrefVel0= sim.getAgentPrefVelocity(0)
        print(" Agent 0 preferred velocity: ", self.initialPrefVel0)
        self.initialPrefVel1= sim.getAgentPrefVelocity(1)
        print(" Agent 1 preferred velocity: ", self.initialPrefVel1)

        print('Simulation has %i agents and %i obstacle vertices in it.' % (sim.getNumAgents(), sim.getNumObstacleVertices()))

        print('Running simulation')

        self.action_space = Discrete(2)#np.array([0,1])

        #observation is composed by: agent's own position (2), own velocity(2), the other agent's -relative- position(2) and -relative- velocity(2), obstacles' -relative- positions/corners(16), and relative goal position(2)

        agent_own_p= sim.getAgentPosition(0)
        agent_own_v= sim.getAgentVelocity(1)
        neighbor_relative_p= np.array(sim.getAgentPosition(1))-np.array(sim.getAgentPosition(0))
        distance_agents= np.absolute(neighbor_relative_p)
        print("Distance between agents is ", distance_agents[0])
        if(distance_agents[0]>10):
            neighbor_relative_p=np.array([np.inf,np.inf])
        neighbor_relative_v= np.array(sim.getAgentVelocity(1))-np.array(sim.getAgentVelocity(0))
        obstacle0_relative_p= np.array([8.0, 29.0])-np.array(agent_own_p)
        obstacle1_relative_p= np.array([-8.0, 29.0])-np.array(agent_own_p)
        obstacle2_relative_p= np.array([-8.0, 28.42])-np.array(agent_own_p)
        obstacle3_relative_p= np.array([8.0, 28.42])-np.array(agent_own_p)
        obstacle4_relative_p= np.array([8.0, 27.58])-np.array(agent_own_p)
        obstacle5_relative_p= np.array([-8.0, 27.58])-np.array(agent_own_p)
        obstacle6_relative_p= np.array([-8.0, 27.0])-np.array(agent_own_p)
        obstacle7_relative_p= np.array([8.0, 27.0])-np.array(agent_own_p)
        goal_relative_p=np.array([10,28])- np.array(agent_own_p)

        sim.doStep()

        self.observation_space = Box((np.full((26,1), -np.inf)), (np.full((26,1), np.inf))) 

        self.state = None


    def stepPhysics(self, force):
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
            (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        return (x, x_dot, theta, theta_dot)



        

    def step(self, action):
        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        positions = ['(%5.3f, %5.3f)' % sim.getAgentPosition(0)]

        for agent_no in range(2):
            random.seed()
            pert_agent =random.random()*RAND_MAX
            angle = pert_agent*2.0 * M_PI / RAND_MAX
            dist= pert_agent*0.01 * M_PI / RAND_MAX

            #Agent 0 is the decision maker, Agent 1 is simple ORCA agent
    
            if agent_no==0:
                if action==0: #move to the goal
                    const=1
                else:
                    const=-1

            else:
                const=-1
            GoalVector = np.array([const*10,28])-np.array(sim.getAgentPosition(agent_no)) 
            unitario = unit_vector(GoalVector)
            sim.setAgentPrefVelocity(agent_no, (unitario[0]+dist*math.cos(angle), unitario[1]+dist*math.sin(angle)))
            
        sim.doStep()
        


        agent_own_p= sim.getAgentPosition(0)
        agent_own_v= sim.getAgentVelocity(1)
        neighbor_relative_p= np.array(sim.getAgentPosition(1))-np.array(sim.getAgentPosition(0))
        distance_agents= np.absolute(neighbor_relative_p)
        print("Distance between agents is ", distance_agents[0])
        if(distance_agents[0]>10):
            neighbor_relative_p=np.array([np.inf,np.inf])
        neighbor_relative_v= np.array(sim.getAgentVelocity(1))-np.array(sim.getAgentVelocity(0))
        obstacle0_relative_p= np.array([8.0, 29.0])-np.array(agent_own_p)
        obstacle1_relative_p= np.array([-8.0, 29.0])-np.array(agent_own_p)
        obstacle2_relative_p= np.array([-8.0, 28.42])-np.array(agent_own_p)
        obstacle3_relative_p= np.array([8.0, 28.42])-np.array(agent_own_p)
        obstacle4_relative_p= np.array([8.0, 27.58])-np.array(agent_own_p)
        obstacle5_relative_p= np.array([-8.0, 27.58])-np.array(agent_own_p)
        obstacle6_relative_p= np.array([-8.0, 27.0])-np.array(agent_own_p)
        obstacle7_relative_p= np.array([8.0, 27.0])-np.array(agent_own_p)
        goal_relative_p=np.array([10,28])- np.array(agent_own_p)

        

        part1= np.concatenate((np.array(agent_own_p),np.array(agent_own_v),np.array(neighbor_relative_p)))#,np.array(agent_own_v),np.array(neighbor_relative_p))
        part2= np.concatenate((neighbor_relative_v, obstacle0_relative_p,obstacle1_relative_p))
        
        part3= np.concatenate((obstacle2_relative_p,obstacle3_relative_p,obstacle4_relative_p))
        part4= np.concatenate((obstacle5_relative_p,obstacle6_relative_p,obstacle7_relative_p))
        #part5= np.concatenate((goal_relative_p))
        part6=np.concatenate((part3, part4, goal_relative_p))

        self.state = np.concatenate((part1,part2,part6))#agent_own_p,agent_own_v,neighbor_relative_p, neighbor_relative_v, obstacle0_relative_p,obstacle1_relative_p,obstacle2_relative_p,obstacle3_relative_p,obstacle4_relative_p,obstacle5_relative_p,obstacle6_relative_p,obstacle7_relative_p,goal_relative_p)
        #print("state size is: ", self.state.size)
        print("Goal-position Difference: ", np.linalg.norm(np.array([10,28])- np.array(sim.getAgentPosition(0))))
        done = np.linalg.norm(np.array([10,28])- np.array(sim.getAgentPosition(0))) < 1
        print("Done is ", done)

        #done = (np.array(10,28)- sim.getAgentPosition(0)) < 1
        done = bool(done)

        if not done:
            reward = 0.0
        else:
            print("Goal-position Difference: ", np.linalg.norm(np.array([10,28])- np.array(sim.getAgentPosition(0))))
            reward = 10.0
            
        return np.array(self.state), reward, done, {}


    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(26,))
        self.pole_fell = False
        #sim.setAgentPosition(0,(-10, 28))
        #print(" SIZE OF STATE: ", self.state.size)
        return np.array(self.state)

#------------------------------------
# Helper functions / classes

def is_float_integer(var) -> bool:
    """Checks if a variable is an integer or float."""
    return np.issubdtype(type(var), np.integer) or np.issubdtype(type(var), np.floating)

class Box:
  def __init__(self, low, high, shape = None):
    self.low = low
    self.high = high 
    if shape is not None:
        shape = tuple(int(dim) for dim in shape)
    elif is_float_integer(low):
        shape = (1,)
    else:
        shape = low.shape
    if is_float_integer(low): self.low = np.full(shape, low, dtype=float) 
    if is_float_integer(high): self.high = np.full(shape, low, dtype=float)
    print("shape", shape)
    self.shape = shape

  def sample(self):
    return np.random.uniform(
            low=self.low, high=self.high, size=self.shape
        )

class Discrete:
    """
    {0,1,...,n-1}

    Example usage:
    self.observation_space = spaces.Discrete(2)
    """
    def __init__(self, n):
        self.n = n
    def sample(self):
        return np.random.randint(self.n)
    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.kind in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int >= 0 and as_int < self.n
    def __repr__(self):
        return "Discrete(%d)" % self.n
    def __eq__(self, other):
        return self.n == other.n
