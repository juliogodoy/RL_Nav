"""
Classic cart-pole system implemented by Rich Sutton et al.
Continuous version by Ian Danforth, with simplification's form S. J. Guy
"""

import math
import numpy as np
from collections import namedtuple, deque
from itertools import count
import rvo2
import pygame
import torch
import transformations
import random
from pygame import gfxdraw


M_PI = 3.14159265358979323846
RAND_MAX = 32767
sim = rvo2.PyRVOSimulator(1/5., 15, 10, 5, 1.3, 0.4, 1.5)

BLUE = (0 , 0 , 255)
CYAN = (0 , 255 , 255)
RED = (255, 0 , 0 )
GREEN = (0 , 255 , 0)
BLACK = (0,0,0)
BROWN = (150, 75, 0)
WHITE = (255,255,255)
FPS = 60
RESOLUTION=500 # ASSUMES SQUARE
room_width = 60 # meters
pixels2meters = lambda x: (room_width * x / RESOLUTION)-(room_width/2)
meters2pixels = lambda x: ((x+room_width/2)/room_width * RESOLUTION)
abs_meters2pixels = lambda x: meters2pixels(x) - meters2pixels(0) 




def draw_circle(surface, x, y, radius, color):
    xr = int(round(x))
    yr = int(round(y))
    rr = int(round(radius))
    gfxdraw.aacircle(surface, xr, yr, rr, color)
    gfxdraw.filled_circle(surface, xr, yr, rr, color)

def draw_range(surface, x, y, radius, color):
    xr = int(round(x))
    yr = int(round(y))
    rr = int(round(radius))
    gfxdraw.aacircle(surface, xr, yr, rr, color)
    #gfxdraw.filled_circle(surface, xr, yr, rr, color)


def draw_AABB(surface, TLX, TLY, BLX, BLY, color):
    left = TLX
    top = TLY
    dimx = abs(TLX - BLX)
    dimy = abs(TLY - BLY)

    gfxdraw.rectangle(surface, [left, top, dimx, dimy], color)
    gfxdraw.box(surface, [left, top, dimx, dimy], color)

def isBoxCircleColliding(xb, yb, wb, hb, xc, yc, rc):
    circleDistance_x = abs(xc - xb)
    circleDistance_y = abs(yc - yb)

    if (circleDistance_x > (wb/2 + rc)): return False
    if (circleDistance_y > (hb/2 + rc)): return False
    if (circleDistance_x <= (wb/2)): return True
    if (circleDistance_y <= (hb/2)): return True

    cornerDistance_sq = (circleDistance_x - wb/2)**2 + (circleDistance_y - hb/2)**2

    return cornerDistance_sq <= (rc**2)

def isCircleCircleColliding(xa,ya,ra, xb,yb,rb):
   return (xa - xb)**2 + (ya - yb)**2 <= (ra + rb)**2

def unit_vector(v):
    return v / np.linalg.norm(v)

class Agent():

    def __init__(self, id, position):
        self.id=id
        self.position=np.array(position)
        if self.id==0:
            self.color=RED
        else:
            self.color=BLUE
        self.radius=0.4

    #def setPosition(agent_no):
    #    self.position=np.array(sim.getAgentPosition(agent_no))

class Obstacle():
    def __init__(self,topleft,bottomright):
        self.topleft=topleft
        self.bottomright=bottomright


class RVOEnv():
    
    def __init__(self):
        ## Main part of the code
        self.render_mode="human"
        self.episode=0
        self.forward=0
        self.stepsEpisode=0
        self.screen=None
        self.background=None
        self.a0 = sim.addAgent((4.5, 7.5))
        self.a1 = sim.addAgent((40, 8), 15, 10, 5, 1.3, 0.4, 1.5, (0, 0))
        self.agents = [Agent(0,sim.getAgentPosition(0))]
        self.agents.append(Agent(1,sim.getAgentPosition(1)))
        self.episodes_goal_reached=0




        # Obstacles are also supported.
        #o1 = sim.addObstacle([(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1)])
        self.o1 = sim.addObstacle([(8.0, 100.0), (-8.0, 100.0), (-8.0, 8.42),(8.0, 8.42)])
        self.o2 = sim.addObstacle([(8.0, 7.58), (-8.0, 7.58), (-8.0, -100.0),(8.0, -100.0)])
        #self.o3 = sim.addObstacle([(8, 20), (7.5, 20), (7.5, 9), (8,9)])
        #self.o4 = sim.addObstacle([(8, 7), (7.5, 7), (7.5, -20), (8,-20)])
        #self.o5 = sim.addObstacle([(-7.5, 20), (-8, 20), (-8, 9), (-7.5,9)])
        #self.o6 = sim.addObstacle([(-7.5, 7), (-8, 7), (-8, -20), (-7.5,-20)])

        sim.processObstacles()


        self.action_space = Discrete(2)#np.array([0,1])

        self.observation_space = Box((np.full((26,1), -np.inf)), (np.full((26,1), np.inf))) 

        self.state = None



    def render(self):
        if self.render_mode is None:
            #gym.logger.warn(
            #    "You are calling render method without specifying any render mode."
            #)
            return

        def offset(i, size, offset=0):
            if i == 0:
                return -(size) - offset
            else:
                return offset

        screen_height = RESOLUTION
        screen_width = RESOLUTION


        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
                self.font = pygame.font.SysFont(None, 14)
            pygame.event.get()
        elif self.screen is None:
            pygame.font.init()
            self.screen = pygame.Surface((screen_width, screen_height))
            self.font = pygame.font.SysFont(None, 14)

        
        #self.agent.update() # rectifies agent sprite to state location

        self.screen.fill(WHITE)
        if not self.background is None:
            surf = pygame.surfarray.make_surface(self.background)
            surf = pygame.transform.scale(surf,RESOLUTION)
            self.screen.blit(surf, (0, 0))

        #for rend in self.render_group.values():
        #    rend(self.screen, self.font)

        obstacle_pos=np.array([-8,8.42])
        obstacle_size=np.array([8,100])
        draw_AABB(self.screen, *meters2pixels(obstacle_pos),*meters2pixels(obstacle_size), BLUE)
        obstacle_pos=np.array([-8,-100])
        obstacle_size=np.array([8,7.58])
        draw_AABB(self.screen, *meters2pixels(obstacle_pos),*meters2pixels(obstacle_size), RED)
        

        #[(8.0, 9.0), (-8.0, 9.0), (-8.0, 8.42),(8.0, 8.42)]

        #for obs in self.boxObstacles:
        #    draw_AABB(self.screen, *meters2pixels(obs.aabb), obs.color)
        indx=0
        for agent in self.agents:
            agent.position=np.array(sim.getAgentPosition(indx))
            indx=indx+1
            draw_circle(self.screen, *meters2pixels(agent.position), abs_meters2pixels(agent.radius), agent.color)
            draw_range(self.screen, *meters2pixels(agent.position), abs_meters2pixels(15), agent.color)
            

        # render collisions if you want

        pygame.display.flip()
        pygame.time.Clock().tick(FPS)

    def step(self, action):
        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        positions = ['(%5.3f, %5.3f)' % sim.getAgentPosition(0)]

        for agent_no in range(2):
            random.seed()
            pert_agent =random.random()*RAND_MAX
            angle = pert_agent*2.0 * M_PI / RAND_MAX
            dist= pert_agent*0.01 * M_PI / RAND_MAX

            #Agent 0 is the decision maker, Agent 1 is simple ORCA agent
            #print("Action: ", action)
    
            if agent_no==0:
                GoalPos=np.array([10,8])
                GoalVector = GoalPos-np.array(sim.getAgentPosition(agent_no))
                if action==0: #move to the goal
                    const=1
                    self.forward=self.forward+1
                    self.stepsEpisode=self.stepsEpisode+1
                    #GoalVector = GoalPos-np.array(sim.getAgentPosition(agent_no)) 
                else:
                    const=-1
                    GoalVector = const*GoalVector
                    self.stepsEpisode=self.stepsEpisode+1
                    #GoalVector = np.array([10,const*-80])-np.array(sim.getAgentPosition(agent_no)) 

            else:
                GoalPos=np.array([50,8])
                const=-1
                GoalVector = GoalPos-np.array(sim.getAgentPosition(agent_no)) 
            #GoalVector = np.array([10,const*8])-np.array(sim.getAgentPosition(agent_no)) 
            unitario = unit_vector(GoalVector)
            sim.setAgentPrefVelocity(agent_no, (unitario[0]+dist*math.cos(angle), unitario[1]+dist*math.sin(angle)))
        #print("Action: ", action)
        #print("Agent 0 preferred velocity: ", sim.getAgentPrefVelocity(0))
        
            
        sim.doStep()
        #print("Agent 0 ORCA velocity: ", sim.getAgentVelocity(0))
        #print("Goal-position Difference agent 0: ", np.linalg.norm(np.array([10,8])- np.array(sim.getAgentPosition(0))))


        agent_own_p= sim.getAgentPosition(0)
        agent_own_v= sim.getAgentVelocity(1)
        neighbor_relative_p= np.array(sim.getAgentPosition(1))-np.array(sim.getAgentPosition(0))
        distance_agents= np.absolute(neighbor_relative_p)
        #print("---------------------------------------------------------------------------")
        #print("Action 0 action: ", action)
        #print("Agent 0 pos: ", sim.getAgentPosition(0), "  - Agent 1 pos: ",sim.getAgentPosition(1))
        #print("Distance between agents is ", distance_agents[0])
        #print("Agent 0 vel: ", sim.getAgentVelocity(0), "  - Agent 1 vel: ",sim.getAgentVelocity(1))
        #print("---------------------------------------------------------------------------")
        neighbor_relative_v= np.array(sim.getAgentVelocity(1))-np.array(sim.getAgentVelocity(0))
        if(np.linalg.norm(neighbor_relative_p)>10):
            neighbor_relative_p=np.array([RAND_MAX,RAND_MAX])
            neighbor_relative_v=np.array([RAND_MAX,RAND_MAX])
        
        #print("Obs: ", neighbor_relative_p)
        obstacle0_relative_p= np.array([8.0, 100.0])-np.array(agent_own_p)
        if(np.linalg.norm(obstacle0_relative_p)>10):
            obstacle0_relative_p=np.array([RAND_MAX,RAND_MAX])

        obstacle1_relative_p= np.array([-8.0, 100.0])-np.array(agent_own_p)

        if(np.linalg.norm(obstacle1_relative_p)>10):
             obstacle1_relative_p=np.array([RAND_MAX,RAND_MAX])

        obstacle2_relative_p= np.array([-8.0, 8.42])-np.array(agent_own_p)
        if(np.linalg.norm(obstacle2_relative_p)>10):
             obstacle2_relative_p=np.array([RAND_MAX,RAND_MAX])

        obstacle3_relative_p= np.array([8.0, 8.42])-np.array(agent_own_p)
        if(np.linalg.norm(obstacle3_relative_p)>10):
            obstacle3_relative_p=np.array([RAND_MAX,RAND_MAX])

        obstacle4_relative_p= np.array([8.0, 7.58])-np.array(agent_own_p)
        if(np.linalg.norm(obstacle4_relative_p)>10):
            obstacle4_relative_p=np.array([RAND_MAX,RAND_MAX])


        obstacle5_relative_p= np.array([-8.0, 7.58])-np.array(agent_own_p)
        if(np.linalg.norm(obstacle5_relative_p)>10):
            obstacle5_relative_p=np.array([RAND_MAX,RAND_MAX])



        obstacle6_relative_p= np.array([-8.0, -100.0])-np.array(agent_own_p)

        if(np.linalg.norm(obstacle6_relative_p)>10):
            obstacle6_relative_p=np.array([RAND_MAX,RAND_MAX])


        obstacle7_relative_p= np.array([8.0, -100.0])-np.array(agent_own_p)
        if(np.linalg.norm(obstacle7_relative_p)>10):
            obstacle7_relative_p=np.array([RAND_MAX,RAND_MAX])




        goal_relative_p=np.array([10,8])- np.array(agent_own_p)

        

        part1= np.concatenate((np.array(agent_own_p),np.array(agent_own_v),np.array(neighbor_relative_p)))#,np.array(agent_own_v),np.array(neighbor_relative_p))
        part2= np.concatenate((neighbor_relative_v, obstacle0_relative_p,obstacle1_relative_p))
        
        part3= np.concatenate((obstacle2_relative_p,obstacle3_relative_p,obstacle4_relative_p))
        part4= np.concatenate((obstacle5_relative_p,obstacle6_relative_p,obstacle7_relative_p))
        #part5= np.concatenate((goal_relative_p))
        part6=np.concatenate((part3, part4, goal_relative_p))

        self.state = np.concatenate((part1,part2,part6))#agent_own_p,agent_own_v,neighbor_relative_p, neighbor_relative_v, obstacle0_relative_p,obstacle1_relative_p,obstacle2_relative_p,obstacle3_relative_p,obstacle4_relative_p,obstacle5_relative_p,obstacle6_relative_p,obstacle7_relative_p,goal_relative_p)
        #print("state size is: ", self.state.size)
        #print("Goal-position Difference agent 0: ", np.linalg.norm(np.array([10,8])- np.array(sim.getAgentPosition(0))))
        #print("Goal-position Difference agent 1: ", np.linalg.norm(np.array([-10,28])- np.array(sim.getAgentPosition(1))))
        done = (np.linalg.norm(np.array([5,8])- np.array(sim.getAgentPosition(0))) < 1)# and (np.linalg.norm(np.array([-10,8])- np.array(sim.getAgentPosition(1))) < 1)
        #print("Done is ", done)

        #done = (np.array(10,28)- sim.getAgentPosition(0)) < 1
        done = bool(done)

        #if done:
            #print("I reached the GOAL!!")

        if not done:
            reward = -1.0
        else:
            #print("I reached the GOAL!!")
            #print("Goal-position Difference agent 0: ", np.linalg.norm(np.array([10,28])- np.array(sim.getAgentPosition(0))))
            #print("Goal-position Difference agent 1: ", np.linalg.norm(np.array([-10,28])- np.array(sim.getAgentPosition(1))))
            
            self.episodes_goal_reached=self.episodes_goal_reached+1
            reward = 100.0
            
        return np.array(self.state), reward, done, {}


    def reset(self):

        if self.episode % 50 ==0:
            print("Goal reached: ", 100*(self.episodes_goal_reached/50), "%")
            self.episodes_goal_reached=0
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(26,))
        #print("EPISODE: ", self.episode)
        #if self.episode>0:
        #    print("Straight-to-goal action rate: ", self.forward/self.stepsEpisode)
        self.forward=0
        self.stepsEpisode=0
        self.episode=self.episode+1
         # Pass either just the position (the other parameters then use
        # the default values passed to the PyRVOSimulator constructor),
        # or pass all available parameters.
        random.seed()
        initialPosProbX= random.random()
        initialPosProbY= random.random()


        sim.setAgentPosition(0, (-10.5+initialPosProbX*10,8))





        #sim.setAgentPosition(0, (4-initialPosProbX*(self.episode/100 ),8))
        #sim.setAgentPosition(0, (5,8))

        #if self.episode>1000:
            #print(" Agent 0 position: ", sim.getAgentPosition(0))

        #if initialPosProb < 0.2:
        #    sim.setAgentPosition(0, (10+initialPosProb,15+initialPosProb))
        #elif initialPosProb<0.4:
        #    sim.setAgentPosition(0, (10+initialPosProb,0+initialPosProb))
        #elif initialPosProb<0.6:
        #    sim.setAgentPosition(0,(13+initialPosProb, 6+initialPosProb))
        #elif initialPosProb<0.8:
        #    sim.setAgentPosition(0,(15+initialPosProb, 4+initialPosProb))
        #else:
        #    sim.setAgentPosition(0,(12+initialPosProb,2+initialPosProb))

        #sim.setAgentPosition(0,(-10,8))




        sim.setAgentPosition(1, (40,8))
        #print(" Agent 0 position: ", sim.getAgentPosition(0))
        #print(" Agent 1 position: ", sim.getAgentPosition(1))

        for agent_no in range(2):
            random.seed()
            pert_agent =random.random()*RAND_MAX
            angle = pert_agent*2.0 * M_PI / RAND_MAX
            dist= pert_agent*0.01 * M_PI / RAND_MAX
            #sim.setAgentPrefVelocity(a0, sim.getAgentPrefVelocity(a0) +  dist * (math.cos(angle), math.sin(angle)))


            if agent_no==0:
                GoalPos=np.array([10,8])
                const=1
                GoalVector = GoalPos-np.array(sim.getAgentPosition(agent_no))
            else:
                const=1
                GoalPos=np.array([50,8])
                GoalVector = GoalPos-np.array(sim.getAgentPosition(agent_no))

            #GoalVector = np.array([10,const*8])-np.array(sim.getAgentPosition(agent_no)) 
            unitario = unit_vector(GoalVector)
            sim.setAgentPrefVelocity(agent_no, (unitario[0]+dist*math.cos(angle), unitario[1]+dist*math.sin(angle)))
                  

        self.initialPrefVel0= sim.getAgentPrefVelocity(0)
        #print("RESET Agent 0 preferred velocity: ", self.initialPrefVel0)
        self.initialPrefVel1= sim.getAgentPrefVelocity(1)
        #print(" Agent 1 preferred velocity: ", self.initialPrefVel1)

        #print('Simulation has %i agents and %i obstacle vertices in it.' % (sim.getNumAgents(), sim.getNumObstacleVertices()))

        #print('Running simulation')

        

        #observation is composed by: agent's own position (2), own velocity(2), the other agent's -relative- position(2) and -relative- velocity(2), obstacles' -relative- positions/corners(16), and relative goal position(2)

        agent_own_p= sim.getAgentPosition(0)
        agent_own_v= sim.getAgentVelocity(1)
        neighbor_relative_p= np.array(sim.getAgentPosition(1))-np.array(sim.getAgentPosition(0))
        if(np.linalg.norm(neighbor_relative_p)>10):
            neighbor_relative_p=np.array([RAND_MAX,RAND_MAX])
            neighbor_relative_v=np.array([RAND_MAX,RAND_MAX])
        
        #print("Obs: ", neighbor_relative_p)
        obstacle0_relative_p= np.array([8.0, 100.0])-np.array(agent_own_p)
        if(np.linalg.norm(obstacle0_relative_p)>10):
            obstacle0_relative_p=np.array([RAND_MAX,RAND_MAX])

        obstacle1_relative_p= np.array([-8.0, 100.0])-np.array(agent_own_p)

        if(np.linalg.norm(obstacle1_relative_p)>10):
             obstacle1_relative_p=np.array([RAND_MAX,RAND_MAX])

        obstacle2_relative_p= np.array([-8.0, 8.42])-np.array(agent_own_p)
        if(np.linalg.norm(obstacle2_relative_p)>10):
             obstacle2_relative_p=np.array([RAND_MAX,RAND_MAX])

        obstacle3_relative_p= np.array([8.0, 8.42])-np.array(agent_own_p)
        if(np.linalg.norm(obstacle3_relative_p)>10):
            obstacle3_relative_p=np.array([RAND_MAX,RAND_MAX])

        obstacle4_relative_p= np.array([8.0, 7.58])-np.array(agent_own_p)
        if(np.linalg.norm(obstacle4_relative_p)>10):
            obstacle4_relative_p=np.array([RAND_MAX,RAND_MAX])


        obstacle5_relative_p= np.array([-8.0, 7.58])-np.array(agent_own_p)
        if(np.linalg.norm(obstacle5_relative_p)>10):
            obstacle5_relative_p=np.array([RAND_MAX,RAND_MAX])



        obstacle6_relative_p= np.array([-8.0, -100.0])-np.array(agent_own_p)

        if(np.linalg.norm(obstacle6_relative_p)>10):
            obstacle6_relative_p=np.array([RAND_MAX,RAND_MAX])


        obstacle7_relative_p= np.array([8.0, -100.0])-np.array(agent_own_p)
        if(np.linalg.norm(obstacle7_relative_p)>10):
            obstacle7_relative_p=np.array([RAND_MAX,RAND_MAX])




        goal_relative_p=np.array([10,8])- np.array(agent_own_p)

        sim.doStep()
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
