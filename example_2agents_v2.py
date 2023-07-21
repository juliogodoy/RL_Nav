#!/usr/bin/env python

import rvo2
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import OpenGL
import math
import OpenGL.GL
import OpenGL.GLUT
import OpenGL.GLU
from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *
print("Imports successful!") # If you see this printed to the console then installation was successful
M_PI = 3.14159265358979323846
RAND_MAX = 32767

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


def idle():
    glutPostRedisplay()

def renderBitmapString(x,y, font, nombre):    
    glRasterPos2f(x,y)
    for c in range (nombre):
        glutBitmapCharacter(font, *c)

def unit_vector(v):
    return v / np.linalg.norm(v)



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

