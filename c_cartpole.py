"""
Classic cart-pole system implemented by Rich Sutton et al.
Continuous version by Ian Danforth, with simplification's form S. J. Guy
"""

import math
import numpy as np


class ContinuousCartPoleEnv():

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 30.0
        self.tau = 0.02  # seconds between state updates
        self.min_action = -1.0
        self.max_action = 1.0

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,)
        )
        self.observation_space = Box(-high, high)

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
        force = self.force_mag * float(action)
        self.state = self.stepPhysics(force)
        x, x_dot, theta, theta_dot = self.state
        done = x < -self.x_threshold or x > self.x_threshold \
            or theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        else:
            reward = -1.0
            self.pole_fell = True
        if self.pole_fell:
            assert("You are Simulating past done, this is wasteful or a mistake (you should call reset)")

        return np.array(self.state), reward, done, {}


    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.pole_fell = False
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


