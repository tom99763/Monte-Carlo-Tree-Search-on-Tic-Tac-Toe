import numpy as np
from copy import deepcopy
from math import sqrt, log
from random import choice


class Node:
    def __init__(self, env, state):
        self.env = deepcopy(env)
        self.state = state
        self.total_value = 0
        self.visit_times = 0
        self.parent = None
        self.children = []
        self.done = self.env.done
        self.actions = []
        self.from_action = None
        self.player = self.state[1]

    def update_visit(self):
        self.visit_times += 1

    def update_total_value(self, value):
        self.total_value += value

    def update_children(self, child):
        self.children.append(child)

    def update_parent(self, parent):
        self.parent = parent

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    # upper confidence bound
    def UCB(self, c):
        exploit = self.total_value/self.visit_times
        explore = c*sqrt(log(self.parent.visit_times)/self.visit_times)

        return exploit+explore

    # record children
    def update_actions(self, action):
        self.actions.append(action)

    def update_from_action(self, action):
        self.from_action = action

    def take_action(self, action):
        env = deepcopy(self.env)

        s_, reward, done, info = env.step(action)

        self.update_actions(action)

        return env, s_
