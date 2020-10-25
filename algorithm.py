import numpy as np
from copy import deepcopy
from math import sqrt, log
from random import choice
from node import Node


class MonteCarloTreeSearch:
    def __init__(self, iters, player='O', gamma=0.9):
        self.iters = iters
        self.player = 'O'
        self.gamma = gamma

    def search(self, state, env):
        root = Node(env, state)
        for i in range(self.iters):
            node = self.tree_policy(root)
            if node.done:
                continue
            reward = self.simulate(node)
            self.backpropagation(node, reward)

        return self.best_child(root, c=0).from_action

    def tree_policy(self, node):
        # run until node is terminal
        while node.done == False:
            if len(node.children) < len(node.env.available_actions()):
                return self.expand(node)
            else:
                node = self.best_child(node)  # assign node and search down

        return node  # done return itself

    def expand(self, node):
        for act in node.env.available_actions():
            if act not in node.actions:
                env, state = node.take_action(act)

                child = Node(env, state)
                # update info
                child.update_parent(node)
                child.update_from_action(act)
                node.update_children(child)

                return child

    def best_child(self, node, c=sqrt(0.5)):
        ucb = []
        for child in node.children:
            if child.visit_times == 0:
                return child
            else:
                # parent visit time is impossible to be 0 because it has children
                ucb.append(child.UCB(c))
        return node.children[np.argmax(ucb)]

    def simulate(self, node):
        env = deepcopy(node.env)
        while env.done == False:
            action = choice(env.available_actions())
            s_, reward, done, info = env.step(action)
        return reward

    def backpropagation(self, node, reward):
        node.update_visit()
        qvalue = reward+self.gamma*node.total_value/node.visit_times
        if self.player == 'O':
            node.update_total_value(qvalue)
        else:
            node.update_total_value(-qvalue)

        if node.is_root() == False:
            self.backpropagation(node.parent, reward)
