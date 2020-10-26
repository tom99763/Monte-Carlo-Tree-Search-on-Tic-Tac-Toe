import numpy as np
from copy import deepcopy
from math import sqrt, log
from random import choice
from node import Node






class tree:
    def __init__(self,iters,env,player=None):
        self.iters=iters
        self.player=player
        self.env=deepcopy(env)
        self.root=Node(self.env,env._get_obs())
    
    def to_child(self,state):
        for child in self.root.children:
            if child.state==state:
                self.root=child
                break
            
    def search(self):
        for i in range(self.iters):
            node=self.tree_policy(self.root)
            if node.done:
                continue
            reward=self.simulate(node)
            self.backpropagation(node,reward)
            
        self.root=self.best_child(self.root,c=0)
        return self.root.from_action
           
    def tree_policy(self,node):
        #run until node is terminal
        while node.done==False:
            if len(node.children)<len(node.env.available_actions()):
                return self.expand(node)
            else:
                node=self.best_child(node) #assign node and search down
                
        return node #done return itself
    
    def expand(self,node):
        for act in node.env.available_actions():
            if act not in node.actions:
                env,state=node.take_action(act)
                
                child=Node(env,state)
                #update info
                child.update_parent(node)
                child.update_from_action(act)
                node.update_children(child)
                
                return child
            
            
    def best_child(self,node,c=sqrt(0.5)):
        ucb=[]
        
        if node.player==self.player:
            for child in node.children:
                if child.visit_times==0:
                    return child
                else:
                    ucb.append(child.UCB(c))
        else:
            for child in node.children:
                if child.visit_times==0:
                    return child
                else:
                    #the less opponent get ,the more we get
                    ucb.append(1-child.UCB(-c))
            
            
        return node.children[np.argmax(ucb)]
    
    
    
    def simulate(self,node):
        env=deepcopy(node.env)
        while env.done==False:
            action=choice(env.available_actions())
            s_,reward,done,info=env.step(action)
            
        return reward
    
    
    def backpropagation(self,node,reward):
        node.update_visit()
        if self.player=='O':
            node.update_total_value(reward)
        else:
            node.update_total_value(-reward)
        
        if node.is_root()==False:
            self.backpropagation(node.parent,reward)
            
            
            
    def reset_root(self):
        self.backup(self.root)
        
    def backup(self,root):
        if root.is_root():
            self.root=root
        else:
            self.backup(root.parent)
