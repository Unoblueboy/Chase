# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 18:45:43 2018

@author: Natha
"""
import numpy as np
from PIL import Image
from random import sample, randint

class ChaseObj(object):
    def __init__(self, x, y, colour, name):
        self.x = x
        self.y = y
        self.colour = colour
        self.name = name

class ChaseEnv(object):
    def __init__(self, size):
        self.size = size
        self.objects = []
        self.action_space_size = 4
        self.state = None
        self.steps = None
        self.done = None
        self.prev_dist = None
        # Generate the position of the 2 players
        # Generate Obstacles
        
    def reset(self):
        self.done = False
        self.steps = 0
        self.objects = []
        player1 = self.new_position([1.0,0,0], "player1")
        self.objects.append(player1)
        player2 = self.new_position([0,1.0,0], "player2")
        self.objects.append(player2)
        self.state = self.getState()
        self.prev_dist = abs(player1.x-player2.x) + abs(player1.y-player2.y)
        return self.state
    
    def new_position(self, colour, name):
        object_pos = [[obj.x, obj.y] for obj in self.objects]
        valid_pos = []
        for x in range(self.size):
            for y in range(self.size):
                if [x,y] not in object_pos:
                    valid_pos.append([x,y])
        pos = sample(valid_pos,1)[0]
        obj = ChaseObj(pos[0],pos[1], colour, name)
        return obj
    
    def playerStep(self, player, action):
        if action == 0: # UP
            player.y = min(self.size-1, player.y+1)
        elif action == 1: # RIGHT
            player.x = min(self.size-1, player.x+1)
        elif action == 2: # DOWN
            player.y = max(0, player.y-1)
        elif action == 3: # LEFT
            player.x = max(0, player.x-1)
    
    def step(self, action):
        # We are going to consider the reward in terms of the 
        # the player being chased, player1
        if self.steps >= 50:
            self.done = True
        self.steps+=1
        
        player1 = self.objects[0]
        action1 = action[0]
        self.playerStep(player1, action1)
        self.objects[0] = player1
        
        player2 = self.objects[1]
        action2 = action[1]
        self.playerStep(player2, action2)
        self.objects[1] = player2
        
        dist = abs(player1.x-player2.x) + abs(player1.y-player2.y)
        reward = np.sign(dist - self.prev_dist)
        if dist == 0:
            reward = -50
            self.done = True
        self.state = self.getState()
        self.prev_dist = dist
        return self.state, reward, self.done
    
    def getState(self):
        state = np.ones([self.size,self.size,3])
        for obj in self.objects:
            state[obj.x, obj.y] = obj.colour
        r = Image.fromarray(state[:,:,0]).resize([112,112], Image.NEAREST).rotate(90)
        r = np.array(r)
        g = Image.fromarray(state[:,:,1]).resize([112,112], Image.NEAREST).rotate(90)
        g = np.array(g)
        b = Image.fromarray(state[:,:,2]).resize([112,112], Image.NEAREST).rotate(90)
        b = np.array(b)
        image = np.stack((r,g,b), axis = 2)
        return image
    
    def sampleActionSpace(self):
        return [randint(0,3), randint(0,3)]
        