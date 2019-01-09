# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 18:45:43 2018

@author: Natha
"""
import numpy as np
from PIL import Image
from random import sample, randint
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.ion()
# %matplotlib qt

class ChaseObj(object):
    def __init__(self, x, y, colour, name):
        self.x = x
        self.y = y
        self.colour = colour
        self.name = name

class ChaseEnv(object):
    def __init__(self, size, imsize = 84):
        self.size = size
        self.imsize = imsize
        self.objects = []
        self.action_space_size = 4
        self.state = None
        self.steps = None
        self.done = None
        # No longer relevant
        # self.prev_dist = None
#        self.fig = plt.figure()
##        self.axs =  self.fig.add_subplot(1,1,1)
#        self.rendering = False
#        self.im = None
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
        # No longer relevant
        # self.prev_dist = abs(player1.x-player2.x) + abs(player1.y-player2.y)
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
        # Also we are going to assume player 1 steps first
        # Then player 2
        self.steps+=1
        reward = 0
        
        player1 = self.objects[0]
        player2 = self.objects[1]
        
        action1 = action[0]
        self.playerStep(player1, action1)
        self.objects[0] = player1
        
        if player1.x == player2.x and player1.y == player2.y:
            self.state = self.getState()
            reward = -1
            self.done = True
            return self.state, reward, self.done
        
        action2 = action[1]
        self.playerStep(player2, action2)
        self.objects[1] = player2
        
        # dist = abs(player1.x-player2.x) + abs(player1.y-player2.y)
        # reward = np.sign(dist - self.prev_dist)
        if player1.x == player2.x and player1.y == player2.y:
            self.state = self.getState()
            reward = -1
            self.done = True
            return self.state, reward, self.done
        
        self.state = self.getState()
        # self.prev_dist = dist
        if self.steps >= 50:
            reward = 1
            self.done = True
            return self.state, reward, self.done
        return self.state, reward, self.done
    
    def getState(self):
        state = np.ones([self.size+2,self.size+2,3])
        state[:,0,:] = state[:,self.size+1,:] = \
        state[self.size+1,:,:] = state[0,:,:] = np.zeros([self.size+2,3])
        for obj in self.objects:
            state[obj.x+1, obj.y+1] = obj.colour
        r = Image.fromarray(state[:,:,0]).resize([self.imsize,self.imsize], Image.NEAREST).rotate(90)
        r = np.array(r)
        g = Image.fromarray(state[:,:,1]).resize([self.imsize,self.imsize], Image.NEAREST).rotate(90)
        g = np.array(g)
        b = Image.fromarray(state[:,:,2]).resize([self.imsize,self.imsize], Image.NEAREST).rotate(90)
        b = np.array(b)
        image = np.stack((r,g,b), axis = 2)
        return image
    
    def sampleActionSpace(self):
        return [randint(0,3), randint(0,3)]
    
    def animate(self, i):
#        self.axs.clear()
        self.im.set_array(self.state)
    
    def render(self, update = 10):
        if not self.rendering:
            self.im = plt.imshow(self.state, animated = True)
            ani = animation.FuncAnimation(plt.gcf(), self.animate, interval = 100, blit=False)
            plt.show()
    
    def close(self):
        plt.ioff()
        
        