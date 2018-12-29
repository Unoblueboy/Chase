# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 20:16:19 2018

@author: Natha
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, Input, optimizers
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt

from IPython.display import clear_output

from random import sample, random, randint

from ChaseEnvironment import ChaseEnv

class DQN(object):
    def __init__(self):
        
        # Need to set up primary network
        self.input_states = Input(shape=[112,112,3], dtype=tf.float32)
        self.layer1 = layers.Conv2D(100, 8)(self.input_states)
        self.layer2 = layers.Conv2D(24, 4)(self.layer1)
        self.layer3 = layers.Flatten()(self.layer2)
        self.layer4 = layers.Dense(32, activation='relu', 
                                      name='layer4')(self.layer3)
        self.layer5 = layers.Dense(32, activation='relu', 
                                      name='layer5')(self.layer4)
        
        self.Value = layers.Dense(1, activation='linear', 
                                      name='value')(self.layer5)
        self.Advantage = layers.Dense(4, activation='linear', 
                                      name='advantage')(self.layer5)
        
        def anon(lis):
            adv = lis[1]
            val = lis[0]
            mean_adv = tf.reduce_mean(adv, axis = 1, keepdims = True)
            return val+adv-mean_adv
        
        self.optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.predict_Q = layers.Lambda(anon)([self.Value, self.Advantage])
        self.model = tf.keras.Model(inputs=self.input_states, outputs=self.predict_Q)
        self.model.compile(optimizer = self.optimizer,
                           loss = 'mse')
    
    def act(self, state, batch_size=1):
        q_values = self.model.predict(state, batch_size=batch_size)
        return np.array(np.argmax(q_values, axis = 1), dtype = np.int32)
    
    def learn(self, states, q_values):
        self.model.fit(states, q_values)
        
class ExperienceBuffer(object):
    def __init__(self, buffer_size = 5000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self, experience):
        if len(self.buffer) >= self.buffer_size:
            del self.buffer[0]
        self.buffer.append(experience)
        
    def sample(self, size):
        return np.array(sample(self.buffer, min(size, len(self.buffer))))
    
env = ChaseEnv(10)

# What up its ya boi, hyper parameters
pre_train_steps = 20
batch_size = 3
buffer_size = 5000
# In terms of steps
main_update_freq = 1
# In terms of episodes 
target_update_freq = 20
start_e = 1
end_e = 0.01
reducing_steps = 1000
step_drop = (start_e - end_e)/reducing_steps
discount = 0.95
tau = 1
save = True
save_freq = 50
tf.reset_default_graph()

# Set up some other avriables
buffer = ExperienceBuffer(buffer_size = buffer_size)
# player_2_buffer = ExperienceBuffer(buffer_size = buffer_size)
total_steps = 0
r_all = []
e = start_e
player_1_losses = []
player_2_losses = []

player_1_main_q = DQN()
player_1_target_q = DQN()
player_1_target_q.model.set_weights(player_1_main_q.model.get_weights())

player_2_main_q = DQN()
player_2_target_q = DQN()
player_2_target_q.model.set_weights(player_2_main_q.model.get_weights())

if save:
    try:
        player_1_main_q.model = load_model('chase 1/player_1_main_q.h5', custom_objects={"tf": tf})
        player_1_target_q.model = load_model('chase 1/player_1_target_q.h5', custom_objects={"tf": tf})
        player_2_main_q.model = load_model('chase 1/player_2_main_q.h5', custom_objects={"tf": tf})
        player_2_target_q.model = load_model('chase 1/player_2_target_q.h5', custom_objects={"tf": tf})
        print("Models Loaded")
    except Exception as ex:
        player_1_main_q.model.save('chase 1/player_1_main_q.h5')
        player_1_target_q.model.save('chase 1/player_1_target_q.h5')
        player_2_main_q.model.save('chase 1/player_2_main_q.h5')
        player_2_target_q.model.save('chase 1/player_2_target_q.h5')
        print("New Models Created")
else:
    player_1_target_q.model.set_weights(player_1_main_q.model.get_weights())
    player_2_target_q.model.set_weights(player_2_main_q.model.get_weights())
    
try:
    ep = 0
    while True:
        ep += 1
        ep_reward = 0
        s = env.reset()
        while True:
            total_steps += 1
            if total_steps <= pre_train_steps or random() < e:
                action1 = randint(0,3)
                action2 = randint(0,3)
            else:
                action1 = player_1_main_q.act(np.reshape(s,[1,112,112,3]))[0]
                action2 = player_2_main_q.act(np.reshape(s,[1,112,112,3]))[0]
            s1, r, done = env.step([action1, action2])
            experience = [s,[action1, action2], r, s1, int(done)]
            buffer.add(experience)
            ep_reward += r
            if total_steps > pre_train_steps:
                if e > end_e:
                    e -= step_drop
                    e = max(e, end_e)
                # can now start training
                if total_steps % main_update_freq == 0:
                    # update main q network
                    # Get a sample of the past experiences
                    batch = buffer.sample(batch_size)


                    for experience in batch:
                        old_state = experience[0]
                        action1 = int(experience[1][0])
                        action2 = int(experience[1][1])
                        reward1 = experience[2]
                        reward2 = -experience[2]
                        new_state = experience[3]
                        ended = experience[4]
                        
                        derived_action1 = player_1_main_q.act(np.reshape(new_state, [1,112,112,3]))[0]
                        q_target1 = player_1_target_q.model.predict(np.reshape(new_state, [1,112,112,3]))[0]
                        q_value1 = q_target1[int(derived_action1)]
                        target1 = reward1 + discount*q_value1
                        if ended:
                            target1 = reward1
                        # print(target)
                        # So need to feed in q_table not just updated values so update
                        # batch q table with new values
                        cur_q_values1 = player_1_main_q.model.predict(np.reshape(old_state, [1,112,112,3]))[0]
                        cur_q_values1[action1] = target1
                        loss_1 = player_1_main_q.model.fit(np.reshape(old_state, [1,112,112,3]), 
                                         np.reshape(cur_q_values1, [1, 4]), 
                                         verbose=0)
                        player_1_losses.append(loss_1.history['loss'][0])
                        
                        derived_action2 = player_2_main_q.act(np.reshape(new_state, [1,112,112,3]))[0]
                        q_target2 = player_2_target_q.model.predict(np.reshape(new_state, [1,112,112,3]))[0]
                        q_value2 = q_target2[int(derived_action2)]
                        target2 = reward2 + discount*q_value2
                        if ended:
                            target2 = reward2
                        # print(target)
                        # So need to feed in q_table not just updated values so update
                        # batch q table with new values
                        cur_q_values2 = player_2_main_q.model.predict(np.reshape(old_state, [1,112,112,3]))[0]
                        cur_q_values2[action2] = target2
                        loss_2 = player_2_main_q.model.fit(np.reshape(old_state, [1,112,112,3]), 
                                         np.reshape(cur_q_values2, [1, 4]), 
                                         verbose=0)
                        player_2_losses.append(loss_2.history['loss'][0])
            s = s1
            if done:
                print("Episode {}: {}, e is {}".format(ep, int(ep_reward), e))
                print("player 1 losses over the last 10 steps: {}".format(
                    sum(player_1_losses[-10:])/10))
                print("player 2 losses over the last 10 steps: {}".format(
                    sum(player_2_losses[-10:])/10))
                if ep % target_update_freq == 0:
                    # update target q network
                    main_weights = player_1_main_q.model.get_weights()
                    weighted_main_weights = []
                    for x in main_weights:
                        weighted_main_weights.append(x*tau)
                    target_weights = player_1_target_q.model.get_weights()
                    weighted_target_weights = []
                    for x in target_weights:
                        weighted_target_weights.append(x*(1-tau))
                    player_1_target_q.model.set_weights(weighted_main_weights + weighted_target_weights)
                    
                    main_weights = player_2_main_q.model.get_weights()
                    weighted_main_weights = []
                    for x in main_weights:
                        weighted_main_weights.append(x*tau)
                    target_weights = player_2_target_q.model.get_weights()
                    weighted_target_weights = []
                    for x in target_weights:
                        weighted_target_weights.append(x*(1-tau))
                    player_2_target_q.model.set_weights(weighted_main_weights + weighted_target_weights)
                    print()
                    print("Updating Target Network")
                if ep % save_freq == 0 and save:
                    print("Saving Networks")
                    player_1_main_q.model.save('chase 1/player_1_main_q.h5')
                    player_1_target_q.model.save('chase 1/player_1_target_q.h5')
                    player_2_main_q.model.save('chase 1/player_2_main_q.h5')
                    player_2_target_q.model.save('chase 1/player_2_target_q.h5')
                break
        r_all.append(ep_reward)
except KeyboardInterrupt:
    if save:
        print("Saving Networks")
        player_1_main_q.model.save('chase 1/player_1_main_q.h5')
        player_1_target_q.model.save('chase 1/player_1_target_q.h5')
        player_2_main_q.model.save('chase 1/player_2_main_q.h5')
        player_2_target_q.model.save('chase 1/player_2_target_q.h5')
        
fig = plt.figure()
a = fig.add_subplot(1, 1, 1)
done = False
s = env.reset()
while True:
    action1 = player_1_main_q.act(np.reshape(s,[1,112,112,3]))[0]
    action2 = player_2_main_q.act(np.reshape(s,[1,112,112,3]))[0]
    s, r, done = env.step([action1, action2])
    plt.imshow(s)
    plt.show()
    plt.gcf().clear()
    if done == True:
        break