# -*- coding: utf-8 -*-
"""
Created on Tue Oct 03 15:36:46 2017

@author: Alex
"""

import numpy as np

class ActorCritic:
    
    def init(self, DOF = 3):
        
        # GENERAL PARAMETERS
        self.n_interval = 9 # define intervals used to define the number of gaussian (kernel)
        self.gaussian_number = self.n_interval + 1 
        self.interval_lenght = 1. / self.n_interval
        self.X_pos = np.zeros([self.gaussian_number, DOF]) # gaussian avarage value
        self.X_rew = np.zeros([self.gaussian_number, 2]) # gaussian avarage value
        self.std_dev = 1. / ((self.n_interval -1) * 2)
        
        # computate gaussian's average values
        for gn in xrange(self.gaussian_number):
            if gn == 0 :
                self.X_pos[0] = 0    
            else:
                self.X_pos[gn] = self.X_pos[gn-1] + self.interval_lenght
                      
        # computate gaussian's average values
        for gn in xrange(self.gaussian_number):
            if gn == 0 :
                self.X_rew[0] = 0    
            else:
                self.X_rew[gn] = self.X_rew[gn-1] + self.interval_lenght
        
        # TIME PARAMETERS
        self.simulation_duration = 1.
        self.delta_time = 0.1
        self.steps = self.simulation_duration / self.delta_time
        self.tau = 1.  
        self.n_trial = 2000
        self.max_trial_movements = 5000
    
        # input units
        self.n_input = self.gaussian_number
        
        # output units
        self.actor_n_output = 3
        self.critic_n_output = 1
        
        # learning parameters
        self.a_eta = 0.1
        self.c_eta = 0.08
        self.discount_factor = 0.98
        
        # ANN's state arrays
        self.reward_state = np.zeros(self.gaussian_number * 2)
        self.position_state = np.zeros([self.gaussian_number * DOF])
        self.actual_state = np.zeros([self.gaussian_number * (DOF+2)])
        self.previous_state = np.zeros([self.gaussian_number * DOF])
        
        # weights
        self.actor_weights = np.zeros([ self.actor_n_output, self.n_input * (DOF+2)])
        self.critic_weights = np.zeros(self.n_input * (DOF+2))
        
        # noise
        self.actual_noise = np.zeros([self.actor_n_output, 1])
        self.previous_noise = np.zeros(self.actor_n_output)
        
        
        self.actor_output = np.zeros(3)
        self.ep3d = np.zeros(3)
    
        # critic output parameters
        self.surprise = np.zeros(1)
        self.actual_critic_output = np.zeros(self.critic_n_output)
        self.previous_critic_output = np.zeros(self.critic_n_output)
        
        self.needed_steps = np.zeros(self.n_trial)
               
    def spreading(self, w, pattern):
        return np.dot(w, pattern)

    def TDerror(self, actual_reward, actual_critic_output, previous_critic_output, discount_factor):
        x = discount_factor * actual_critic_output - previous_critic_output 
        return x + actual_reward 
        
    def act_training(self, a_eta, surprise, previous_state, previous_noise):    
        return a_eta * surprise * np.outer(previous_state, previous_noise)
        
    def crit_training(self, c_eta, surprise, previous_state):
        return c_eta * surprise * previous_state