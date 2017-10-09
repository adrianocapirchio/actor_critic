# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 00:13:21 2017

@author: Alex
"""

import numpy as np
import matplotlib.pyplot as plt
import math_utils as utils
from ActorCritic import ActorCritic
from Wrist import Wrist

if __name__ == "__main__":
    
    #♣ INIT 2d plotting         
    fig1   = plt.figure("Workspace",figsize=(50,50))
    ax1    = fig1.add_subplot(111)
    ax1.set_xlim([-1,1])
    ax1.set_ylim([-1,1])
    
    
    
    bg = ActorCritic()
    wrist = Wrist()
    
    bg.init()
    wrist.init()
    
    # start trials
    for trial in xrange(bg.n_trial):
        
        # temperature magnitude
        T = 1 * utils.clipped_exp(- trial * 0.2 / float(bg.n_trial))
        
        if trial == 0:
            agent, = ax1.plot(wrist.agent_starting_2dposition[0], wrist.agent_starting_2dposition[1], 'o')
            
        if trial > 0:
            agent.remove()
            agent, = ax1.plot(wrist.agent_starting_2dposition[0], wrist.agent_starting_2dposition[1], 'o')
        #    plt.pause(0.1)
        
        #◘ place reward
         
        reward_position = utils.placeReward(trial) 
       # print reward_position
        reward, = ax1.plot([reward_position[0]] , [reward_position[1]], "x") 
        bg.reward_state = np.reshape(utils.gaussian(reward_position, bg.X_rew, bg.std_dev), (20))
        
        #  print reward_position 
        # start movements
        for movement in xrange(bg.max_trial_movements):
            
            # init first movement's parameters
            if movement == 0:
                
                wrist.actual_3dposition = np.array([0.5,0.5,0.5])
                wrist.previous_3dposition = wrist.actual_3dposition.copy()
                
                wrist.actual_2dposition = np.array( [0. , 0.])
                wrist.previous_2dposition = wrist.actual_2dposition.copy()
                
            # compute actual state
            bg.position_state = np.reshape(utils.gaussian(wrist.actual_3dposition, bg.X_pos, bg.std_dev), (30))
            bg.actual_state = np.hstack([bg.position_state, bg.reward_state])
            
            # compute actor output
            bg.actor_output = utils.sigmoid(bg.spreading(bg.actor_weights, bg.actual_state))
            
            # storage old noise and compute new noise
            bg.previous_noise = bg.actual_noise.copy()
            bg.actual_noise = np.squeeze(utils.computate_noise(bg.previous_noise, bg.delta_time, bg.tau) * T) 
            
            # compute 3D EPs
            bg.ep3d = utils.Cut_range(bg.actor_output + bg.actual_noise, 0, 1) 
            
            if movement > 0:
                wrist.previous_3derror = wrist.actual_3derror.copy()
                wrist.previous_3dposition = wrist.actual_3dposition.copy() 
                wrist.previous_3dvelocity = wrist.actual_3dvelocity.copy()
                wrist.previous_3dacceleration = wrist.actual_3dacceleration.copy()
            
            # compute 3d movement 
            wrist.actual_3dposition = wrist.move3d(bg.ep3d)
            
            # conversion to 2d movement
            wrist.delta_2dposition = utils.conversion2d(utils.change_range(wrist.actual_3dposition , 0, 1, -1, 1)) 
                                                    
            # compute 2d final position 
            wrist.actual_2dposition[0] = wrist.actual_2dposition[0] - wrist.delta_2dposition[0]
     #       wrist.next_2dposition[0] = utils.Cut_range(wrist.next_2dposition[0], -0.99999, 0.99999)       
            wrist.actual_2dposition[1] = wrist.actual_2dposition[1] + wrist.delta_2dposition[1]
            wrist.actual_2dposition = utils.Cut_range(wrist.actual_2dposition, -0.99999, 0.99999) 
            agent.set_data(wrist.actual_2dposition[0], wrist.actual_2dposition[1])
          #  plt.pause(0.1)
                
            # storage old state 
            bg.previous_state = bg.actual_state.copy()
            
            # compute actual state
            bg.position_state = np.reshape(utils.gaussian(wrist.actual_3dposition, bg.X_pos, bg.std_dev), (30))
            bg.actual_state = np.hstack([bg.position_state, bg.reward_state])
            
            if movement > 0:
                
                # compute reward distance
                distance = utils.Distance(wrist.actual_2dposition[0],reward_position[0],wrist.actual_2dposition[1],reward_position[1])

                # get the reward
                if distance < 0.1:
                    print "presa"
                    print movement
                    actual_reward = 1    
                else:
                    actual_reward = 0
                    
                # storage old critic output, compute new and get surprise
                bg.previous_critic_output = bg.actual_critic_output.copy()
                bg.actual_critic_output = bg.spreading(bg.critic_weights,bg.actual_state)
                bg.surprise = bg.TDerror(actual_reward, bg.actual_critic_output, bg.previous_critic_output, bg.discount_factor)
                
                # weights adjustement
                bg.actor_weights +=  bg.act_training(bg.a_eta, bg.surprise, bg.previous_state, bg.previous_noise).T
                bg.critic_weights +=  bg.crit_training(bg.c_eta, bg.surprise, bg.previous_state)
                
                # storage min movements to get reward
                bg.needed_steps[trial] = movement
                               
                               
                # end trial if agent got reward                   
                if actual_reward == 1:
                    break 
        
        if actual_reward == 0:
            print "******fail******"
            
   # # PLOT TRIAL'S MOVEMENTS TO GET REWARD            
    plt.figure(figsize=(80, 4), num=4, dpi=80)
    plt.title('number of movement to get reward')
    plt.xlim([0, bg.n_trial])
    plt.ylim([0, bg.max_trial_movements])
    plt.xticks(np.arange(0,bg.n_trial, 100))
    plt.plot(bg.needed_steps)
    
#    np.savetxt("C:\Users\Alex\Desktop\motor_learning_control_model\data\actor_weights", (bg.actor_weights))
 #   np.savetxt("C:\Users\Alex\Desktop\motor_learning_control_model\data\critic_weights", (bg.critic_weights))