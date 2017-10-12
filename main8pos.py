# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 00:13:21 2017
@author: Alex
"""

import numpy as np
import matplotlib.pyplot as plt
import math_utils as utils
from ActorCritic8pos import ActorCritic
from Wrist import Wrist
from mpl_toolkits.mplot3d import *

if __name__ == "__main__":
    
    bg = ActorCritic()
    wrist = Wrist()
    
    bg.init()
    wrist.init(bg.gaussian_number)
    
    startPlotting = 0
    # start trials
    for trial in xrange(bg.n_trial):
        
        #◘ place reward
        wrist.reward_position = utils.placeReward(trial) 
        
        # temperature magnitude
        T = 1 * utils.clipped_exp(- trial * 0.2 / float(bg.n_trial))
        
        if trial == startPlotting:
              #♣ INIT 2d plotting         
             fig1   = plt.figure("Workspace",figsize=(80,80))
             
             
             
             circle1 = plt.Circle((5 , 8), 0.5, color = 'yellow') 
             circle2 = plt.Circle((7 , 7), 0.5, color = 'yellow')
             circle3 = plt.Circle((8 , 5), 0.5, color = 'yellow')
             circle4 = plt.Circle((7 , 3), 0.5, color = 'yellow')
             circle5 = plt.Circle((5 , 2), 0.5, color = 'yellow')
             circle6 = plt.Circle((3 , 3), 0.5, color = 'yellow')
             circle7 = plt.Circle((2 , 5), 0.5, color = 'yellow')
             circle8 = plt.Circle((3 , 7), 0.5, color = 'yellow')
        
             ax1 = fig1.add_subplot(222)
             
             ax1.set_xlim([0,10])
             ax1.set_ylim([0,10])
             ax1.add_artist(circle1)
             ax1.add_artist(circle2)
             ax1.add_artist(circle3)
             ax1.add_artist(circle4)
             ax1.add_artist(circle5)
             ax1.add_artist(circle6)
             ax1.add_artist(circle7)
             ax1.add_artist(circle8)
             reward, = ax1.plot(wrist.reward_position[0] , wrist.reward_position[1], 'x', color='r')
             agent, = ax1.plot(wrist.actual_2dposition[0], wrist.actual_2dposition[1], 'o')
             
             ax2  = fig1.add_subplot(221, projection='3d')
             # set limits
             ax2.set_xlim([0,1])
             ax2.set_ylim([0,1])
             ax2.set_zlim([0,1])
             # set ticks
             ax2.set_xticks(np.arange(0, 1, 0.1))
             ax2.set_yticks(np.arange(0, 1, 0.1))
             ax2.set_zticks(np.arange(0, 1, 0.1))
             # add counters
             text1 = plt.figtext(.9, .1, "trial = %s" % (0), style='italic', bbox={'facecolor':'green'})
             text2 = plt.figtext(.1, .9, "movement = %s" % (0), style='italic', bbox={'facecolor':'red'})
             
             agent3d, = ax2.plot([wrist.actual_3dposition[0]], [wrist.actual_3dposition[1]], [wrist.actual_3dposition[2]], 'o', color="blue")
             
             
        if trial > startPlotting:
             text1.set_text("trial = %s" % (trial))
             agent3d.remove()
             agent3d, = ax2.plot([wrist.actual_3dposition[0]], [wrist.actual_3dposition[1]], [wrist.actual_3dposition[2]], 'o', color="blue")
             reward.set_data(wrist.reward_position[0], wrist.reward_position[1])
             agent.set_data(wrist.actual_2dposition[0], wrist.actual_2dposition[1])
             plt.pause(0.1)
    
        wrist.reward_state = np.reshape(utils.gaussian(wrist.reward_position, bg.X_rew, bg.std_dev * 10), bg.gaussian_number * 2) 
        
       
        # start movements
        for movement in xrange(bg.max_trial_movements):
            
            # init first movement's parameters
            if movement == 0:
                
                wrist.actual_3dposition = np.array([0.5,0.5,0.5])
                wrist.previous_3dposition = wrist.actual_3dposition.copy()
                
                wrist.actual_2dposition = np.array( [5. , 5.])
                wrist.previous_2dposition = wrist.actual_2dposition.copy()
                
            if trial > startPlotting:
                agent.set_data(wrist.actual_2dposition[0], wrist.actual_2dposition[1])
                plt.pause(0.1)
                
            # compute actual state
            wrist.position3d_state = np.reshape(utils.gaussian(wrist.actual_3dposition, bg.X_pos, bg.std_dev), bg.gaussian_number * 3)
            wrist.position2d_state = np.reshape(utils.gaussian(wrist.actual_2dposition, bg.X_rew, bg.std_dev * 10), bg.gaussian_number * 2)
            bg.actual_state = np.hstack([wrist.position3d_state, wrist.position2d_state, wrist.reward_state])
            
            # compute actor output
            bg.actor_output = utils.sigmoid(bg.spreading(bg.actor_weights, bg.actual_state))
            
            
            # storage old noise and compute new noise
            bg.previous_noise = bg.actual_noise.copy()
            bg.actual_noise = utils.computate_noise(bg.previous_noise, bg.delta_time, bg.tau) * T
          #  bg.actual_noise = utils.Cut_range(bg.actual_noise, -0.5 , 0.5)
            
            # compute 3D EPs
            bg.ep3d = utils.Cut_range(bg.actor_output + bg.actual_noise, 0., 1.) 
            
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
            wrist.actual_2dposition[1] = wrist.actual_2dposition[1] + wrist.delta_2dposition[1]
            wrist.actual_2dposition = utils.Cut_range(wrist.actual_2dposition, 0, 10.) 
            
            if trial > startPlotting:   
                text2.set_text("movement = %s" % (movement))
                agent3d.remove()
                agent3d, = ax2.plot([wrist.actual_3dposition[0]], [wrist.actual_3dposition[1]], [wrist.actual_3dposition[2]], 'o', color="blue")
                agent.set_data(wrist.actual_2dposition[0], wrist.actual_2dposition[1])
                plt.pause(0.1)
                
            # storage old state 
            bg.previous_state = bg.actual_state.copy()
            
            # compute actual state
            wrist.position3d_state = np.reshape(utils.gaussian(wrist.actual_3dposition, bg.X_pos, bg.std_dev), bg.gaussian_number * 3)
            wrist.position2d_state = np.reshape(utils.gaussian(wrist.actual_2dposition, bg.X_rew, bg.std_dev * 10), bg.gaussian_number * 2)
            bg.actual_state = np.hstack([wrist.position3d_state, wrist.position2d_state, wrist.reward_state])
            
            if movement > 0:
                
                # compute reward distance
                distance = utils.Distance(wrist.actual_2dposition[0],wrist.reward_position[0],wrist.actual_2dposition[1],wrist.reward_position[1])
                if distance < 0.6:
                    print "TRIAL" , trial, "REWARD-->movement" , movement
                    bg.actual_reward = 1
                    
                    bg.needed_steps[trial] = movement
        #        elif  np.max(wrist.actual_2dposition) >= 10:
               #     print "OUTSIDE"
        #            bg.actual_reward = -1
        #        elif  np.min(wrist.actual_2dposition) <= 0:
             #       print "OUTSIDE"
         #           bg.actual_reward = -1
                else:
                    bg.actual_reward = 0    
                
                # storage old critic output, compute new and get surprise
                bg.previous_critic_output = bg.actual_critic_output.copy()
                bg.actual_critic_output = bg.spreading(bg.critic_weights,bg.actual_state)
                bg.surprise = bg.TDerror(bg.actual_reward, bg.actual_critic_output, bg.previous_critic_output, bg.discount_factor)
          
                # weights adjustement
                bg.actor_weights +=  bg.act_training(bg.a_eta, bg.surprise, bg.previous_state, bg.previous_noise).T
                bg.critic_weights +=  bg.crit_training(bg.c_eta, bg.surprise, bg.previous_state)
                # storage min movements to get reward
                if bg.actual_reward == 1:
                    break
        #        if bg.actual_reward == -1:
         #           bg.needed_steps[trial] = bg.max_trial_movements
          #          break
                
        if bg.actual_reward == 0:
            print "TRIAL" , trial, "---->TOO MANY MOVEMENTS"
            bg.needed_steps[trial] = bg.max_trial_movements      
        bg.trial5[trial%5] = movement
        if trial % 5 == 4:
            value5= np.sum(bg.trial5) / 5
            bg.avaragemovements[trial/5] = value5
        
   # # PLOT TRIAL'S MOVEMENTS TO GET REWARD            
    plt.figure(figsize=(120, 4), num=4, dpi=80)
    plt.title('number of movement to get reward')
    plt.xlim([0, bg.n_trial])
    plt.ylim([0, np.max(bg.avaragemovements)])
    plt.xticks(np.arange(0,bg.n_trial, 100))
    plt.plot(bg.needed_steps)
    
#    np.savetxt("C:\Users\Alex\Desktop\motor_learning_control_model\data\actor_weights", (bg.actor_weights))
 #   np.savetxt("C:\Users\Alex\Desktop\motor_learning_control_model\data\critic_weights", (bg.critic_weights))