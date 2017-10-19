# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 00:13:21 2017
@author: Alex
"""


import numpy as np
import matplotlib.pyplot as plt
import math_utils as utils
from basalGanglia import ActorCritic
from Wrist import Wrist
from mpl_toolkits.mplot3d import *
from maze import Maze8Arm 

if __name__ == "__main__":
    
    
    startPlotting = 75000
    # INIT OBJECTS1
    bg = ActorCritic()
    wrist = Wrist()
    maze = Maze8Arm()
    bg.init()
    wrist.init(bg.gaussN)
    maze.init()
    
    # START TRIAL
    for trial in xrange(bg.trialN):
        
        bg.actRew=0
        
        #◘ place reward
        wrist.reward_position = utils.placeReward(trial) 
        
        # temperature magnitude
        T = 1 * utils.clipped_exp(- trial / float(bg.trialN))
        
        maze.outside = 0   
        # INIT PLOTTING
        if trial == startPlotting:
            
            #♣ INIT plotting         
            fig1   = plt.figure("Workspace",figsize=(80,80), )
            gs = plt.GridSpec(1, 2, width_ratios=[3, 3]) 
             
            # MAZE COORDINATES
                       
            # north track
            line1 = plt.Line2D([4.5,4.5] , [6.2,9.5] , color = 'black')
            line2 = plt.Line2D([4.5,5.5] , [9.5,9.5] , color = 'black')
            line3 = plt.Line2D([5.5,5.5] , [6.2,9.5] , color = 'black')
            line4 = plt.Line2D([4.5,5.5] , [6.2,6.2] , color = 'silver')
            circle1 = plt.Circle((5 , 9), 0.5, color = 'springgreen') 
            edgecircle1 = plt.Circle((5 , 9), 0.5, color = 'black', fill = False)
            # north east track 
            line5 = plt.Line2D([5.5,8], [6.2,8.7] , color = 'black')
            line6 = plt.Line2D([8,8.7], [8.7,8] , color = 'black')
            line7 = plt.Line2D([8.7,6.2], [8,5.5] , color = 'black')
            line8 = plt.Line2D([5.5,6.2], [6.2,5.5] , color = 'silver')   
            circle2 = plt.Circle((8 , 8), 0.5, color = 'springgreen')
            edgecircle2 = plt.Circle((8 , 8), 0.5, color = 'black', fill = False)  
            # east track   
            line9 = plt.Line2D([6.20,9.5], [5.5,5.5] , color = 'black')
            line10 = plt.Line2D([9.5,9.5], [5.5,4.5] , color = 'black')
            line11 = plt.Line2D([9.5,6.2], [4.5,4.5] , color = 'black')
            line12 = plt.Line2D([6.2,6.2], [5.5,4.5] , color = 'silver')
            circle3 = plt.Circle((9 , 5), 0.5, color = 'springgreen')
            edgecircle3 = plt.Circle((9 , 5), 0.5, color = 'black', fill = False)
            # south east track 
            line13 = plt.Line2D([6.2 , 8.7] , [4.5,2] , color = 'black')
            line14 = plt.Line2D([8.7,8] , [ 2, 1.3] , color = 'black')
            line15 = plt.Line2D([8 , 5.5], [1.3 , 3.8] , color = 'black')
            line16 = plt.Line2D([5.5,6.2], [3.8,4.5] , color = 'silver')
            circle4 = plt.Circle((8 , 2), 0.5, color = 'springgreen')
            edgecircle4 = plt.Circle((8 , 2), 0.5, color = 'black', fill = False)
            # south track  
            line17 = plt.Line2D([5.5,5.5], [3.8,0.5] , color = 'black')
            line18 = plt.Line2D([5.5,4.5], [0.5,0.5] , color = 'black')
            line19 = plt.Line2D([4.5,4.5], [0.5,3.8] , color = 'black')
            line20 = plt.Line2D([4.5,5.5], [3.8,3.8] , color = 'silver')
            circle5 = plt.Circle((5 , 1), 0.5, color = 'springgreen')
            edgecircle5 = plt.Circle((5 , 1), 0.5, color = 'black', fill = False)
            # south west track
            line21 = plt.Line2D([4.5,2], [3.8,1.3] , color = 'black')
            line22 = plt.Line2D([2,1.3], [1.3,2] , color = 'black')
            line23 = plt.Line2D([1.3,3.8], [2.,4.5] , color = 'black')
            line24 = plt.Line2D([4.5,3.8], [3.8,4.5] , color = 'silver')
            circle6 = plt.Circle((2 , 2), 0.5, color = 'springgreen')
            edgecircle6 = plt.Circle((2 , 2), 0.5, color = 'black', fill = False)
            # west track 
            line25 = plt.Line2D([3.8,0.5], [4.5,4.5] , color = 'black')
            line26 = plt.Line2D([0.5,0.5], [4.5,5.5] , color = 'black')
            line27 = plt.Line2D([0.5,3.8], [5.5,5.5] , color = 'black')
            line28 = plt.Line2D([3.8,3.8], [4.5,5.5] , color = 'silver')
            circle7 = plt.Circle((1 , 5), 0.5, color = 'springgreen')
            edgecircle7 = plt.Circle((1 , 5), 0.5, color = 'black', fill = False)
            # NORTH WEST TRACK
            line29 = plt.Line2D([3.8,1.3], [5.5,8] , color = 'black')
            line30 = plt.Line2D([1.3, 2], [8,8.7] , color = 'black')
            line31 = plt.Line2D([2.,4.5], [8.7,6.2] , color = 'black')
            line32 = plt.Line2D([3.8,4.5], [5.5,6.2] , color = 'silver')
            circle8 = plt.Circle((2 , 8), 0.5, color = 'springgreen')
            edgecircle8 = plt.Circle((2 , 8), 0.5, color = 'black', fill = False)
            
            # INIT 2d PLOTTING
            ax1 = fig1.add_subplot(gs[0])
            ax1.set_xlim([0,10])
            ax1.set_ylim([0,10]) 
            # north track
            ax1.add_artist(line1)
            ax1.add_artist(line2)
            ax1.add_artist(line3)
            ax1.add_artist(line4)
            ax1.add_artist(circle1)
            ax1.add_artist(edgecircle1)
            #north east track
            ax1.add_artist(line5)
            ax1.add_artist(line6)
            ax1.add_artist(line7)
            ax1.add_artist(line8)
            ax1.add_artist(circle2)
            ax1.add_artist(edgecircle2)
            #east track
            ax1.add_artist(line9)
            ax1.add_artist(line10)
            ax1.add_artist(line11)
            ax1.add_artist(line12)
            ax1.add_artist(circle3)
            ax1.add_artist(edgecircle3)
            #south east track
            ax1.add_artist(line13)
            ax1.add_artist(line14)
            ax1.add_artist(line15)
            ax1.add_artist(line16)
            ax1.add_artist(circle4)
            ax1.add_artist(edgecircle4)
            #south track
            ax1.add_artist(line17)
            ax1.add_artist(line18)
            ax1.add_artist(line19)
            ax1.add_artist(line20)
            ax1.add_artist(circle5)
            ax1.add_artist(edgecircle5)
            #south west track
            ax1.add_artist(line21)
            ax1.add_artist(line22)
            ax1.add_artist(line23)
            ax1.add_artist(line24)
            ax1.add_artist(circle6)
            ax1.add_artist(edgecircle6)
            #west track 
            ax1.add_artist(line25)
            ax1.add_artist(line26)
            ax1.add_artist(line27)
            ax1.add_artist(line28)
            ax1.add_artist(circle7)
            ax1.add_artist(edgecircle7)
            #north west track
            ax1.add_artist(line29)
            ax1.add_artist(line30)
            ax1.add_artist(line31)
            ax1.add_artist(line32)
            ax1.add_artist(circle8)
            ax1.add_artist(edgecircle8)
            #plot agent & reward
            reward, = ax1.plot(wrist.reward_position[0] , wrist.reward_position[1], 'x', color='r')
            agent, = ax1.plot(wrist.actual_2dposition[0], wrist.actual_2dposition[1], 'o')
            
            
            # INIT 3D PLOT
            ax2  = fig1.add_subplot(gs[1], projection='3d')
            
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
            # plot agent
            agent3d, = ax2.plot([wrist.actual_3dposition[0]], [wrist.actual_3dposition[1]], [wrist.actual_3dposition[2]], 'o', color="blue")
            
            
        if trial > startPlotting:
             text1.set_text("trial = %s" % (trial))
             agent3d.remove()
             agent3d, = ax2.plot([wrist.actual_3dposition[0]], [wrist.actual_3dposition[1]], [wrist.actual_3dposition[2]], 'o', color="blue")
             reward.set_data(wrist.reward_position[0], wrist.reward_position[1])
             agent.set_data(wrist.actual_2dposition[0], wrist.actual_2dposition[1])
             plt.pause(0.1)
    
        wrist.reward_state = np.reshape(utils.gaussian(wrist.reward_position, bg.rewX, bg.std_dev * 10), bg.gaussN * 2) 
        
        
       
        # start movements
        for movement in xrange(bg.trialMov):
            
            
            
            # init first movement's parameters
            if movement == 0:
                
                wrist.actual_3dposition = np.array([0.5,0.5,0.5])
                wrist.previous_3dposition = wrist.actual_3dposition.copy()
                
                wrist.actual_2dposition = np.array( [5. , 5.])
                wrist.previous_2dposition = wrist.actual_2dposition.copy()
                
            if trial > startPlotting:
                agent.set_data(wrist.actual_2dposition[0], wrist.actual_2dposition[1])
                plt.pause(0.05)
                
            # compute actual state
            wrist.position3d_state = np.reshape(utils.gaussian(wrist.actual_3dposition, bg.posX, bg.std_dev), bg.gaussN * 3)
          #  wrist.position2d_state = np.reshape(utils.gaussian(wrist.actual_2dposition, bg.rewX, bg.std_dev * 10), bg.gaussN * 2)
            bg.actState = np.hstack([wrist.position3d_state , wrist.reward_state])
            
            # compute actor output
            bg.actOut = utils.sigmoid(bg.spreading(bg.actW, bg.actState))
                    
            # storage old noise and compute new noise
            bg.prvNoise = bg.actNoise.copy()
            bg.actNoise= utils.computate_noise(bg.prvNoise, bg.delT, bg.tau) * T
         #♠   bg.actual_noise = utils.Cut_range(bg.actNoise, -0.5 , 0.5)
            
            # compute 3D EPs
            bg.ep3D = utils.Cut_range(bg.actOut + bg.actNoise, 0., 1.) 
            
            if movement > 0:
                wrist.saveMov3d()
            
             
            # compute 3d movement 
            wrist.actual_3dposition = wrist.move3d(bg.ep3D)
            # conversion to 2d movement
            wrist.delta_2dposition = utils.conversion2d(utils.change_range(wrist.actual_3dposition , 0, 1, -1, 1))
                                                    
            # compute 2d final position 
            wrist.next_2dposition[0] = wrist.actual_2dposition[0] - wrist.delta_2dposition[0]       
            wrist.next_2dposition[1] = wrist.actual_2dposition[1] + wrist.delta_2dposition[1]
           # maze.verifyOutside(wrist.next_2dposition)
            wrist.next_2dposition = maze.imposeLBounds(wrist.actual_2dposition, wrist.next_2dposition)          
            wrist.next_2dposition = maze.imposeHbounds(wrist.actual_2dposition, wrist.next_2dposition)
       #     wrist.next_2dposition = utils.Cut_range(wrist.next_2dposition, 0., 10)
            
            wrist.actual_2dposition = wrist.next_2dposition.copy()
            
            if trial > startPlotting:   
                text2.set_text("movement = %s" % (movement))
                agent3d.remove()
                agent3d, = ax2.plot([wrist.actual_3dposition[0]], [wrist.actual_3dposition[1]], [wrist.actual_3dposition[2]], 'o', color="blue")
                agent.set_data(wrist.actual_2dposition[0], wrist.actual_2dposition[1])
                plt.pause(0.05)
                
            # storage old state 
            bg.prvState = bg.actState.copy()
            
            # compute actual state
            wrist.position3d_state = np.reshape(utils.gaussian(wrist.actual_3dposition, bg.posX, bg.std_dev), bg.gaussN* 3)
          #  wrist.position2d_state = np.reshape(utils.gaussian(wrist.actual_2dposition, bg.rewX, bg.std_dev * 10), bg.gaussN* 2)
            bg.actState = np.hstack([wrist.position3d_state,wrist.reward_state])
            
            if movement > 0:
                
                # compute reward distance
                distance = utils.Distance(wrist.actual_2dposition[0],wrist.reward_position[0],wrist.actual_2dposition[1],wrist.reward_position[1])
                
                
                if distance < 1.0:
             #       print "TRIAL" , trial, "REWARD-->movement" , movement
                    bg.actRew = 1                    
                    bg.trial200[trial%200] = movement
                    #   bg.rewMov[trial] = movement # storage min movements to get reward    
                
             #   elif maze.outside == 1:
              #      bg.actRew = -0.05
              #   #   print "****OUTSIDE****", movement 
              #      bg.trial200[trial%200] = bg.trialMov  
         #       else:
          #          bg.actRew = 0
                
                # compute critic output and surprise
                bg.prvCritOut = bg.actCritOut.copy()
                bg.actCritOut = bg.spreading(bg.critW,bg.actState)           
                bg.surp = bg.computeTdError()
                
              #  print bg.actRew
                # training
                bg.actW +=  bg.trainAct().T
                bg.critW +=  bg.trainCrit()
               
              #  print maze.outside
                if bg.actRew == 1:
                    break
             #   elif bg.actRew == -0.05:
              #      break
                
        if bg.actRew == 0:
          #  print "TRIAL" , trial, "---->TOO MANY MOVEMENTS"
   #         bg.rewMov[trial] = bg.trialMov
            bg.trial200[trial%200] = bg.trialMov
        
        if trial % 200 == 199:
            value200= np.sum(bg.trial200) / 200.
            print "avarage100", value200, "trial" , trial
            bg.avgMov[trial/200] = value200
        
    # PLOT TRIAL'S MOVEMENTS TO GET REWARD            
   # plt.figure(figsize=(120, 4), num=2, dpi=160)
   # plt.title('number of movement to get reward')
   # plt.xlim([0, bg.n_trial])
   # plt.ylim([0, np.max(bg.max_trial_movements)])
   # plt.xticks(np.arange(0,bg.n_trial, 100))
   # plt.plot(bg.needed_steps)
    
    plt.figure(figsize=(120, 4), num=3, dpi=160)
    plt.title('avarage 200')
    plt.xlim([0, bg.trialN/ 200])
    plt.ylim([0, np.max(bg.avgMov)])
    plt.xticks(np.arange(0,bg.trialN/200, 25))
    plt.plot(bg.avgMov)
    
#    np.savetxt("C:\Users\Alex\Desktop\motor_learning_control_model\data\actor_weights", (bg.actor_weights))
 #   np.savetxt("C:\Users\Alex\Desktop\motor_learning_control_model\data\critic_weights", (bg.critic_weights))
