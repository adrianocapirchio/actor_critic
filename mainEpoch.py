# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:18:32 2017

@author: Alex
"""

import numpy as np
import matplotlib.pyplot as plt
import math_utils as utils
from basalGanglia import ActorCritic
from cerebellum import Cerebellum 
from Wrist import Wrist
from mpl_toolkits.mplot3d import *
from maze import Maze8Arm 
import os

# NET OPTIONS
training = True
multiNet = False
SL = True
cerebellum = True

GANGLIA_NOISE = True
CEREBELLUM_NOISE = True

### INPUTS
VISION = True
GOAL_VISION= True
AGENT_VISION = True
WRIST_PROPRIOCEPTION = True

calibration = 1.
loadData = False
saveData = True

randomGoal = False
returnGoal = False

maxEpoch = 500
maxTrial = 8
maxStep = 100

ratio = 5

startPlotting = 500

if __name__ == "__main__":
    
    # create objects
    bg = ActorCritic()
    wrist = Wrist()
    maze = Maze8Arm()
    
    #init objects
    wrist.init()
    maze.init(returnGoal)
    bg.init(VISION ,AGENT_VISION, GOAL_VISION, WRIST_PROPRIOCEPTION,  multiNet, maxStep, wrist.wristRange)
    
    if SL == True:
        cb = Cerebellum()
        cb.init(multiNet, bg.currState)
    
    avg6EpochAcc = np.zeros(6)
    avg6EpochTime = np.ones(6) * (maxStep /10)
    finalAvgAcc = np.zeros(maxEpoch/6)
    finalAvgTime = np.ones(maxEpoch/6) * (maxStep /10)
    epochAccurancy = np.zeros(len(maze.rewPosList))
    epochGoalTime = np.ones(len(maze.rewPosList)) * (maxStep /10)
    epochAvgAcc = np.zeros(1)
    epochAvgTime = np.zeros(1)
    
    # load saved weights
    if loadData == True:
        
        mydir = os.getcwd
        os.chdir("C:\Users\Alex\Desktop\motorLearningSistem\data")
        
        if multiNet == True:
            
            if SL == True:
                
                bg.actW0 = np.loadtxt('bg.actW0cereb.txt')
                bg.actW1 = np.loadtxt('bg.actW1cereb.txt')
                bg.actW2 = np.loadtxt('bg.actW2cereb.txt')
                bg.actW3 = np.loadtxt('bg.actW3cereb.txt')
                bg.actW4 = np.loadtxt('bg.actW4cereb.txt')
                bg.actW5 = np.loadtxt('bg.actW5cereb.txt')
                bg.actW6 = np.loadtxt('bg.actW6cereb.txt')
                bg.actW7 = np.loadtxt('bg.actW7cereb.txt')
            
                bg.critW0 = np.loadtxt('bg.critW0cereb.txt')
                bg.critW1 = np.loadtxt('bg.critW1cereb.txt')
                bg.critW2 = np.loadtxt('bg.critW2cereb.txt')
                bg.critW3 = np.loadtxt('bg.critW3cereb.txt')
                bg.critW4 = np.loadtxt('bg.critW4cereb.txt')
                bg.critW5 = np.loadtxt('bg.critW5cereb.txt')
                bg.critW6 = np.loadtxt('bg.critW6cereb.txt')
                bg.critW7 = np.loadtxt('bg.critW7cereb.txt')
                
                cb.w0 = np.loadtxt('cb.w0.txt')
                cb.w1 = np.loadtxt('cb.w1.txt')
                cb.w2 = np.loadtxt('cb.w2.txt')
                cb.w3 = np.loadtxt('cb.w3.txt')
                cb.w4 = np.loadtxt('cb.w4.txt')
                cb.w5 = np.loadtxt('cb.w5.txt')
                cb.w6 = np.loadtxt('cb.w6.txt')
                cb.w7 = np.loadtxt('cb.w7.txt')
                
            else:
                bg.actW0 = np.loadtxt('bg.actW0.txt')
                bg.actW1 = np.loadtxt('bg.actW1.txt')
                bg.actW2 = np.loadtxt('bg.actW2.txt')
                bg.actW3 = np.loadtxt('bg.actW3.txt')
                bg.actW4 = np.loadtxt('bg.actW4.txt')
                bg.actW5 = np.loadtxt('bg.actW5.txt')
                bg.actW6 = np.loadtxt('bg.actW6.txt')
                bg.actW7 = np.loadtxt('bg.actW7.txt')
            
                bg.critW0 = np.loadtxt('bg.critW0.txt')
                bg.critW1 = np.loadtxt('bg.critW1.txt')
                bg.critW2 = np.loadtxt('bg.critW2.txt')
                bg.critW3 = np.loadtxt('bg.critW3.txt')
                bg.critW4 = np.loadtxt('bg.critW4.txt')
                bg.critW5 = np.loadtxt('bg.critW5.txt')
                bg.critW6 = np.loadtxt('bg.critW6.txt')
                bg.critW7 = np.loadtxt('bg.critW7.txt')
        
        else:    
            if SL == True:
                bg.actW = np.loadtxt('bg.actWcereb.txt')
                bg.critW = np.loadtxt('bg.critWcereb.txt')
                cb.w = np.loadtxt('cb.w.txt')
                
            else:
                bg.actW = np.loadtxt('bg.actW.txt')
                bg.critW = np.loadtxt('bg.critW.txt')
                
    # START TRIALS
    for epoch in xrange(maxEpoch):
        
     #   bg.actRew = 0
        
     #   print " NEW EPOCH ************************************", epoch
     
        bg.actRew *= 0
        bg.surp *= 0 
        bg.prvState *= 0
        bg.currState *= 0
        
        bg.stateBuff *= 0
        bg.desOutBuff *= 0
        bg.surpBuff *= 0 
          
        bg.prvCritOut *= 0
        bg.currCritOut *= 0
        
        bg.prvNoise *= 0               
        bg.currActOut = np.array([0.5,0.5,0.5]).copy()        
                
        if cerebellum == True:
            cb.prvNoise *= 0
            cb.currOut = np.array([0.5,0.5,0.5]).copy()
        
        epochAccurancy = np.zeros(len(maze.rewPosList))
        epochGoalTime = np.ones(len(maze.rewPosList)) * (maxStep /10)
        
       
        
        maze.curr2DPos = np.array([5. , 5.]).copy()
                    
        wrist.curr3DPos = np.array([0.5,0.5,0.5]).copy() 
        wrist.prv3DPos = np.array([0.5,0.5,0.5]).copy()
        wrist.ep = np.array([0.5,0.5,0.5]).copy()
        wrist.prvEp = np.array([0.5,0.5,0.5]).copy()
        wrist.curr3DErr *= 0
        wrist.force *= 0
        wrist.curr3DAcc *= 0
        wrist.curr3DVel *= 0
        wrist.prv3DVel *= 0
        wrist.prv3DErr *= 0
        
        
        # temperature magnitude
        if epoch < 10000:    
            T = 1. * utils.clipped_exp(- epoch / float(maxEpoch))
            
        # INIT PLOTTING
        if epoch == startPlotting:
            
            #♣ INIT plotting         
            fig1   = plt.figure("Workspace",figsize=(80,80), dpi=120)
            gs = plt.GridSpec(1, 2, width_ratios=[3, 3]) 
             
            # MAZE COORDINATES                   
            # north track
            line1 = plt.Line2D([4.5,4.5] , [6.2,9.5] , color = 'black')
            line2 = plt.Line2D([4.5,5.5] , [9.5,9.5] , color = 'black')
            line3 = plt.Line2D([5.5,5.5] , [6.2,9.5] , color = 'black')
            line4 = plt.Line2D([4.5,5.5] , [6.2,6.2] , color = 'silver')
            circle1 = plt.Circle((5 , 9.24), 0.7, color = 'springgreen') 
            edgecircle1 = plt.Circle((5 , 9.24), 0.7, color = 'black', fill = False)
            # north east track 
            line5 = plt.Line2D([5.5,8], [6.2,8.7] , color = 'black')
            line6 = plt.Line2D([8,8.7], [8.7,8] , color = 'black')
            line7 = plt.Line2D([8.7,6.2], [8,5.5] , color = 'black')
            line8 = plt.Line2D([5.5,6.2], [6.2,5.5] , color = 'silver')   
            circle2 = plt.Circle((8 , 8), 0.7, color = 'springgreen')
            edgecircle2 = plt.Circle((8 , 8), 0.7, color = 'black', fill = False)  
            # east track   
            line9 = plt.Line2D([6.20,9.5], [5.5,5.5] , color = 'black')
            line10 = plt.Line2D([9.5,9.5], [5.5,4.5] , color = 'black')
            line11 = plt.Line2D([9.5,6.2], [4.5,4.5] , color = 'black')
            line12 = plt.Line2D([6.2,6.2], [5.5,4.5] , color = 'silver')
            circle3 = plt.Circle((9.24 , 5), 0.7, color = 'springgreen')
            edgecircle3 = plt.Circle((9.24 , 5), 0.7, color = 'black', fill = False)
            # south east track 
            line13 = plt.Line2D([6.2 , 8.7] , [4.5,2] , color = 'black')
            line14 = plt.Line2D([8.7,8] , [ 2, 1.3] , color = 'black')
            line15 = plt.Line2D([8 , 5.5], [1.3 , 3.8] , color = 'black')
            line16 = plt.Line2D([5.5,6.2], [3.8,4.5] , color = 'silver')
            circle4 = plt.Circle((8 , 2), 0.7, color = 'springgreen')
            edgecircle4 = plt.Circle((8 , 2), 0.7, color = 'black', fill = False)
            # south track  
            line17 = plt.Line2D([5.5,5.5], [3.8,0.5] , color = 'black')
            line18 = plt.Line2D([5.5,4.5], [0.5,0.5] , color = 'black')
            line19 = plt.Line2D([4.5,4.5], [0.5,3.8] , color = 'black')
            line20 = plt.Line2D([4.5,5.5], [3.8,3.8] , color = 'silver')
            circle5 = plt.Circle((5 , 0.76), 0.7, color = 'springgreen')
            edgecircle5 = plt.Circle((5 , 0.76), 0.7, color = 'black', fill = False)
            # south west track
            line21 = plt.Line2D([4.5,2], [3.8,1.3] , color = 'black')
            line22 = plt.Line2D([2,1.3], [1.3,2] , color = 'black')
            line23 = plt.Line2D([1.3,3.8], [2.,4.5] , color = 'black')
            line24 = plt.Line2D([4.5,3.8], [3.8,4.5] , color = 'silver')
            circle6 = plt.Circle((2 , 2), 0.7, color = 'springgreen')
            edgecircle6 = plt.Circle((2 , 2), 0.7, color = 'black', fill = False)
            # west track 
            line25 = plt.Line2D([3.8,0.5], [4.5,4.5] , color = 'black')
            line26 = plt.Line2D([0.5,0.5], [4.5,5.5] , color = 'black')
            line27 = plt.Line2D([0.5,3.8], [5.5,5.5] , color = 'black')
            line28 = plt.Line2D([3.8,3.8], [4.5,5.5] , color = 'silver')
            circle7 = plt.Circle((0.76 , 5), 0.7, color = 'springgreen')
            edgecircle7 = plt.Circle((0.76 , 5), 0.7, color = 'black', fill = False)
            # NORTH WEST TRACK
            line29 = plt.Line2D([3.8,1.3], [5.5,8] , color = 'black')
            line30 = plt.Line2D([1.3, 2], [8,8.7] , color = 'black')
            line31 = plt.Line2D([2.,4.5], [8.7,6.2] , color = 'black')
            line32 = plt.Line2D([3.8,4.5], [5.5,6.2] , color = 'silver')
            circle8 = plt.Circle((2 , 8), 0.7, color = 'springgreen')
            edgecircle8 = plt.Circle((2 , 8), 0.7, color = 'black', fill = False)
            
            circle9 = plt.Circle((5 , 5), 0.7, color = 'orange')
            # INIT 2d PLOTTING
            ax1 = fig1.add_subplot(gs[0])
            ax1.set_xlim([0,10])
            ax1.set_ylim([0,10]) 
            # north track
            ax1.add_artist(line1)
            ax1.add_artist(line2)
            ax1.add_artist(line3)
      #     ax1.add_artist(line4)
            ax1.add_artist(circle1)
            ax1.add_artist(edgecircle1)
            #north east track
            ax1.add_artist(line5)
            ax1.add_artist(line6)
            ax1.add_artist(line7)
       #     ax1.add_artist(line8)
            ax1.add_artist(circle2)
            ax1.add_artist(edgecircle2)
            #east track
            ax1.add_artist(line9)
            ax1.add_artist(line10)
            ax1.add_artist(line11)
        #    ax1.add_artist(line12)
            ax1.add_artist(circle3)
            ax1.add_artist(edgecircle3)
            #south east track
            ax1.add_artist(line13)
            ax1.add_artist(line14)
            ax1.add_artist(line15)
         #   ax1.add_artist(line16)
            ax1.add_artist(circle4)
            ax1.add_artist(edgecircle4)
            #south track
            ax1.add_artist(line17)
            ax1.add_artist(line18)
            ax1.add_artist(line19)
          #  ax1.add_artist(line20)
            ax1.add_artist(circle5)
            ax1.add_artist(edgecircle5)
            #south west track
            ax1.add_artist(line21)
            ax1.add_artist(line22)
            ax1.add_artist(line23)
          #  ax1.add_artist(line24)
            ax1.add_artist(circle6)
            ax1.add_artist(edgecircle6)
            #west track 
            ax1.add_artist(line25)
            ax1.add_artist(line26)
            ax1.add_artist(line27)
          #  ax1.add_artist(line28)
            ax1.add_artist(circle7)
            ax1.add_artist(edgecircle7)
            #north west track
            ax1.add_artist(line29)
            ax1.add_artist(line30)
            ax1.add_artist(line31)
           # ax1.add_artist(line32)
            ax1.add_artist(circle8)
            ax1.add_artist(edgecircle8)
            
            ax1.add_artist(circle9)
            #plot agent & reward
            reward, = ax1.plot(maze.rewPos[0] , maze.rewPos[1], 'x', color='r')
            agent, = ax1.plot(maze.curr2DPos[0], maze.curr2DPos[1], 'o')
            
            
            # INIT 3D PLOT
            ax2  = fig1.add_subplot(gs[1], projection='3d')
            # set limits
            ax2.set_xlim([0,1])
            ax2.set_ylim([0,1])
            ax2.set_zlim([0,1])
            # set ticks
            ax2.set_xticks(np.arange(0, 1, 0.1))
            ax2.set_yticks(np.arange(0, 1, 0.1), )
            ax2.set_zticks(np.arange(0, 1, 0.1),)
            
            
            ax2.set_xlabel("FE")
            ax2.set_ylabel("RUD")
            ax2.set_zlabel("PS")
            
            
            # add counters
            text1 = plt.figtext(.9, .1, "trial = %s" % (0), style='italic', bbox={'facecolor':'green'})
            text2 = plt.figtext(.1, .9, "step = %s" % (0), style='italic', bbox={'facecolor':'red'})
            text3 = plt.figtext(.1, .1, "epoch = %s" % (0), style='italic', bbox={'facecolor':'yellow'})
            # plot agent
            X = [0.5,0.5,0.5] 
            Y = [0.0,0.5,0.5]
            Z = [0.5,0.5,0.5]
            agent3d, = ax2.plot3D(X, Y, Z, color="blue")
            
        if epoch > startPlotting:                
            text3.set_text("epoch = %s" % (epoch+1))
            plt.pause(0.01)
            
        
        for trial in xrange(len(maze.rewPosList)):
        #    print "NEW TRIAL", trial
        #    bg.actRew *= 0
        #    bg.surp *= 0 
        #    bg.prvState *= 0
        #    bg.currState *= 0
        #    bg.stateBuff *= 0
        #    bg.desOutBuff *= 0
        #    bg.surpBuff *= 0           
        #    bg.prvCritOut *= 0
        #    bg.currCritOut *= 0
        #    bg.prvNoise *= 0               
        #    bg.currActOut = np.array([0.5,0.5,0.5]).copy()        
        #        
        #    if cerebellum == True:
        #        cb.prvNoise *= 0
       #         cb.currOut = np.array([0.5,0.5,0.5]).copy()
                
       #     maze.curr2DPos = np.array([5. , 5.]).copy()
                    
       #     wrist.curr3DPos = np.array([0.5,0.5,0.5]).copy() 
        #    wrist.prv3DPos = np.array([0.5,0.5,0.5]).copy()
         #   wrist.ep = np.array([0.5,0.5,0.5]).copy()
         #   wrist.prvEp = np.array([0.5,0.5,0.5]).copy()
         #   wrist.curr3DErr *= 0
         #   wrist.force *= 0
         #   wrist.curr3DAcc *= 0
         #   wrist.curr3DVel *= 0
         #   wrist.prv3DVel *= 0
         #   wrist.prv3DErr *= 0
            
            if returnGoal == False:   
                maze.setGoal(trial)
            else:
                maze.setReturnGoal(trial)
      #      print "EPOCH", epoch, "trial", trial, maze.rewPos
            
            if epoch > startPlotting:
                text1.set_text("trial = %s" % (trial+1))
                reward.set_data(maze.rewPos[0], maze.rewPos[1])
                plt.pause(0.01)
    
        
            if multiNet == True:
            
                bg.critW *= 0
                bg.actW *= 0
                if SL == True:
                    cb.w *= 0
        
                if maze.rewPos == maze.rewList[0]:
        #        print 1
                    bg.critW = bg.critW0.copy()
                    bg.actW = bg.actW0.copy()
                    if SL == True:
                        cb.w = cb.w0.copy()
                elif maze.rewPos == maze.rewList[1]:
       #         print 2
                    bg.critW = bg.critW1.copy()
                    bg.actW = bg.actW1.copy()
                    if SL == True:
                        cb.w = cb.w1.copy()
                elif maze.rewPos == maze.rewList[2]:
        #        print 3
                    bg.critW = bg.critW2.copy()
                    bg.actW = bg.actW2.copy()
                    if SL == True:
                        cb.w = cb.w2.copy()
                elif maze.rewPos == maze.rewList[3]:
        #        print 4
                    bg.critW = bg.critW3.copy()
                    bg.actW = bg.actW3.copy()
                    if SL == True:
                        cb.w = cb.w3.copy()
                elif maze.rewPos == maze.rewList[4]:
         #       print 5
                    bg.critW = bg.critW4.copy()
                    bg.actW = bg.actW4.copy()
                    if SL == True:
                        cb.w = cb.w4.copy()
                elif maze.rewPos == maze.rewList[5]:
         #       print 6
                    bg.critW = bg.critW5.copy()
                    bg.actW = bg.actW5.copy()
                    if SL == True:
                        cb.w = cb.w5.copy()
                elif maze.rewPos == maze.rewList[6]:
         #       print 7
                    bg.critW = bg.critW6.copy()
                    bg.actW = bg.actW6.copy()
                    if SL == True:
                        cb.w = cb.w6.copy()
                elif maze.rewPos == maze.rewList[7]:
         #       print 8
                    bg.critW = bg.critW7.copy()
                    bg.actW = bg.actW7.copy()
                    if SL == True:
                        cb.w = cb.w7.copy()
                elif maze.rewPos == maze.rewList[8]:
         #       print 8
                    bg.critW = bg.critW8.copy()
                    bg.actW = bg.actW8.copy()
                    if SL == True:
                        cb.w = cb.w8.copy()
            
            if VISION == True:
                if GOAL_VISION == True:
                    bg.compGoalVisionState(maze.rewPos)
            #    print 1 , maze.rewPos , bg.goalVisionState, "*****************************"
                    if AGENT_VISION == True:
                        bg.visionState = np.hstack([bg.agentVisionState, bg.goalVisionState]).copy()
                    else:
                        bg.visionState = bg.goalVisionState.copy()
                        
            # start steps
            for step in xrange(maxStep):
                
                if step == 0:
                    bg.actRew *= 0
                    
                    bg.stateBuff *= 0
                    bg.desOutBuff *= 0
                    bg.surpBuff *= 0     
                    
                    if returnGoal == False:
                        
                        bg.surp *= 0 
                    
                        
                    
                        bg.prvCritOut *= 0
                        bg.currCritOut *= 0
                        bg.prvState *= 0
                        bg.prvNoise *= 0 
                        bg.currState *= 0
                                  
                        bg.currActOut = np.array([0.5,0.5,0.5]).copy()
                        wrist.ep = np.array([0.5,0.5,0.5]).copy()
                        wrist.prvEp = np.array([0.5,0.5,0.5]).copy()
                        
                        if cerebellum == True:
                       # cb.prvNoise *= 0
                            cb.currOut = np.array([0.5,0.5,0.5]).copy()
                        
                        maze.curr2DPos = np.array([5. , 5.]).copy()
                        
                        wrist.curr3DPos = np.array([0.5,0.5,0.5]).copy() 
                        wrist.prv3DPos = np.array([0.5,0.5,0.5]).copy()
                        wrist.curr3DErr *= 0
                        wrist.force *= 0
                        wrist.curr3DAcc *= 0
                        wrist.curr3DVel *= 0
                        wrist.prv3DVel *= 0
                        wrist.prv3DErr *= 0
                        
                        
                        
                        
                        
                if step > 0:
                # save old values 
          #      maze.posBuff[:,step] = maze.curr2DPos.copy()
                
                    wrist.prvEp = wrist.ep.copy()
                    bg.prvState = bg.currState.copy()
                    bg.prvCritOut = bg.currCritOut.copy()
                    bg.prvActOut = bg.currActOut.copy()
                    bg.prvNoise = bg.currNoise.copy()
                
                    if SL == True:
                        if cerebellum == True:
                            bg.desOutBuff[:,step] = wrist.prvEp.copy()  
                            bg.stateBuff[:,step] = bg.prvState.copy()
                
                    
                    
                    wrist.saveMov3d()
                
                wrist.move3d()
            
               # compute 2d final position 
                maze.delta2DPos = (utils.conversion2d(utils.change_range(wrist.curr3DPos , 0, 1, -1, 1))).copy()* calibration# conversion to 2d step
                maze.next2DPos[0] = (maze.curr2DPos[0] - maze.delta2DPos[0]).copy()       
                maze.next2DPos[1] = (maze.curr2DPos[1] + maze.delta2DPos[1]).copy()    
                maze.imposeOutBounds()          
                maze.imposeInBounds()
                maze.curr2DPos = maze.next2DPos.copy()
                
                if epoch > startPlotting:   
                    text2.set_text("step = %s" % (step+1))
                    agent.set_data(maze.curr2DPos[0], maze.curr2DPos[1])
                    X = [0.5,0.5,wrist.curr3DPos[0].copy()] 
                    Y = [0.0,0.5,wrist.curr3DPos[1].copy()]
                    Z = [0.5,0.5,wrist.curr3DPos[2].copy()]        #    ax2.plot_wireframe(X,Y,Z)
                    agent3d.remove()
                    agent3d, = ax2.plot3D(X, Y, Z, color="blue")
                    plt.pause(0.01)  
                    
                if WRIST_PROPRIOCEPTION == True:
                    bg.compWristState(wrist.curr3DPos)

                if VISION == True:
                    if AGENT_VISION == True:
                        bg.compAgentVisionState(maze.curr2DPos)
                        if GOAL_VISION == True:
                            bg.visionState = np.hstack([bg.agentVisionState, bg.goalVisionState]).copy()
                        else:
                            bg.visionState = bg.agentVisionState.copy()
                    else:
                        bg.visionState = bg.goalVisionState.copy()
            
                if WRIST_PROPRIOCEPTION == True:
                    if VISION == True:
                        bg.currState = np.hstack([bg.wristState, bg.visionState]).copy()        
                    else:
                        bg.currState = bg.wristState.copy()
                else:
                    if VISION == True:
                        bg.currState = bg.visionState.copy()
                        
                maze.computeDistance()
                
                if step > 0:
                    
                    if maze.rewPos == maze.rewList[1]:
                        if maze.distance < 1.:
                            bg.actRew = 1                     
                            bg.currCritOut *= 0
                        else:    
                            bg.spreadCrit()
                    else:
                        if maze.distance < 0.7:
                            bg.actRew = 1                     
                            bg.currCritOut *= 0
                        else:    
                            bg.spreadCrit()
                    bg.compSurprise()
                    
                if bg.currCritOut > 2.:
                    print trial, step, bg.currCritOut
                    
                if SL == True:
                    if step % ratio == 0:
                        bg.spreadAct()
                        if GANGLIA_NOISE== True:
                            bg.computate_noise(T)   
                          #  print "ganglia turn" , bg.currActOut
                        wrist.ep = (utils.Cut_range(bg.currActOut + bg.currNoise, 0., 1.)).copy()                  
                    else:                    
                        if cerebellum == True:
                            cb.spreading(bg.currState)
                            bg.spreadAct()
                            if CEREBELLUM_NOISE== True:
                                cb.computate_noise()
                                wrist.ep = (utils.Cut_range(cb.currOut + bg.currNoise + cb.currNoise * T, 0., 1.)).copy()
                            else:
                                wrist.ep = (utils.Cut_range(cb.currOut + bg.currNoise, 0., 1.)).copy()               
                        else:
                            wrist.ep = (utils.Cut_range(bg.currActOut + bg.currNoise, 0., 1.)).copy()             
                else:
                    bg.spreadAct()
                    if GANGLIA_NOISE == True:
                        bg.computate_noise(T)
                    wrist.ep = (utils.Cut_range(bg.currActOut + bg.currNoise, 0., 1.)).copy()
                
                if training == True:
                    if step > 0:                       
                        bg.trainCrit() 
                        bg.trainAct()
                        if SL == True:
                            if cerebellum == True:     
                                if step %ratio == 1:
                                    bg.trainAct()
                                else:
                                    bg.trainAct2(wrist.prvEp)
                           
                                if bg.actRew == 1:
                                    for i in xrange(step+1): 
                                        cb.trainCb(bg.stateBuff[:,i], bg.desOutBuff[:,i])
                            else:
                                bg.trainAct()
                                        
                    if multiNet == True:
                        if maze.rewPos == maze.rewList[0]:
                            bg.critW0 = bg.critW.copy()
                            bg.actW0 = bg.actW.copy()
                            if SL == True:
                                cb.w0 = cb.w.copy()
                        elif maze.rewPos == maze.rewList[1]:
                            bg.critW1 = bg.critW.copy()
                            bg.actW1 = bg.actW.copy()
                            if SL == True:
                                cb.w1 = cb.w.copy()
                        elif maze.rewPos== maze.rewList[2]:
                            bg.critW2 = bg.critW.copy()
                            bg.actW2 = bg.actW.copy()
                            if SL == True:
                                cb.w2 = cb.w.copy()
                        elif maze.rewPos== maze.rewList[3]:
                            bg.critW3 = bg.critW.copy()
                            bg.actW3 = bg.actW.copy()
                            if SL == True:
                                cb.w3 = cb.w.copy()
                        elif maze.rewPos== maze.rewList[4]:
                            bg.critW4 = bg.critW.copy()
                            bg.actW4 = bg.actW.copy()
                            if SL == True:
                                cb.w4 = cb.w.copy()
                        elif maze.rewPos== maze.rewList[5]:
                            bg.critW5 = bg.critW.copy()
                            bg.actW5 = bg.actW.copy()
                            if SL == True:
                                cb.w5 = cb.w.copy()
                        elif maze.rewPos== maze.rewList[6]:
                            bg.critW6 = bg.critW.copy()
                            bg.actW6 = bg.actW.copy()
                            if SL == True:
                                cb.w6 = cb.w.copy()
                        elif maze.rewPos== maze.rewList[7]:
                            bg.critW7 = bg.critW.copy()
                            bg.actW7 = bg.actW.copy()
                            if SL == True:
                                cb.w7 = cb.w.copy()
                        elif maze.rewPos== maze.rewList[8]:
                            bg.critW8 = bg.critW.copy()
                            bg.actW8 = bg.actW.copy()
                            if SL == True:
                                cb.w8 = cb.w.copy()
                                
                if bg.actRew == 1:
             #       print "WINNER************", trial%len(maze.rewPosList)
                    epochAccurancy[trial%len(maze.rewPosList)] = 1.
                  #  print trial%len(maze.rewPosList)
                    if SL == True:
                        epochGoalTime[trial%len(maze.rewPosList)] = (step +1) / 10.# storage min movements to get reward                 
                    #        print epochGoalTime[trial]
                    else:
                        epochGoalTime[trial%len(maze.rewPosList)] = (step +1) / 2.
               #     print "WINNER************"# epochAccurancy, epochGoalTime    
                    break
                
            if bg.actRew == 0:    
          #      print trial%len(maze.rewPosList)
              #  print "loser************",trial%len(maze.rewPosList)
                epochAccurancy[trial%len(maze.rewPosList)] = 0.
                if SL == True:            
                    epochGoalTime[trial%len(maze.rewPosList)] = maxStep/10.  
               #         print epochGoalTime[trial]
                else:
                    epochGoalTime[trial%len(maze.rewPosList)] = maxStep/ 2.
              #  print "LOSER" #epochAccurancy, epochGoalTime
             #   break
            
          #  print trial % len(maze.rewPosList), len(maze.rewPosList)-1
     #       if trial % len(maze.rewPosList) == len(maze.rewPosList)-1:
      #          print "************"
            epochAvgAcc = (float(np.sum(epochAccurancy)) / len(maze.rewPosList)) * 100
            epochAvgTime = (float(np.sum(epochGoalTime)) / len(maze.rewPosList))               
         #   print "epoch" , epoch, "avarage steps", epochAvgTime, "average ACC", epochAvgAcc
            
            avg6EpochAcc[epoch%6] = epochAvgAcc
            avg6EpochTime[epoch%6] = epochAvgTime
            #    print "epoch" , epoch, "avarage steps", epochAvgTime, "average ACC", avg6EpochAcc[epoch%6]             
            if returnGoal == True:
                if bg.actRew == 0:
               # print "LOSERRRRR"
                    break
           
        if epoch % 6 == 5:
            finalAvgAcc[(epoch/6)%(maxEpoch/6)] = np.sum(avg6EpochAcc) / 6.
            finalAvgTime[(epoch/6)%(maxEpoch/6)] = np.sum(avg6EpochTime) / 6.
            print "******avg 6 epoch" , epoch, "avarage steps", np.sum(avg6EpochTime) / 6., "accurancy" , np.sum(avg6EpochAcc) / 6., "%"
            
    plt.figure(figsize=(120, 4), num=3, dpi=160)
    plt.title('average time in 100 epoch')
    plt.xlim([0, maxEpoch/6])
    if SL == True:
        plt.ylim([0, maxStep/10.])
    else:
        plt.ylim([0, maxStep/2.])
    plt.xlabel("epochs")
    plt.ylabel("s")
    plt.xticks(np.arange(0,maxEpoch/6, 25))
    plt.plot(finalAvgTime)
    
    plt.figure(figsize=(120, 4), num=4 ,dpi=160)
    plt.title('% accurancy')
    plt.xlim([0,maxEpoch/6])
    plt.ylim([0,101])
    plt.xlabel("epochs")
    plt.ylabel("accurancy %")
    plt.xticks(np.arange(0,maxEpoch/6, 25))
    plt.plot(finalAvgAcc)
                
            