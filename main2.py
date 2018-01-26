# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:44:07 2018

@author: Alex
"""

import numpy as np
import matplotlib.pyplot as plt
import math_utils as utils
from basalGanglia import ActorCritic
from cerebellum import Cerebellum 
from wrist import Wrist
from mpl_toolkits.mplot3d import *
from maze import Maze8Arm 
import os

# NET OPTIONS
TRAINING = True
GANGLIA_NOISE = True
MULTINET = False


SL = True
CEREBELLUM = True
INTRALAMINAR_NUCLEI = True
FASTLEARNING = False

### INPUTS
VISION = True
GOAL_VISION= True
AGENT_VISION = True
WRIST_PROPRIOCEPTION = True


loadData = False
saveData = True

randomGoal = False
returnGoal = True
perturbation = False
pertMag = 0.

maxEpoch = 1200
maxStep = 50

ACTOR_CEREBELLUM_RATIO = 4
CRITIC_CEREBELLUM_RATIO = 1

if CEREBELLUM == True:
    K = 1. / ACTOR_CEREBELLUM_RATIO
else:
    K = 1.
    
startPerturbation = 1500
endPerturbation = 1500

startPlotting = 1195


if __name__ == "__main__":
    
    # create objects
    bg = ActorCritic()
    wrist = Wrist()
    maze = Maze8Arm()
    
    #init objects
    maze.init(returnGoal, maxStep, maxEpoch)
    wrist.init(maxStep, maze.maxTrial, maxEpoch)
    bg.init(VISION ,AGENT_VISION, GOAL_VISION, WRIST_PROPRIOCEPTION,  MULTINET, maxStep, wrist.wristRange, maze.rewList)
    
    if CEREBELLUM == True:
        cb = Cerebellum()
        cb.init(MULTINET, bg.currState, bg.visionState, bg.wristState, maze.rewList, bg.activeSistems)
    
    avg10EpochAcc = np.zeros(10)
    avg10EpochTime = np.ones(10) * (maxStep /10)
    finalAvgAcc = np.zeros(maxEpoch/10)
    finalAvgTime = np.ones(maxEpoch/10) * (maxStep /10)
    epochAccurancy = np.zeros(len(maze.rewPosList))
    epochGoalTime = np.ones(len(maze.rewPosList)) * (maxStep /10)
    epochAvgAcc = np.zeros(1)
    epochAvgTime = np.zeros(1)
    
    avg10EpochFwdVisionError = np.zeros(10)
    finalAvgFwdVisionError = np.zeros(maxEpoch/10)
    epochFwdVisionError= np.zeros(len(maze.rewPosList))
    epochAvgFwdVisionError = np.zeros(1)
        
    # load saved weights
    if loadData == True:
        
        mydir = os.getcwd
        os.chdir("C:\Users\Alex\Desktop\lastVersion\data")
        
        if SL == True:       
            if MULTINET == True:       
                if CEREBELLUM == True:                   
                    if INTRALAMINAR_NUCLEI == True:                
                        bg.multiActW == np.load('bg.multinetActWIntralaminar.npy')
                        bg.multiCritW = np.load('bg.multinetCritWIntralaminar.npy')
                        cb.multiCerebW= np.load('cb.multinetWIntralaminar.npy')
                    else:                         
                        bg.multiActW == np.load('bg.multinetActWCereb.txt')
                        bg.multiCritW = np.load('bg.multinetCritWCereb.txt')
                        cb.multiCerebW= np.load('cb.multinetWCereb.txt')
                else:                    
                    bg.multiActW == np.load('bg.multinetActWGanglia.txt')
                    bg.multiCritW = np.load('bg.multinetCritWGanglia.txt')
            else:   
                if CEREBELLUM == True:
                    if INTRALAMINAR_NUCLEI == True:
                        bg.actW = np.load('bg.uninetActWIntralaminar.txt')
                        bg.critW = np.load('bg.uninetCritWIntralaminar.txt')
                        cb.w = np.load('cb.uninetWIntralaminar.txt')
                    else:
                        bg.actW = np.load('bg.uninetActWCereb.txt')
                        bg.critW = np.load('bg.uninetCritWCereb.txt')
                        cb.w = np.load('cb.uninetWCereb.txt')
                else:
                    bg.actW = np.load('bg.uninetActWGanglia.txt')
                    bg.critW = np.load('bg.uninetCritWGanglia.txt')
                
    # START ITERATING EPOCHS 
    for epoch in xrange(maxEpoch):
        
        if (epoch >= startPerturbation and epoch <= endPerturbation):
            perturbation = True
        else: 
            perturbation = False
        
        epochAccurancy = np.zeros(len(maze.rewPosList))
        epochGoalTime = np.ones(len(maze.rewPosList)) * (maxStep /10)
     
        maze.curr2DPos = np.array([0. , 0.])
        
        bg.epochReset()     
                
        if CEREBELLUM == True:
            cb.epochReset()
        
        wrist.epochReset()
        
        
        # temperature magnitude

        if epochAvgAcc < 100.0:    
            T = 1. - epoch / float(maxEpoch) #* utils.clipped_exp((- epoch * 1.) / float(maxEpoch))
       #     print T
        else:
            T = 1. - epochAvgAcc / 100.# + 0.01
        
        
        # INIT PLOTTING
        if epoch == startPlotting:
            
            #♣ INIT plotting         
            fig1   = plt.figure("Workspace",figsize=(80,80), dpi=120)
            gs = plt.GridSpec(1, 2, width_ratios=[3, 3]) 
            
            # add counters
            text1 = plt.figtext(.22, .02, "trial = %s" % (0), style='italic', bbox={'facecolor':'green'})
            text2 = plt.figtext(.12, .02, "step = %s" % (0), style='italic', bbox={'facecolor':'red'})
            text3 = plt.figtext(.02, .02, "epoch = %s" % (0), style='italic', bbox={'facecolor':'yellow'})
            
            
            # INIT 2d PLOTTING
            ax1 = fig1.add_subplot(gs[0])
            ax1.set_xlim([-1,1])
            ax1.set_ylim([-1,1]) 
            
            circle1 = plt.Circle((0.0 , 0.85), 0.15, color = 'yellow') 
            edgecircle1 = plt.Circle((0.0 , 0.85), 0.15, color = 'black', fill = False)  
            circle2 = plt.Circle((0.6 , 0.6), 0.15, color = 'yellow')
            edgecircle2 = plt.Circle((0.6 , 0.6), 0.15, color = 'black', fill = False)  
            circle3 = plt.Circle((0.85 , 0.0), 0.15, color = 'yellow')
            edgecircle3 = plt.Circle((0.85 , 0.0), 0.15, color = 'black', fill = False)
            circle4 = plt.Circle((0.6 , -0.6), 0.15, color = 'yellow')
            edgecircle4 = plt.Circle((0.6 , -0.6), 0.15, color = 'black', fill = False)
            circle5 = plt.Circle((0.0 , -0.85), 0.15, color = 'yellow')
            edgecircle5 = plt.Circle((0.0 , -0.85), 0.15, color = 'black', fill = False)
            circle6 = plt.Circle((-0.6 , -0.6), 0.15, color = 'yellow')
            edgecircle6 = plt.Circle((-0.6 , -0.6), 0.15, color = 'black', fill = False)
            circle7 = plt.Circle((-0.85 , 0.0), 0.15, color = 'yellow')
            edgecircle7 = plt.Circle((-0.85 , 0.0), 0.15, color = 'black', fill = False)
            circle8 = plt.Circle((-0.6 , 0.6), 0.15, color = 'yellow')
            edgecircle8 = plt.Circle((-0.6 , 0.6), 0.15, color = 'black', fill = False)            
            circle9 = plt.Circle((0.0 , 0.0), 0.15, color = 'yellow')
            edgecircle9 = plt.Circle((0.0 , 0.0), 0.15, color = 'black', fill = False)
            
            ax1.add_artist(circle1)
            ax1.add_artist(edgecircle1)
            ax1.add_artist(circle2)
            ax1.add_artist(edgecircle2)
            ax1.add_artist(circle3)
            ax1.add_artist(edgecircle3)
            ax1.add_artist(circle4)
            ax1.add_artist(edgecircle4)
            ax1.add_artist(circle5)
            ax1.add_artist(edgecircle5)
            ax1.add_artist(circle6)
            ax1.add_artist(edgecircle6)
            ax1.add_artist(circle7)
            ax1.add_artist(edgecircle7)
            ax1.add_artist(circle8)
            ax1.add_artist(edgecircle8)          
            ax1.add_artist(circle9)
            ax1.add_artist(edgecircle9)
            
            #plot agent & reward
            reward, = ax1.plot(maze.rewPos[0] , maze.rewPos[1], 'x', color='r')
            agent, = ax1.plot(maze.curr2DPos[0], maze.curr2DPos[1], 'o')
            agentCircle = plt.Circle((maze.curr2DPos[0] , maze.curr2DPos[1]), 0.15, color = 'blue')
            rewardCircle = plt.Circle((maze.rewPos[0], maze.rewPos[1]), 0.15, color = 'red')
            
            ax1.add_artist(agentCircle)
            ax1.add_artist(rewardCircle)
            
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
            ax2.set_xlabel("PS")
            ax2.set_ylabel("RUD")
            ax2.set_zlabel("FE")
            
            # plot agent
            X = [0.5,0.5,0.5] 
            Y = [0.0,0.5,0.5]
            Z = [0.5,0.5,0.5]
            agent3d, = ax2.plot3D(X, Y, Z, color="blue")
            
        if epoch > startPlotting:                
            text3.set_text("epoch = %s" % (epoch+1))
            plt.pause(0.01)        
            
        for trial in xrange(maze.maxTrial):

            maze.posBuff *= 0
            wrist.posBuff *= 0
            maze.setGoal(trial)
            
       #     maze.rewPos = maze.rewPosList[0].copy()
            
            bg.trialReset()
            
            if CEREBELLUM == True:
                cb.trialReset()
                    
            if returnGoal == False:                  
                maze.curr2DPos = np.array([0. , 0.])
                wrist.trialReset()
                
            else: 
                if trial == 0:
                    maze.curr2DPos = np.array([0. , 0.])
                    wrist.trialReset()
                else:
                    wrist.prvEp =  wrist.ep.copy()
             #○       wrist.ep = wrist.curr3DPos.copy()

            if epoch > startPlotting:
                text1.set_text("trial = %s" % (trial+1))
                rewardCircle.remove()
                rewardCircle = plt.Circle((maze.rewPos[0], maze.rewPos[1]), 0.15, color = 'red')
                ax1.add_artist(rewardCircle)
                agentCircle.remove()
                agentCircle = plt.Circle((maze.curr2DPos[0] , maze.curr2DPos[1]), 0.15, color = 'blue')
                ax1.add_artist(agentCircle)
                reward.set_data(maze.rewPos[0], maze.rewPos[1])
                plt.pause(0.01)
            
            # LOAD NET 
            if MULTINET == True:           
                bg.critW *= 0
                bg.actW *= 0
                if CEREBELLUM == True:
                    cb.w *= 0    
                for i in xrange(len(maze.rewList)):
               #     print i
                    if maze.rewPos == maze.rewList[i]:
                        bg.critW = bg.multiCritW[:,i].copy()
                        bg.actW = bg.multiActW[:,:,i].copy()
                        if CEREBELLUM == True:
                            cb.w = cb.multiCerebW[:,:,i].copy()
                            cb.fwdVisionW = cb.multiFwdVisionW[:,:,i].copy()
                        break
            
            # GOAL VISION
            if VISION == True:
                if GOAL_VISION == True:
                    bg.compGoalVisionState(utils.change_range(np.array([maze.rewPos]), -1., 1., 0., 1.))
            #    print 1 , maze.rewPos , bg.goalVisionState, "*****************************"
                    if AGENT_VISION == True:
                        bg.visionState = np.hstack([bg.agentVisionState, bg.goalVisionState]).copy()
                    else:
                        bg.visionState = bg.goalVisionState.copy()
                        
            # start steps
            for step in xrange(maxStep):
                
                if perturbation == True:
                    if cb.trialFwdError > 1:
                        T = 1.
                
                if step > 0:  
                    bg.prvCritOut = bg.currCritOut.copy() 
                    bg.prvState = bg.currState.copy() 
                    
                    if SL == True:
                        if CEREBELLUM == True:
                            cb.prvFwdVisionError = cb.fwdVisionError.copy()
                            cb.prvFwdVisionState = cb.fwdVisionState.copy()                            
                            wrist.prvEp = wrist.ep.copy()
                            bg.desOutBuff[:,step] = wrist.prvEp.copy()  
                            bg.stateBuff[:,step] = bg.prvState.copy()
                            if INTRALAMINAR_NUCLEI == True: 
                                bg.prvTrainOut = bg.trainOut.copy()
                                cb.prvOut = cb.currOut.copy()       
         
                if step > (ACTOR_CEREBELLUM_RATIO - 1):    
                    if step % ACTOR_CEREBELLUM_RATIO == 1:  
                        bg.prvNoise = bg.currNoise.copy()
 
    
                            
             
    
                            
                
                wrist.move3d()
                
         #       wrist.curr3DPos = np.array([0.5,1.,0.5])
                
                maze.curr2DPos = utils.conversion2d(utils.change_range(wrist.curr3DPos , 0, 1, -1, 1), perturbation, pertMag)

                
                maze.posBuff[:,step] = maze.curr2DPos.copy()
                wrist.posBuff[:,step] = wrist.curr3DPos.copy()
                
                if epoch > startPlotting:   
                    text2.set_text("step = %s" % (step+1))
                    agent.set_data(maze.curr2DPos[0], maze.curr2DPos[1])
                    agentCircle.remove()
                    agentCircle = plt.Circle((maze.curr2DPos[0] , maze.curr2DPos[1]), 0.15, color = 'blue')
                    ax1.add_artist(agentCircle)
                    X = [0.5,0.5,wrist.curr3DPos[0]]
                    Y = [0.0,0.5,wrist.curr3DPos[1]]
                    Z = [0.5,0.5,wrist.curr3DPos[2]]
                    agent3d.remove()
                    agent3d, = ax2.plot3D(X, Y, Z, color="blue")
                    plt.pause(0.01) 
                    
                    
                    
                    
                    
                    
                if WRIST_PROPRIOCEPTION == True:
                    bg.compWristState(wrist.curr3DPos)
                    

                if VISION == True:
                    if AGENT_VISION == True:
                        
                        
                        if FASTLEARNING == True:
                            if perturbation == False:
                                bg.compAgentVisionState(utils.change_range(maze.curr2DPos, -1., 1., 0., 1.))
                            else:
                                bg.compAgentVisionState(utils.change_range(maze.curr2DPos, -1., 1., 0., 1.))
                        else:
                            bg.compAgentVisionState(utils.change_range(maze.curr2DPos, -1., 1., 0., 1.))
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
                        
                
                if CEREBELLUM == True:
                    cb.fwdVisionError = np.sum(np.sqrt((bg.visionState - cb.estVision)**2))
                    cb.trialFwdError += cb.fwdVisionError 
                    
                        
                        
                            
                maze.computeDistance()
                
                if step > 0:
                    
                    if maze.distance < 0.15:
                        bg.actRew = 1
                        bg.currCritOut *= 0.
                    else:
                        bg.spreadCrit()

                    bg.compSurprise()  
                
                
                    
                maze.posBuff[:,step] = maze.curr2DPos.copy()
                wrist.posBuff[:,step] = wrist.curr3DPos.copy()
                
                
                  
                
                if CEREBELLUM == True:
                    if step % ACTOR_CEREBELLUM_RATIO == 0:
                        bg.spreadAct()       
                        if GANGLIA_NOISE== True:
                            bg.computate_noise(T)  
                    cb.spreading(bg.currState)                    
                    netOut = (K * bg.currActOut) + ((1-K) * cb.currOut)
                    wrist.ep = netOut + bg.currNoise   
                    if INTRALAMINAR_NUCLEI == True:
                        bg.trainOut = utils.sigmoid(np.dot(bg.actW.T, bg.currState))                          
                else:
                    if step % ACTOR_CEREBELLUM_RATIO == 0:
                        bg.spreadAct()
                        if GANGLIA_NOISE== True:
                            bg.computate_noise(T)  
                    netOut = K * bg.currActOut
                    wrist.ep = netOut + bg.currNoise
                    
              #  print bg.currNoise    
                    
                
                if CEREBELLUM == True:
                    bg.compEpState(wrist.ep)
                    cb.fwdVisionState = np.hstack([bg.visionState, bg.epState])
                    cb.fwdVision()
                        
                        
                        
                if TRAINING == True:
                    if step > 0: 
                        
                        if INTRALAMINAR_NUCLEI == True:    
                            bg.actW += bg.ACT_ETA * bg.surp * (np.outer(bg.prvState , K * bg.prvNoise) +  np.outer(bg.prvState, (1-K) * T * (cb.prvOut - bg.prvTrainOut)))
                        else:                            
                            bg.actW += bg.ACT_ETA * bg.surp * np.outer(bg.prvState, bg.prvNoise)
               #         if (step % CRITIC_CEREBELLUM_RATIO == 0):
                        bg.critW += bg.CRIT_ETA * bg.surp * bg.prvState
                            
                             
                        
                    if CEREBELLUM == True:
                        cb.trainFwdVision(bg.visionState)
                        if (bg.actRew == 1):
                            for i in xrange(step+1): 
                                cb.trainCb(bg.stateBuff[:,i], bg.desOutBuff[:,i], T)
                              #      cb.trainCb2(bg.stateBuff[:,i], bg.desOutBuff[:,i], bg.surpBuff[i], T)
                                    
                        
                        
                        
                            
                    if MULTINET == True:
                        for i in xrange(len(maze.rewList)):
                            if maze.rewPos == maze.rewList[i]:
                                bg.multiCritW[:,i] = bg.critW.copy()
                                bg.multiActW[:,:,i] = bg.actW.copy()
                                if CEREBELLUM == True:
                                    cb.multiCerebW[:,:,i] = cb.w.copy()
                                    cb.multiFwdVisionW[:,:,i] = cb.fwdVisionW.copy()
                                break
                                                                            
                           
                            
                maze.trajectories[:,:,trial,epoch] = maze.posBuff.copy()
                wrist.trajectories[:,:,trial,epoch] = wrist.posBuff.copy()
                
                                
                if bg.actRew == 1:
                    epochAccurancy[trial%len(maze.rewPosList)] = 1.
                    epochGoalTime[trial%len(maze.rewPosList)] = (step +1) / 10.# storage min movements to get reward                 
                    if CEREBELLUM == True:
                        epochFwdVisionError[trial%len(maze.rewPosList)] = cb.trialFwdError / float(step)
                    break
                
                
                
            if bg.actRew == 0:    
                epochAccurancy[trial%len(maze.rewPosList)] = 0.          
                epochGoalTime[trial%len(maze.rewPosList)] = maxStep/10.  
                if CEREBELLUM == True:
                    epochFwdVisionError[trial%len(maze.rewPosList)] = cb.trialFwdError / float(maxStep)
             
            epochAvgAcc = (float(np.sum(epochAccurancy)) / len(maze.rewPosList)) * 100
            epochAvgTime = (float(np.sum(epochGoalTime)) / len(maze.rewPosList)) 
            if CEREBELLUM == True:
                epochAvgFwdVisionError = (float(np.sum(epochFwdVisionError)) / len(maze.rewPosList))              
            
            avg10EpochAcc[epoch%10] = epochAvgAcc
            avg10EpochTime[epoch%10] = epochAvgTime
            if CEREBELLUM == True:              
                avg10EpochFwdVisionError[epoch%10] = epochAvgFwdVisionError
           
        if epoch % 10 == 9:
            finalAvgAcc[(epoch/10)%(maxEpoch/10)] = np.sum(avg10EpochAcc) / 10.
            finalAvgTime[(epoch/10)%(maxEpoch/10)] = np.sum(avg10EpochTime) / 10.
            if CEREBELLUM == True:
                finalAvgFwdVisionError[(epoch/10)%(maxEpoch/10)] = np.sum(avg10EpochFwdVisionError) / 10.
            print "******avg 10 epoch" , epoch, "avarage steps", np.sum(avg10EpochTime) / 10., "accurancy" , np.sum(avg10EpochAcc) / 10., "%"
            
    plt.figure(figsize=(120, 4), num=3, dpi=160)
    plt.title('average time in 10 epoch')
    plt.xlim([0, maxEpoch/10])
    plt.ylim([0, maxStep/10.])
    plt.xlabel("epochs")
    plt.ylabel("s")
    plt.xticks(np.arange(0,maxEpoch/10, 25))
    plt.plot(finalAvgTime)
    
    plt.figure(figsize=(120, 4), num=4 ,dpi=160)
    plt.title('% accurancy')
    plt.xlim([0,maxEpoch/10])
    plt.ylim([0,101])
    plt.xlabel("epochs")
    plt.ylabel("accurancy %")
    plt.xticks(np.arange(0,maxEpoch/10, 25))
    plt.plot(finalAvgAcc)
    
    if CEREBELLUM == True:
        plt.figure(figsize=(120, 4) ,num=5, dpi=160)
        plt.title('avg FWDvision Error')
        plt.xlim([0,maxEpoch/10])
        plt.xlabel("epochs")
        plt.ylabel("FWD Vision Error")
        plt.xticks(np.arange(0,maxEpoch/10, 25))
        plt.plot(finalAvgFwdVisionError) 
    
    
    
    if saveData == True:
        mydir = os.getcwd
        os.chdir("C:\Users\Alex\Desktop\lastVersion\data")
        
        if SL == True:
        
            if MULTINET == True:
                
                if CEREBELLUM == True:                  
                    if INTRALAMINAR_NUCLEI == True:
                        np.save("bg.multinetActWCerebNoise", (bg.multiActW))
                        np.save("bg.multinetCritWCerebNoise", (bg.multiCritW))
                        np.save("cb.multinetWCerebNoise", (cb.multiCerebW))           
                        np.save("maze.multinetCerebNoiseTrajectories", (maze.trajectories)) 
                        np.save("wrist.multinetCerebNoiseTrajectories", (wrist.trajectories))
                    else:                   
                        np.save("bg.multinetActWCereb.txt", (bg.multiActW))
                        np.save("bg.multinetCritWCereb.txt", (bg.multiCritW))
                        np.save("cb.multinetWCereb.txt", (cb.multiCerebW))
                        np.save("maze.multinetCerebTrajectories", (maze.trajectories))
                        np.save("wrist.multinetCerebTrajectories", (wrist.trajectories))
                else:                   
                    np.save("bg.multinetActWGanglia.txt", (bg.multiActW))
                    np.save("bg.multinetCritWGanglia.txt", (bg.multiCritW))
                    np.save("maze.multinetGangliaTrajectories", (maze.trajectories))
                    np.save("wrist.multinetGangliaTrajectories", (wrist.trajectories))
        
            else:
                if CEREBELLUM == True:
                    if INTRALAMINAR_NUCLEI == True:
                        np.save("bg.uninetActWCerebNoise.txt", (bg.actW))
                        np.save("bg.uninetCritWCerebNoise.txt", (bg.critW))
                        np.save("cb.uninetWCerebNoise.txt", (cb.w))
                        np.save("maze.uninetCerebNoiseTrajectories", (maze.trajectories)) 
                        np.save("wrist.uninetCerebNoiseTrajectories", (wrist.trajectories))
                    else:
                        np.save("bg.uninetActWCereb.txt", (bg.actW))
                        np.save("bg.uninetCritWCereb.txt", (bg.critW))
                        np.save("cb.uninetWCereb.txt", (cb.w))
                        np.save("maze.uninetCerebTrajectories", (maze.trajectories)) 
                        np.save("wrist.uninetCerebTrajectories", (wrist.trajectories))
                else:
                    np.save("bg.uninetGangliaActW.txt", (bg.actW))
                    np.save("bg.uninetGangliaCritW.txt", (bg.critW))                        
                    np.save("maze.uninetGangliaTrajectories", (maze.trajectories))
                    np.save("wrist.uninetGangliaTrajectories", (wrist.trajectories))